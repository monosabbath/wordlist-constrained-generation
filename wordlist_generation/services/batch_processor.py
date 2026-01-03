import json
import os
import time
import uuid
import logging
from typing import Any, Dict, List

import torch
from fastapi import BackgroundTasks, HTTPException
from pydantic import ValidationError

from wordlist_generation.api.routers.models import ChatCompletionRequest
from wordlist_generation.inference.generation import (
    extract_and_reorder_messages,
    normalize_max_new_tokens,
    getgen_kwargs,
)
from wordlist_generation.inference.vocab_constraints.constraints import (
    get_stop_ids,
    build_regexp_prefix_fn,
)

logger = logging.getLogger("batch-jobs")

# Job expiration time in seconds (24 hours)
JOB_EXPIRATION_SECONDS = 24 * 60 * 60


class BatchProcessor:
    def __init__(self, settings, model_service):
        self.settings = settings
        self.model_service = model_service
        self.job_status: Dict[str, Dict[str, Any]] = {}

    def _cleanup_expired_jobs(self):
        """Remove jobs older than JOB_EXPIRATION_SECONDS."""
        cutoff = int(time.time()) - JOB_EXPIRATION_SECONDS
        expired = [
            jid for jid, info in self.job_status.items()
            if info.get("submitted_at", 0) < cutoff
        ]
        for jid in expired:
            info = self.job_status.pop(jid, {})
            # Clean up output file if it exists
            output_path = info.get("output_path")
            if output_path and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired job(s).")

    def _process_job(
        self,
        job_id: str,
        input_path: str,
        output_path: str,
        job_config: Dict[str, Any],
    ):
        logger.info(f"[Job {job_id}] Starting processing...")
        try:
            self.job_status[job_id]["status"] = "processing"

            # 1. Read and parse the input file
            with open(input_path, "r", encoding="utf-8") as f:
                try:
                    raw_requests = json.load(f)
                    if not isinstance(raw_requests, list):
                        raise ValueError("Input file must contain a JSON list.")
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON: {e}")

            # 2. Prepare prompts/messages
            logger.info(f"[Job {job_id}] Preparing {len(raw_requests)} prompts...")
            messages_list: List[List[Dict[str, str]]] = []
            custom_ids: List[str | None] = []
            for i, req_data in enumerate(raw_requests):
                try:
                    req = ChatCompletionRequest(**req_data)
                    messages = extract_and_reorder_messages(req.messages)
                    messages_list.append(messages)
                    custom_ids.append(req.custom_id)
                except ValidationError as e:
                    logger.warning(f"[Job {job_id}] Skipping request {i}: Invalid format. {e}")
                except Exception as e:
                    logger.warning(f"[Job {job_id}] Skipping request {i}: Error processing. {e}")

            # 3. Shared generation kwargs
            tokenizer = self.model_service.tokenizer
            model = self.model_service.model
            stop_ids = get_stop_ids(tokenizer)
            
            max_new_tokens = normalize_max_new_tokens(job_config.get("max_tokens", 512), self.settings.ALLOWED_MAX_NEW_TOKENS)
            
            # Constrained vocab prefix
            vocab_lang = job_config.get("vocab_lang")
            vocab_n_words = job_config.get("vocab_n_words")
            pf = None
            if vocab_lang and vocab_n_words:
                logger.info(f"[Job {job_id}] Building constrained vocab for {vocab_lang} ({vocab_n_words} words)")
                pf = build_regexp_prefix_fn(
                    tokenizer=tokenizer,
                    lang=vocab_lang,
                    n_words=vocab_n_words,
                    wordlist_dir=self.settings.WORDLIST_DIR,
                )
                if not pf:
                    raise ValueError(f"Constrained vocabulary config failed for lang '{vocab_lang}'.")

            gen_kwargs = getgen_kwargs(
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                stop_ids=stop_ids,
                num_beams=job_config.get("num_beams", 10),
                length_penalty=job_config.get("length_penalty", 1.0),
                prefix_fn=pf,
                temperature=job_config.get("temperature"),
                top_p=job_config.get("top_p"),
                top_k=job_config.get("top_k"),
                repetition_penalty=job_config.get("repetition_penalty"),
            )

            # 4. Run generation
            logger.info(f"[Job {job_id}] Running generation...")
            results: List[str] = []

            # Build inputs in batches
            batch_size = self.settings.BATCH_JOB_PIPELINE_SIZE
            for i in range(0, len(messages_list), batch_size):
                batch_messages = messages_list[i:i + batch_size]
                
                inputs = tokenizer.apply_chat_template(
                    batch_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.settings.MAX_INPUT_TOKENS,
                ).to(model.device)
                
                input_len = inputs["input_ids"].shape[1]

                with self.model_service.gpu_gate:
                    with torch.inference_mode():
                        outputs = model.generate(**inputs, **gen_kwargs)
                
                if hasattr(outputs, "sequences"):
                    generated_sequences = outputs.sequences
                else:
                    generated_sequences = outputs

                for seq in generated_sequences:
                    text = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
                    results.append(text)

            # 5. Save output file
            logger.info(f"[Job {job_id}] Formatting and saving results...")
            created = int(time.time())
            final_output = []
            for i, text in enumerate(results):
                resp = {
                    "id": f"chatcmpl-batch-{job_id}-{i}",
                    "custom_id": custom_ids[i] if i < len(custom_ids) else None,
                    "object": "chat.completion",
                    "created": created,
                    "model": self.settings.MODEL_NAME,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "total_tokens": None,
                    },
                }
                final_output.append(resp)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_output, f, indent=2)

            self.job_status[job_id]["status"] = "completed"
            logger.info(f"[Job {job_id}] Processing complete.")
        except Exception as e:
            logger.error(f"[Job {job_id}] Processing FAILED: {e}")
            self.job_status[job_id]["status"] = "failed"
            self.job_status[job_id]["error"] = str(e)
        finally:
            try:
                if os.path.exists(input_path):
                    os.remove(input_path)
                    logger.info(f"[Job {job_id}] Cleaned up input file.")
            except Exception as e:
                logger.warning(f"[Job {job_id}] Failed to clean up input file: {e}")

    def enqueue(
        self,
        background_tasks: BackgroundTasks,
        file,
        max_tokens: int,
        num_beams: int,
        length_penalty: float,
        vocab_lang: str | None,
        vocab_n_words: int | None,
        # Sampling params
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ):
        # Clean up expired jobs periodically
        self._cleanup_expired_jobs()

        job_id = str(uuid.uuid4())
        input_path = os.path.join(self.settings.BATCH_JOB_TEMP_DIR, f"{job_id}_input.json")
        output_path = os.path.join(self.settings.BATCH_JOB_TEMP_DIR, f"{job_id}_output.json")

        try:
            with open(input_path, "wb") as f:
                f.write(file.file.read())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save input file: {e}")

        job_config = {
            "max_tokens": max_tokens,
            "num_beams": num_beams,
            "length_penalty": length_penalty,
            "vocab_lang": vocab_lang,
            "vocab_n_words": vocab_n_words,
            # Sampling params
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }

        self.job_status[job_id] = {
            "status": "pending",
            "input_path": input_path,
            "output_path": output_path,
            "submitted_at": int(time.time()),
            "config": job_config,
        }

        # Schedule processing; GPU-bound work will be serialized by the gate
        background_tasks.add_task(
            self._process_job, job_id, input_path, output_path, job_config
        )

        return {
            "job_id": job_id,
            "status": "pending",
            "message": "Batch job accepted and queued for processing.",
        }
