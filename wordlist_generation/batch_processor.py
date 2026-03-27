import json
import os
import time
import uuid
from typing import Any, Dict, List

from fastapi import BackgroundTasks, HTTPException
from pydantic import ValidationError

from wordlist_generation.api.routers.models import ChatCompletionRequest
from wordlist_generation.inference.runner import (
    prepare_messages,
    build_chat_inputs,
    build_prefix_fn,
    build_presence_penalty_processor,
    build_soft_constraint_setup,
    build_generation_kwargs,
    unwrap_generated_sequences,
    decode_sequences,
    generate_sequences,
)
from wordlist_generation.inference.vocab_constraints.constraints import get_stop_ids


JOB_EXPIRATION_SECONDS = 24 * 60 * 60


class BatchProcessor:
    def __init__(self, settings, model_service):
        self.settings = settings
        self.model_service = model_service
        self.job_status: Dict[str, Dict[str, Any]] = {}

    def _cleanup_expired_jobs(self):
        cutoff = int(time.time()) - JOB_EXPIRATION_SECONDS
        expired = [jid for jid, info in self.job_status.items() if info.get("submitted_at", 0) < cutoff]
        for jid in expired:
            info = self.job_status.pop(jid, {})
            output_path = info.get("output_path")
            if output_path and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass

    def _process_job(
        self,
        job_id: str,
        input_path: str,
        output_path: str,
        job_config: Dict[str, Any],
    ):
        try:
            self.job_status[job_id]["status"] = "processing"

            with open(input_path, "r", encoding="utf-8") as f:
                raw_requests = json.load(f)
                if not isinstance(raw_requests, list):
                    raise ValueError("Input file must contain a JSON list.")

            messages_list: List[List[Dict[str, str]]] = []
            custom_ids: List[str | None] = []
            for req_data in raw_requests:
                try:
                    req = ChatCompletionRequest(**req_data)
                    messages_list.append(prepare_messages(req.messages))
                    custom_ids.append(req.custom_id)
                except (ValidationError, Exception):
                    continue

            tokenizer = self.model_service.tokenizer
            model = self.model_service.model
            stop_ids = get_stop_ids(tokenizer)

            prefix_fn_n = build_prefix_fn(
                tokenizer=tokenizer,
                wordlist_dir=self.settings.WORDLIST_DIR,
                vocab_lang=job_config.get("vocab_lang"),
                vocab_n_words=job_config.get("vocab_n_words"),
            )
            if (job_config.get("vocab_lang") and job_config.get("vocab_n_words")) and not prefix_fn_n:
                raise ValueError(f"Constrained vocabulary config failed for lang '{job_config.get('vocab_lang')}'.")

            prefix_for_generate, vocab_logits_processor = build_soft_constraint_setup(
                tokenizer=tokenizer,
                settings=self.settings,
                vocab_lang=job_config.get("vocab_lang"),
                vocab_n_words=job_config.get("vocab_n_words"),
                prefix_fn_n=prefix_fn_n,
                vocab_constraint_mode=job_config.get("vocab_constraint_mode"),
                vocab_soft_tier2_max_rank_multiplier=job_config.get("vocab_soft_tier2_max_rank_multiplier"),
                vocab_soft_tier2_penalty=job_config.get("vocab_soft_tier2_penalty"),
                vocab_soft_tier3_penalty=job_config.get("vocab_soft_tier3_penalty"),
            )

            presence_penalty_val = job_config.get("presence_penalty")

            results: List[str] = []
            batch_size = self.settings.BATCH_JOB_PIPELINE_SIZE

            for i in range(0, len(messages_list), batch_size):
                batch_messages = messages_list[i : i + batch_size]
                inputs, input_len = build_chat_inputs(
                    tokenizer=tokenizer,
                    messages=batch_messages,
                    max_input_tokens=self.settings.MAX_INPUT_TOKENS,
                    device=model.device,
                    enable_thinking=self.settings.ENABLE_THINKING,
                )

                # Build per-batch logits processors (presence penalty needs prompt_len)
                batch_processors = list(vocab_logits_processor) if vocab_logits_processor else []
                pp = build_presence_penalty_processor(
                    presence_penalty=presence_penalty_val,
                    prompt_len=input_len,
                )
                if pp is not None:
                    batch_processors.append(pp)

                gen_kwargs, _max_new_tokens = build_generation_kwargs(
                    tokenizer=tokenizer,
                    allowed_max_new_tokens=self.settings.ALLOWED_MAX_NEW_TOKENS,
                    requested_max_tokens=job_config.get("max_tokens", 512),
                    stop_ids=stop_ids,
                    num_beams=job_config.get("num_beams", 10),
                    length_penalty=job_config.get("length_penalty", 1.0),
                    prefix_fn=prefix_for_generate,
                    logits_processor=batch_processors or None,
                    temperature=job_config.get("temperature"),
                    top_p=job_config.get("top_p"),
                    top_k=job_config.get("top_k"),
                    repetition_penalty=job_config.get("repetition_penalty"),
                )

                outputs = generate_sequences(model_service=self.model_service, inputs=inputs, gen_kwargs=gen_kwargs)
                generated_sequences = unwrap_generated_sequences(outputs)
                results.extend(
                    decode_sequences(
                        tokenizer=tokenizer,
                        generated_sequences=generated_sequences,
                        input_len=input_len,
                        stop_ids=stop_ids,
                    )
                )

            created = int(time.time())
            final_output = []
            for i, text in enumerate(results):
                final_output.append(
                    {
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
                        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                    }
                )

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_output, f, indent=2)

            self.job_status[job_id]["status"] = "completed"
        except Exception as e:
            self.job_status[job_id]["status"] = "failed"
            self.job_status[job_id]["error"] = str(e)
        finally:
            try:
                if os.path.exists(input_path):
                    os.remove(input_path)
            except Exception:
                pass

    def enqueue(
        self,
        background_tasks: BackgroundTasks,
        file,
        max_tokens: int,
        num_beams: int,
        length_penalty: float,
        vocab_lang: str | None,
        vocab_n_words: int | None,
        vocab_constraint_mode: str | None,
        vocab_soft_tier2_max_rank_multiplier: float | None,
        vocab_soft_tier2_penalty: float | None,
        vocab_soft_tier3_penalty: float | None,
        presence_penalty: float,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ):
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
            "vocab_constraint_mode": vocab_constraint_mode,
            "vocab_soft_tier2_max_rank_multiplier": vocab_soft_tier2_max_rank_multiplier,
            "vocab_soft_tier2_penalty": vocab_soft_tier2_penalty,
            "vocab_soft_tier3_penalty": vocab_soft_tier3_penalty,
            "presence_penalty": presence_penalty,
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

        background_tasks.add_task(self._process_job, job_id, input_path, output_path, job_config)
        return {"job_id": job_id, "status": "pending", "message": "Batch job accepted and queued for processing."}
