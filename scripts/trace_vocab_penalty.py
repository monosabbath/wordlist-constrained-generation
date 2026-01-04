from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper, TemperatureLogitsWarper

from wordlist_generation.model_service import ModelService
from wordlist_generation.settings import Settings
from wordlist_generation.inference.vocab_constraints.constraints import build_regexp_prefix_fn
from wordlist_generation.inference.vocab_constraints.logits_processor import SoftPrefixConstraintLogitsProcessor


@dataclass
class StepDiagnostics:
    penalty: float
    temperature: float
    top_k: int
    top_p: float
    allowed_size: int
    mass_allowed: float
    mass_disallowed: float
    in_topk_allowed: int
    in_topk_disallowed: int
    kth_logit: float


def _make_toy_wordlist_dir() -> str:
    d = tempfile.mkdtemp(prefix="toy_wordlists_")
    p = Path(d) / "toy.txt"
    # Small, common-ish set; the exact content is less important than having *some* constraint.
    p.write_text("\n".join(["the", "and", "to", "of", "a", "in", "is", "it", "you", "that", "hello", "world"]) + "\n", encoding="utf-8")
    return d


def _pick_lang(wordlist_dir: str, requested: str | None) -> str:
    if requested:
        return requested
    files = sorted(Path(wordlist_dir).glob("*.txt"))
    for f in files:
        return f.stem
    return "toy"


def _topk_partition(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, float]:
    # scores: [vocab]
    if k <= 0 or k >= int(scores.numel()):
        return torch.ones_like(scores, dtype=torch.bool), float("-inf")
    v, _ = torch.topk(scores, k)
    kth = float(v[-1].item())
    return scores >= v[-1], kth


def _softmax_mass(mask: torch.Tensor, probs: torch.Tensor) -> float:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    return float(probs[mask].sum().item())


def diagnose_one_step(
    *,
    ms: ModelService,
    prefix_fn,
    prompt: str,
    penalties: list[float],
    temperature: float,
    top_k: int,
    top_p: float,
):
    tok = ms.tokenizer
    model = ms.model
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    input_ids: torch.LongTensor = inputs["input_ids"]

    with torch.inference_mode():
        raw = model(**inputs).logits[:, -1, :].float().detach()  # [1, vocab]

    allowed_ids = set(prefix_fn(0, input_ids[0]))
    allowed_mask = torch.zeros(raw.shape[-1], dtype=torch.bool, device=raw.device)
    if allowed_ids:
        allowed_mask[torch.tensor(list(allowed_ids), device=raw.device, dtype=torch.long)] = True

    # Warpers are applied after processors during sampling.
    warpers = LogitsProcessorList()
    if temperature and temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(float(temperature)))
    if top_k and top_k > 0:
        warpers.append(TopKLogitsWarper(int(top_k)))
    if top_p and top_p < 1.0:
        warpers.append(TopPLogitsWarper(float(top_p)))

    out: list[StepDiagnostics] = []
    for p in penalties:
        proc = SoftPrefixConstraintLogitsProcessor(prefix_allowed_tokens_fn=prefix_fn, penalty=float(p))
        scores = proc(input_ids, raw.clone()).squeeze(0)

        # Simulate warpers as generate() would for sampling.
        warped = warpers(input_ids, scores.unsqueeze(0)).squeeze(0)

        probs = torch.softmax(warped, dim=-1)

        # top-k membership after warpers (the sharp, discrete part)
        topk_mask, kth = _topk_partition(warped, top_k)
        in_topk_allowed = int((topk_mask & allowed_mask).sum().item())
        in_topk_disallowed = int((topk_mask & (~allowed_mask)).sum().item())

        out.append(
            StepDiagnostics(
                penalty=float(p),
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                allowed_size=int(allowed_mask.sum().item()),
                mass_allowed=_softmax_mass(allowed_mask, probs),
                mass_disallowed=_softmax_mass(~allowed_mask, probs),
                in_topk_allowed=in_topk_allowed,
                in_topk_disallowed=in_topk_disallowed,
                kth_logit=float(kth),
            )
        )

    return out


def diagnose_one_step_beam_sampling(
    *,
    ms: ModelService,
    prefix_fn,
    prompt: str,
    penalties: list[float],
    temperature: float,
    top_k: int,
    top_p: float,
    num_beams: int,
    trials: int,
):
    tok = ms.tokenizer
    model = ms.model
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    input_ids: torch.LongTensor = inputs["input_ids"]

    with torch.inference_mode():
        logits = model(**inputs).logits[:, -1, :].float().detach()  # [1, vocab]
    log_probs = F.log_softmax(logits, dim=-1)  # what _beam_search starts from

    allowed_ids = set(prefix_fn(0, input_ids[0]))
    allowed_mask = torch.zeros(log_probs.shape[-1], dtype=torch.bool, device=log_probs.device)
    if allowed_ids:
        allowed_mask[torch.tensor(list(allowed_ids), device=log_probs.device, dtype=torch.long)] = True

    # Mimic the v5 beam_search path: processors are applied to log_probs.
    # Ask transformers to prepare a generation config (incl. special-token tensors) and build the processor chain.
    base_cfg, _ = model._prepare_generation_config(
        generation_config=None,
        do_sample=True,
        num_beams=int(num_beams),
        num_return_sequences=1,
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
    )
    model._prepare_special_tokens(base_cfg, device=input_ids.device)

    beams_to_keep = max(2, 1 + 0) * int(num_beams)  # matches _beam_search for n_eos_tokens=0

    rows = []
    for p in penalties:
        lp = log_probs.clone()
        custom = LogitsProcessorList(
            [SoftPrefixConstraintLogitsProcessor(prefix_allowed_tokens_fn=prefix_fn, penalty=float(p))]
        )
        proc = model._get_logits_processor(
            generation_config=base_cfg,
            input_ids_seq_length=int(input_ids.shape[1]),
            prefix_allowed_tokens_fn=None,
            logits_processor=custom,
            device=str(model.device),
            model_kwargs=None,
        )
        lp2 = proc(input_ids, lp)

        probs = torch.softmax(lp2.squeeze(0), dim=-1)
        mass_allowed = _softmax_mass(allowed_mask, probs)
        mass_disallowed = _softmax_mass(~allowed_mask, probs)

        # Estimate how often multinomial(beams_to_keep) draws *any* disallowed token.
        # (In _get_top_k_continuations, sampling is on accumulated_log_probs; for the first step this is equivalent.)
        any_disallowed = 0
        if trials > 0:
            for _ in range(int(trials)):
                idx = torch.multinomial(probs, num_samples=beams_to_keep, replacement=False)
                if bool((~allowed_mask[idx]).any().item()):
                    any_disallowed += 1
        p_any = (any_disallowed / float(trials)) if trials > 0 else float("nan")

        rows.append(
            {
                "penalty": float(p),
                "allowed_size": int(allowed_mask.sum().item()),
                "mass_allowed": float(mass_allowed),
                "mass_disallowed": float(mass_disallowed),
                "beams_to_keep": int(beams_to_keep),
                "p_any_disallowed_in_draws": float(p_any),
                "processor_chain": [type(x).__name__ for x in proc],
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Write a short sentence:")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--n-words", type=int, default=200)
    ap.add_argument("--wordlist-dir", default=None)
    ap.add_argument("--penalties", default="15.0,15.1,15.2,15.3,16.0")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--num-beams", type=int, default=5)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--mode", choices=("sample", "beam"), default="beam")
    args = ap.parse_args()

    s = Settings()
    ms = ModelService.from_settings(s)

    # Default to an on-the-fly toy wordlist to avoid reading large repo wordlists.
    wordlist_dir = args.wordlist_dir or _make_toy_wordlist_dir()
    lang = _pick_lang(wordlist_dir, args.lang)
    if lang == "es":
        raise SystemExit("Refusing to read wordlists/es.txt (repo rule). Use --wordlist-dir with a smaller wordlist.")
    prefix_fn = build_regexp_prefix_fn(
        tokenizer=ms.tokenizer,
        lang=lang,
        n_words=int(args.n_words),
        wordlist_dir=str(wordlist_dir),
    )
    if prefix_fn is None:
        raise SystemExit(f"Failed to build prefix_fn for lang={lang} wordlist_dir={wordlist_dir}")

    penalties = [float(x.strip()) for x in args.penalties.split(",") if x.strip()]

    print(f"model={s.MODEL_NAME!r} wordlist_dir={wordlist_dir!r} lang={lang!r} n_words={args.n_words}")
    print(f"prompt={args.prompt!r}")

    if args.mode == "sample":
        rows = diagnose_one_step(
            ms=ms,
            prefix_fn=prefix_fn,
            prompt=args.prompt,
            penalties=penalties,
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
        )
        print(
            "penalty\tallowed_size\tmass_allowed\tmass_disallowed\tin_topk_allowed\tin_topk_disallowed\tkth_logit"
        )
        for r in rows:
            print(
                f"{r.penalty:.3f}\t{r.allowed_size}\t{r.mass_allowed:.6f}\t{r.mass_disallowed:.6f}\t"
                f"{r.in_topk_allowed}\t{r.in_topk_disallowed}\t{r.kth_logit:.4f}"
            )
    else:
        rows = diagnose_one_step_beam_sampling(
            ms=ms,
            prefix_fn=prefix_fn,
            prompt=args.prompt,
            penalties=penalties,
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            num_beams=int(args.num_beams),
            trials=int(args.trials),
        )
        print("penalty\tallowed_size\tmass_allowed\tmass_disallowed\tbeams_to_keep\tp(any disallowed in draws)")
        for r in rows:
            print(
                f"{r['penalty']:.3f}\t{r['allowed_size']}\t{r['mass_allowed']:.6f}\t{r['mass_disallowed']:.6f}\t"
                f"{r['beams_to_keep']}\t{r['p_any_disallowed_in_draws']:.3f}"
            )
        if rows:
            print("processor_chain=", rows[0]["processor_chain"])


if __name__ == "__main__":
    main()
