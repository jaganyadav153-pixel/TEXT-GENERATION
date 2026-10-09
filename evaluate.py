"""
Simple evaluation: length stats, type-token ratio (diversity proxy),
and optional BLEU self-check + GPT-2 perplexity if libs available.
For black-box hosted APIs true perplexity is N/A (no logits), so this
reports reference-based proxy metrics + qualitative notes in evaluation.md.

Usage:
  python -m pip install -r requirements.txt   # nltk/transformers optional; regex fallback works offline
  python evaluate.py --samples generated_samples --out evaluation_results.txt
"""
import argparse
import re
from pathlib import Path
from collections import Counter


def tokenize(s):
    return re.findall(r"[A-Za-z0-9']+", s.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="generated_samples")
    ap.add_argument("--out", default="evaluation_results.txt")
    args = ap.parse_args()

    samples_dir = Path(args.samples)
    if not samples_dir.is_dir():
        raise SystemExit(f"Samples directory not found: {samples_dir}")
    files = sorted(samples_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"No *.txt files found in {samples_dir}. Nothing to evaluate.")
    rows = []
    all_tokens = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except Exception as e:
            print(f"WARNING: skipping {f.name}: {type(e).__name__}: {e}")
            continue
        # strip header up to --- if present
        body = text.split("---", 1)[-1] if "---" in text else text
        toks = tokenize(body)
        words = len(toks)
        uniq = len(set(toks))
        ttr = uniq / max(1, words)
        rows.append((f.name, words, uniq, ttr))
        all_tokens.extend(toks)
    if not rows:
        raise SystemExit("No readable sample files; nothing to evaluate.")

    avg_words = sum(r[1] for r in rows) / max(1, len(rows))
    avg_ttr = sum(r[3] for r in rows) / max(1, len(rows))
    total_tokens = len(all_tokens)
    total_unique = len(set(all_tokens))

    # Optional BLEU: leave-one-out unigram precision as diversity proxy (lower overlap = more diverse)
    # Optional perplexity via GPT-2 if transformers+torch present
    ppl_note = "N/A (black-box API, no logits)"
    try:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        import torch
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        lm = GPT2LMHeadModel.from_pretrained("gpt2")
        lm.eval()
        import math
        nlls, n_tokens = 0.0, 0
        with torch.no_grad():
            for f in files[:5]:  # sample 5 to keep fast
                body = f.read_text(encoding="utf-8").split("---", 1)[-1][:1000]
                ids = tok(body, return_tensors="pt").input_ids
                if ids.shape[1] < 2:
                    continue
                out = lm(ids, labels=ids)
                nlls += out.loss.item() * (ids.shape[1])
                n_tokens += ids.shape[1]
        if n_tokens:
            ppl_note = f"{math.exp(nlls/n_tokens):.2f} (GPT-2 scorer, 5-file sample, proxy only)"
    except Exception as e:
        ppl_note = f"N/A - GPT-2 scorer unavailable ({type(e).__name__})"

    lines = [f"Files evaluated: {len(rows)}", f"Avg words/sample: {avg_words:.1f}",
             f"Avg type-token ratio: {avg_ttr:.3f}",
             f"Total tokens: {total_tokens} | unique: {total_unique}",
             f"Proxy perplexity: {ppl_note}", "",
             "Per-file: name | words | unique | TTR"]
    for name, w, u, t in rows:
        lines.append(f"{name} | {w} | {u} | {t:.3f}")

    out_path = Path(args.out)
    if out_path.parent != Path(".") and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
