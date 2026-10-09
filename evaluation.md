# Evaluation — Text Generation (22 samples: GPT-5 + Gemini 3 Pro)

## Quantitative (proxy, from `evaluate.py`)

- Files: 22 (11× gpt-5, 11× gemini-3-pro-preview), avg 97.0 words/sample, total 2133 tokens / 1189 unique.
- Mean type-token ratio (TTR): 0.814 — high diversity; lowest P10 (0.608, code repetition) and P16 (0.656, ingredient repetition) as expected; highest P04 (1.00, short haikus).
- BLEU: N/A — no human references; not computed to avoid misleading scores.
- Perplexity: N/A — black-box hosted APIs (OpenAI/Gemini) expose no logits. Optional GPT-2 scorer unavailable offline (`ModuleNotFoundError`); would be proxy-only even if run. See `evaluation_results.txt`.

Run: `python evaluate.py --samples generated_samples --out evaluation_results.txt`

## Qualitative Summary (198 words)

Samples average 97 words, mean type-token ratio 0.814 (high lexical diversity, n=22, 2133 tokens). True perplexity and BLEU are N/A for black-box APIs without logits/references; GPT-2 scorer not installed, so proxy-perplexity is N/A — qualitative review used instead.

Coherence: Strong instruction-following (21/22 respect length/format; P08 3 bullets in 26 words). Creative pieces (P01, P02, P21) maintain arc and twist; technical pieces (P10, P11, P20) are correct — factorial, clean_text, attention formula. Tone control good across formal email, LinkedIn post, dialogue. GPT-5 outputs slightly more structured (headers, code blocks); Gemini outputs slightly more lyrical in poetry/fiction.

Failure modes: (1) Illustrative statistic in P14 is invented by design — risk of hallucinated facts if label removed. (2) P12/P17 dates/prices are plausible but unverified — needs fact-check for production. (3) Code (P11) strips punctuation naively (`gpttest`) and lacks tests. (4) Haiku syllable counts approximate, not strict. (5) No refusal/hedging tests; safety not stressed.

Overall: fluent, diverse, task-appropriate; main risks are factuality and overconfidence, typical of GPT-5/Gemini-class models. Mitigate with retrieval, unit tests, fact labels.
