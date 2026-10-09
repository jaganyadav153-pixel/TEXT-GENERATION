# Task 1 — Text Generation (GPT-5+, Gemini 3+ / Similar)

22 diverse prompts + generated samples using GPT-5 and Gemini 3 Pro (hosted APIs), with reproducible inference script + notebook.

## 1. Model / Checkpoint Used

**P01, P03, P05, P07, P09, P11, P13, P15, P17, P19, P21 → `gpt-5` (OpenAI)**
- Type: Hosted API
- Source: https://platform.openai.com/docs/models/gpt-5
- License: Proprietary commercial — API terms at https://openai.com/policies/terms-of-use . No weights redistributed.

**P02, P04, P06, P08, P10, P12, P14, P16, P18, P20, P22 → `gemini-3-pro-preview` (Google Gemini 3 Pro)**
- Type: Hosted API
- Source: https://ai.google.dev/gemini-api/docs
- License: Proprietary commercial — API terms at https://ai.google.dev/terms . No weights redistributed.

Each file in `generated_samples/` states its model in the header, e.g.:
```
Prompt ID: P01
Title: Creative Story Opening
Model: gpt-5 (OpenAI GPT-5 family, hosted API)
```

Local fallback (optional, not used for samples): `gpt2` from HuggingFace — https://huggingface.co/openai-community/gpt2 — MIT license.

## 2. Repo Structure

```
.
├── prompts.txt              # 22 prompts, format: ID ||| Title ||| Prompt
├── generated_samples/       # P01.txt ... P22.txt (plain text, header + output)
├── inference.py             # Inference for gpt-5 / gemini-3 / local
├── inference.ipynb          # Notebook with API call examples (GPT-5 + Gemini 3)
├── evaluate.py              # Length, TTR, proxy-perplexity stats
├── evaluation_results.txt   # Output of evaluate.py
├── evaluation.md            # Concise evaluation (<=300 words)
├── requirements.txt
└── README.md
```

## 3. How to Reproduce

```bash
python -m pip install -r requirements.txt
# NOTE: use `python -m pip` (bare `pip.exe` is blocked by Device Guard on some Windows machines).
# `--model mock` and `evaluate.py` need zero dependencies and run offline.
```

### A. OpenAI GPT-5
```bash
set OPENAI_API_KEY=sk-...
python inference.py --prompts prompts.txt --out generated_samples --model gpt-5 --max-tokens 300
```
```python
from openai import OpenAI
client = OpenAI()  # uses OPENAI_API_KEY
resp = client.chat.completions.create(
  model="gpt-5",
  messages=[{"role":"user","content":"Write the opening paragraph of a mystery novel set in a lighthouse."}],
  max_completion_tokens=300)
print(resp.choices[0].message.content)
```

### B. Google Gemini 3
```bash
set GEMINI_API_KEY=...
python inference.py --prompts prompts.txt --out generated_samples --model gemini-3 --max-tokens 300
```
```python
import google.generativeai as genai
genai.configure(api_key="...")  # GEMINI_API_KEY
model = genai.GenerativeModel("gemini-3-pro-preview")
resp = model.generate_content(
  "Write three linked haikus about sunrise, noon, sunset over a desert.",
  generation_config={"max_output_tokens":300, "temperature":0.7})
print(resp.text)
```

See `inference.ipynb` for runnable notebook version of A + B.

### C. Local checkpoint (offline, no key)
```bash
python inference.py --prompts prompts.txt --out generated_samples --model local --local-model gpt2
```

### C2. Mock (offline placeholder, no key, no deps — for pipeline testing)
```bash
python inference.py --prompts prompts.txt --out tmp_mock --model mock
```

### D. Evaluate
```bash
python evaluate.py --samples generated_samples --out evaluation_results.txt
```

## 4. Evaluation Summary (198 words)

See `evaluation.md` for full version + `evaluation_results.txt` for numbers.

Samples average 97 words, mean type-token ratio 0.814 (high lexical diversity, n=22, 2133 tokens). True perplexity and BLEU are N/A for black-box APIs without logits/references; GPT-2 scorer not installed, so proxy-perplexity is N/A — qualitative review used instead.

Coherence: Strong instruction-following (21/22 respect length/format; P08 3 bullets in 26 words). Creative pieces (P01, P02, P21) maintain arc and twist; technical pieces (P10, P11, P20) are correct — factorial, clean_text, attention formula. Tone control good across formal email, LinkedIn post, dialogue. GPT-5 outputs slightly more structured (headers, code blocks); Gemini outputs slightly more lyrical in poetry/fiction.

Failure modes: (1) Illustrative statistic in P14 is invented by design — risk of hallucinated facts if label removed. (2) P12/P17 dates/prices are plausible but unverified — needs fact-check for production. (3) Code (P11) strips punctuation naively (`gpttest`) and lacks tests. (4) Haiku syllable counts approximate, not strict. (5) No refusal/hedging tests; safety not stressed.

Overall: fluent, diverse, task-appropriate; main risks are factuality and overconfidence, typical of GPT-5/Gemini-class models. Mitigate with retrieval, unit tests, fact labels.

## 5. Prompts

See `prompts.txt` — 22 prompts covering story, flash-fiction, poetry, haiku, product, email, letter, summarization, ELI10, code explain/generate, history, philosophy, persuasion, dialogue, recipe, travel, interview, social, technical, fable, counterfactual.
