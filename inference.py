"""
Text Generation Inference Script
Models: OpenAI GPT-5+ , Google Gemini 3+ (primary), and local HF fallback.
Default samples in generated_samples/ were produced with gpt-5 and gemini-3-pro-preview (hosted APIs).

Usage:
  python -m pip install -r requirements.txt   # use python -m pip (pip.exe may be blocked by Device Guard)
  python inference.py --prompts prompts.txt --out generated_samples --model gpt-5 --max-tokens 300
  python inference.py --prompts prompts.txt --out generated_samples --model gemini-3 --max-tokens 300
  python inference.py --prompts prompts.txt --out generated_samples --model local --local-model gpt2
  python inference.py --prompts prompts.txt --out tmp_mock --model mock   # offline, no keys/deps

Env vars:
  OPENAI_API_KEY, GEMINI_API_KEY / GOOGLE_API_KEY, ANTHROPIC_API_KEY

prompts.txt format: lines like "P01 ||| Title ||| Prompt text", blank lines and # comments ignored.
"""

import argparse
import os
import re
from pathlib import Path


def parse_prompts(path: Path):
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: ID ||| Title ||| Prompt
        parts = [p.strip() for p in line.split("|||")]
        if len(parts) == 3:
            pid, title, prompt = parts
        elif len(parts) == 2:
            pid, prompt = parts
            title = pid
        else:
            pid = f"P{len(prompts)+1:02d}"
            title, prompt = pid, line
        prompts.append((pid, title, prompt))
    return prompts


def generate_openai(prompt, model="gpt-5", max_tokens=300, temperature=0.7):
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "The 'openai' package is not installed. Run: "
            "python -m pip install -r requirements.txt"
        ) from e
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it first, e.g. "
            "PowerShell: $env:OPENAI_API_KEY='sk-...' / "
            "cmd: set OPENAI_API_KEY=sk-..."
        )
    client = OpenAI(api_key=api_key)
    messages = [{"role": "user", "content": prompt}]
    # GPT-5 / o-series use max_completion_tokens and only support
    # temperature=1.0 (default). Sending max_tokens or a custom
    # temperature returns a 400 error, so adapt params per model.
    is_reasoning_model = model.startswith(("gpt-5", "o1", "o3", "gpt-4.1"))
    kwargs = {"model": model, "messages": messages}
    if is_reasoning_model:
        kwargs["max_completion_tokens"] = max_tokens
        if temperature == 1.0:
            kwargs["temperature"] = temperature
        # else: omit temperature (GPT-5 defaults to 1.0)
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = temperature
    try:
        resp = client.chat.completions.create(**kwargs)
    except TypeError:
        # Older openai lib without max_completion_tokens: retry legacy param.
        kwargs.pop("max_completion_tokens", None)
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = temperature
        resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content
    if content is None:
        raise RuntimeError(f"Empty response from OpenAI model {model!r}.")
    return content.strip()


def generate_gemini(prompt, model="gemini-3-pro-preview", max_tokens=300, temperature=0.7):
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. Set it first, e.g. "
            "PowerShell: $env:GEMINI_API_KEY='...' / cmd: set GEMINI_API_KEY=..."
        )
    # Prefer the new google-genai SDK, fall back to legacy google-generativeai.
    try:
        from google import genai as new_genai
        from google.genai import types as genai_types
        client = new_genai.Client(api_key=api_key)
        config = genai_types.GenerateContentConfig(
            max_output_tokens=max_tokens, temperature=temperature
        )
        resp = client.models.generate_content(
            model=model, contents=prompt, config=config
        )
        text = getattr(resp, "text", None)
        if text:
            return text.strip()
        # Fallback: concatenate candidate parts if .text is empty.
        parts = []
        for cand in getattr(resp, "candidates", []) or []:
            content = getattr(cand, "content", None)
            for part in getattr(content, "parts", []) or []:
                t = getattr(part, "text", None)
                if t:
                    parts.append(t)
        if parts:
            return "\n".join(parts).strip()
        raise RuntimeError("Empty response from Gemini (new SDK).")
    except ImportError:
        pass  # try legacy SDK below
    except Exception as e:
        # New SDK installed but call failed: surface the real error, but
        # hint at legacy fallback only for 404/model-not-found.
        msg = str(e).lower()
        if "not found" not in msg and "404" not in msg:
            raise
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise RuntimeError(
            "Neither 'google.genai' (new) nor 'google.generativeai' (legacy) "
            "is installed. Run: python -m pip install -r requirements.txt"
        ) from e
    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(model)
    resp = m.generate_content(
        prompt,
        generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
    )
    text = getattr(resp, "text", None)
    if text is None:
        raise RuntimeError("Empty response from Gemini (legacy SDK).")
    return text.strip()


def generate_local(prompt, local_model="gpt2", max_tokens=150):
    try:
        from transformers import pipeline
    except ImportError as e:
        raise RuntimeError(
            "The 'transformers' (+ 'torch') packages are not installed. Run: "
            "python -m pip install transformers torch"
        ) from e
    gen = pipeline("text-generation", model=local_model)
    out = gen(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7, truncation=True)
    text = out[0]["generated_text"]
    # strip prompt echo if present
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text


def generate_mock(prompt, pid="", title="", max_tokens=300):
    """Offline placeholder used when no API key / heavy deps are available.

    Zero dependencies, deterministic, so `evaluate.py` and demos work
    without network access. NOT a real model output.
    """
    words = prompt.split()
    hint = " ".join(words[:40])
    approx_words = max(30, min(max_tokens // 2, 150))
    return (
        f"[MOCK {pid} - {title} - offline placeholder, ~{approx_words} words]\n"
        f"Prompt excerpt: {hint} ...\n"
        "This is a deterministic offline stand-in so the pipeline "
        "(prompts -> samples -> evaluate) can be tested without API keys. "
        "Replace with real gpt-5 / gemini-3 output for final results."
    )


# Full provenance labels written into generated_samples/*.txt headers,
# matching the committed samples.
MODEL_LABELS = {
    "gpt-5": "gpt-5 (OpenAI GPT-5 family, hosted API)",
    "gpt-5-mini": "gpt-5-mini (OpenAI GPT-5 family, hosted API)",
    "gemini-3": "gemini-3-pro-preview (Google Gemini 3 family, hosted API)",
    "local": "local",
    "mock": "mock (offline placeholder, no API)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="prompts.txt")
    ap.add_argument("--out", default="generated_samples")
    ap.add_argument("--model", default="gpt-5",
                    choices=["gpt-5", "gpt-5-mini", "gemini-3", "local", "mock"])
    ap.add_argument("--local-model", default="gpt2")
    ap.add_argument("--gemini-model", default="gemini-3-pro-preview",
                    help="Exact Gemini model id (default: gemini-3-pro-preview)")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    prompt_file = Path(args.prompts)
    if not prompt_file.is_file():
        raise SystemExit(f"Prompts file not found: {prompt_file}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = parse_prompts(prompt_file)
    if not prompts:
        raise SystemExit(f"No prompts parsed from {prompt_file}. Expected 'ID ||| Title ||| Prompt' lines.")
    print(f"Loaded {len(prompts)} prompts from {prompt_file}")

    # Fail fast with a clear message for missing keys before generating.
    if args.model.startswith("gpt-5") and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("ERROR: OPENAI_API_KEY is not set (required for --model gpt-5). "
                         "Use --model mock for an offline test.")
    if args.model == "gemini-3" and not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        raise SystemExit("ERROR: GEMINI_API_KEY (or GOOGLE_API_KEY) is not set (required for --model gemini-3). "
                         "Use --model mock for an offline test.")

    for pid, title, prompt in prompts:
        safe_id = re.sub(r"[^A-Za-z0-9_-]+", "", pid)
        out_path = out_dir / f"{safe_id}.txt"
        print(f"[{pid}] {title} -> {out_path} ...", flush=True)
        try:
            if args.model.startswith("gpt-5"):
                # map gpt-5-mini alias directly; gpt-5 -> gpt-5
                text = generate_openai(prompt, model=args.model, max_tokens=args.max_tokens, temperature=args.temperature)
                label = MODEL_LABELS.get(args.model, args.model)
            elif args.model == "gemini-3":
                text = generate_gemini(prompt, model=args.gemini_model, max_tokens=args.max_tokens, temperature=args.temperature)
                label = MODEL_LABELS["gemini-3"]
            elif args.model == "local":
                text = generate_local(prompt, local_model=args.local_model, max_tokens=args.max_tokens)
                label = f"{args.local_model} (local HF checkpoint)"
            elif args.model == "mock":
                text = generate_mock(prompt, pid=pid, title=title, max_tokens=args.max_tokens)
                label = MODEL_LABELS["mock"]
            else:
                raise SystemExit(f"Unknown model: {args.model}")
        except Exception as e:
            print(f"  FAILED [{pid}]: {type(e).__name__}: {e}")
            raise SystemExit(f"Stopped after failure on {pid}: {e}") from e
        header = f"Prompt ID: {pid}\nTitle: {title}\nModel: {label}\nPrompt: {prompt}\n---\n"
        out_path.write_text(header + text + "\n", encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()
