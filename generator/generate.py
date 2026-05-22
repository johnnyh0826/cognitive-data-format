"""
Cognitive Data Format — Token Generator
Uses Llama3 via Ollama to generate CDF tokens from a word list.

Usage:
    python generator/generate.py --single apple
    python generator/generate.py --all
    python generator/generate.py --count 50
"""

import json
import re
import time
import argparse
import requests
import jsonschema
from pathlib import Path

SCHEMA_PATH  = Path(__file__).parent.parent / "schema" / "cdf_schema_v1.json"
OUTPUT_PATH  = Path(__file__).parent.parent / "data" / "generated"
OLLAMA_URL   = "http://localhost:11434/api/generate"
MODEL        = "llama3"

# Full word list — existing + new custom words
WORD_LIST = [
    # Original 50
    "table", "door", "window", "knife", "bread",
    "stone", "rope", "wheel", "cup", "lamp",
    "book", "mirror", "key", "clock", "box",
    "rain", "wind", "tree", "grass", "sand",
    "ice", "cloud", "sun", "moon", "leaf",
    "father", "child", "friend", "stranger", "teacher",
    "home", "time", "light", "shadow", "silence",
    "pain", "memory", "dream", "fear", "hope",
    "run", "fall", "press", "strike", "lead",
    "bank", "bat", "bark", "match", "spring",
    "pool", "pitch", "post", "kind", "plant",
    # Custom additions
    "blueberry", "car", "phone", "laptop", "steak",
    "bible", "money", "tv", "shoes", "desk",
    "fan", "cards", "desktop", "printer", "paper",
    "sofa", "photo", "spoon", "fork", "vacuum",
    "battery", "socks", "hat", "sneakers", "golf",
    "basketball", "tires", "cement", "star", "planet",
    "gate", "monitor", "salt", "snacks", "potato",
    "pineapple", "rice", "coffee", "tea", "sugar",
    "milk", "butter", "oil", "wine", "beer",
    "hammer", "nail", "brush", "paint", "glass",
]


def load_schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def already_generated(word):
    """Check if this token already exists in the output directory."""
    name = word.replace(" ", "_")
    return (OUTPUT_PATH / f"{name}.cdf.json").exists()


def build_prompt(word):
    return f"""You are building a Cognitive Data Format (CDF) database — a structured dictionary where every word carries ALL the meanings it has accumulated across different contexts in human language, culture, history, and daily life.

Your job: generate a CDF token for the word "{word}".

CRITICAL RULES:
- You MUST find EVERY genuinely distinct meaning this word has in real human usage
- Do NOT stop at the obvious first meaning — dig deeper
- Think across ALL domains: science, culture, slang, technology, law, sports, food, geography, history, medicine, finance, military, music, art
- Each track must be a MEANINGFULLY DIFFERENT meaning — not just a slight variation
- Minimum 2 tracks, target 3-5 tracks for any word with multiple real uses
- If a word has 6 genuine meanings, include all 6
- Meanings must be real and verifiable — not invented
- Keep each meaning concise — one clear sentence
- Return ONLY valid JSON. No explanation. No markdown. No code blocks.

EXAMPLES OF GOOD TRACK DEPTH:
- "spring": season / mechanical coil / water source / to jump / to release someone from prison
- "bank": financial institution / river edge / to tilt an aircraft / blood bank / to bank a shot in pool
- "bark": tree covering / dog sound / sailing ship / to speak sharply / a type of chocolate coating

Now generate the CDF token for: "{word}"

Output this exact JSON structure:

{{
  "token": "{word}",
  "tracks": [
    {{
      "id": "01",
      "context": "the primary domain this meaning belongs to",
      "meaning": "plain language definition — one clear sentence",
      "added": "first",
      "source": "llm"
    }},
    {{
      "id": "02",
      "context": "another distinct domain",
      "meaning": "plain language definition — one clear sentence",
      "added": "accumulated",
      "source": "llm"
    }}
  ]
}}

Return ONLY the JSON. Nothing else."""


def call_ollama(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.5,
            "top_p": 0.95,
            "num_predict": 1200,
        }
    }, timeout=180)
    response.raise_for_status()
    return response.json().get("response", "")


def extract_json(raw):
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end   = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(cleaned[start:end])
    raise ValueError(f"Could not extract JSON from: {raw[:300]}")


def generate_token(word, schema):
    prompt = build_prompt(word)
    raw    = call_ollama(prompt)
    token  = extract_json(raw)
    jsonschema.validate(instance=token, schema=schema)
    return token


def save_token(token):
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    name = token["token"].replace(" ", "_")
    path = OUTPUT_PATH / f"{name}.cdf.json"
    with open(path, "w") as f:
        json.dump(token, f, indent=2)
    return path


def run_batch(words, schema):
    # Filter out already generated words
    new_words = [w for w in words if not already_generated(w)]
    skipped   = len(words) - len(new_words)

    print(f"\nCognitive Data Format — Token Generator")
    print(f"Model:    {MODEL} via Ollama")
    print(f"Total:    {len(words)} words")
    print(f"Skipped:  {skipped} already generated")
    print(f"New:      {len(new_words)} to generate")
    print(f"{'─' * 55}")

    if not new_words:
        print("  All tokens already generated. Nothing to do.")
        return

    valid  = 0
    failed = []

    for i, word in enumerate(new_words, 1):
        print(f"  [{i:>3}/{len(new_words)}]  {word:<20}", end="", flush=True)
        t = time.time()
        try:
            token  = generate_token(word, schema)
            save_token(token)
            tracks = len(token["tracks"])
            print(f"  ✓  {tracks} tracks  ({round(time.time()-t, 1)}s)")
            valid += 1
        except Exception as e:
            print(f"  ✗  FAILED  ({round(time.time()-t, 1)}s)  {str(e)[:60]}")
            failed.append(word)

    print(f"{'─' * 55}")
    print(f"  Generated: {valid}/{len(new_words)}")
    print(f"  Failed:    {len(failed)}")
    print(f"  Total in database: {valid + (len(words) - len(new_words))}")
    if failed:
        print(f"  Retry: {', '.join(failed)}")
    print(f"  Output: {OUTPUT_PATH}\n")


def main():
    parser = argparse.ArgumentParser(description="CDF Token Generator")
    parser.add_argument("--single", type=str, help="Generate one token and print it")
    parser.add_argument("--all",    action="store_true", help="Generate all words — skips existing")
    parser.add_argument("--count",  type=int, help="Generate first N new words only")
    args = parser.parse_args()

    schema = load_schema()

    if args.single:
        print(f"\nGenerating: {args.single}\n{'─'*40}")
        try:
            token = generate_token(args.single, schema)
            print(json.dumps(token, indent=2))
        except Exception as e:
            print(f"FAILED: {e}")
        return

    words = WORD_LIST
    if args.count:
        # Only count new words
        new_words = [w for w in words if not already_generated(w)]
        words = new_words[:args.count]

    run_batch(words, schema)


if __name__ == "__main__":
    main()
