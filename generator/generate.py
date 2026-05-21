"""
Cognitive Data Format — Token Generator
Uses Llama3 via Ollama to generate CDF tokens from a word list.

Usage:
    python generate.py --single apple
    python generate.py --all
    python generate.py --count 50
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

# Starter word list — 50 common tokens across diverse categories
WORD_LIST = [
    # Concrete objects
    "table", "door", "window", "knife", "bread",
    "stone", "rope", "wheel", "cup", "lamp",
    "book", "mirror", "key", "clock", "box",
    # Nature
    "rain", "wind", "tree", "grass", "sand",
    "ice", "cloud", "sun", "moon", "leaf",
    # People and relationships
    "father", "child", "friend", "stranger", "teacher",
    # Abstract
    "home", "time", "light", "shadow", "silence",
    "pain", "memory", "dream", "fear", "hope",
    # Verbs used as nouns
    "run", "fall", "press", "strike", "lead",
    # Common ambiguous words
    "bank", "bat", "bark", "match", "spring",
    "pool", "pitch", "post", "kind", "plant"
]


def load_schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def build_prompt(word):
    return f"""You are building a Cognitive Data Format (CDF) database.

CDF is like a dictionary where every word carries ALL the meanings it has accumulated across different contexts in human language and culture — not just its first or most common meaning.

Your job: generate a CDF token for the word "{word}".

Rules:
- Find ALL distinct real-world contexts where "{word}" has a different meaning
- Each context gets its own track
- The first track (id: "01") is the original or most fundamental meaning
- All other tracks (id: "02", "03" etc) are meanings accumulated over time
- Meanings must be genuinely different — not just slight variations
- Between 2 and 5 tracks maximum
- Keep meanings concise — one clear sentence each
- Return ONLY valid JSON. No explanation. No markdown. No code blocks.

Output this exact structure:

{{
  "token": "{word}",
  "tracks": [
    {{
      "id": "01",
      "context": "the domain this meaning belongs to",
      "meaning": "plain language definition in this context",
      "added": "first",
      "source": "llm"
    }},
    {{
      "id": "02",
      "context": "another domain",
      "meaning": "plain language definition in this context",
      "added": "accumulated",
      "source": "llm"
    }}
  ]
}}

Generate the CDF token for: {word}"""


def call_ollama(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 800}
    }, timeout=120)
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
    print(f"\nCognitive Data Format — Token Generator")
    print(f"Model:  {MODEL} via Ollama")
    print(f"Words:  {len(words)}")
    print(f"{'─' * 55}")

    valid = 0
    failed = []

    for i, word in enumerate(words, 1):
        print(f"  [{i:>3}/{len(words)}]  {word:<20}", end="", flush=True)
        t = time.time()
        try:
            token = generate_token(word, schema)
            save_token(token)
            tracks = len(token["tracks"])
            print(f"  ✓  {tracks} tracks  ({round(time.time()-t, 1)}s)")
            valid += 1
        except Exception as e:
            print(f"  ✗  FAILED  ({round(time.time()-t, 1)}s)  {str(e)[:60]}")
            failed.append(word)

    print(f"{'─' * 55}")
    print(f"  Valid:   {valid}/{len(words)}")
    print(f"  Failed:  {len(failed)}")
    if failed:
        print(f"  Retry:   {', '.join(failed)}")
    print(f"  Output:  {OUTPUT_PATH}\n")


def main():
    parser = argparse.ArgumentParser(description="CDF Token Generator")
    parser.add_argument("--single", type=str, help="Generate one token and print it")
    parser.add_argument("--all",    action="store_true", help="Generate all 50 starter tokens")
    parser.add_argument("--count",  type=int, help="Generate first N tokens only")
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
        words = words[:args.count]

    run_batch(words, schema)


if __name__ == "__main__":
    main()
