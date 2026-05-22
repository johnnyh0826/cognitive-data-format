"""
Cognitive Data Format — Token Generator
Uses Llama3 via Ollama to generate CDF tokens.

USAGE:
    python generator/generate.py                        # use wordlist.txt
    python generator/generate.py mywords.txt            # any .txt file
    python generator/generate.py document.pdf           # extract words from PDF
    python generator/generate.py notes.docx             # extract words from Word doc
    python generator/generate.py list.csv               # extract words from CSV
    python generator/generate.py --single bank          # generate one word
    python generator/generate.py --count 10             # generate first 10 new words

HOW TO ADD MORE WORDS:
    Drop any file into the folder and pass it as an argument.
    Delete the file when done. The generator skips already-generated words.
"""

import json
import re
import sys
import time
import argparse
import requests
import jsonschema
from pathlib import Path

SCHEMA_PATH   = Path(__file__).parent.parent / "schema" / "cdf_schema_v1.json"
OUTPUT_PATH   = Path(__file__).parent.parent / "data" / "generated"
WORDLIST_PATH = Path(__file__).parent.parent / "wordlist.txt"
OLLAMA_URL    = "http://localhost:11434/api/generate"
MODEL         = "llama3"


# ─── File readers ────────────────────────────────────────────────────────────

def read_txt(path):
    """Read words from a plain .txt file — one word per line."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [
            line.strip().lower()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]


def read_pdf(path):
    """Extract all words from a PDF and return unique single words."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        text = " ".join(page.extract_text() or "" for page in reader.pages)
        return extract_words_from_text(text)
    except ImportError:
        print("ERROR: pypdf not installed. Run: pip install pypdf")
        sys.exit(1)


def read_docx(path):
    """Extract all words from a Word document."""
    try:
        import docx
        doc = docx.Document(str(path))
        text = " ".join(para.text for para in doc.paragraphs)
        return extract_words_from_text(text)
    except ImportError:
        print("ERROR: python-docx not installed. Run: pip install python-docx")
        sys.exit(1)


def read_csv(path):
    """Extract words from a CSV — reads all cells."""
    import csv
    words = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            for cell in row:
                cell = cell.strip().lower()
                if cell and " " not in cell and cell.isalpha():
                    words.append(cell)
    return deduplicate(words)


def extract_words_from_text(text):
    """
    Extract meaningful single words from raw text.
    Filters to alphabetic words between 3-20 chars.
    Removes common stopwords so we get content words only.
    """
    stopwords = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "her",
        "was", "one", "our", "out", "day", "get", "has", "him", "his", "how",
        "its", "may", "new", "now", "old", "see", "two", "way", "who", "did",
        "does", "had", "has", "have", "been", "with", "that", "this", "from",
        "they", "will", "what", "when", "your", "said", "each", "she", "use",
        "more", "also", "into", "than", "then", "them", "some", "would",
        "make", "like", "time", "just", "know", "take", "people", "year",
        "good", "very", "even", "most", "tell", "much", "want", "well",
        "also", "back", "after", "could", "these", "first", "those", "only",
        "over", "such", "here", "through", "where", "should", "about",
        "there", "think", "every", "never", "under", "other", "right",
        "come", "work", "both", "little", "being", "because", "going",
        "still", "down", "give", "long", "think", "hand", "high", "place",
        "hold", "real", "life", "word", "last", "next", "seem", "hard",
        "open", "example", "begin", "life", "always", "those", "both",
        "paper", "together", "got", "group", "often", "run", "important"
    }

    words = re.findall(r'\b[a-zA-Z]{3,20}\b', text)
    words = [w.lower() for w in words]
    words = [w for w in words if w not in stopwords]
    return deduplicate(words)


def deduplicate(words):
    """Remove duplicates while preserving order."""
    seen = set()
    result = []
    for w in words:
        if w not in seen:
            seen.add(w)
            result.append(w)
    return result


def load_words(source_path=None):
    """
    Load words from the given file path.
    Supports: .txt, .pdf, .docx, .csv
    Defaults to wordlist.txt if no path given.
    """
    if source_path is None:
        path = WORDLIST_PATH
        if not path.exists():
            print(f"ERROR: wordlist.txt not found. Add words to wordlist.txt or pass a file.")
            sys.exit(1)
    else:
        path = Path(source_path)
        if not path.exists():
            print(f"ERROR: File not found — {path}")
            sys.exit(1)

    ext = path.suffix.lower()
    print(f"  Reading: {path.name} ({ext})")

    if ext in (".txt", ".md", ".csv") and ext != ".csv":
        return read_txt(path), path
    elif ext == ".pdf":
        return read_pdf(path), path
    elif ext == ".docx":
        return read_docx(path), path
    elif ext == ".csv":
        return read_csv(path), path
    else:
        # Try as plain text for unknown types
        try:
            return read_txt(path), path
        except Exception:
            print(f"ERROR: Unsupported file type — {ext}")
            print("Supported: .txt .pdf .docx .csv")
            sys.exit(1)


# ─── Schema and generation ────────────────────────────────────────────────────

def load_schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def already_generated(word):
    name = word.replace(" ", "_")
    return (OUTPUT_PATH / f"{name}.cdf.json").exists()


def build_prompt(word):
    return f"""You are building a Cognitive Data Format (CDF) database — a structured dictionary where every word carries ALL the meanings it has accumulated across different contexts in human language, culture, history, and daily life.

Your job: generate a CDF token for the word "{word}".

CRITICAL RULES:
- Find EVERY genuinely distinct meaning this word has in real human usage
- Do NOT stop at the obvious first meaning — dig deeper
- Think across ALL domains: science, culture, slang, technology, law, sports, food, geography, history, medicine, finance, military, music, art
- Each track must be a MEANINGFULLY DIFFERENT meaning — not just a slight variation
- Minimum 2 tracks, target 3-5 tracks for any word with multiple real uses
- Meanings must be real and verifiable — not invented
- Keep each meaning concise — one clear sentence
- Return ONLY valid JSON. No explanation. No markdown. No code blocks.

EXAMPLES OF GOOD TRACK DEPTH:
- "spring": season / mechanical coil / water source / to jump / to release from prison
- "bank": financial institution / river edge / to tilt aircraft / blood bank / pool shot
- "bark": tree covering / dog sound / sailing ship / to speak sharply

Generate the CDF token for: "{word}"

Output this exact JSON structure:

{{
  "token": "{word}",
  "tracks": [
    {{
      "id": "01",
      "context": "primary domain",
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
        "options": {"temperature": 0.5, "top_p": 0.95, "num_predict": 1200}
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
    raise ValueError(f"Could not extract JSON: {raw[:200]}")


def generate_token(word, schema):
    raw   = call_ollama(build_prompt(word))
    token = extract_json(raw)
    jsonschema.validate(instance=token, schema=schema)
    return token


def save_token(token):
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    name = token["token"].replace(" ", "_")
    path = OUTPUT_PATH / f"{name}.cdf.json"
    with open(path, "w") as f:
        json.dump(token, f, indent=2)
    return path


# ─── Batch runner ─────────────────────────────────────────────────────────────

def run_batch(words, schema, source_name="wordlist.txt"):
    new_words = [w for w in words if not already_generated(w)]
    skipped   = len(words) - len(new_words)

    print(f"\nCognitive Data Format — Token Generator")
    print(f"Model:      {MODEL} via Ollama")
    print(f"Source:     {source_name} ({len(words)} words)")
    print(f"Skipped:    {skipped} already generated")
    print(f"Generating: {len(new_words)} new tokens")
    print(f"{'─' * 55}")

    if not new_words:
        print("  All words already generated. Nothing to do.")
        total = len(list(OUTPUT_PATH.glob("*.cdf.json")))
        print(f"  Total in database: {total} tokens\n")
        return

    valid  = 0
    failed = []

    for i, word in enumerate(new_words, 1):
        print(f"  [{i:>3}/{len(new_words)}]  {word:<22}", end="", flush=True)
        t = time.time()
        try:
            token  = generate_token(word, schema)
            save_token(token)
            tracks = len(token["tracks"])
            print(f"  ✓  {tracks} tracks  ({round(time.time()-t,1)}s)")
            valid += 1
        except Exception as e:
            print(f"  ✗  FAILED  ({round(time.time()-t,1)}s)  {str(e)[:50]}")
            failed.append(word)

    total = len(list(OUTPUT_PATH.glob("*.cdf.json")))
    print(f"{'─' * 55}")
    print(f"  Generated:  {valid}/{len(new_words)}")
    print(f"  Failed:     {len(failed)}")
    print(f"  Total in database: {total} tokens")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CDF Token Generator — accepts .txt .pdf .docx .csv or uses wordlist.txt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generator/generate.py                  # use wordlist.txt
  python generator/generate.py mywords.txt      # plain text file
  python generator/generate.py notes.pdf        # extract words from PDF
  python generator/generate.py doc.docx         # extract words from Word doc
  python generator/generate.py --single spring  # generate one word
  python generator/generate.py --count 20       # generate first 20 new words
        """
    )
    parser.add_argument("file",    nargs="?", help="Input file (.txt .pdf .docx .csv)")
    parser.add_argument("--single", type=str, help="Generate one word and print it")
    parser.add_argument("--count",  type=int, help="Generate first N new words only")
    args = parser.parse_args()

    schema = load_schema()

    # Single word mode
    if args.single:
        print(f"\nGenerating: {args.single}\n{'─'*40}")
        try:
            token = generate_token(args.single, schema)
            print(json.dumps(token, indent=2))
        except Exception as e:
            print(f"FAILED: {e}")
        return

    # Load words from file or wordlist.txt
    words, source_path = load_words(args.file)

    # Apply count limit to new words only
    if args.count:
        new_words = [w for w in words if not already_generated(w)]
        words = new_words[:args.count]

    run_batch(words, schema, source_name=source_path.name)


if __name__ == "__main__":
    main()
