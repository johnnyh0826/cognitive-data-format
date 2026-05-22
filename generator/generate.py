"""
Cognitive Data Format — Token Generator
Two-phase generation:
  Phase 1 — WordNet: clean authoritative base meanings
  Phase 2 — LLM: accumulated meanings WordNet doesn't capture
  Phase 3 — Deduplicate and write CDF token

Usage:
    python generator/generate.py                  # use wordlist.txt
    python generator/generate.py mywords.txt      # any .txt file
    python generator/generate.py document.pdf     # extract words from PDF
    python generator/generate.py notes.docx       # extract words from Word doc
    python generator/generate.py --single bank    # generate one word
    python generator/generate.py --count 10       # generate first 10 new words
"""

import json
import re
import sys
import time
import argparse
import requests
import jsonschema
from pathlib import Path

# WordNet
import nltk
try:
    from nltk.corpus import wordnet as wn
    wn.synsets("test")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.corpus import wordnet as wn

SCHEMA_PATH   = Path(__file__).parent.parent / "schema" / "cdf_schema_v1.json"
OUTPUT_PATH   = Path(__file__).parent.parent / "data" / "generated"
WORDLIST_PATH = Path(__file__).parent.parent / "wordlist.txt"
OLLAMA_URL    = "http://localhost:11434/api/generate"
MODEL         = "llama3"

# WordNet POS labels → CDF domain hints
POS_LABELS = {"n": "noun", "v": "verb", "a": "adjective", "r": "adverb", "s": "adjective"}

# WordNet lexname → cleaner domain label
LEXNAME_MAP = {
    "noun.act": "action", "noun.animal": "biology", "noun.artifact": "object",
    "noun.attribute": "property", "noun.body": "anatomy", "noun.cognition": "psychology",
    "noun.communication": "communication", "noun.event": "event", "noun.feeling": "emotion",
    "noun.food": "food", "noun.group": "group", "noun.location": "geography",
    "noun.motive": "motivation", "noun.object": "object", "noun.person": "person",
    "noun.phenomenon": "phenomenon", "noun.plant": "biology", "noun.possession": "finance",
    "noun.process": "process", "noun.quantity": "measurement", "noun.relation": "relationship",
    "noun.shape": "geometry", "noun.state": "state", "noun.substance": "chemistry",
    "noun.time": "time", "verb.body": "anatomy", "verb.change": "change",
    "verb.cognition": "psychology", "verb.communication": "communication",
    "verb.competition": "sports", "verb.consumption": "consumption",
    "verb.contact": "action", "verb.creation": "creation", "verb.emotion": "emotion",
    "verb.motion": "movement", "verb.perception": "perception", "verb.possession": "finance",
    "verb.social": "social", "verb.stative": "state", "verb.weather": "weather",
}


# ─── Phase 1: WordNet ─────────────────────────────────────────────────────────

def get_wordnet_tracks(word):
    """
    Extract clean base meanings from WordNet.
    Returns top 2-3 most common synsets as CDF tracks.
    Filters to nouns and verbs only — most relevant for disambiguation.
    """
    synsets = wn.synsets(word, pos=[wn.NOUN, wn.VERB])
    if not synsets:
        synsets = wn.synsets(word)

    tracks = []
    seen_definitions = set()

    for synset in synsets:
        definition = synset.definition()

        # Skip near-duplicate definitions
        definition_key = definition[:40].lower()
        if definition_key in seen_definitions:
            continue
        seen_definitions.add(definition_key)

        # Get domain label from lexname
        lexname = synset.lexname() or ""
        context = LEXNAME_MAP.get(lexname, lexname.split(".")[-1] if "." in lexname else "general")

        # Use the lemma name to improve context when possible
        lemma_name = synset.name().split(".")[0].replace("_", " ")
        if lemma_name != word and len(lemma_name) > 3:
            context = lemma_name

        tracks.append({
            "context": context,
            "meaning": definition,
            "source": "wordnet"
        })

        if len(tracks) >= 3:
            break

    return tracks


# ─── Phase 2: LLM accumulated meanings ───────────────────────────────────────

def build_llm_prompt(word, existing_meanings):
    existing_text = "\n".join(f"- {m}" for m in existing_meanings)
    return f"""You are enriching a Cognitive Data Format (CDF) token for the word "{word}".

The following meanings are already captured from a dictionary:
{existing_text}

Your task: identify 1-2 additional common meanings that a fluent English speaker would recognise but that are NOT covered above.

Rules:
- Only add meanings clearly different from the ones listed
- Standard, commonly known meanings only — no obscure or rare usages
- If no additional common meanings exist, return an empty tracks array
- Return ONLY valid JSON. No explanation. No markdown.

Output this exact JSON:
{{
  "additional_tracks": [
    {{
      "context": "domain label",
      "meaning": "clear plain language definition"
    }}
  ]
}}

If nothing to add: {{"additional_tracks": []}}
Return ONLY the JSON."""


def get_llm_tracks(word, existing_meanings):
    """Ask LLM for meanings not covered by WordNet."""
    try:
        prompt   = build_llm_prompt(word, existing_meanings)
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 400}
        }, timeout=60)
        response.raise_for_status()
        raw = response.json().get("response", "")

        cleaned = re.sub(r"```(?:json)?", "", raw).strip().replace("```", "").strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end   = cleaned.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(cleaned[start:end])
            else:
                return []

        return data.get("additional_tracks", [])

    except Exception:
        return []


# ─── Phase 3: Assemble CDF token ─────────────────────────────────────────────

def assemble_token(word, wordnet_tracks, llm_tracks):
    """Combine WordNet and LLM tracks into a valid CDF token."""
    all_tracks = []
    seen = set()

    for i, track in enumerate(wordnet_tracks + llm_tracks):
        meaning = track.get("meaning", "").strip()
        meaning_key = meaning[:40].lower()

        if meaning_key in seen or not meaning:
            continue
        seen.add(meaning_key)

        track_id = str(len(all_tracks) + 1).zfill(2)
        all_tracks.append({
            "id": track_id,
            "context": track.get("context", "general"),
            "meaning": meaning,
            "added": "first" if i == 0 else "accumulated",
            "source": track.get("source", "llm")
        })

        if len(all_tracks) >= 4:
            break

    if not all_tracks:
        return None

    return {
        "token": word,
        "tracks": all_tracks
    }


# ─── File readers ─────────────────────────────────────────────────────────────

def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [
            line.strip().lower()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]


def read_pdf(path):
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        text = " ".join(page.extract_text() or "" for page in reader.pages)
        return extract_words_from_text(text)
    except ImportError:
        print("ERROR: pypdf not installed. Run: pip install pypdf")
        sys.exit(1)


def read_docx(path):
    try:
        import docx
        doc = docx.Document(str(path))
        text = " ".join(para.text for para in doc.paragraphs)
        return extract_words_from_text(text)
    except ImportError:
        print("ERROR: python-docx not installed. Run: pip install python-docx")
        sys.exit(1)


def read_csv(path):
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
    stopwords = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "her",
        "was", "one", "our", "out", "day", "get", "has", "him", "his", "how",
        "its", "may", "new", "now", "old", "see", "two", "way", "who", "did",
        "does", "had", "have", "been", "with", "that", "this", "from", "they",
        "will", "what", "when", "your", "said", "each", "she", "use", "more",
        "also", "into", "than", "then", "them", "some", "would", "make", "like",
        "just", "know", "take", "very", "even", "most", "back", "after", "could",
        "these", "first", "those", "only", "over", "such", "here", "should",
        "about", "there", "think", "every", "never", "under", "other", "right",
        "come", "both", "little", "being", "because", "going", "still", "down",
    }
    words = re.findall(r'\b[a-zA-Z]{3,20}\b', text)
    words = [w.lower() for w in words if w.lower() not in stopwords]
    return deduplicate(words)


def deduplicate(words):
    seen = set()
    result = []
    for w in words:
        if w not in seen:
            seen.add(w)
            result.append(w)
    return result


def load_words(source_path=None):
    if source_path is None:
        path = WORDLIST_PATH
        if not path.exists():
            print(f"ERROR: wordlist.txt not found.")
            sys.exit(1)
    else:
        path = Path(source_path)
        if not path.exists():
            print(f"ERROR: File not found — {path}")
            sys.exit(1)

    ext = path.suffix.lower()
    print(f"  Reading: {path.name} ({ext})")

    if ext == ".pdf":
        return read_pdf(path), path
    elif ext == ".docx":
        return read_docx(path), path
    elif ext == ".csv":
        return read_csv(path), path
    else:
        return read_txt(path), path


# ─── Schema and storage ───────────────────────────────────────────────────────

def load_schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def already_generated(word):
    name = word.replace(" ", "_")
    return (OUTPUT_PATH / f"{name}.cdf.json").exists()


def save_token(token):
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    name = token["token"].replace(" ", "_")
    path = OUTPUT_PATH / f"{name}.cdf.json"
    with open(path, "w") as f:
        json.dump(token, f, indent=2)
    return path


# ─── Main generation ──────────────────────────────────────────────────────────

def generate_token(word, schema):
    """Full two-phase generation: WordNet + LLM."""
    # Phase 1 — WordNet
    wordnet_tracks = get_wordnet_tracks(word)

    # Phase 2 — LLM adds what WordNet missed
    existing_meanings = [t["meaning"] for t in wordnet_tracks]
    llm_tracks = get_llm_tracks(word, existing_meanings)

    # Phase 3 — Assemble
    token = assemble_token(word, wordnet_tracks, llm_tracks)
    if not token:
        raise ValueError(f"Could not generate any tracks for '{word}'")

    jsonschema.validate(instance=token, schema=schema)
    return token


def run_batch(words, schema, source_name="wordlist.txt"):
    new_words = [w for w in words if not already_generated(w)]
    skipped   = len(words) - len(new_words)

    print(f"\nCognitive Data Format — Token Generator")
    print(f"Mode:       WordNet + {MODEL} via Ollama")
    print(f"Source:     {source_name} ({len(words)} words)")
    print(f"Skipped:    {skipped} already generated")
    print(f"Generating: {len(new_words)} new tokens")
    print(f"{'─' * 60}")

    if not new_words:
        print("  All words already generated. Add more words to continue.")
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
            sources = set(tr.get("source","?") for tr in token["tracks"])
            src_str = "+".join(sorted(sources))
            print(f"  ✓  {tracks} tracks [{src_str}]  ({round(time.time()-t,1)}s)")
            valid += 1
        except Exception as e:
            print(f"  ✗  FAILED  ({round(time.time()-t,1)}s)  {str(e)[:50]}")
            failed.append(word)

    total = len(list(OUTPUT_PATH.glob("*.cdf.json")))
    print(f"{'─' * 60}")
    print(f"  Generated:  {valid}/{len(new_words)}")
    print(f"  Failed:     {len(failed)}")
    print(f"  Total in database: {total} tokens")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="CDF Generator — WordNet + LLM two-phase generation"
    )
    parser.add_argument("file",     nargs="?", help="Input file (.txt .pdf .docx .csv)")
    parser.add_argument("--single", type=str,  help="Generate one word and print it")
    parser.add_argument("--count",  type=int,  help="Generate first N new words only")
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

    words, source_path = load_words(args.file)

    if args.count:
        new_words = [w for w in words if not already_generated(w)]
        words = new_words[:args.count]

    run_batch(words, schema, source_name=source_path.name)


if __name__ == "__main__":
    main()
