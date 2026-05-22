"""
Cognitive Data Format — Token Generator
Three-phase generation:
  Phase 1 — Free Dictionary API (dictionaryapi.dev): modern comprehensive meanings
  Phase 2 — WordNet: fills academic gaps
  Phase 3 — LLM: catches anything both miss
  Phase 4 — Deduplicate and write CDF token

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
DICT_API_URL  = "https://api.dictionaryapi.dev/api/v2/entries/en"
MODEL         = "llama3"

HEADERS = {"User-Agent": "CDF-Generator/1.0 (cognitive-data-format; educational)"}



def get_dictionary_tracks(word):
    """
    Fetch meanings from Free Dictionary API (dictionaryapi.dev).
    Returns clean tracks with modern definitions.
    """
    try:
        r = requests.get(f"{DICT_API_URL}/{word}", headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return []

        data = r.json()
        tracks = []
        seen  = set()

        for entry in data:
            for meaning in entry.get("meanings", []):
                pos = meaning.get("partOfSpeech", "")
                for defn in meaning.get("definitions", []):
                    definition = defn.get("definition", "").strip()
                    if not definition:
                        continue

                    # Clean HTML tags if any
                    definition = re.sub(r'<[^>]+>', '', definition)

                    # Skip near-duplicates
                    key = definition[:40].lower()
                    if key in seen:
                        continue
                    seen.add(key)

                    context = "general"  # relabel_tracks will assign correct label via Claude

                    tracks.append({
                        "context": context,
                        "meaning": definition,
                        "source": "dictionary"
                    })

                    # no ceiling on dictionary phase either

        return tracks

    except Exception:
        return []


# ─── Phase 2: WordNet fallback ────────────────────────────────────────────────

def get_wordnet_tracks(word, existing_meanings):
    """
    Get WordNet meanings not already covered by the dictionary.
    Only used as fallback when dictionary returns less than 2 meanings.
    """
    synsets = wn.synsets(word, pos=[wn.NOUN, wn.VERB])
    if not synsets:
        synsets = wn.synsets(word)

    tracks = []
    seen   = set(m[:40].lower() for m in existing_meanings)

    for synset in synsets:
        definition = synset.definition()
        key = definition[:40].lower()

        if key in seen:
            continue
        seen.add(key)

        context = "general"  # relabel_tracks will assign correct label via Claude

        tracks.append({
            "context": context,
            "meaning": definition,
            "source": "wordnet"
        })

        if len(tracks) >= 2:
            break

    return tracks


# ─── Phase 3: Claude API — accumulated meanings ──────────────────────────────

def get_claude_client():
    """Get Anthropic client — lazy init so missing key only fails when needed."""
    try:
        import anthropic
        return anthropic.Anthropic()
    except ImportError:
        print("ERROR: anthropic not installed. Run: pip install anthropic")
        sys.exit(1)


def call_claude(prompt, client, max_tokens=400):
    """Call Claude API with retry on overload."""
    import anthropic
    for attempt in range(3):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except anthropic.APIStatusError as e:
            if e.status_code == 529 and attempt < 2:
                time.sleep(5 + attempt * 3)
                continue
            raise
    return ""


def get_llm_tracks(word, existing_meanings, client=None):
    """Ask Claude for meanings not covered by dictionary or WordNet."""
    if client is None:
        return []

    existing_text = "\n".join(f"- {m}" for m in existing_meanings)
    prompt = f"""You are building a Cognitive Data Format (CDF) database.

Word: "{word}"

Meanings already captured from dictionary sources:
{existing_text}

Task: List ALL remaining distinct meanings a fluent English speaker would recognise that are NOT covered above.
Focus especially on: verbs, informal uses, domain-specific uses, and compound meanings (e.g. "lead" as a metal vs. a verb vs. a role).

Rules:
- Include meanings even if partially similar to existing ones — err on the side of inclusion
- Do NOT limit to 1-2 — return as many as genuinely exist and are missing above
- Each must have a specific domain label — never use "general"
- Return ONLY valid JSON — no explanation, no markdown

Format:
{{"additional_tracks": [{{"context": "specific domain label", "meaning": "clear one sentence definition"}}]}}

Or if nothing to add:
{{"additional_tracks": []}}"""

    try:
        raw = call_claude(prompt, client)
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


# ─── Phase 4: Frequency scoring ──────────────────────────────────────────────

def get_frequency_scores(word, tracks, client=None):
    """Ask Claude to estimate real-world usage frequency per track."""
    if client is None:
        return None

    track_list = "\n".join(
        f"  Track {t['id']} — {t['context']}: {t['meaning'][:80]}"
        for t in tracks
    )
    ids  = [t["id"] for t in tracks]
    empty = {id: 0.0 for id in ids}

    prompt = f"""For the word "{word}", estimate what proportion of everyday real-world usage each meaning below represents.

{track_list}

Rules:
- Each value must be between 0.0 and 1.0
- All values must sum to exactly 1.0
- Base estimates on how commonly each meaning appears in everyday English
- Return ONLY valid JSON — no explanation

{{"frequencies": {json.dumps(empty)}}}"""

    try:
        raw = call_claude(prompt, client, max_tokens=200)
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().replace("```", "").strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end   = cleaned.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(cleaned[start:end])
            else:
                return None

        frequencies = data.get("frequencies", {})
        total = sum(float(v) for v in frequencies.values())
        if total == 0:
            return None
        return {k: round(float(v) / total, 3) for k, v in frequencies.items()}

    except Exception:
        return None


# ─── Phase 5: Assemble CDF token ─────────────────────────────────────────────

def assemble_token(word, all_tracks):
    """Combine all tracks, deduplicate, and build final CDF token."""
    final_tracks = []
    seen_meanings = set()
    seen_contexts = set()  # enforce unique context labels

    for i, track in enumerate(all_tracks):
        meaning = track.get("meaning", "").strip()
        context = track.get("context", "general")
        key = meaning[:40].lower()

        if key in seen_meanings or not meaning:
            continue
        if context in seen_contexts:  # skip if this context label already used
            continue

        seen_meanings.add(key)
        seen_contexts.add(context)

        track_id = str(len(final_tracks) + 1).zfill(2)
        final_tracks.append({
            "id": track_id,
            "context": context,
            "meaning": meaning,
            "added": "first" if len(final_tracks) == 0 else "accumulated",
            "source": track.get("source", "llm")
        })

        # no ceiling — track count reflects real semantic load of the word

    if not final_tracks:
        return None

    return {"token": word, "tracks": final_tracks}


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
        "the","and","for","are","but","not","you","all","can","her","was","one",
        "our","out","day","get","has","him","his","how","its","may","new","now",
        "old","see","two","way","who","did","does","had","have","been","with",
        "that","this","from","they","will","what","when","your","said","each",
        "she","use","more","also","into","than","then","them","some","would",
        "make","like","just","know","take","very","even","most","back","after",
        "could","these","first","those","only","over","such","here","should",
        "about","there","think","every","never","under","other","right","come",
        "both","little","being","because","going","still","down",
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

    if ext == ".pdf":    return read_pdf(path), path
    elif ext == ".docx": return read_docx(path), path
    elif ext == ".csv":  return read_csv(path), path
    else:                return read_txt(path), path


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



# ─── Pure LLM generation ─────────────────────────────────────────────────────

def get_llm_tracks_pure(word, client=None):
    """Ask Claude to generate all tracks from scratch — no dictionary, no WordNet."""
    if client is None:
        return []

    prompt = f"""You are building a Cognitive Data Format (CDF) token database.

Word: "{word}"

Task: List ALL distinct meanings a fluent English speaker would recognise for this word.
Cover every major domain this word appears in — nouns, verbs, informal uses, technical uses.

Examples for "pitch":
- music: the highness or lowness of a sound or musical note
- business: a sales presentation made to persuade someone to buy or invest
- sports: the act of throwing a ball toward a batter in baseball
- construction: a dark sticky substance made from tar, used for waterproofing
- geography: the angle or slope of a roof or surface

Rules:
- Be exhaustive — include all common meanings, not just the most obvious one
- Each context label must be unique and specific (never use "general")
- Use the most precise domain label that fits — labels are open vocabulary, not restricted to a fixed list
- Good labels: finance, geography, botany, zoology, music, sports, mechanics, clothing, accommodation, chemistry, medicine, military, computing, law, printing, metallurgy, carpentry, nautical, geology, psychology, informal, agriculture, etc.
- Be as specific as possible — "botany" is better than "biology", "zoology" is better than "animal"
- Each meaning must be a clear one-sentence definition
- Return ONLY valid JSON — no explanation, no markdown

Format:
{{"tracks": [{{"context": "specific domain", "meaning": "clear one sentence definition"}}]}}"""

    try:
        raw = call_claude(prompt, client, max_tokens=1200)
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
        tracks = data.get("tracks", [])
        for t in tracks:
            t["source"] = "llm"
        return tracks
    except Exception:
        return []

# ─── Context relabeling ───────────────────────────────────────────────────────

def relabel_tracks(word, tracks, client):
    """Ask Claude to assign correct unique domain labels to all tracks at once."""
    if not tracks or client is None:
        return tracks

    track_list = "\n".join(
        f"  {i+1}. {t['meaning']}"
        for i, t in enumerate(tracks)
    )
    prompt = f"""For the word "{word}", assign the single most specific domain label to each meaning below.

{track_list}

Rules:
- Labels are open vocabulary — use the most precise domain label that fits, do not restrict to a fixed list
- Be specific: "botany" not "biology", "zoology" not "animal", "mechanics" not "physics"
- Never use "general", "governance" for a bank branch, or "sports" for a card game
- Each label must be unique across all tracks — if two meanings share a domain pick the closest alternative for the less important one
- Return ONLY valid JSON — no explanation, no markdown
- Array must have exactly {len(tracks)} labels in the same order as the meanings above

{{"labels": ["label1", "label2", ...]}}"""

    try:
        raw = call_claude(prompt, client, max_tokens=300)
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().replace("```", "").strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end   = cleaned.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(cleaned[start:end])
            else:
                return tracks
        labels = data.get("labels", [])
        if len(labels) == len(tracks):
            for i, track in enumerate(tracks):
                track["context"] = labels[i]
        return tracks
    except Exception:
        return tracks  # fall back to existing labels if Claude fails

def generate_token(word, schema, client=None):
    """Full generation: Dictionary + WordNet + Claude + Frequency."""

    # Phase 1 — Claude API only (pure LLM mode)
    dict_tracks = []
    wordnet_tracks = []
    llm_tracks = get_llm_tracks_pure(word, client=client)

    # Phase 2 — Assemble and deduplicate
    token = assemble_token(word, llm_tracks)
    if not token:
        raise ValueError(f"Could not generate any tracks for '{word}'")

    # Phase 6 — Claude frequency scoring
    frequencies = get_frequency_scores(word, token["tracks"], client=client)
    if frequencies:
        for track in token["tracks"]:
            if track["id"] in frequencies:
                track["frequency"] = frequencies[track["id"]]

    jsonschema.validate(instance=token, schema=schema)
    return token


def run_batch(words, schema, source_name="wordlist.txt"):
    new_words = [w for w in words if not already_generated(w)]
    skipped   = len(words) - len(new_words)

    # Init Claude client for bootstrap
    client = get_claude_client()

    print(f"\nCognitive Data Format — Token Generator")
    print(f"Mode:       Claude API only (pure LLM)")
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
            token  = generate_token(word, schema, client=client)
            save_token(token)
            tracks  = len(token["tracks"])
            sources = set(tr.get("source", "?") for tr in token["tracks"])
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
        description="CDF Generator — Dictionary API + WordNet + LLM"
    )
    parser.add_argument("file",     nargs="?", help="Input file (.txt .pdf .docx .csv)")
    parser.add_argument("--single", type=str,  help="Generate one word and print it")
    parser.add_argument("--count",  type=int,  help="Generate first N new words only")
    args = parser.parse_args()

    schema = load_schema()

    if args.single:
        print(f"\nGenerating: {args.single}\n{'─'*40}")
        client = get_claude_client()
        try:
            token = generate_token(args.single, schema, client=client)
            path = save_token(token)
            print(json.dumps(token, indent=2))
            print(f"\n  Saved: {path}")
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
