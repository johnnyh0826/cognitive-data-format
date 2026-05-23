# ─── DROP-IN REPLACEMENTS FOR benchmark.py ───────────────────────────────────
# Replace parse_response() and build_raw_prompt_ollama() with these versions.
# Also add build_raw_prompt_ollama() and call_ollama() if not already present.
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

# Keyword map for plain-text context extraction — mirrors generator's approach
CONTEXT_KEYWORDS = {
    "finance":        ["financial", "money", "deposit", "loan", "bank", "fund", "invest", "capital", "credit"],
    "geography":      ["land", "slope", "river", "lake", "shore", "coast", "region", "terrain", "ground"],
    "chemistry":      ["metal", "element", "chemical", "compound", "pb", "lead pipe", "pipe", "toxic", "mineral"],
    "mechanics":      ["spring", "coil", "elastic", "machine", "device", "snap", "tension", "mechanical"],
    "seasons":        ["season", "winter", "summer", "bloom", "flower", "spring arrived", "warm", "autumn"],
    "hydrology":      ["water", "stream", "bubbled", "source", "ground water", "current", "flow", "downstream"],
    "sales":          ["pitch", "investor", "presentation", "business", "sales", "proposal", "deck"],
    "music":          ["note", "tone", "pitch", "musician", "instrument", "melody", "scale", "flat", "sharp"],
    "sports":         ["game", "team", "match", "player", "field", "pitch", "kickoff", "ball", "bat", "score"],
    "investigation":  ["detective", "clue", "lead", "case", "crime", "mystery", "solve", "evidence"],
    "performance":    ["role", "play", "actor", "lead role", "theatre", "film", "starring", "cast"],
    "animal":         ["dog", "bark", "creature", "flew", "cave", "dusk", "mammal", "bat"],
    "botany":         ["tree", "oak", "bark", "trunk", "plant", "wood", "grove", "grooved"],
    "fire":           ["match", "light", "candle", "struck", "flame", "ignite", "burn"],
    "similarity":     ["match", "colour", "shade", "same", "identical", "correspond"],
    "swimming":       ["pool", "swim", "splash", "water", "afternoon", "children"],
    "transport":      ["car pool", "commut", "board", "aircraft", "passenger", "vehicle", "row", "boat"],
    "billiards":      ["pool", "eight ball", "pocket", "cue", "billiard"],
    "labour":         ["strike", "worker", "pay", "demand", "union", "wage"],
    "bowling":        ["strike", "pin", "bowl", "knocked down", "lane"],
    "military":       ["strike", "target", "dawn", "attack", "combat", "draft", "notice"],
    "media":          ["press", "reported", "election", "news", "journalist", "publication"],
    "clothing":       ["suit", "pressed", "dry clean", "iron", "fabric", "garment"],
    "governance":     ["board", "directors", "committee", "quarterly", "meeting", "organisation"],
    "material":       ["board", "wood", "flat", "plank", "cut", "surface"],
    "conflict":       ["row", "argument", "terrible", "couple", "dispute", "fight"],
    "arrangement":    ["row", "garden", "seeds", "straight", "planted", "line"],
    "document":       ["file", "submitted", "tax", "deadline", "record", "paper"],
    "tools":          ["file", "smooth", "metal", "edge", "tool", "abrasive"],
    "electricity":    ["current", "wire", "electrician", "voltage", "circuit", "electric"],
    "measurement":    ["scale", "weighed", "ingredients", "kitchen", "quantity", "measure"],
    "writing":        ["draft", "report", "submitted", "review", "document", "wrote"],
    "weather":        ["draft", "cold", "blew", "gap", "door", "wind", "air"],
    "penalty":        ["fine", "parking", "overstay", "ticket", "infraction", "charged"],
    "quality":        ["fine", "finest", "craftsmanship", "quality", "excellent"],
    "accommodation":  ["flat", "rented", "city centre", "apartment", "tenant"],
    "time":           ["current", "status", "present", "now", "today", "ongoing"],
}


def parse_plain_text(raw, word):
    """
    Extract context and meaning from Llama3 plain-text response.
    Uses keyword matching — same approach as the generator.
    Returns dict with context, meaning, confidence=0.5 (unscored).
    """
    text = raw.lower()
    scores = {}
    for domain, keywords in CONTEXT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[domain] = score

    if not scores:
        return None

    best_context = max(scores, key=scores.get)

    # Extract a short meaning from the response — first sentence that contains the word
    sentences = re.split(r'[.!?\n]', raw)
    meaning = next(
        (s.strip() for s in sentences if word.lower() in s.lower() and len(s.strip()) > 10),
        raw[:120].strip()
    )

    return {
        "context": best_context,
        "meaning": meaning,
        "confidence": 0.5   # plain text — no explicit confidence stated
    }


def parse_response(raw, word=None):
    """
    Parse model response — handles both JSON (Claude) and plain text (Llama3).
    word param optional but improves plain-text extraction when provided.
    """
    # Step 1 — try clean JSON parse
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Step 2 — try to find JSON block within prose
    start = cleaned.find("{")
    end   = cleaned.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end])
        except json.JSONDecodeError:
            pass

    # Step 3 — plain text fallback (Llama3 / Ollama)
    if word:
        return parse_plain_text(raw, word)

    return None


# ─── OLLAMA SUPPORT ──────────────────────────────────────────────────────────

def call_ollama(prompt, model=OLLAMA_MODEL, timeout=30):
    """Call local Ollama model. Returns plain text response."""
    import requests
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout
        )
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        return ""
    except Exception:
        return ""


def build_raw_prompt_ollama(sentence, word):
    """
    Simplified raw prompt for Ollama — plain English, no JSON required.
    Matches the same task as build_raw_prompt() but in a format Llama3 handles.
    """
    return f"""What does the word "{word}" mean in this sentence?

Sentence: "{sentence}"

Answer in one sentence. Start with the domain (e.g. finance, geography, sports) then explain the meaning."""


def build_cdf_prompt_ollama(sentence, word, token):
    """
    Simplified CDF prompt for Ollama — plain text track list, no JSON required.
    """
    lines = [f"  {t['context']}: {t['meaning']}" for t in token["tracks"]]
    tracks_text = "\n".join(lines)

    return f"""What does the word "{word}" mean in this sentence?

Sentence: "{sentence}"

Choose from these known meanings:
{tracks_text}

Answer in one sentence. Say which meaning fits and why."""
