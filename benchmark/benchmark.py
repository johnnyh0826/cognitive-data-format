"""
Cognitive Data Format — Benchmark
Compares disambiguation accuracy and confidence using Claude API.

Usage:
    python benchmark/benchmark.py --count 10
    python benchmark/benchmark.py
"""

import json
import re
import time
import argparse
import anthropic
from pathlib import Path
from datetime import datetime

GENERATED_PATH = Path(__file__).parent.parent / "data" / "generated"
RESULTS_PATH   = Path(__file__).parent.parent / "data" / "benchmark"
MODEL_RAW      = "claude-sonnet-4-6"   # Condition A — unbounded reasoning, needs full model
MODEL_CDF      = "claude-haiku-4-5-20251001"  # Condition B — bounded selection, lightweight

# Anthropic API pricing per million tokens (as of May 2026)
PRICING = {
    "claude-sonnet-4-6": {
        "input":  3.00,   # $ per 1M input tokens
        "output": 15.00   # $ per 1M output tokens
    },
    "claude-haiku-4-5-20251001": {
        "input":  0.80,   # $ per 1M input tokens
        "output": 4.00    # $ per 1M output tokens
    }
}

def calc_cost(model, input_tokens, output_tokens):
    """Calculate API cost in USD for a single call."""
    p = PRICING.get(model, {"input": 0, "output": 0})
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000

TEST_CASES = [
    ("I need to visit the bank before it closes at 5pm", "bank", "finance", ["geography"]),
    ("The fisherman sat on the bank watching his line", "bank", "geography", ["finance"]),
    ("The spring snapped under the weight of the machine", "spring", "mechanics", ["seasons"]),
    ("We went hiking when spring arrived and the flowers bloomed", "spring", "seasons", ["mechanics"]),
    ("Fresh water bubbled up from the spring in the hillside", "spring", "hydrology", ["seasons"]),
    ("The sales team prepared a pitch for the new investors", "pitch", "sales", ["music", "sports"]),
    ("The musician adjusted the pitch before the performance", "pitch", "music", ["sales", "sports"]),
    ("The players warmed up on the pitch before kickoff", "pitch", "sports", ["music", "sales"]),
    ("The detective followed every lead in the case", "lead", "investigation", ["chemistry"]),
    ("The pipes were made of lead and needed replacing", "lead", "chemistry", ["investigation"]),
    ("She took the lead role in the school play", "lead", "performance", ["chemistry"]),
    ("The dog let out a sharp bark at the stranger", "bark", "animal", ["botany"]),
    ("The bark of the oak tree was rough and deeply grooved", "bark", "botany", ["animal"]),
    ("She struck a match to light the candle", "match", "fire", ["sports"]),
    ("The two teams played a match that lasted three hours", "match", "sports", ["fire"]),
    ("The colours don't match — try a different shade", "match", "similarity", ["sports"]),
    ("The children splashed in the pool all afternoon", "pool", "swimming", ["billiards"]),
    ("The company set up a car pool to reduce commuting costs", "pool", "transport", ["swimming"]),
    ("He sank the eight ball in the corner pocket playing pool", "pool", "billiards", ["swimming"]),
    ("The workers went on strike demanding better pay", "strike", "labour", ["bowling"]),
    ("He knocked down all ten pins for a strike", "strike", "bowling", ["labour"]),
    ("The press reported on the election results", "press", "media", ["mechanics"]),
    ("She used the press to extract olive oil from the harvest", "press", "mechanics", ["media"]),
    ("He sent his suit to be dry cleaned and pressed", "press", "clothing", ["media"]),
    ("The batter swung the bat and hit a home run", "bat", "sports", ["animal"]),
    ("A bat flew out of the cave at dusk", "bat", "animal", ["sports"]),
    ("The board of directors met to discuss quarterly results", "board", "governance", ["transport"]),
    ("Passengers began to board the aircraft", "board", "transport", ["governance"]),
    ("The couple had a terrible row about money", "row", "conflict", ["transport"]),
    ("They rowed the boat across the lake at sunrise", "row", "transport", ["conflict"]),
    ("She submitted the tax file before the deadline", "file", "document", ["tools"]),
    ("He used a file to smooth the rough metal edge", "file", "tools", ["document"]),
    ("The current carried the boat downstream rapidly", "current", "hydrology", ["electricity"]),
    ("The electrician checked the current in the wire", "current", "electricity", ["hydrology"]),
    ("She weighed the ingredients on a kitchen scale", "scale", "measurement", ["music"]),
    ("The musician practiced scales for an hour each morning", "scale", "music", ["measurement"]),
    ("He submitted the first draft of the report for review", "draft", "writing", ["military"]),
    ("The young men received their draft notice in the mail", "draft", "military", ["writing"]),
    ("A cold draft blew in through the gap under the door", "draft", "weather", ["writing"]),
    ("She received a parking fine for overstaying the limit", "fine", "penalty", ["quality"]),
    ("The craftsmanship was of the finest quality", "fine", "quality", ["penalty"]),
    ("It was a fine sunny day with no clouds", "fine", "weather", ["penalty"]),
    ("The tyre went flat halfway through the journey", "flat", "mechanics", ["music"]),
    ("She rented a flat in the city centre", "flat", "accommodation", ["mechanics"]),
    ("The musician played a note that was slightly flat", "flat", "music", ["mechanics"]),
    ("She planted seeds in a long straight row in the garden", "row", "arrangement", ["conflict"]),
    ("He cut the wood using a long flat board", "board", "material", ["governance"]),
    ("The map was drawn to a scale of one to fifty thousand", "scale", "geography", ["measurement"]),
    ("What is the current status of the project?", "current", "time", ["hydrology"]),
    ("The strike hit the target precisely at dawn", "strike", "military", ["labour"]),
]


def load_cdf_token(word):
    path = GENERATED_PATH / f"{word}.cdf.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def format_tracks(token):
    lines = []
    for track in token["tracks"]:
        freq = track.get("frequency")
        freq_str = f" [usage: {freq:.0%}]" if freq is not None else ""
        lines.append(f"  Track {track['id']} — {track['context']}: {track['meaning']}{freq_str}")
    return "\n".join(lines)


def build_raw_prompt(sentence, word):
    return f"""You are a precise word sense disambiguation system.

Sentence: "{sentence}"
Target word: "{word}"

Identify exactly what "{word}" means in this specific sentence.

Respond with ONLY this JSON — no explanation, no markdown:
{{"context": "one word domain label e.g. finance geography mechanics music", "meaning": "one sentence definition for this usage", "confidence": 0.95}}"""


def build_cdf_prompt(sentence, word, token):
    tracks_text = format_tracks(token)
    return f"""You are a precise word sense disambiguation system using Cognitive Data Format (CDF).

Sentence: "{sentence}"
Target word: "{word}"

Known meanings (CDF tracks):
{tracks_text}

Select which track best matches the meaning of "{word}" in this sentence.

Respond with ONLY this JSON — no explanation, no markdown:
{{"selected_track": "context label from the tracks above", "meaning": "the meaning from that track", "confidence": 0.95}}"""


def call_claude(prompt, client, model=None):
    if model is None:
        model = MODEL_RAW
    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    usage = {
        "input_tokens":  response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens":  response.usage.input_tokens + response.usage.output_tokens
    }
    return response.content[0].text, usage


def parse_response(raw):
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end   = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
    return None


def is_correct(result, correct_context, sentence=None, word=None, client=None):
    """
    Semantic correctness judge — asks Claude whether the selected meaning
    matches the expected context. Falls back to string match if no client.
    """
    if not result:
        return False

    # Fast path — exact string match
    text = json.dumps(result).lower()
    if correct_context.lower() in text:
        return True

    # Semantic judge via Claude
    if client is None:
        return False

    selected_context = result.get("selected_track", result.get("context", ""))
    selected_meaning = result.get("meaning", "")

    prompt = f"""You are evaluating a word sense disambiguation result.

Word: "{word}"
Sentence: "{sentence}"
Expected domain: "{correct_context}"
Selected domain: "{selected_context}"
Selected meaning: "{selected_meaning}"

Question: Is the selected meaning semantically correct for this sentence, even if the domain label differs from the expected one?

Rules:
- Judge the MEANING, not the label string
- "zoology" and "animal" for a dog barking are both correct
- "mechanics" and "engineering" for a spring coil are both correct
- Return ONLY valid JSON — no explanation

{{"correct": true}} or {{"correct": false}}"""

    try:
        response = client.messages.create(
            model=MODEL_CDF,
            max_tokens=20,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().replace("```", "").strip()
        data = json.loads(cleaned)
        return bool(data.get("correct", False))
    except Exception:
        return False


def parse_response(raw):
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end   = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
    return None


# Synonym map — benchmark expected label → acceptable alternatives in generated tokens
CONTEXT_SYNONYMS = {
    # spring
    "mechanics":     ["mechanics", "mechanical", "engineering", "physics", "machinery",
                      "engineering", "construction", "technology", "coil", "elastic"],
    "seasons":       ["seasons", "season", "time", "nature", "biology", "weather",
                      "seasonal", "spring", "climate", "calendar"],
    "hydrology":     ["hydrology", "geography", "water", "geology", "nature",
                      "geology", "earth", "landscape", "source", "freshwater"],
    # bank
    "finance":       ["finance", "financial", "banking", "economics", "business",
                      "monetary", "institution", "commerce", "credit"],
    "geography":     ["geography", "geographical", "terrain", "land", "environment",
                      "landscape", "earth", "riverbank", "slope", "geology", "topography",
                      "cartography", "navigation", "physical"],
    # pitch
    "sales":         ["sales", "business", "commerce", "marketing", "commercial",
                      "presentation", "proposal", "persuasion", "pitch"],
    "music":         ["music", "musical", "acoustics", "audio", "sound",
                      "tone", "frequency", "note", "acoustic"],
    "sports":        ["sports", "athletics", "baseball", "football", "cricket",
                      "sport", "game", "field", "ground", "pitch"],
    # lead
    "investigation": ["investigation", "law", "detective", "criminology", "forensics",
                      "crime", "police", "clue", "journalism", "reporting", "media"],
    "chemistry":     ["chemistry", "chemical", "metallurgy", "material", "physics",
                      "metal", "element", "mineral", "geology", "materials"],
    "performance":   ["performance", "theatre", "acting", "entertainment", "drama",
                      "role", "lead", "acting", "film", "music", "entertainment"],
    # bark
    "animal":        ["animal", "biology", "zoology", "dog", "canine", "creature",
                      "mammal", "sound", "vocalization", "behaviour"],
    "botany":        ["botany", "biology", "plant", "tree", "nature", "forestry",
                      "wood", "flora", "horticulture", "arboriculture"],
    # match
    "similarity":    ["similarity", "matching", "comparison", "general", "informal",
                      "colour", "color", "design", "fashion", "appearance", "visual"],
    # pool
    "transport":     ["transport", "nautical", "movement", "travel", "rowing",
                      "commute", "vehicle", "carpool", "sharing", "driving", "road"],
    # press
    "clothing":      ["clothing", "fashion", "garment", "textile", "laundry",
                      "fabric", "ironing", "tailoring", "dry cleaning", "apparel"],
    # bat
    "animal":        ["animal", "biology", "zoology", "mammal", "creature", "wildlife",
                      "nocturnal", "flying", "nature"],
    # board
    "governance":    ["governance", "business", "corporate", "management", "organisation",
                      "committee", "directors", "executive", "administration", "council"],
    "transport":     ["transport", "nautical", "movement", "travel", "rowing",
                      "boarding", "vehicle", "ship", "aircraft", "travel"],
    # row
    "conflict":      ["conflict", "argument", "dispute", "social", "informal",
                      "quarrel", "disagreement", "fight", "altercation", "british"],
    # file
    "tools":         ["tools", "metalwork", "craft", "construction", "manufacturing",
                      "tool", "abrasive", "smoothing", "workshop", "hardware"],
    "document":      ["document", "computing", "technology", "records", "filing",
                      "data", "storage", "archive", "office", "paperwork"],
    # draft
    "weather":       ["weather", "climate", "atmosphere", "air", "wind",
                      "breeze", "airflow", "ventilation", "draught", "cold"],
    "military":      ["military", "warfare", "combat", "army", "conscription",
                      "draft", "service", "enlistment", "armed forces", "war"],
    "writing":       ["writing", "document", "communication", "publishing", "editing",
                      "manuscript", "version", "copy", "text", "composition"],
    # fine
    "penalty":       ["penalty", "law", "legal", "finance", "regulation",
                      "fine", "ticket", "infraction", "charge", "fee", "punishment"],
    "quality":       ["quality", "description", "general", "informal", "adjective",
                      "excellent", "superior", "good", "high quality", "refined"],
    # flat
    "accommodation": ["accommodation", "housing", "real estate", "property", "living",
                      "apartment", "residence", "dwelling", "home", "rental"],
    # scale
    "measurement":   ["measurement", "science", "mathematics", "scaling",
                      "ratio", "proportion", "map", "cartography", "geography",
                      "size", "magnitude", "scope", "range"],
    # strike
    "labour":        ["labour", "labor", "employment", "industrial", "union",
                      "worker", "strike", "protest", "industrial action", "walkout"],
    "bowling":       ["bowling", "sports", "athletics", "cricket", "baseball",
                      "pin", "alley", "ball", "frame", "game"],
    "media":         ["media", "press", "journalism", "communication", "publishing",
                      "newspaper", "broadcast", "news", "reporter", "outlet"],
    "arrangement":   ["arrangement", "agriculture", "farming", "gardening", "layout",
                      "garden", "row", "line", "planting", "sequence"],
    "material":      ["material", "construction", "woodworking", "manufacturing", "building",
                      "timber", "lumber", "plank", "wood", "structural"],
    "time":          ["time", "temporal", "current", "present", "general",
                      "contemporary", "existing", "ongoing", "now", "modern"],
    "billiards":     ["billiards", "sports", "games", "recreation", "pool",
                      "snooker", "cue", "table", "ball", "leisure"],
    "swimming":      ["swimming", "sports", "recreation", "leisure", "pool",
                      "aquatic", "water", "swim", "lido", "natatorium"],
    "electricity":   ["electricity", "electrical", "electronics", "physics", "technology",
                      "current", "voltage", "circuit", "power", "electron"],
}



def run_benchmark(test_cases, client):
    print(f"\nCognitive Data Format — Disambiguation Benchmark")
    print(f"Model A (Raw): {MODEL_RAW}")
    print(f"Model B (CDF): {MODEL_CDF}")
    print(f"Test cases: {len(test_cases)}")
    print(f"{'─' * 65}")
    print(f"  {'#':<4} {'Word':<12} {'A:Raw':<10} {'B:CDF':<10} {'A Conf':<10} {'B Conf':<10} Notes")
    print(f"{'─' * 65}")

    results = []

    for i, (sentence, word, correct_context, distractors) in enumerate(test_cases, 1):
        token = load_cdf_token(word)

        # Condition A — Raw (Sonnet — unbounded reasoning)
        t_a = time.time()
        try:
            raw_a, usage_a = call_claude(build_raw_prompt(sentence, word), client, model=MODEL_RAW)
            result_a  = parse_response(raw_a)
            correct_a = is_correct(result_a, correct_context, sentence, word, client)
            conf_a    = float(result_a.get("confidence", 0.0)) if result_a else 0.0
            time_a    = round(time.time() - t_a, 1)
            tok_in_a  = usage_a["input_tokens"]
            tok_out_a = usage_a["output_tokens"]
            cost_a    = calc_cost(MODEL_RAW, tok_in_a, tok_out_a)
        except Exception as e:
            print(f"\n  ERROR A [{word}]: {e}")
            correct_a, conf_a, time_a, tok_in_a, tok_out_a, cost_a = False, 0.0, 0.0, 0, 0, 0.0

        # Condition B — CDF
        t_b = time.time()
        if token:
            try:
                raw_b, usage_b = call_claude(build_cdf_prompt(sentence, word, token), client, model=MODEL_CDF)
                result_b  = parse_response(raw_b)
                correct_b = is_correct(result_b, correct_context, sentence, word, client)
                conf_b    = float(result_b.get("confidence", 0.0)) if result_b else 0.0
                time_b    = round(time.time() - t_b, 1)
                tok_in_b  = usage_b["input_tokens"]
                tok_out_b = usage_b["output_tokens"]
                cost_b    = calc_cost(MODEL_CDF, tok_in_b, tok_out_b)
            except Exception as e:
                print(f"\n  ERROR B [{word}]: {e}")
                correct_b, conf_b, time_b, tok_in_b, tok_out_b, cost_b = False, 0.0, 0.0, 0, 0, 0.0
        else:
            correct_b, conf_b, time_b, tok_in_b, tok_out_b, cost_b = None, 0.0, 0.0, 0, 0, 0.0

        a_str = "✓" if correct_a else "✗"
        b_str = "✓" if correct_b else ("?" if correct_b is None else "✗")
        print(f"  {i:<4} {word:<12} {a_str:<10} {b_str:<10} {conf_a:<10.2f} {conf_b:.2f}")

        results.append({
            "sentence": sentence,
            "word": word,
            "correct_context": correct_context,
            "condition_a": {"correct": correct_a, "confidence": conf_a, "time": time_a,
                            "tokens_in": tok_in_a, "tokens_out": tok_out_a, "cost_usd": cost_a},
            "condition_b": {"correct": correct_b, "confidence": conf_b, "time": time_b,
                            "tokens_in": tok_in_b, "tokens_out": tok_out_b, "cost_usd": cost_b},
            "cdf_available": token is not None
        })

    valid     = [r for r in results if r["cdf_available"]]
    a_correct = sum(1 for r in valid if r["condition_a"]["correct"])
    b_correct = sum(1 for r in valid if r["condition_b"]["correct"])
    a_conf    = sum(r["condition_a"]["confidence"] for r in valid) / max(len(valid), 1)
    b_conf    = sum(r["condition_b"]["confidence"] for r in valid) / max(len(valid), 1)
    a_time     = sum(r["condition_a"]["time"] for r in valid) / max(len(valid), 1)
    b_time     = sum(r["condition_b"]["time"] for r in valid) / max(len(valid), 1)
    a_tok_in   = sum(r["condition_a"]["tokens_in"]  for r in valid) / max(len(valid), 1)
    a_tok_out  = sum(r["condition_a"]["tokens_out"] for r in valid) / max(len(valid), 1)
    b_tok_in   = sum(r["condition_b"]["tokens_in"]  for r in valid) / max(len(valid), 1)
    b_tok_out  = sum(r["condition_b"]["tokens_out"] for r in valid) / max(len(valid), 1)
    a_cost     = sum(r["condition_a"]["cost_usd"]   for r in valid) / max(len(valid), 1)
    b_cost     = sum(r["condition_b"]["cost_usd"]   for r in valid) / max(len(valid), 1)
    acc_gain   = round((b_correct - a_correct) / max(len(valid), 1) * 100, 1)
    conf_gain  = round((b_conf - a_conf) * 100, 1)
    tok_saving = round((1 - (b_tok_in + b_tok_out) / max(a_tok_in + a_tok_out, 1)) * 100, 1)
    cost_saving = round((1 - b_cost / max(a_cost, 0.000001)) * 100, 1)

    print(f"{'─' * 65}")
    print(f"\n  RESULTS ({len(valid)} test cases with CDF available)")
    print(f"{'─' * 65}")
    print(f"  {'Metric':<30} {'Condition A (Raw)':<22} {'Condition B (CDF)'}")
    print(f"  {'Accuracy':<30} {a_correct}/{len(valid)} ({round(a_correct/max(len(valid),1)*100)}%){'':>12} {b_correct}/{len(valid)} ({round(b_correct/max(len(valid),1)*100)}%)")
    print(f"  {'Avg Confidence':<30} {round(a_conf,3):<22} {round(b_conf,3)}")
    print(f"  {'Avg Response Time':<30} {round(a_time,1)}s{'':<20} {round(b_time,1)}s")
    print(f"  {'Avg Input Tokens':<30} {round(a_tok_in):<22} {round(b_tok_in)}")
    print(f"  {'Avg Output Tokens':<30} {round(a_tok_out):<22} {round(b_tok_out)}")
    print(f"  {'Avg Total Tokens':<30} {round(a_tok_in+a_tok_out):<22} {round(b_tok_in+b_tok_out)}")
    print(f"  {'Avg Cost per Query':<30} ${a_cost*1000:.4f}/1k queries{'':<6} ${b_cost*1000:.4f}/1k queries")
    print(f"{'─' * 65}")
    print(f"\n  Accuracy improvement:   {'+' if acc_gain >= 0 else ''}{acc_gain}%")
    print(f"  Confidence improvement: {'+' if conf_gain >= 0 else ''}{conf_gain}%")
    print(f"  Cost saving:            {'+' if cost_saving >= 0 else ''}{cost_saving}% cheaper per query")
    print(f"  Speed improvement:      {round((1 - b_time/max(a_time,0.001))*100, 1)}% faster\n")

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_raw": MODEL_RAW,
        "model_cdf": MODEL_CDF,
        "total_cases": len(test_cases),
        "valid_cases": len(valid),
        "condition_a": {"accuracy": a_correct/max(len(valid),1), "avg_confidence": a_conf,
                        "avg_time": a_time, "avg_tokens_in": a_tok_in, "avg_tokens_out": a_tok_out,
                        "avg_cost_usd": a_cost},
        "condition_b": {"accuracy": b_correct/max(len(valid),1), "avg_confidence": b_conf,
                        "avg_time": b_time, "avg_tokens_in": b_tok_in, "avg_tokens_out": b_tok_out,
                        "avg_cost_usd": b_cost},
        "accuracy_improvement_pct": acc_gain,
        "confidence_improvement_pct": conf_gain,
        "token_efficiency_gain_pct": tok_saving,
        "cost_saving_pct": cost_saving,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="CDF Benchmark — Claude API")
    parser.add_argument("--count", type=int, help="Run first N test cases only")
    parser.add_argument("--model", type=str, help="Claude model to use")
    args = parser.parse_args()

    if args.model:
        global MODEL_RAW, MODEL_CDF
        MODEL_RAW = args.model
        MODEL_CDF = args.model

    client = anthropic.Anthropic()
    cases  = TEST_CASES[:args.count] if args.count else TEST_CASES
    summary = run_benchmark(cases, client)

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_file = str(RESULTS_PATH / f"benchmark_claude_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved: {output_file}\n")


if __name__ == "__main__":
    main()
