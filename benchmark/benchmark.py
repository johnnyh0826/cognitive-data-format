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
MODEL          = "claude-sonnet-4-6"

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
        lines.append(f"  Track {track['id']} — {track['context']}: {track['meaning']}")
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


def call_claude(prompt, client):
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


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


def is_correct(result, correct_context):
    if not result:
        return False
    text = json.dumps(result).lower()
    return correct_context.lower() in text


def run_benchmark(test_cases, client):
    print(f"\nCognitive Data Format — Disambiguation Benchmark")
    print(f"Model:      {MODEL}")
    print(f"Test cases: {len(test_cases)}")
    print(f"{'─' * 65}")
    print(f"  {'#':<4} {'Word':<12} {'A:Raw':<10} {'B:CDF':<10} {'A Conf':<10} {'B Conf'}")
    print(f"{'─' * 65}")

    results = []

    for i, (sentence, word, correct_context, distractors) in enumerate(test_cases, 1):
        token = load_cdf_token(word)

        # Condition A — Raw
        t_a = time.time()
        try:
            raw_a     = call_claude(build_raw_prompt(sentence, word), client)
            result_a  = parse_response(raw_a)
            correct_a = is_correct(result_a, correct_context)
            conf_a    = float(result_a.get("confidence", 0.0)) if result_a else 0.0
            time_a    = round(time.time() - t_a, 1)
        except Exception as e:
            print(f"\n  ERROR A [{word}]: {e}")
            correct_a, conf_a, time_a = False, 0.0, 0.0

        # Condition B — CDF
        t_b = time.time()
        if token:
            try:
                raw_b     = call_claude(build_cdf_prompt(sentence, word, token), client)
                result_b  = parse_response(raw_b)
                correct_b = is_correct(result_b, correct_context)
                conf_b    = float(result_b.get("confidence", 0.0)) if result_b else 0.0
                time_b    = round(time.time() - t_b, 1)
            except Exception as e:
                print(f"\n  ERROR B [{word}]: {e}")
                correct_b, conf_b, time_b = False, 0.0, 0.0
        else:
            correct_b, conf_b, time_b = None, 0.0, 0.0

        a_str = "✓" if correct_a else "✗"
        b_str = "✓" if correct_b else ("?" if correct_b is None else "✗")
        print(f"  {i:<4} {word:<12} {a_str:<10} {b_str:<10} {conf_a:<10.2f} {conf_b:.2f}")

        results.append({
            "sentence": sentence,
            "word": word,
            "correct_context": correct_context,
            "condition_a": {"correct": correct_a, "confidence": conf_a, "time": time_a},
            "condition_b": {"correct": correct_b, "confidence": conf_b, "time": time_b},
            "cdf_available": token is not None
        })

    valid     = [r for r in results if r["cdf_available"]]
    a_correct = sum(1 for r in valid if r["condition_a"]["correct"])
    b_correct = sum(1 for r in valid if r["condition_b"]["correct"])
    a_conf    = sum(r["condition_a"]["confidence"] for r in valid) / max(len(valid), 1)
    b_conf    = sum(r["condition_b"]["confidence"] for r in valid) / max(len(valid), 1)
    a_time    = sum(r["condition_a"]["time"] for r in valid) / max(len(valid), 1)
    b_time    = sum(r["condition_b"]["time"] for r in valid) / max(len(valid), 1)
    acc_gain  = round((b_correct - a_correct) / max(len(valid), 1) * 100, 1)
    conf_gain = round((b_conf - a_conf) * 100, 1)

    print(f"{'─' * 65}")
    print(f"\n  RESULTS ({len(valid)} test cases with CDF available)")
    print(f"{'─' * 65}")
    print(f"  {'Metric':<30} {'Condition A (Raw)':<22} {'Condition B (CDF)'}")
    print(f"  {'Accuracy':<30} {a_correct}/{len(valid)} ({round(a_correct/max(len(valid),1)*100)}%){'':>12} {b_correct}/{len(valid)} ({round(b_correct/max(len(valid),1)*100)}%)")
    print(f"  {'Avg Confidence':<30} {round(a_conf,3):<22} {round(b_conf,3)}")
    print(f"  {'Avg Response Time':<30} {round(a_time,1)}s{'':<20} {round(b_time,1)}s")
    print(f"{'─' * 65}")
    print(f"\n  Accuracy improvement:   {'+' if acc_gain >= 0 else ''}{acc_gain}%")
    print(f"  Confidence improvement: {'+' if conf_gain >= 0 else ''}{conf_gain}%\n")

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": MODEL,
        "total_cases": len(test_cases),
        "valid_cases": len(valid),
        "condition_a": {"accuracy": a_correct/max(len(valid),1), "avg_confidence": a_conf, "avg_time": a_time},
        "condition_b": {"accuracy": b_correct/max(len(valid),1), "avg_confidence": b_conf, "avg_time": b_time},
        "accuracy_improvement_pct": acc_gain,
        "confidence_improvement_pct": conf_gain,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="CDF Benchmark — Claude API")
    parser.add_argument("--count", type=int, help="Run first N test cases only")
    parser.add_argument("--model", type=str, help="Claude model to use")
    args = parser.parse_args()

    if args.model:
        global MODEL
        MODEL = args.model

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
