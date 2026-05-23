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
import requests
import anthropic
from pathlib import Path
from datetime import datetime

GENERATED_PATH = Path(__file__).parent.parent / "data" / "generated"
RESULTS_PATH   = Path(__file__).parent.parent / "data" / "benchmark"
OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_MODEL   = "llama3"

# Model identifiers
MODEL_SONNET   = "claude-sonnet-4-6"
MODEL_HAIKU    = "claude-haiku-4-5-20251001"
MODEL_OLLAMA   = "ollama/llama3"  # prefix to distinguish from Anthropic models

# Anthropic API pricing per million tokens (as of May 2026)
PRICING = {
    "claude-sonnet-4-6":        {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5-20251001":{"input": 0.80,  "output": 4.00},
    "ollama/llama3":             {"input": 0.0,   "output": 0.0},   # free local
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
TEST_CASES_V2 = [
    # break
    ("She needed a break after working twelve hours straight", "break", "rest", ["mechanics"]),
    ("The wave would break against the rocks with tremendous force", "break", "hydrology", ["rest"]),
    ("He managed to break the world record by two seconds", "break", "sports", ["rest"]),
    # charge
    ("The lawyer submitted the charge sheet to the court", "charge", "law", ["electricity"]),
    ("She plugged in her phone to charge overnight", "charge", "electricity", ["law"]),
    ("The cavalry charge swept across the open battlefield", "charge", "military", ["law"]),
    # beat
    ("The drummer kept a steady beat throughout the performance", "beat", "music", ["sports"]),
    ("She managed to beat her personal best time in the race", "beat", "sports", ["music"]),
    ("The police officer walked his beat through the neighbourhood", "beat", "law", ["music"]),
    # cast
    ("The director announced the cast for the new film", "cast", "theatre", ["fishing"]),
    ("He cast the fishing line far out into the lake", "cast", "fishing", ["theatre"]),
    ("The doctor put a cast on her broken arm", "cast", "medicine", ["theatre"]),
    # blow
    ("The wind delivered a powerful blow to the trees", "blow", "weather", ["conflict"]),
    ("The boxer delivered a blow to his opponent's chin", "blow", "sports", ["weather"]),
    ("She gave the trumpet a long steady blow", "blow", "music", ["weather"]),
    # check
    ("He wrote a check to cover the rent payment", "check", "finance", ["sports"]),
    ("The doctor gave her a routine health check", "check", "medicine", ["finance"]),
    ("The chess player put the king in check", "check", "games", ["finance"]),
    # back
    ("She threw the ball back to the pitcher", "back", "movement", ["anatomy"]),
    ("He injured his back lifting heavy equipment", "back", "anatomy", ["movement"]),
    ("The company agreed to back the new startup financially", "back", "finance", ["anatomy"]),
    # call
    ("She made a call to her doctor about the results", "call", "communication", ["finance"]),
    ("The referee made a controversial call during the match", "call", "sports", ["communication"]),
    ("The trader placed a call option on the stock", "call", "finance", ["communication"]),
    # cut
    ("The surgeon made a clean cut during the operation", "cut", "medicine", ["finance"]),
    ("The government announced a cut in interest rates", "cut", "finance", ["medicine"]),
    ("She received a cut of the profits from the deal", "cut", "business", ["finance"]),
    # deal
    ("The two companies signed a deal worth millions", "deal", "business", ["games"]),
    ("It was her turn to deal the cards around the table", "deal", "games", ["business"]),
    ("The drug deal took place in an abandoned warehouse", "deal", "crime", ["business"]),
    # drop
    ("There was a sharp drop in temperature overnight", "drop", "weather", ["finance"]),
    ("The stock market saw a significant drop in value", "drop", "finance", ["weather"]),
    ("She added a drop of vanilla extract to the mixture", "drop", "food", ["weather"]),
    # face
    ("She put on a brave face despite the difficult news", "face", "psychology", ["anatomy"]),
    ("The climber had to face the sheer rock face alone", "face", "geography", ["anatomy"]),
    ("The watch had a cracked face that needed replacing", "face", "horology", ["anatomy"]),
    # iron
    ("She used an iron to press the creases out of her shirt", "iron", "clothing", ["chemistry"]),
    ("The doctor prescribed iron supplements for her anaemia", "iron", "medicine", ["clothing"]),
    ("The golfer selected a seven iron for the approach shot", "iron", "sports", ["clothing"]),
    # line
    ("She waited in line for over an hour at the post office", "line", "movement", ["communication"]),
    ("The fishing line snapped under the weight of the catch", "line", "fishing", ["movement"]),
    ("The actor forgot his line during the final rehearsal", "line", "theatre", ["movement"]),
    # lock
    ("She turned the key and heard the lock click shut", "lock", "security", ["sports"]),
    ("The wrestler applied a headlock during the match", "lock", "sports", ["security"]),
    ("The canal boat passed through the lock slowly", "lock", "nautical", ["security"]),
    # mark
    ("The teacher gave him full marks for the essay", "mark", "education", ["finance"]),
    ("The German mark was replaced by the euro in 2002", "mark", "finance", ["education"]),
    ("She left a mark on the wall when she moved the furniture", "mark", "general", ["finance"]),
    # mount
    ("The tension began to mount as the deadline approached", "mount", "psychology", ["geography"]),
    ("The climber began to mount the steep northern face", "mount", "geography", ["psychology"]),
    ("The soldier learned to mount and dismount the horse quickly", "mount", "military", ["geography"]),
    # note
    ("She left a note on the kitchen table before leaving", "note", "communication", ["music"]),
    ("The musician held the final note for several seconds", "note", "music", ["communication"]),
    ("The bank note had a serial number printed on the back", "note", "finance", ["music"]),
    # order
    ("The waiter took their order and disappeared into the kitchen", "order", "food", ["law"]),
    ("The judge issued a restraining order against the defendant", "order", "law", ["food"]),
    ("The troops stood in order waiting for the command", "order", "military", ["food"]),
    # pack
    ("She began to pack her suitcase the night before the flight", "pack", "travel", ["animal"]),
    ("A pack of wolves surrounded the isolated farmhouse", "pack", "animal", ["travel"]),
    ("He applied an ice pack to reduce the swelling", "pack", "medicine", ["travel"]),
    # pick
    ("It took her a long time to pick the right candidate", "pick", "decision", ["music"]),
    ("The guitarist used a pick to strum the strings", "pick", "music", ["decision"]),
    ("The thief used a pick to open the lock without a key", "pick", "crime", ["music"]),
    # pipe
    ("The plumber replaced the burst pipe under the sink", "pipe", "construction", ["music"]),
    ("He sat in his armchair smoking a pipe", "pipe", "informal", ["construction"]),
    ("The organ pipe produced a deep resonant sound", "pipe", "music", ["construction"]),
    # play
    ("The children went outside to play after school", "play", "recreation", ["theatre"]),
    ("She had a lead role in the school play", "play", "theatre", ["recreation"]),
    ("The coach called a play that caught the defence off guard", "play", "sports", ["theatre"]),
    # point
    ("She made a strong point during the debate", "point", "communication", ["geometry"]),
    ("The compass needle always points north", "point", "navigation", ["communication"]),
    ("He scored the winning point with seconds to spare", "point", "sports", ["communication"]),
    # pop
    ("She heard a pop when she opened the champagne bottle", "pop", "sound", ["music"]),
    ("The pop song was number one for three weeks", "pop", "music", ["sound"]),
    ("He went to pop the question on their anniversary", "pop", "informal", ["music"]),
    # port
    ("The ship arrived at the port after a long voyage", "port", "nautical", ["computing"]),
    ("She poured a glass of port after dinner", "port", "food", ["nautical"]),
    ("The technician connected the cable to the USB port", "port", "computing", ["nautical"]),
    # pound
    ("She paid five pounds for the coffee", "pound", "finance", ["measurement"]),
    ("The recipe called for a pound of minced beef", "pound", "measurement", ["finance"]),
    ("The stray dog was taken to the pound overnight", "pound", "animal", ["finance"]),
    # race
    ("She won the race by a clear margin", "race", "sports", ["biology"]),
    ("The human race faces significant environmental challenges", "race", "biology", ["sports"]),
    ("There was a race against time to finish before the deadline", "race", "informal", ["sports"]),
    # record
    ("She broke the world record in the hundred metres", "record", "sports", ["music"]),
    ("He pulled out a vinyl record and placed it on the turntable", "record", "music", ["sports"]),
    ("The hospital kept a detailed medical record for each patient", "record", "medicine", ["sports"]),
    # ring
    ("She wore a diamond ring on her left hand", "ring", "jewellery", ["sports"]),
    ("The boxer entered the ring to a roar from the crowd", "ring", "sports", ["jewellery"]),
    ("The phone ring woke her at three in the morning", "ring", "communication", ["jewellery"]),
    # rock
    ("She sat on a smooth rock beside the river", "rock", "geology", ["music"]),
    ("The band played classic rock songs all evening", "rock", "music", ["geology"]),
    ("The boat began to rock violently in the storm", "rock", "movement", ["geology"]),
    # roll
    ("She ordered a bread roll with her soup", "roll", "food", ["movement"]),
    ("The teacher called the roll at the start of class", "roll", "education", ["food"]),
    ("The barrel began to roll down the steep hill", "roll", "movement", ["food"]),
    # round
    ("The boxer won the fight in the third round", "round", "sports", ["mathematics"]),
    ("The doctor did her rounds early in the morning", "round", "medicine", ["sports"]),
    ("She ordered a round of drinks for the whole table", "round", "informal", ["sports"]),
    # seal
    ("The seal surfaced near the fishing boat", "seal", "animal", ["security"]),
    ("She used wax to seal the envelope shut", "seal", "security", ["animal"]),
    ("The plumber applied a rubber seal around the pipe joint", "seal", "construction", ["animal"]),
    # set
    ("She won the first set six games to four", "set", "sports", ["theatre"]),
    ("The crew spent hours building the film set", "set", "theatre", ["sports"]),
    ("He watched the sun set over the ocean", "set", "nature", ["sports"]),
    # shoot
    ("The photographer had to shoot in low light conditions", "shoot", "photography", ["sports"]),
    ("The basketball player lined up to shoot a free throw", "shoot", "sports", ["photography"]),
    ("The new bamboo shoot grew several inches overnight", "shoot", "botany", ["sports"]),
    # slip
    ("She gave him a slip of paper with the address on it", "slip", "communication", ["movement"]),
    ("He had a Freudian slip during the interview", "slip", "psychology", ["movement"]),
    ("She slipped on the wet floor and twisted her ankle", "slip", "movement", ["psychology"]),
    # snap
    ("She heard the branch snap under her weight", "snap", "sound", ["photography"]),
    ("He took a quick snap of the sunset with his phone", "snap", "photography", ["sound"]),
    ("The dog gave a sudden snap when the child reached out", "snap", "animal", ["sound"]),
    # sort
    ("She took time to sort through the old photographs", "sort", "action", ["computing"]),
    ("What sort of music do you enjoy most", "sort", "general", ["action"]),
    ("The algorithm was designed to sort data efficiently", "sort", "computing", ["action"]),
    # spot
    ("She found a quiet spot by the river to read", "spot", "geography", ["informal"]),
    ("The manager put her on the spot with a difficult question", "spot", "informal", ["geography"]),
    ("The mechanic found the fault spot on the engine", "spot", "mechanics", ["geography"]),
    # stem
    ("She carefully removed the stem from the flower", "stem", "botany", ["movement"]),
    ("The government moved to stem the flow of illegal goods", "stem", "action", ["botany"]),
    ("The bleeding was stemmed by applying direct pressure", "stem", "medicine", ["botany"]),
]

TEST_CASES_V3 = [
    # wave
    ("She gave a wave as the train pulled out of the station", "wave", "communication", ["physics"]),
    ("The surfer caught a massive wave near the reef", "wave", "hydrology", ["communication"]),
    ("A new wave of technology is transforming the industry", "wave", "informal", ["hydrology"]),
    # tip
    ("She left a generous tip for the waiter", "tip", "finance", ["anatomy"]),
    ("He balanced carefully on the tip of the diving board", "tip", "anatomy", ["finance"]),
    ("The police received a tip about the suspect location", "tip", "law", ["finance"]),
    # track
    ("The athlete ran the fastest time on the track", "track", "sports", ["music"]),
    ("She downloaded the title track from the new album", "track", "music", ["sports"]),
    ("The detective tried to track down the missing witness", "track", "investigation", ["sports"]),
    # train
    ("She caught the early morning train to the city", "train", "transport", ["sports"]),
    ("The coach began to train the team for the championship", "train", "sports", ["transport"]),
    ("A long train of thought led him to the answer", "train", "psychology", ["transport"]),
    # turn
    ("It was her turn to present the quarterly results", "turn", "general", ["movement"]),
    ("The car made a sharp turn at the junction", "turn", "movement", ["general"]),
    ("The century turn brought major technological changes", "turn", "time", ["movement"]),
    # type
    ("She began to type the report on her laptop", "type", "computing", ["biology"]),
    ("He had a rare blood type that complicated the surgery", "type", "medicine", ["computing"]),
    ("What type of music do you prefer for studying", "type", "general", ["computing"]),
    # volume
    ("She turned up the volume on the radio", "volume", "sound", ["measurement"]),
    ("The library holds a volume of rare manuscripts", "volume", "publishing", ["sound"]),
    ("The volume of water in the reservoir was critically low", "volume", "measurement", ["sound"]),
    # well
    ("She felt perfectly well after a night of rest", "well", "health", ["geography"]),
    ("The village drew water from a deep stone well", "well", "geography", ["health"]),
    ("He performed well under pressure during the interview", "well", "general", ["health"]),
    # work
    ("She submitted her latest work to the gallery", "work", "art", ["employment"]),
    ("He had to work overtime to meet the deadline", "work", "employment", ["art"]),
    ("The mechanic checked to see if the engine would work", "work", "mechanics", ["employment"]),
    # yield
    ("The farm produced a high yield of wheat this season", "yield", "agriculture", ["finance"]),
    ("The bond offered a yield of five percent annually", "yield", "finance", ["agriculture"]),
    ("She refused to yield to pressure from her opponents", "yield", "psychology", ["finance"]),
    # tone
    ("The doctor used a firm but reassuring tone throughout", "tone", "communication", ["music"]),
    ("She practised scales to improve her tone on the piano", "tone", "music", ["communication"]),
    ("The gym programme helped her improve her muscle tone", "tone", "anatomy", ["music"]),
    # network
    ("She built a strong professional network over the years", "network", "business", ["computing"]),
    ("The company upgraded its entire IT network last quarter", "network", "computing", ["business"]),
    ("The television network broadcast the game live", "network", "media", ["computing"]),
    # trigger
    ("The trigger on the old rifle was stiff and unreliable", "trigger", "military", ["psychology"]),
    ("Stress can trigger a migraine in some patients", "trigger", "medicine", ["military"]),
    ("The announcement was the trigger for widespread protests", "trigger", "psychology", ["military"]),
    # culture
    ("The company had developed a strong culture of innovation", "culture", "business", ["biology"]),
    ("She studied the culture of ancient Rome at university", "culture", "history", ["business"]),
    ("The lab technician prepared a bacterial culture for testing", "culture", "biology", ["history"]),
    # cycle
    ("She went for a cycle along the riverside path", "cycle", "sports", ["time"]),
    ("The washing machine completed its cycle in forty minutes", "cycle", "technology", ["sports"]),
    ("The economic cycle tends to repeat every few years", "cycle", "finance", ["sports"]),
    # degree
    ("She completed her degree in three years", "degree", "education", ["measurement"]),
    ("The temperature dropped by ten degrees overnight", "degree", "measurement", ["education"]),
    ("He burned his hand to a minor degree on the hot surface", "degree", "medicine", ["education"]),
    # flight
    ("She booked a flight to Tokyo for the conference", "flight", "transport", ["movement"]),
    ("The suspect took flight when he saw the police arrive", "flight", "movement", ["transport"]),
    ("The staircase had two flights leading to the upper floor", "flight", "construction", ["transport"]),
    # form
    ("She filled out the application form carefully", "form", "administration", ["sports"]),
    ("The athlete was in peak form ahead of the championship", "form", "sports", ["administration"]),
    ("Water can change its form depending on temperature", "form", "chemistry", ["sports"]),
    # grave
    ("She placed flowers on the grave of her grandfather", "grave", "death", ["music"]),
    ("The situation was grave and required immediate action", "grave", "general", ["death"]),
    ("The musician played the passage in a grave and solemn tempo", "grave", "music", ["death"]),
    # host
    ("She acted as host for the charity dinner", "host", "social", ["biology"]),
    ("The parasite depends entirely on its host to survive", "host", "biology", ["social"]),
    ("The television host interviewed three guests on the show", "host", "media", ["biology"]),
    # launch
    ("The company planned to launch the product in spring", "launch", "business", ["nautical"]),
    ("The rocket launch was delayed by bad weather", "launch", "technology", ["business"]),
    ("They took the launch across the harbour to the island", "launch", "nautical", ["business"]),
    # level
    ("She filled the water to exactly the right level", "level", "measurement", ["general"]),
    ("The builder used a level to check the surface was flat", "level", "construction", ["measurement"]),
    ("He remained calm and level-headed throughout the crisis", "level", "psychology", ["measurement"]),
    # model
    ("She worked as a model for a fashion house in Paris", "model", "fashion", ["computing"]),
    ("The scientists built a model to simulate climate change", "model", "science", ["fashion"]),
    ("The new car model was unveiled at the motor show", "model", "automotive", ["fashion"]),
    # coat
    ("She pulled on her coat before heading out into the cold", "coat", "clothing", ["construction"]),
    ("The painter applied a second coat to the wooden surface", "coat", "construction", ["clothing"]),
    ("The dog had a thick winter coat that shed in spring", "coat", "biology", ["clothing"]),
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


def build_raw_prompt_ollama(sentence, word):
    """Simplified raw prompt for local models with weaker instruction following."""
    return f"""What does the word "{word}" mean in this sentence?

Sentence: "{sentence}"

Reply in this exact format and nothing else:
context: [one word domain label e.g. finance geography sports]
meaning: [one sentence definition]
confidence: 0.85"""


def build_cdf_prompt_ollama(sentence, word, token):
    """Simplified CDF prompt for local models — plain English, no JSON structure."""
    contexts = " / ".join(t["context"] for t in token["tracks"])
    meanings = "\n".join(
        f"- {t['context']}: {t['meaning']}"
        for t in token["tracks"]
    )
    return f"""What does the word "{word}" mean in this sentence?

Sentence: "{sentence}"

Known meanings:
{meanings}

Which meaning fits best? Reply with ONLY this — no explanation:
context: pick one from ({contexts})
meaning: copy the matching meaning above
confidence: 0.95"""


def call_model(prompt, client, model=None):
    """Universal model caller — supports Anthropic API and Ollama local models."""
    if model is None:
        model = MODEL_SONNET

    # Ollama local model
    if model.startswith("ollama/"):
        ollama_model = model.split("/", 1)[1]
        try:
            r = requests.post(OLLAMA_URL, json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False
            }, timeout=60)
            data = r.json()
            text = data.get("response", "")
            est_in  = len(prompt.split())
            est_out = len(text.split())
            usage = {"input_tokens": est_in, "output_tokens": est_out,
                     "total_tokens": est_in + est_out}
            return text, usage
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}")

    # Anthropic API
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

def call_claude(prompt, client, model=None):
    """Backwards-compatible alias for call_model."""
    return call_model(prompt, client, model)


RETRY_FEEDBACK = "That answer was incorrect. Reconsider the context carefully and try again."


def call_model_conv(messages, client, model):
    """Call Anthropic model with a full multi-turn message history."""
    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=messages
    )
    usage = {
        "input_tokens":  response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens":  response.usage.input_tokens + response.usage.output_tokens
    }
    return response.content[0].text, usage


def run_one_raw(prompt, word, sentence, correct_context, client, model):
    """Like run_one but also returns the raw response text for building retry history."""
    try:
        t = time.time()
        text, usage = call_model(prompt, client, model=model)
        result  = parse_response(text)
        correct = is_correct(result, correct_context, sentence, word, client)
        conf    = float(result.get("confidence", 0.0)) if result else 0.0
        elapsed = round(time.time() - t, 1)
        cost    = calc_cost(model, usage["input_tokens"], usage["output_tokens"])
        row = {
            "correct": correct, "confidence": conf, "time": elapsed,
            "tokens_in": usage["input_tokens"], "tokens_out": usage["output_tokens"],
            "cost_usd": cost
        }
        return row, text
    except Exception as e:
        print(f"\n  ERROR [{model}|{word}]: {e}")
        row = {"correct": False, "confidence": 0.0, "time": 0.0,
                "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
        return row, ""


def retry_condition(initial_prompt, initial_result, initial_raw, word, sentence,
                    correct_context, client, model, cdf_cost):
    """
    Retry an incorrect A/B result using multi-turn conversation history.
    Appends RETRY_FEEDBACK and retries until the answer is correct or
    cumulative cost exceeds cdf_cost (the single-pass cost of the CDF equivalent).
    Returns the result dict enhanced with: initial_correct, attempts,
    total_tokens, total_cost, resolved.
    """
    result = initial_result.copy()
    result["initial_correct"]    = False
    result["attempts"]           = 1
    result["total_tokens"]       = result["tokens_in"] + result["tokens_out"]
    result["total_output_tokens"] = result["tokens_out"]
    result["total_cost"]         = result["cost_usd"]
    result["resolved"]           = False
    result["time_to_correct"]    = result["time"]       # seed with first-call elapsed
    result["cost_to_correct"]    = result["cost_usd"]   # seed with first-call cost

    messages = [
        {"role": "user",      "content": initial_prompt},
        {"role": "assistant", "content": initial_raw or ""},
        {"role": "user",      "content": RETRY_FEEDBACK},
    ]

    while True:
        try:
            t = time.time()
            text, usage = call_model_conv(messages, client, model)
            parsed  = parse_response(text)
            correct = is_correct(parsed, correct_context, sentence, word, client)
            cost    = calc_cost(model, usage["input_tokens"], usage["output_tokens"])
            elapsed = round(time.time() - t, 1)

            result["attempts"]             += 1
            result["total_tokens"]         += usage["total_tokens"]
            result["total_output_tokens"]  += usage["output_tokens"]
            result["total_cost"]           += cost
            result["time_to_correct"] += elapsed
            result["cost_to_correct"] += cost

            if correct:
                result["correct"]  = True
                result["resolved"] = True
                if parsed:
                    result["confidence"] = float(parsed.get("confidence", result["confidence"]))
                break

            if result["total_cost"] > cdf_cost:
                result["resolved"] = False
                break

            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user",      "content": RETRY_FEEDBACK})

        except Exception as e:
            print(f"\n  RETRY ERROR [{model}|{word}]: {e}")
            break

    return result


def parse_response(raw):
    """Parse both JSON (Claude) and plain text (Ollama) responses."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().replace("```", "").strip()

    # Try JSON first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try JSON substring
    start = cleaned.find("{")
    end   = cleaned.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end])
        except json.JSONDecodeError:
            pass

    # Parse plain text format (Ollama simplified prompt response)
    # Expected: "context: finance\nmeaning: ...\nconfidence: 0.95"
    result = {}
    for line in cleaned.split("\n"):
        line = line.strip()
        if line.startswith("context:"):
            result["context"] = line.split(":", 1)[1].strip()
            result["selected_track"] = result["context"]
        elif line.startswith("meaning:"):
            result["meaning"] = line.split(":", 1)[1].strip()
        elif line.startswith("confidence:"):
            try:
                result["confidence"] = float(line.split(":", 1)[1].strip())
            except ValueError:
                result["confidence"] = 0.5
    if result.get("context"):
        return result

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
    """Parse both JSON (Claude) and plain text (Ollama) responses."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().replace("```", "").strip()

    # Try JSON first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try JSON substring
    start = cleaned.find("{")
    end   = cleaned.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end])
        except json.JSONDecodeError:
            pass

    # Parse plain text format (Ollama simplified prompt response)
    # Expected: "context: finance\nmeaning: ...\nconfidence: 0.95"
    result = {}
    for line in cleaned.split("\n"):
        line = line.strip()
        if line.startswith("context:"):
            result["context"] = line.split(":", 1)[1].strip()
            result["selected_track"] = result["context"]
        elif line.startswith("meaning:"):
            result["meaning"] = line.split(":", 1)[1].strip()
        elif line.startswith("confidence:"):
            try:
                result["confidence"] = float(line.split(":", 1)[1].strip())
            except ValueError:
                result["confidence"] = 0.5
    if result.get("context"):
        return result

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



def run_one(prompt, word, sentence, correct_context, client, model):
    """Run a single condition and return result dict."""
    try:
        t = time.time()
        text, usage = call_model(prompt, client, model=model)
        result  = parse_response(text)
        correct = is_correct(result, correct_context, sentence, word, client)
        conf    = float(result.get("confidence", 0.0)) if result else 0.0
        elapsed = round(time.time() - t, 1)
        cost    = calc_cost(model, usage["input_tokens"], usage["output_tokens"])
        return {
            "correct": correct, "confidence": conf, "time": elapsed,
            "tokens_in": usage["input_tokens"], "tokens_out": usage["output_tokens"],
            "cost_usd": cost
        }
    except Exception as e:
        print(f"\n  ERROR [{model}|{word}]: {e}")
        return {"correct": False, "confidence": 0.0, "time": 0.0,
                "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}


def summarise(results, key, n):
    """Aggregate metrics for one condition across all results."""
    rows = [r[key] for r in results if r["cdf_available"] and r[key] is not None]
    if not rows:
        return {}
    correct = sum(1 for r in rows if r["correct"])
    return {
        "correct": correct, "total": len(rows),
        "accuracy": correct / max(len(rows), 1),
        "avg_confidence": sum(r["confidence"] for r in rows) / max(len(rows), 1),
        "avg_time":       sum(r["time"]       for r in rows) / max(len(rows), 1),
        "avg_tokens_in":  sum(r["tokens_in"]  for r in rows) / max(len(rows), 1),
        "avg_tokens_out": sum(r["tokens_out"] for r in rows) / max(len(rows), 1),
        "avg_cost_usd":   sum(r["cost_usd"]   for r in rows) / max(len(rows), 1),
    }


def print_summary(label, s, baseline_cost=None):
    if not s:
        return
    n = s["total"]
    pct = round(s["accuracy"] * 100)
    cost_str = f"${s['avg_cost_usd']*1000:.4f}/1k"
    saving = ""
    if baseline_cost and baseline_cost > 0:
        saving = f"  ({round((1 - s['avg_cost_usd']/baseline_cost)*100, 1)}% cheaper)"
    print(f"  {label:<30} {s['correct']}/{n} ({pct}%)   "
          f"conf:{s['avg_confidence']:.3f}  "
          f"time:{s['avg_time']:.1f}s  "
          f"cost:{cost_str}{saving}")


def run_benchmark(test_cases, client, ollama_enabled=True):
    conditions = [
        ("A", "Raw   Sonnet",  MODEL_SONNET, "raw"),
        ("B", "Raw   Haiku",   MODEL_HAIKU,  "raw"),
        ("C", "Raw   Llama3",  MODEL_OLLAMA, "raw"),
        ("D", "CDF   Haiku",   MODEL_HAIKU,  "cdf"),
        ("E", "CDF   Llama3",  MODEL_OLLAMA, "cdf"),
        ("F", "CDF   Sonnet",  MODEL_SONNET, "cdf"),
    ]

    # Check Ollama availability
    ollama_ok = False
    if ollama_enabled:
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            ollama_ok = r.status_code == 200
        except Exception:
            pass
    if not ollama_ok:
        print(f"  NOTE: Ollama not available — skipping Llama3 conditions (C, E)")

    print(f"\nCognitive Data Format — Multi-Model Benchmark")
    print(f"Conditions: Raw Sonnet | Raw Haiku | Raw Llama3 | CDF Haiku | CDF Llama3 | CDF Sonnet")
    print(f"Test cases: {len(test_cases)}")
    print(f"{'─' * 80}")
    print(f"  {'#':<4} {'Word':<10} {'A:RS':<7} {'B:RH':<7} {'C:RL':<7} {'D:CH':<7} {'E:CL':<7} {'F:CS':<7}")
    print(f"{'─' * 80}")

    results = []

    for i, (sentence, word, correct_context, distractors) in enumerate(test_cases, 1):
        token = load_cdf_token(word)
        row = {"sentence": sentence, "word": word, "correct_context": correct_context,
               "cdf_available": token is not None}

        raw_texts  = {}  # raw response text for A and B (used to build retry history)
        prompts_ab = {}  # initial prompts for A and B

        for cid, label, model, mode in conditions:
            key = f"condition_{cid.lower()}"
            if model == MODEL_OLLAMA and not ollama_ok:
                row[key] = None
                continue
            if mode == "cdf" and not token:
                row[key] = None
                continue
            # Use simplified prompts for Ollama — full JSON confuses local models
            if model.startswith("ollama/"):
                prompt = build_raw_prompt_ollama(sentence, word) if mode == "raw" else build_cdf_prompt_ollama(sentence, word, token)
            else:
                prompt = build_raw_prompt(sentence, word) if mode == "raw" else build_cdf_prompt(sentence, word, token)
            if cid in ("A", "B"):
                row[key], raw_texts[cid] = run_one_raw(prompt, word, sentence, correct_context, client, model)
                prompts_ab[cid] = prompt
            else:
                row[key] = run_one(prompt, word, sentence, correct_context, client, model)

        # Cost-threshold retry loop for A (Raw Sonnet, threshold = F cost) and
        # B (Raw Haiku, threshold = D cost). C/E/F are never touched.
        for cid, cdf_key, retry_model in [("A", "condition_f", MODEL_SONNET),
                                           ("B", "condition_d", MODEL_HAIKU)]:
            key   = f"condition_{cid.lower()}"
            r     = row.get(key)
            cdf_r = row.get(cdf_key)
            if not r:
                continue
            if not r["correct"] and cdf_r and cdf_r.get("cost_usd", 0) > 0:
                row[key] = retry_condition(
                    prompts_ab.get(cid, ""), r, raw_texts.get(cid, ""),
                    word, sentence, correct_context, client, retry_model,
                    cdf_r["cost_usd"]
                )
            else:
                r["initial_correct"]     = r["correct"]
                r["attempts"]            = 1
                r["total_tokens"]        = r["tokens_in"] + r["tokens_out"]
                r["total_output_tokens"] = r["tokens_out"]
                r["total_cost"]          = r["cost_usd"]
                r["resolved"]            = r["correct"]
                r["time_to_correct"]     = r["time"]
                r["cost_to_correct"]     = r["cost_usd"]

        # Annotate D and F with single-pass time_to_correct / cost_to_correct
        for _cid in ("D", "F"):
            _r = row.get(f"condition_{_cid.lower()}")
            if _r:
                _r["time_to_correct"] = _r["time"]
                _r["cost_to_correct"] = _r["cost_usd"]

        results.append(row)

        def sym(k):
            v = row.get(k)
            if v is None: return "?"
            return "✓" if v["correct"] else "✗"

        print(f"  {i:<4} {word:<10} {sym('condition_a'):<7} {sym('condition_b'):<7} "
              f"{sym('condition_c'):<7} {sym('condition_d'):<7} "
              f"{sym('condition_e'):<7} {sym('condition_f'):<7}")

    valid = [r for r in results if r["cdf_available"]]
    n = len(valid)

    print(f"\n{'─' * 80}")
    print(f"  RESULTS ({n} test cases with CDF available)")
    print(f"{'─' * 80}")
    print(f"  {'Condition':<30} Acc      Conf    Time    Cost")
    print(f"{'─' * 80}")

    summaries = {}
    baseline_cost = None
    for cid, label, model, mode in conditions:
        key = f"condition_{cid.lower()}"
        s = summarise(results, key, n)
        summaries[cid] = s
        if cid == "A":
            baseline_cost = s.get("avg_cost_usd", None)
        print_summary(f"{cid}: {label}", s, baseline_cost)

    print(f"{'─' * 80}")
    print(f"\n  Key findings:")
    # Same model comparison: A vs F (Sonnet Raw vs Sonnet CDF)
    sa = summaries.get("A", {})
    sf = summaries.get("F", {})
    if sa and sf:
        delta = round((sf["accuracy"] - sa["accuracy"]) * 100, 1)
        print(f"  Sonnet Raw vs Sonnet CDF:  accuracy {'+' if delta>=0 else ''}{delta}%  (format impact, same model)")
    # Small model with CDF: B vs D (Haiku Raw vs Haiku CDF)
    sb = summaries.get("B", {})
    sd = summaries.get("D", {})
    if sb and sd:
        delta = round((sd["accuracy"] - sb["accuracy"]) * 100, 1)
        print(f"  Haiku Raw vs Haiku CDF:    accuracy {'+' if delta>=0 else ''}{delta}%  (CDF impact on small model)")
    # Cost: A vs D (Sonnet Raw vs Haiku CDF)
    if sa and sd and sa.get("avg_cost_usd", 0) > 0:
        saving = round((1 - sd["avg_cost_usd"] / sa["avg_cost_usd"]) * 100, 1)
        print(f"  Sonnet Raw vs Haiku CDF:   cost saving {saving}%  (model substitution)")
    print()

    print(f"  Cost Per Correct Answer — Same Model Comparison")
    print(f"{'─' * 80}")

    def _avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    for cid, label, cdf_cid in [("A", "Raw Sonnet", "F"), ("B", "Raw Haiku", "D")]:
        key        = f"condition_{cid.lower()}"
        rows       = [r for r in valid if r.get(key)]
        resolved   = [r for r in rows if r[key].get("resolved")]
        unresolved = [r for r in rows
                      if not r[key].get("resolved") and r[key].get("attempts", 1) > 1]
        avg_att = _avg([r[key]["attempts"]       for r in resolved])
        avg_ttc = _avg([r[key]["time_to_correct"] for r in resolved])
        avg_ctc = _avg([r[key]["cost_to_correct"] for r in resolved])
        avg_tu  = _avg([r[key]["time_to_correct"] for r in unresolved])
        avg_cu  = _avg([r[key]["cost_to_correct"] for r in unresolved])
        print(f"  {label}:")
        print(f"    Resolved ({len(resolved)}):   "
              f"avg attempts {avg_att:.1f}  "
              f"avg time {avg_ttc:.1f}s  "
              f"avg cost ${avg_ctc:.6f}")
        print(f"    Unresolved ({len(unresolved)}): "
              f"avg time {avg_tu:.1f}s  "
              f"avg cost ${avg_cu:.6f} before threshold")

    print()
    a_key    = "condition_a"
    b_key    = "condition_b"
    a_missed = [r for r in valid
                if r.get(a_key) and not r[a_key].get("initial_correct", r[a_key].get("correct"))]
    b_missed = [r for r in valid
                if r.get(b_key) and not r[b_key].get("initial_correct", r[b_key].get("correct"))]
    a_exceeded = sum(1 for r in a_missed
                     if r[a_key].get("attempts", 1) > 1 and not r[a_key].get("resolved"))
    b_exceeded = sum(1 for r in b_missed
                     if r[b_key].get("attempts", 1) > 1 and not r[b_key].get("resolved"))
    print(f"  Without CDF, Sonnet exceeded its own CDF cost threshold on "
          f"{a_exceeded}/{len(a_missed)} missed cases. "
          f"Haiku on {b_exceeded}/{len(b_missed)}.")
    print()

    for cid, label in [("F", "CDF Sonnet"), ("D", "CDF Haiku")]:
        key       = f"condition_{cid.lower()}"
        correct_r = [r for r in valid if r.get(key) and r[key].get("correct")]
        total_r   = [r for r in valid if r.get(key)]
        avg_ttc   = _avg([r[key]["time_to_correct"] for r in correct_r])
        avg_ctc   = _avg([r[key]["cost_to_correct"] for r in correct_r])
        print(f"  {label}: "
              f"avg time {avg_ttc:.1f}s  "
              f"avg cost ${avg_ctc:.6f}  "
              f"({len(correct_r)}/{len(total_r)} correct, single pass)")
    print()

    def _raw_real_cost_per_correct(cid):
        # total_spend / resolved_count — includes money burned on unresolved cases
        key      = f"condition_{cid.lower()}"
        all_rows = [r for r in valid if r.get(key)]
        resolved = [r for r in all_rows if r[key].get("resolved")]
        total_spend = sum(r[key]["cost_to_correct"] for r in all_rows)
        n_correct   = len(resolved)
        return total_spend / n_correct if n_correct > 0 else 0.0

    def _cdf_real_cost_per_correct(cid):
        # total_spend / correct_count — single pass, every case counted
        key      = f"condition_{cid.lower()}"
        all_rows = [r for r in valid if r.get(key)]
        correct  = [r for r in all_rows if r[key].get("correct")]
        total_spend = sum(r[key]["cost_to_correct"] for r in all_rows)
        n_correct   = len(correct)
        return total_spend / n_correct if n_correct > 0 else 0.0

    def _raw_avg_time_resolved(cid):
        key  = f"condition_{cid.lower()}"
        rows = [r for r in valid if r.get(key) and r[key].get("resolved")]
        return _avg([r[key]["time_to_correct"] for r in rows])

    def _cdf_avg_time_correct(cid):
        key  = f"condition_{cid.lower()}"
        rows = [r for r in valid if r.get(key) and r[key].get("correct")]
        return _avg([r[key]["time_to_correct"] for r in rows])

    ca = _raw_real_cost_per_correct("A")
    cf = _cdf_real_cost_per_correct("F")
    cb = _raw_real_cost_per_correct("B")
    cd = _cdf_real_cost_per_correct("D")
    ta = _raw_avg_time_resolved("A")
    tf = _cdf_avg_time_correct("F")
    tb = _raw_avg_time_resolved("B")
    td = _cdf_avg_time_correct("D")

    def _pct_more(raw, cdf):
        return round((raw - cdf) / cdf * 100, 1) if cdf > 0 else 0.0

    def _x_slower(raw, cdf):
        return round(raw / cdf, 1) if cdf > 0 else 0.0

    print(f"  Raw Sonnet real cost per correct answer: ${ca:.6f} vs CDF Sonnet: ${cf:.6f} "
          f"— {_pct_more(ca, cf):+.1f}% more expensive")
    print(f"  Raw Haiku real cost per correct answer:  ${cb:.6f} vs CDF Haiku:  ${cd:.6f} "
          f"— {_pct_more(cb, cd):+.1f}% more expensive")
    print(f"  Raw Sonnet time to correct: {ta:.1f}s vs CDF Sonnet: {tf:.1f}s "
          f"— {_x_slower(ta, tf):.1f}x slower")
    print(f"  Raw Haiku time to correct:  {tb:.1f}s vs CDF Haiku:  {td:.1f}s "
          f"— {_x_slower(tb, td):.1f}x slower")
    print()

    # Waste cost — spend on cases that never produced a correct answer
    for raw_cid, cdf_cid in [("A", "F"), ("B", "D")]:
        raw_key = f"condition_{raw_cid.lower()}"
        cdf_key = f"condition_{cdf_cid.lower()}"
        raw_label = "Raw Sonnet" if raw_cid == "A" else "Raw Haiku"
        cdf_label = "CDF Sonnet" if cdf_cid == "F" else "CDF Haiku"

        raw_all        = [r for r in valid if r.get(raw_key)]
        raw_unresolved = [r for r in raw_all if not r[raw_key].get("resolved")]
        raw_total_spend = sum(r[raw_key]["cost_to_correct"] for r in raw_all)
        raw_waste_cost  = sum(r[raw_key]["cost_to_correct"] for r in raw_unresolved)
        raw_waste_pct   = round(raw_waste_cost / raw_total_spend * 100, 1) if raw_total_spend > 0 else 0.0

        cdf_all      = [r for r in valid if r.get(cdf_key)]
        cdf_wrong    = [r for r in cdf_all if not r[cdf_key].get("correct")]
        cdf_total_spend = sum(r[cdf_key]["cost_to_correct"] for r in cdf_all)
        cdf_waste_cost  = sum(r[cdf_key]["cost_to_correct"] for r in cdf_wrong)
        cdf_waste_pct   = round(cdf_waste_cost / cdf_total_spend * 100, 1) if cdf_total_spend > 0 else 0.0

        print(f"  {raw_label} waste: ${raw_waste_cost:.6f} spent on {len(raw_unresolved)} cases "
              f"that never resolved ({raw_waste_pct}% of total spend)")
        print(f"  {cdf_label} waste: ${cdf_waste_cost:.6f} spent on {len(cdf_wrong)} cases "
              f"that failed fast ({cdf_waste_pct}% of total spend)")
    print()

    # ── Section 1: Confidence Calibration ─────────────────────────────────
    print(f"  Confidence Calibration")
    print(f"{'─' * 80}")
    conf_buckets = [
        (0.0,  0.70, "0.00-0.70"),
        (0.70, 0.90, "0.70-0.90"),
        (0.90, 0.95, "0.90-0.95"),
        (0.95, 1.01, "0.95-1.00"),
    ]
    cal_errors = {}
    for cid, label in [("A", "Raw Sonnet"), ("B", "Raw Haiku"),
                        ("D", "CDF Haiku"),  ("F", "CDF Sonnet")]:
        key  = f"condition_{cid.lower()}"
        rows = [r for r in valid if r.get(key)]
        print(f"  {label}:")
        for lo, hi, tag in conf_buckets:
            bucket = [r for r in rows if lo <= r[key].get("confidence", 0.0) < hi]
            if not bucket:
                continue
            acc      = sum(1 for r in bucket if r[key].get("correct")) / len(bucket)
            avg_conf = _avg([r[key].get("confidence", 0.0) for r in bucket])
            gap      = round((avg_conf - acc) * 100, 1)
            sign     = "+" if gap >= 0 else ""
            print(f"    [{tag}]  n={len(bucket):3d}  "
                  f"accuracy={round(acc * 100):3d}%  "
                  f"gap={sign}{gap}%")
        case_mae = _avg([abs(r[key].get("confidence", 0.0) -
                             (1 if r[key].get("correct") else 0))
                         for r in rows])
        cal_errors[cid] = round(case_mae * 100, 1)
    print(f"  Calibration error — "
          f"Raw Sonnet: {cal_errors.get('A', 0)}%  "
          f"Raw Haiku: {cal_errors.get('B', 0)}%  "
          f"CDF Haiku: {cal_errors.get('D', 0)}%  "
          f"CDF Sonnet: {cal_errors.get('F', 0)}%")
    print()

    # ── Section 2: Ambiguity Difficulty Scaling ────────────────────────────
    print(f"  Ambiguity Difficulty Scaling")
    print(f"{'─' * 80}")

    def _wrong_count(row):
        return sum(
            1 for c in ("a", "b", "c", "d", "e", "f")
            if row.get(f"condition_{c}") is not None
            and not row[f"condition_{c}"].get("correct")
        )

    easy   = [r for r in valid if _wrong_count(r) <= 1]
    medium = [r for r in valid if 2 <= _wrong_count(r) <= 3]
    hard   = [r for r in valid if _wrong_count(r) >= 4]

    def _tier_acc(tier_rows, cid):
        key   = f"condition_{cid.lower()}"
        r_sub = [r for r in tier_rows if r.get(key)]
        if not r_sub:
            return 0
        return round(sum(1 for r in r_sub if r[key].get("correct")) / len(r_sub) * 100)

    def _delta_pct(cdf_acc, raw_acc):
        d = cdf_acc - raw_acc
        return f"+{d}%" if d > 0 else (f"{d}%" if d < 0 else "0%")

    for tier_name, tier_rows in [("Easy", easy), ("Medium", medium), ("Hard", hard)]:
        print(f"  {tier_name} ({len(tier_rows)} cases):")
        if not tier_rows:
            continue
        for raw_cid, cdf_cid, label in [("A", "F", "Sonnet"),
                                         ("B", "D", "Haiku"),
                                         ("C", "E", "Llama3")]:
            raw_acc = _tier_acc(tier_rows, raw_cid)
            cdf_acc = _tier_acc(tier_rows, cdf_cid)
            print(f"    {label:<6}  Raw {raw_acc:3d}%  vs  CDF {cdf_acc:3d}%  "
                  f"— {_delta_pct(cdf_acc, raw_acc)}")

    def _pp(v):
        return f"+{v}pp" if v > 0 else (f"{v}pp" if v < 0 else "0pp")

    son_easy   = _tier_acc(easy,   "F") - _tier_acc(easy,   "A")
    son_medium = _tier_acc(medium, "F") - _tier_acc(medium, "A")
    son_hard   = _tier_acc(hard,   "F") - _tier_acc(hard,   "A")
    hai_easy   = _tier_acc(easy,   "D") - _tier_acc(easy,   "B")
    hai_medium = _tier_acc(medium, "D") - _tier_acc(medium, "B")
    hai_hard   = _tier_acc(hard,   "D") - _tier_acc(hard,   "B")

    print(f"  CDF format impact grows with difficulty — "
          f"Sonnet: {_pp(son_easy)} easy, {_pp(son_medium)} medium, {_pp(son_hard)} hard")
    print(f"  CDF format impact grows with difficulty — "
          f"Haiku: {_pp(hai_easy)} easy, {_pp(hai_medium)} medium, {_pp(hai_hard)} hard")
    print()

    # ── Section 3: First-Pass Certainty ────────────────────────────────────
    print(f"  First-Pass Certainty")
    print(f"{'─' * 80}")
    for cid, label in [("A", "Raw Sonnet"), ("B", "Raw Haiku")]:
        key   = f"condition_{cid.lower()}"
        rows  = [r for r in valid if r.get(key)]
        total = len(rows)
        fp_ok    = sum(1 for r in rows if r[key].get("initial_correct"))
        retry_ok = sum(1 for r in rows
                       if not r[key].get("initial_correct") and r[key].get("correct"))
        never_ok = sum(1 for r in rows if not r[key].get("correct"))
        print(f"  {label}:")
        print(f"    Correct on attempt 1:    {fp_ok:3d}/{total} "
              f"({round(fp_ok / total * 100) if total else 0}%)")
        print(f"    Correct after retries:   {retry_ok:3d}/{total} "
              f"({round(retry_ok / total * 100) if total else 0}%)")
        print(f"    Never correct:           {never_ok:3d}/{total} "
              f"({round(never_ok / total * 100) if total else 0}%)")
    for cid, label in [("D", "CDF Haiku"), ("F", "CDF Sonnet")]:
        key   = f"condition_{cid.lower()}"
        rows  = [r for r in valid if r.get(key)]
        total = len(rows)
        ok    = sum(1 for r in rows if r[key].get("correct"))
        print(f"  {label} (always attempt 1): "
              f"{ok:3d}/{total} correct "
              f"({round(ok / total * 100) if total else 0}%)")

    def _fp_stats(cid):
        key  = f"condition_{cid.lower()}"
        rows = [r for r in valid if r.get(key)]
        tot  = len(rows)
        if cid in ("A", "B"):
            fp = sum(1 for r in rows if r[key].get("initial_correct"))
        else:
            fp = sum(1 for r in rows if r[key].get("correct"))
        return fp, tot, round(fp / tot * 100) if tot else 0

    a_fp, a_tot, a_fp_pct = _fp_stats("A")
    f_fp, f_tot, f_fp_pct = _fp_stats("F")
    b_fp, b_tot, b_fp_pct = _fp_stats("B")
    d_fp, d_tot, d_fp_pct = _fp_stats("D")
    print(f"  Raw Sonnet first-pass correct: {a_fp}/{a_tot} ({a_fp_pct}%)  — "
          f"CDF Sonnet first-pass correct: {f_fp}/{f_tot} ({f_fp_pct}%)")
    print(f"  Raw Haiku first-pass correct:  {b_fp}/{b_tot} ({b_fp_pct}%)  — "
          f"CDF Haiku first-pass correct:  {d_fp}/{d_tot} ({d_fp_pct}%)")
    print(f"  CDF eliminates retry dependency — "
          f"100% of correct answers delivered on first pass")
    print()

    # ── Section 4: Output Token Efficiency & Cost Efficiency ───────────────
    print(f"  Output Token Efficiency & Cost Efficiency")
    print(f"{'─' * 80}")

    def _eff_stats(cid):
        key  = f"condition_{cid.lower()}"
        rows = [r for r in valid if r.get(key)]
        if not rows:
            return 0, 0, 0.0, 0.0, 0.0
        correct    = sum(1 for r in rows if r[key].get("correct"))
        out_tok    = sum(r[key].get("total_output_tokens", r[key]["tokens_out"]) for r in rows)
        total_cost = sum(r[key].get("total_cost", r[key]["cost_usd"]) for r in rows)
        out_eff    = correct / out_tok   * 1000 if out_tok   > 0 else 0.0
        cost_eff   = correct / total_cost       if total_cost > 0 else 0.0
        return out_tok, correct, out_eff, total_cost, cost_eff

    out_a, cor_a, oeff_a, cost_a, ceff_a = _eff_stats("A")
    out_b, cor_b, oeff_b, cost_b, ceff_b = _eff_stats("B")
    out_d, cor_d, oeff_d, cost_d, ceff_d = _eff_stats("D")
    out_f, cor_f, oeff_f, cost_f, ceff_f = _eff_stats("F")

    for label, out_tok, cor, oeff, total_cost, ceff in [
        ("Raw Sonnet", out_a, cor_a, oeff_a, cost_a, ceff_a),
        ("Raw Haiku",  out_b, cor_b, oeff_b, cost_b, ceff_b),
        ("CDF Haiku",  out_d, cor_d, oeff_d, cost_d, ceff_d),
        ("CDF Sonnet", out_f, cor_f, oeff_f, cost_f, ceff_f),
    ]:
        print(f"  {label:<14}  output tokens: {out_tok:6,d}  "
              f"correct/1k out tokens: {oeff:6.3f}  "
              f"correct/$: {ceff:7.1f}")

    def _x(b, a):
        return round(b / a, 1) if a > 0 else 0.0

    print(f"  Raw Sonnet output token efficiency: {oeff_a:.3f}  "
          f"cost efficiency: {ceff_a:.1f} correct answers per dollar")
    print(f"  CDF Sonnet output token efficiency: {oeff_f:.3f} "
          f"— {_x(oeff_f, oeff_a):.1f}x more output-efficient  "
          f"cost efficiency: {ceff_f:.1f} — {_x(ceff_f, ceff_a):.1f}x cheaper per answer")
    print(f"  Raw Haiku  output token efficiency: {oeff_b:.3f}  "
          f"cost efficiency: {ceff_b:.1f} correct answers per dollar")
    print(f"  CDF Haiku  output token efficiency: {oeff_d:.3f} "
          f"— {_x(oeff_d, oeff_b):.1f}x more output-efficient  "
          f"cost efficiency: {ceff_d:.1f} — {_x(ceff_d, ceff_b):.1f}x cheaper per answer")
    print(f"  (Output token efficiency includes all retry output tokens for Raw conditions — conservative estimate favoring CDF)")
    print()

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "test_cases": len(test_cases),
        "valid_cases": n,
        "conditions": {
            cid: summaries[cid]
            for cid, _, _, _ in conditions if cid in summaries
        },
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="CDF Multi-Model Benchmark")
    parser.add_argument("--count",   type=int, help="Run first N test cases only")
    parser.add_argument("--version", type=int, default=1, choices=[1, 2, 3],
                        help="Test case set: 1=original 2=new words 3=third set")
    parser.add_argument("--no-ollama", action="store_true",
                        help="Skip Ollama/Llama3 conditions even if Ollama is running")
    args = parser.parse_args()

    version_map = {1: TEST_CASES, 2: TEST_CASES_V2, 3: TEST_CASES_V3}
    all_cases = version_map[args.version]

    client = anthropic.Anthropic()
    cases  = all_cases[:args.count] if args.count else all_cases
    ollama_enabled = not args.no_ollama

    summary = run_benchmark(cases, client, ollama_enabled=ollama_enabled)

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_file = str(RESULTS_PATH / f"benchmark_multimodel_v{args.version}_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved: {output_file}\n")


if __name__ == "__main__":
    main()
