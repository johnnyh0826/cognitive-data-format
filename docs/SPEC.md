# Cognitive Data Format (CDF)
**Version 1.0 | May 2026 | Author: Ke He**

---

## The Problem

Current data formats store words as passive strings. The word "apple" is just five characters. When an AI reads it, it must search billions of statistical relationships to guess which apple you mean — the fruit, the company, or the city.

This is computationally expensive, probabilistic, and frequently wrong.

## The Idea

A dictionary gives you one meaning per word. Your brain accumulates meanings over time — the first meaning you learned, then every additional context you encountered throughout life.

CDF applies this to data. Every token carries all the meanings it has accumulated across every context it exists in. The AI reads the token, sees a finite list of meanings, and picks the right one instantly.

**Not infinite guessing. Finite selection.**

---

## The Format

A CDF token is a JSON object with two fields:

- `token` — the word
- `tracks` — its accumulated meanings

```json
{
  "token": "apple",
  "tracks": [
    {
      "id": "01",
      "context": "agriculture",
      "meaning": "a round edible fruit grown on trees",
      "added": "first",
      "source": "wikidata"
    },
    {
      "id": "02",
      "context": "technology",
      "meaning": "Apple Inc — a consumer electronics corporation",
      "added": "accumulated",
      "source": "wikidata"
    },
    {
      "id": "03",
      "context": "geography",
      "meaning": "informal name for New York City",
      "added": "accumulated",
      "source": "wikidata"
    }
  ]
}
```

---

## Field Specification

### Token level

| Field | Type | Required | Description |
|---|---|---|---|
| `token` | string | Yes | The word or concept |
| `tracks` | array | Yes | All accumulated meanings — minimum 1 |

### Track level

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | Yes | Track number — format: `01`, `02`, `03` |
| `context` | string | Yes | The domain this meaning belongs to |
| `meaning` | string | Yes | Plain language definition in this context |
| `added` | string | No | `first` for original meaning, `accumulated` for later additions |
| `source` | string | No | Where this meaning came from e.g. `wikidata`, `llm`, `user` |

---

## How It Changes Retrieval

**Current semantic retrieval:**
```
query → embed into vector → search millions of vectors
→ cosine similarity → approximate nearest neighbour
```
Complexity: O(N² · D) — scales with every token against every other token.

**CDF retrieval:**
```
query → read token → scan k tracks → pick highest probability match
```
Complexity: O(k) where k = number of tracks (typically 2-5).

The search space collapses from unbounded to finite.

---

## The Accumulation Rule

Meanings are never overwritten. They are only added.

When a word takes on a new meaning in culture, technology, or personal use — a new track is appended. All existing tracks remain intact. Old data stays valid. New data immediately gains the new meaning.

```
1970: apple → Track 01: fruit
1976: apple → Track 02: tech company (appended)
1990: apple → Track 03: New York slang (appended)
```

This solves vector drift — the corruption of old memories when new meanings are learned.

---

## How AI Uses CDF

An LLM reading CDF data no longer needs to guess meaning from context. It reads the track list and runs a simple probability selection across k options only.

```
sentence: "I upgraded my iOS on my..."
token:    "apple"
tracks:   [fruit, tech company, New York]
result:   tech company — 99% confidence
```

The model's effective reasoning load drops by the ratio of k to the full vocabulary size.

---

## Versioning

- This document covers CDF v1.0
- Schema file: `cdf_schema_v1.json`
- New tracks added to existing tokens → no version change
- New fields added to schema → minor version bump (v1.1)
- Breaking structural changes → major version (v2.0)

---

## Repository Structure

```
cognitive-data-format/
├── schema/
│   └── cdf_schema_v1.json     — formal JSON Schema
├── tokens/
│   ├── apple.cdf.json         — hand-authored example tokens
│   ├── chair.cdf.json
│   ├── fire.cdf.json
│   ├── mother.cdf.json
│   └── water.cdf.json
├── validator/
│   └── validate.py            — validates tokens against schema
├── generator/
│   └── generate.py            — generates tokens via Llama3/Ollama
├── data/generated/            — machine-generated token database
└── docs/
    └── SPEC.md                — this document
```

---

## Running the Validator

```bash
pip install jsonschema requests

# Validate a single token
python validator/validate.py tokens/apple.cdf.json

# Validate all tokens
python validator/validate.py --all
```

## Generating Tokens

```bash
# Make sure Ollama is running
ollama serve

# Test with one word
python generator/generate.py --single bank

# Generate all 50 starter tokens
python generator/generate.py --all

# Generate first 10 only
python generator/generate.py --count 10
```

---

*Cognitive Data Format — The Third Layer of Data*
*Version 1.0 | Ke He | May 2026*
