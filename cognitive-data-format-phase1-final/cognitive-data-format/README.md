# Cognitive Data Format (CDF)

**The Third Layer of Data**
*Author: Ke He | Version 1.0 | May 2026*

---

A word is not one meaning. It is every meaning it has ever accumulated.

CDF is a token format where every word carries its own finite list of meanings — one per context. Instead of an AI searching billions of statistical relationships to guess what a word means, it reads the token's track list and selects the right meaning instantly.

```json
{
  "token": "apple",
  "tracks": [
    { "id": "01", "context": "agriculture", "meaning": "a round edible fruit" },
    { "id": "02", "context": "technology",  "meaning": "Apple Inc — electronics corporation" },
    { "id": "03", "context": "geography",   "meaning": "informal name for New York City" }
  ]
}
```

## Quick Start

```bash
pip install jsonschema requests

# Validate all example tokens
python validator/validate.py --all

# Generate tokens using Llama3 (Ollama must be running)
python generator/generate.py --single bank
python generator/generate.py --all
```

## Why This Matters

Current semantic retrieval: `O(N² · D)` — every token against every other token.

CDF retrieval: `O(k)` — where k is the number of tracks. Typically 2–5.

The search space collapses from unbounded to finite.

## Structure

```
/schema      — cdf_schema_v1.json
/tokens      — 5 hand-authored example tokens
/validator   — validate.py
/generator   — generate.py (Llama3 via Ollama)
/data        — generated token database
/docs        — SPEC.md
```

---
*CDF v1.0 | May 2026*
