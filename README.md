# Cognitive Data Format (CDF)

**The Third Layer of Data**
*Author: Ke He — Independent Researcher — May 2026*
*License: CC BY-NC 4.0 — Commercial use requires permission*

---

## The Idea

A dictionary gives a word one meaning. Your brain accumulates many — every new context you encounter in life adds to your understanding of a word without erasing what came before. The word *apple* starts as a fruit. Later it becomes a technology company. Later still a nickname for New York City. Each meaning is added. Nothing is overwritten.

Current AI systems have neither property. They treat tokens as naked character sequences and reconstruct meaning at inference time through computationally expensive attention mechanisms operating across billions of parameters — answering questions about word meaning that any structured dictionary already knows.

**CDF gives machines the same accumulation principle.** Every token carries its own finite list of meanings — one per real-world context. The AI reads the list and selects the right one. Disambiguation drops from open-ended statistical inference to bounded closed-set selection.

---

## The Format

A CDF token is a JSON object. Every word carries a track registry — its accumulated meanings across every context it exists in.

```json
{
  "token": "bank",
  "tracks": [
    {
      "id": "01",
      "context": "finance",
      "meaning": "a financial institution that accepts deposits and provides loans",
      "added": "first",
      "source": "llm"
    },
    {
      "id": "02",
      "context": "geography",
      "meaning": "the side of a river or lake",
      "added": "accumulated",
      "source": "llm"
    }
  ]
}
```

That is the entire format. A token with a track list. Simple. Deterministic. Self-describing.

---

## Why It Matters

Current semantic retrieval complexity:
```
O(N² · D)  —  every token against every other token
```

CDF retrieval complexity:
```
O(k)  —  where k = number of tracks (typically 2–5)
```

The search space collapses from unbounded to finite. For the 15–20% of natural language tokens that are genuinely ambiguous, CDF eliminates the disambiguation computation almost entirely.

| Problem | Current Approach | CDF |
|---|---|---|
| Word disambiguation | Attention across full context | Track selection — O(k) |
| Memory retrieval | Vector similarity search | Flat index lookup |
| Context window | Large history required | Token is self-contained |
| Hallucination | Common on ambiguous inputs | Near-zero — deterministic |
| Vector drift | Corrupts memory over time | Impossible — tracks are append-only |

---

## Quick Start

```bash
# Clone
git clone https://github.com/johnnyh0826/cognitive-data-format
cd cognitive-data-format

# Install
pip install jsonschema requests

# Validate the example tokens
python validator/validate.py --all

# Generate tokens using Llama3 (Ollama must be running)
ollama serve
python generator/generate.py --single spring
python generator/generate.py --all
```

---

## Repository Structure

```
cognitive-data-format/
├── schema/
│   └── cdf_schema_v1.json          — formal JSON Schema definition
├── tokens/
│   ├── apple.cdf.json              — hand-authored examples
│   ├── chair.cdf.json
│   ├── fire.cdf.json
│   ├── mother.cdf.json
│   └── water.cdf.json
├── data/
│   └── generated/                  — machine-generated token database
├── validator/
│   └── validate.py                 — validates any token against the schema
├── generator/
│   └── generate.py                 — generates tokens via Llama3/Ollama
├── papers/
│   ├── CDF_Academic_Paper_Ke_He_2026.docx
│   ├── CDF_Engineer_Version_Ke_He_2026.docx
│   └── CDF_Industry_Whitepaper_Ke_He_2026.docx
└── docs/
    └── SPEC.md                     — full specification
```

---

## Papers

Three versions of the full CDF specification and research paper are available in `/papers`:

| Document | Audience | Contents |
|---|---|---|
| Academic Paper | Researchers | Full technical specification, theoretical efficiency analysis, references |
| Engineer Version | Developers | Implementation guide, code examples, benchmark methodology |
| Industry Whitepaper | Enterprise | Business case, cost impact, licensing terms |

---

## Current Status

| Item | Status |
|---|---|
| Schema v1.0 | Complete |
| Example tokens (5) | Complete — hand-authored |
| Generated token database | 55 tokens — growing |
| Validator | Complete |
| Generator (Llama3/Ollama) | Complete |
| Empirical benchmark | In progress |
| arXiv preprint | Pending |
| PyPI package | Planned — Q3 2026 |
| Binary format v2 | Planned — Q4 2026 |

---

## The Accumulation Rule

Meanings are never overwritten. They are only appended.

When a word acquires a new meaning in culture, technology, or personal use — a new track is added. All existing tracks remain intact. Old documents stay valid. New documents immediately gain access to the new meaning.

This solves **vector drift** — the corruption of AI memory that occurs when new information is incorporated into vector databases, warping the mathematical representation of older memories.

---

## License

**Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)**

Free for research, personal use, and open source projects.
Commercial use requires explicit written permission.

Commercial licensing: johnnyh0826@gmail.com

---

## Contact

**Ke He**
Independent Researcher
johnnyh0826@gmail.com
github.com/johnnyh0826/cognitive-data-format

---

*Cognitive Data Format v1.0 — May 2026*
*The person who defines and publishes the standard owns the concept.*
