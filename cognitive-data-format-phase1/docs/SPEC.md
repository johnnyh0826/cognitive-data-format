# Cognitive Data Format — Specification
**Version 1.0 | May 20, 2026**

---

## What Is Cognitive Data?

Every current data format records what happened. Cognitive Data records what was *understood* to have happened — which is exactly how human memory and intelligence actually work.

There are three layers of data:

| Layer | Source | Nature | Example |
|---|---|---|---|
| Raw Data | Other people | Inherited label | Name: Apple |
| Metadata | Observation | Inherited description | Round, red, 8cm, sweet |
| Cognitive Data | Personal event | Owned meaning | First bite — cold, loud, held by mother, age 4 |

Cognitive data is never neutral. Every record carries the perspective of the observer at the moment of the event.

---

## The Cognitive Data Record (CDR)

The atomic unit of the format. Every concept in the system has exactly one CDR as its root record, generated at first encounter.

A CDR answers the question: *What did this observer understand about this concept, and under what conditions was that understanding formed?*

---

## Field Specification

### Top Level

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | Yes | Unique record identifier. Format: `cdr_{entity}_{001}` |
| `entity` | string | Yes | The concept being recorded e.g. `apple` |
| `raw` | object | Yes | The inherited label for this concept |
| `meta` | object | Yes | Descriptive attributes from observation |
| `cognitive` | object | Yes | The first encounter event — meaning, context, interpretation |

### raw

| Field | Type | Required | Description |
|---|---|---|---|
| `raw.label` | string | Yes | The inherited name given to this concept |
| `raw.source` | string | No | Where the label came from e.g. `wikidata` |

### meta

| Field | Type | Required | Description |
|---|---|---|---|
| `meta.properties` | object | Yes | Key-value descriptive attributes from observation |
| `meta.category` | string | No | Classification e.g. `fruit`, `furniture`, `element` |
| `meta.source` | string | No | Where metadata was pulled from |

### cognitive

| Field | Type | Required | Description |
|---|---|---|---|
| `cognitive.event_id` | string | Yes | Reference to the triggering event record. Format: `evt_{001}` |
| `cognitive.observer_id` | string | Yes | The agent that encountered this entity. Format: `observer_{001}` |
| `cognitive.timestamp` | datetime | Yes | ISO 8601 — when the encounter occurred |
| `cognitive.environment` | string | Yes | Location context e.g. `kitchen_morning` |
| `cognitive.perception` | object | Yes | Sensory data at moment of encounter |
| `cognitive.context.co_present_entities` | array | Yes | Other entities present at time of encounter |
| `cognitive.context.observer_state` | string | Yes | Agent state: `curious` / `neutral` / `alert` / `cautious` |
| `cognitive.context.preceding_event` | string | No | ID of event that occurred just before this one |
| `cognitive.interpretation.meaning` | string | Yes | Derived understanding in plain language |
| `cognitive.interpretation.confidence` | float | Yes | 0.0 to 1.0 — certainty of interpretation |
| `cognitive.interpretation.emotional_weight` | float | Yes | 0.0 to 1.0 — emotional significance |
| `cognitive.links` | array | No | IDs of related CDRs and events |

---

## Annotated Example CDR

```json
{
  "id": "cdr_apple_001",          // unique ID — entity + sequence number
  "entity": "apple",              // the concept this record belongs to

  "raw": {
    "label": "apple",             // the inherited name
    "source": "wikidata"          // where the label came from
  },

  "meta": {
    "properties": {               // descriptive attributes from observation
      "shape": "round",
      "colors": ["red", "green", "yellow"],
      "size_cm": 8,
      "texture": "smooth",
      "taste": "sweet-tart",
      "edible": true
    },
    "category": "fruit",
    "source": "wikidata"
  },

  "cognitive": {
    "event_id": "evt_001",           // which event triggered this CDR
    "observer_id": "observer_001",   // who encountered the entity
    "timestamp": "2026-01-01T08:00:00Z",
    "environment": "kitchen_morning",

    "perception": {                  // sensory data at moment of encounter
      "visual": "round, red, glossy surface catching the morning light",
      "auditory": "loud sharp crack on first bite",
      "tactile": "smooth skin, heavier than expected, cold from the bowl",
      "olfactory": "faint sweetness, clean and fresh",
      "thermal": "cool against the palm"
    },

    "context": {
      "co_present_entities": ["table", "bowl", "morning_light", "mother"],
      "observer_state": "curious",   // one of: curious, neutral, alert, cautious
      "preceding_event": null        // nothing came before this
    },

    "interpretation": {
      "meaning": "edible object, pleasant to handle, associated with safety and care",
      "confidence": 0.82,            // 82% certain of this interpretation
      "emotional_weight": 0.6        // moderately significant emotionally
    },

    "links": ["cdr_fruit_001", "evt_001"]
  }
}
```

---

## Versioning Policy

- This document describes **CDF v1.0**
- The schema file is `cdr_schema_v1.json`
- Breaking changes increment the major version: v2.0
- Additive changes (new optional fields) increment the minor version: v1.1
- Every schema version is committed to the GitHub repository with a version tag
- Implementations must declare which schema version they target using `$schema`

---

## Folder Structure

```
cognitive-data-format/
├── schema/
│   └── cdr_schema_v1.json      # The formal JSON Schema definition
├── examples/
│   ├── apple.cdr.json
│   ├── chair.cdr.json
│   ├── fire.cdr.json
│   ├── water.cdr.json
│   └── mother.cdr.json
├── validator/
│   └── validate.py             # Validates any CDR against the schema
└── docs/
    └── SPEC.md                 # This document
```

---

## Running the Validator

```bash
pip install jsonschema

# Validate a single CDR
python validator/validate.py examples/apple.cdr.json

# Validate all example CDRs
python validator/validate.py --all
```

---

*Cognitive Data Format is an open standard. The person who defines and publishes the standard owns the concept.*
*— CDF v1.0*
