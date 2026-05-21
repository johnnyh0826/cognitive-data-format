# Cognitive Data Format

**The Third Layer of Data**

Cognitive Data Format (CDF) captures not just what something is called or how it is described, but the depth of understanding formed through a first encounter — driven by context, perception, and meaning.

| Layer | Source | Nature |
|---|---|---|
| Raw Data | Inherited | The name/label of a thing |
| Metadata | Observation | Properties and descriptions |
| Cognitive Data | Simulated event | The first encounter — meaning, context, interpretation |

## Quick Start

```bash
pip install jsonschema
python validator/validate.py --all
```

## Structure

```
/schema      — JSON Schema definition (cdr_schema_v1.json)
/examples    — 5 hand-authored CDRs (apple, chair, fire, water, mother)
/validator   — Validation script
/docs        — SPEC.md — human-readable specification
```

## Status

Phase 1 complete. Schema locked. 5/5 example CDRs valid.

---
*Version 1.0 | May 20, 2026*
