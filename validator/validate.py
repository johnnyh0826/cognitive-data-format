"""
Cognitive Data Format — Token Validator
Validates any CDF token file against cdf_schema_v1.json

Usage:
    python validate.py tokens/apple.cdf.json
    python validate.py --all
"""

import json
import sys
import argparse
from pathlib import Path

try:
    import jsonschema
    from jsonschema import validate, ValidationError
except ImportError:
    print("ERROR: jsonschema not installed. Run: pip install jsonschema")
    sys.exit(1)

SCHEMA_PATH = Path(__file__).parent.parent / "schema" / "cdf_schema_v1.json"
TOKENS_PATH = Path(__file__).parent.parent / "tokens"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def validate_token(path, schema):
    token = load_json(path)
    try:
        validate(instance=token, schema=schema)
        track_count = len(token.get("tracks", []))
        print(f"  ✓  VALID     {path.name:<30} {track_count} tracks")
        return True
    except ValidationError as e:
        field = " -> ".join(str(p) for p in e.absolute_path) or "root"
        print(f"  ✗  INVALID   {path.name:<30} {field} — {e.message}")
        return False


def main():
    parser = argparse.ArgumentParser(description="CDF Token Validator")
    parser.add_argument("token_file", nargs="?", help="Path to a CDF token file")
    parser.add_argument("--all", action="store_true", help="Validate all tokens")
    args = parser.parse_args()

    if not args.token_file and not args.all:
        parser.print_help()
        sys.exit(1)

    schema = load_json(SCHEMA_PATH)

    print(f"\nCognitive Data Format — Token Validator v1")
    print(f"Schema: {SCHEMA_PATH.name}")
    print(f"{'─' * 55}")

    results = []

    if args.all:
        files = sorted(TOKENS_PATH.glob("*.cdf.json"))
        if not files:
            print(f"No token files found in {TOKENS_PATH}")
            sys.exit(1)
        for f in files:
            results.append(validate_token(f, schema))
    else:
        results.append(validate_token(Path(args.token_file), schema))

    print(f"{'─' * 55}")
    passed = sum(results)
    total  = len(results)

    if passed == total:
        print(f"  Result: {passed}/{total} tokens valid — Phase 1 complete\n")
    else:
        print(f"  Result: {passed}/{total} tokens valid — fix errors before proceeding\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
