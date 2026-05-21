"""
Cognitive Data Format — CDR Validator
Phase 1 Deliverable

Usage:
    python validate.py examples/apple.cdr.json
    python validate.py --all

Validates any CDR JSON file against cdr_schema_v1.json
Returns VALID or INVALID with specific error messages.
"""

import json
import sys
import os
import argparse
from pathlib import Path

try:
    import jsonschema
    from jsonschema import validate, ValidationError, SchemaError
except ImportError:
    print("ERROR: jsonschema not installed. Run: pip install jsonschema")
    sys.exit(1)

SCHEMA_PATH = Path(__file__).parent.parent / "schema" / "cdr_schema_v1.json"
EXAMPLES_PATH = Path(__file__).parent.parent / "examples"


def load_json(path: Path) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found — {path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {path} — {e}")
        sys.exit(1)


def validate_cdr(cdr_path: Path, schema: dict) -> bool:
    cdr = load_json(cdr_path)
    filename = cdr_path.name

    try:
        validate(instance=cdr, schema=schema)
        print(f"  ✓  VALID     {filename}")
        return True
    except ValidationError as e:
        print(f"  ✗  INVALID   {filename}")
        print(f"           Field:   {' -> '.join(str(p) for p in e.absolute_path) or 'root'}")
        print(f"           Error:   {e.message}")
        return False
    except SchemaError as e:
        print(f"  ✗  SCHEMA ERROR   {filename}")
        print(f"           {e.message}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Cognitive Data Record Validator — Phase 1"
    )
    parser.add_argument(
        "cdr_file",
        nargs="?",
        help="Path to a CDR JSON file to validate"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all CDR files in the examples/ directory"
    )
    args = parser.parse_args()

    if not args.cdr_file and not args.all:
        parser.print_help()
        sys.exit(1)

    # Load schema
    if not SCHEMA_PATH.exists():
        print(f"ERROR: Schema not found at {SCHEMA_PATH}")
        sys.exit(1)

    schema = load_json(SCHEMA_PATH)
    print(f"\nCognitive Data Format — CDR Validator")
    print(f"Schema: {SCHEMA_PATH.name}")
    print(f"{'─' * 50}")

    results = []

    if args.all:
        cdr_files = sorted(EXAMPLES_PATH.glob("*.cdr.json"))
        if not cdr_files:
            print(f"No CDR files found in {EXAMPLES_PATH}")
            sys.exit(1)
        for cdr_file in cdr_files:
            results.append(validate_cdr(cdr_file, schema))
    else:
        cdr_path = Path(args.cdr_file)
        results.append(validate_cdr(cdr_path, schema))

    print(f"{'─' * 50}")
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"  Result: {passed}/{total} records valid — Phase 1 schema confirmed\n")
    else:
        print(f"  Result: {passed}/{total} records valid — fix errors before proceeding\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
