"""
Cognitive Data Format — Token Validator
Validates CDF token files against cdf_schema_v1.json

Usage:
    python validator/validate.py                    # validate everything
    python validator/validate.py --all              # validate everything
    python validator/validate.py --generated        # validate generated tokens only
    python validator/validate.py --examples         # validate example tokens only
    python validator/validate.py tokens/apple.cdf.json  # validate single file
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

SCHEMA_PATH   = Path(__file__).parent.parent / "schema" / "cdf_schema_v1.json"
TOKENS_PATH   = Path(__file__).parent.parent / "tokens"
GENERATED_PATH = Path(__file__).parent.parent / "data" / "generated"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def validate_token(path, schema):
    try:
        token = load_json(path)
        validate(instance=token, schema=schema)
        track_count = len(token.get("tracks", []))
        sources = set(t.get("source", "?") for t in token.get("tracks", []))
        src_str = "+".join(sorted(sources))
        print(f"  ✓  VALID     {path.name:<35} {track_count} tracks [{src_str}]")
        return True
    except ValidationError as e:
        field = " -> ".join(str(p) for p in e.absolute_path) or "root"
        print(f"  ✗  INVALID   {path.name:<35} {field} — {e.message}")
        return False
    except Exception as e:
        print(f"  ✗  ERROR     {path.name:<35} {e}")
        return False


def validate_folder(folder, schema, label):
    files = sorted(folder.glob("*.cdf.json")) + sorted(folder.glob("*.json"))
    files = list(dict.fromkeys(files))  # deduplicate

    if not files:
        print(f"  No token files found in {folder}")
        return 0, 0

    print(f"\n  {label} ({len(files)} files)")
    print(f"  {'─' * 60}")

    passed = sum(validate_token(f, schema) for f in files)
    return passed, len(files)


def main():
    parser = argparse.ArgumentParser(description="CDF Token Validator")
    parser.add_argument("file",       nargs="?",      help="Path to a single CDF token file")
    parser.add_argument("--all",      action="store_true", help="Validate all tokens (default)")
    parser.add_argument("--generated", action="store_true", help="Validate generated tokens only")
    parser.add_argument("--examples", action="store_true", help="Validate example tokens only")
    args = parser.parse_args()

    schema = load_json(SCHEMA_PATH)

    print(f"\nCognitive Data Format — Token Validator")
    print(f"Schema: {SCHEMA_PATH.name} (v1.1 — frequency field removed)")

    # Single file mode
    if args.file:
        print(f"{'─' * 65}")
        result = validate_token(Path(args.file), schema)
        print(f"{'─' * 65}")
        print(f"  Result: {'VALID' if result else 'INVALID'}\n")
        sys.exit(0 if result else 1)

    # Folder validation
    total_passed = 0
    total_files  = 0

    if args.generated:
        p, f = validate_folder(GENERATED_PATH, schema, "Generated tokens — data/generated/")
        total_passed += p
        total_files  += f

    elif args.examples:
        p, f = validate_folder(TOKENS_PATH, schema, "Example tokens — tokens/")
        total_passed += p
        total_files  += f

    else:
        # Default — validate everything
        p, f = validate_folder(TOKENS_PATH, schema, "Example tokens — tokens/")
        total_passed += p
        total_files  += f

        p, f = validate_folder(GENERATED_PATH, schema, "Generated tokens — data/generated/")
        total_passed += p
        total_files  += f

    print(f"\n{'─' * 65}")
    print(f"  Total:   {total_passed}/{total_files} valid")

    if total_passed == total_files:
        print(f"  Result:  All tokens valid ✓\n")
    else:
        failed = total_files - total_passed
        print(f"  Result:  {failed} invalid — fix before committing\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
