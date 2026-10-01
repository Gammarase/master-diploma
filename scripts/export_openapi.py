"""
Write the API's OpenAPI document to ``api/docs/openapi.yaml``.

Usage:
    python scripts/export_openapi.py            # write the file
    python scripts/export_openapi.py --check    # exit 1 if the file is stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.app import create_app  # noqa: E402

OUTPUT = ROOT / "api" / "docs" / "openapi.yaml"


class _Dumper(yaml.SafeDumper):
    """Writes multi-line strings as readable ``|`` blocks."""


def _str_representer(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_Dumper.add_representer(str, _str_representer)


def render() -> str:
    spec = create_app().openapi()
    return yaml.dump(spec, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="only check the file is current")
    args = parser.parse_args()
    text = render()
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != text:
            print(f"{OUTPUT} is out of date; run scripts/export_openapi.py")
            return 1
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
