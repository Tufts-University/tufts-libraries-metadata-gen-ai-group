#!/usr/bin/env python3
import json
from pathlib import Path

TEMPLATE_FILE = Path("rbms-crosswalk-3b.template")
OUTPUT_FILE   = Path("rbms-crosswalk-3b.modelfile")

TERMS_FILE    = Path("rbms_terms.json")
CHANGED_FILE  = Path("rbms_changed.json")
DELETED_FILE  = Path("rbms_deleted.json")

def main():
    template = TEMPLATE_FILE.read_text(encoding="utf-8")

    # Load raw JSON text – DO NOT pretty-print, we want compact
    rbms_terms_text   = TERMS_FILE.read_text(encoding="utf-8").strip()
    rbms_changed_text = CHANGED_FILE.read_text(encoding="utf-8").strip()
    rbms_deleted_text = DELETED_FILE.read_text(encoding="utf-8").strip()

    # Optionally validate that they are valid JSON
    json.loads(rbms_terms_text)
    json.loads(rbms_changed_text)
    json.loads(rbms_deleted_text)

    filled = (
        template
        .replace("__RBMS_TERMS_JSON__", rbms_terms_text)
        .replace("__RBMS_CHANGED_JSON__", rbms_changed_text)
        .replace("__RBMS_DELETED_JSON__", rbms_deleted_text)
    )

    OUTPUT_FILE.write_text(filled, encoding="utf-8")
    print(f"Wrote {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
