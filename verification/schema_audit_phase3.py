"""Verify immutable contracts and standards rules in built wheel/sdist imports."""

from schema_audit_phase1 import main

if __name__ == "__main__":
    raise SystemExit(main(include_standards=True))
