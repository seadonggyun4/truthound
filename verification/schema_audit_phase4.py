"""Execute contract and audit A/B rule tests from built wheel and sdist."""

from schema_audit_phase1 import main

if __name__ == "__main__":
    raise SystemExit(main(include_implementation=True))
