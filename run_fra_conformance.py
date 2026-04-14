"""Run the FRA conformance pytest targets configured in `fra_conformance.yaml`."""

from __future__ import annotations

import sys

import pytest

from tests.fra_conformance.config import load_conformance_config


def main(argv: list[str] | None = None) -> int:
    config = load_conformance_config()
    pytest_args = list(config["pytest_args"])
    pytest_args.extend(sys.argv[1:] if argv is None else argv)
    if not pytest_args:
        pytest_args = ["tests/fra_conformance"]
    return pytest.main(pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
