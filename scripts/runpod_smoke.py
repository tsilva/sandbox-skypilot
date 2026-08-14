#!/usr/bin/env python3
"""Compatibility wrapper for the provider-neutral smoke test."""

from cloud_smoke import main

if __name__ == "__main__":
    raise SystemExit(main())
