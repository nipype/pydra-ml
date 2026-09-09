#!/usr/bin/env python
"""Thin shim so versioneer can compute the version and freeze it at build time.

All other metadata lives in pyproject.toml.
"""
import os
import sys

from setuptools import setup

# PEP 517 build backends exec this file without the project root on
# sys.path, so the vendored versioneer.py next to this file wouldn't
# otherwise be importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import versioneer  # noqa: E402

if __name__ == "__main__":
    setup(version=versioneer.get_version(), cmdclass=versioneer.get_cmdclass())
