#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Pydra: Dataflow Engine"""
import sys

from setuptools import setup

import versioneer

# Give setuptools a hint to complain if it's too old a version
# 30.3.0 allows us to put most metadata in setup.cfg
# Should match pyproject.toml
SETUP_REQUIRES = ["setuptools >= 30.3.0"]
# This enables setuptools to install wheel on-the-fly
SETUP_REQUIRES += ["wheel"] if "bdist_wheel" in sys.argv else []

if __name__ == "__main__":
    setup(
        name="pydra_ml",
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        setup_requires=SETUP_REQUIRES,
    )
