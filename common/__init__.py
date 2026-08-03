"""Shared library for the agentic age-gate detection pipeline.

Modules here are IMPORTED by scripts/ and spike/ — never run directly.
They exist so the collection script and all classification scripts share
ONE schema and ONE set of helpers and can never fork the data model.
"""
