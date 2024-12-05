#!/bin/bash

# Generate model files
python scripts/generate_naive_bayes_model.py

# Run tests
python -m pytest -v tests/
