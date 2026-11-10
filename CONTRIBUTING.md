# Contributing Guide

## Setup

git clone https://github.com/soundarya-satal/BA-Project.git
cd BA-Project
python -m venv .venv
pip install -r requirements.txt

## Workflow
1. Create a feature branch: git checkout -b feature/name
2. Run EDA: python eda_summary.py
3. Engineer features: use helpers in ts_features.py and sentiment_utils.py
4. Evaluate model: use model_metrics.py
5. Open a Pull Request against main

## Code Style
- Follow PEP 8
- Add docstrings to all public functions
- Commit messages: type: description  (feat / fix / docs / chore / style)
