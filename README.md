# Smart MCQ Solver Challenge

This project explores how different NLP and deep learning approaches can solve multiple-choice questions by ranking the top three most likely answers.

## Overview
The repository compares several approaches for MCQ answering, including:
- TF-IDF + CatBoost as a classical baseline
- a transformer encoder trained from scratch
- fine-tuned SciBERT
- fine-tuned DeBERTa-v3-base

The best-performing model was the fine-tuned DeBERTa-v3-base, which achieved strong validation performance and was used for the final submission.

## Project Structure
- `src/scratch_model/` – training, inference, configuration, dataset handling, and transformer components
- `src/pretrained/` – notebooks and experiments for pretrained models
- `reports/` – project reports and milestone documents

## Setup
1. Create and activate a Python environment.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the scratch transformer model:
   ```bash
   python src/scratch_model/train.py
   ```
4. Generate predictions:
   ```bash
   python src/scratch_model/inference.py
   ```

You can also run the notebook in `src/pretrained/` for the pretrained-model experiments.

## Key Results
- Best validation MAP@3: approximately 0.9988 with DeBERTa-v3-base
- Kaggle Leaderboard score 0.75270

## Notes
- The main report is available in the `reports/` folder.

## Deployment
- [Click Here](https://huggingface.co/spaces/ani017/MCQ-solver-scratch) to view the deployment on HF spaces