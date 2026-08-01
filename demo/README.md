# HuggingFace Spaces Demo — Setup Guide

## How to deploy the ThumaCheck demo on HuggingFace Spaces

### 1. Create a Space
Go to https://huggingface.co/new-space and create:
- **Name**: thumacheck-demo
- **SDK**: Streamlit
- **Visibility**: Public

### 2. Upload files
Upload the following to the Space:
- `demo/app_demo.py` → rename to `app.py`
- `models/expert_v5.joblib` (or your latest model)
- `models/tfidf_expert.joblib`
- `requirements_demo.txt` → rename to `requirements.txt`

### 3. The demo loads pre-trained models with sample texts
No MongoDB needed — the demo uses hardcoded examples + live text input.

### 4. Add the Space URL to your README
Once deployed, add the badge:
```markdown
[![Demo](https://img.shields.io/badge/Demo-HuggingFace-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/YOUR_USERNAME/thumacheck-demo)
```
