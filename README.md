# 🤖 Sentiment Analyzer Bot: Data Curation & Model Optimization

A production-grade case study on curating, cleaning, and formatting user complaint data to train a high-accuracy AI sentiment classification model.

## 📊 Project Overview
* **Role:** Data AI Trainer & Annotator
* **Dataset Size:** 300+ Raw User Complaints
* **Objective:** Clean messy, unstructured user feedback and transform it into high-quality training data for an AI Sentiment Analyzer.

---

## 🛠️ Phase 1: High-Standard Data Curation
To ensure the AI strictly adheres to sentiment rules, the data underwent a rigorous annotation pipeline:

1. **Text Preprocessing:** Removed duplicates, fixed typos, and stripped irrelevant metadata/emojis.
2. **Edge-Case Tagging:** Handled ambiguous complaints (e.g., sarcasm or mixed feedback) by creating a strict guidelines matrix:
   * **Positive:** Feature requests paired with compliments.
   * **Neutral:** General inquiries without emotional tone.
   * **Negative:** Direct complaints, system bugs, and churn threats.

---

## 💻 Phase 2: Python Implementation (Mock Code)
Below is the core script structure used to validate formatting and distribution before feeding the dataset into the LLM training pipeline:

```python
import json

# Sample curated dataset structure
training_data = [
    {
        "text": "Your system crashed twice during my checkout process. Fix this ASAP!",
        "label": "Negative",
        "confidence_score": 0.98
    },
    {
        "text": "The new update looks clean, but I still miss the old layout option.",
        "label": "Neutral",
        "confidence_score": 0.85
    }
]

def validate_dataset(data):
    for entry in data:
        assert "text" in entry and "label" in entry, "Invalid data format!"
    print(f"✅ Validation Success: {len(data)} entries are clean and ready.")

validate_dataset(training_data)
