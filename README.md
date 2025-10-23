# Fake News Classifier

A machine learning project demonstrating the complete **ML Lifecycle** from scoping to deployment. This classifier identifies fake news articles using fine-tuned DistilBERT, a lightweight transformer model optimized for text classification.

📓 **[View Full Implementation in fake_news_model.ipynb](fake_news_model.ipynb)**

---

## 📁 Project Files

- **`fake_news_model.ipynb`** - Complete implementation covering all ML lifecycle phases: problem scoping, data preprocessing, model training with DistilBERT, evaluation, and deployment preparation
- **`requirements.txt`** - Python dependencies including transformers, torch, scikit-learn, and other required libraries
- **`tokenizer_server.py`** - Flask API server for serving the trained model with endpoints for text prediction and health monitoring

---

## 📈 Results

- **Test Accuracy:** 95.4%
- **Precision:** 95.1%
- **Recall:** 95.7%
- **F1-Score:** 95.4%

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/pilavci-ds/fake_news_classifier_ML_lifecycle.git

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook fake_news_model.ipynb
```

---

## 📁 Project Structure

```
├── fake_news_model.ipynb    # Main implementation notebook
├── data/                    # Dataset files
├── models/                  # Trained models
└── requirements.txt         # Dependencies
```

---

## 👤 Author

**GitHub:** [@pilavci-ds](https://github.com/pilavci-ds)