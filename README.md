# NLP Projects

Two natural language processing projects exploring text classification from different angles — one using classic machine learning, one using a transformer model. Both were built and evaluated on real-world datasets, with a focus on understanding *why* each model succeeds or fails, not just reporting an accuracy number.

---

## 📰 Fake News Detection

A binary classifier that labels news articles as **real** or **fake**, using classic NLP and machine learning rather than deep learning.

**Approach**
- Dataset: [Fake and Real News Dataset](https://www.kaggle.com) (Kaggle) — 44,898 articles (23,481 fake, 21,417 real)
- Preprocessing: merged title + body text, lowercased, stripped URLs/special characters, removed stopwords
- Features: TF-IDF vectorization (5,000 features)
- Model: Logistic Regression

**Results**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Fake | 0.99 | 0.99 | 0.99 |
| Real | 0.99 | 0.99 | 0.99 |

**Overall accuracy: 98.89%**

**What stood out**
A simple Logistic Regression model outperformed the BERT-based sentiment model (below) by a wide margin — largely because fake news tends to use exaggerated, emotional language that's easy to separate linguistically, while sentiment is inherently more nuanced. Testing on made-up headlines mostly confirmed this ("Scientists discover vaccine that cures all diseases overnight" → correctly flagged as fake), but the model also mislabeled a plausible real headline ("Apple reported quarterly earnings of 90 billion dollars" → incorrectly flagged as fake), likely due to similar-sounding clickbait phrasing in the training data. A good reminder that high accuracy on a test set doesn't mean the model has learned everything it needs to.

**Tech:** Python, Pandas, Scikit-learn, TF-IDF, Regex

---

## 💬 Sentiment Analysis with BERT

A three-class sentiment classifier (positive / negative / neutral) built by fine-tuning a pre-trained BERT model on real tweets.

**Approach**
- Dataset: [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval) (HuggingFace) — ~45.6K training tweets, 2K validation, 12.3K test
- Tokenized with `BertTokenizer`, padded/truncated to 128 tokens, with attention masks
- Fine-tuned pre-trained BERT with a classification head added on top
- Training: learning rate 2e-5, batch size 16, 3 epochs, GPU-accelerated (CUDA)

**Results**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.64 | 0.68 | 0.66 |
| Neutral | 0.73 | 0.68 | 0.70 |
| Positive | 0.76 | 0.80 | 0.78 |

**Overall accuracy: 73%** (consistent with published benchmarks for this dataset)

**What stood out**
The model performed best on positive tweets and struggled most with negative ones — negative sentiment is often expressed indirectly, making it harder to pick up from surface-level language patterns. One test case captured BERT's core limitation well: given the input *"I drank 5 liters thumbsup"*, the model predicted positive sentiment, picking up on the word "thumbsup" without understanding that drinking 5 liters of anything is actually harmful. It's a clean example of a model that's fluent in language patterns but has no real-world common sense.

**Tech:** Python, PyTorch, HuggingFace Transformers, Scikit-learn, CUDA

---

## 🔍 Takeaway

Putting these two side by side made one thing clear: model complexity doesn't guarantee better results — task difficulty and how well the data patterns match the model's strengths matter more. Fake news detection is a comparatively "easier" pattern-matching problem, which is why a simple TF-IDF + Logistic Regression pipeline hit 99%, while nuanced sentiment classification is a genuinely harder problem even for a large pre-trained transformer like BERT.

---

## 📂 Structure

```
NLP-Projects/
├── fake-news-detection/
│   ├── News Detection.ipynb
│   └── FakeNews_Detection.docx
├── sentiment-analysis/
│   ├── Sentimental Analusis.ipynb
│   └── BERT_Sentiment_Analysis.docx
└── README.md
```
