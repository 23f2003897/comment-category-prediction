# 💬 Comment Category Prediction

A multi-class text classification project built for Kaggle competition. Classifies user comments into 4 categories using NLP and machine learning techniques.

---

## 📌 Problem Statement

Classify user comments into 4 categories (labels 0–3) based on comment text and metadata like upvotes, downvotes, emoticons, and demographic features. The main challenge is severe class imbalance — Label 0 has 114,000 samples while Label 3 has only 5,500 (~21× ratio).

---

## 📂 Project Structure

```
├── notebook.ipynb       # Main Kaggle notebook
├── README.md                   # Project documentation
└── report.pdf                # Detailed project report
```

---

## 📊 Dataset

| Split | Rows | Columns |
|-------|------|---------|
| Train | 198,000 | 15 |
| Test  | 102,000 | 14 |

**Key columns:** `comment`, `upvote`, `downvote`, `emoticon_1/2/3`, `race`, `religion`, `gender`, `disability`, `created_date`, `label`

---

## ⚙️ Preprocessing

- Missing values in `race`, `religion`, `gender` (73% missing) filled with `"unknown"` and label encoded
- Single missing `comment` filled with empty string
- 48 features engineered from raw data
- `StandardScaler` applied to numerical features
- Word-level TF-IDF (12,000 features) + Character-level TF-IDF (5,000 features)

---

## 🔧 Feature Engineering

| Feature | Description |
|---------|-------------|
| `text_length` | Character count of comment |
| `word_count` | Number of words |
| `lexical_density` | Unique words / total words |
| `caps_ratio` | Ratio of uppercase characters |
| `log_upvote` | Log-transformed upvote count |
| `vote_ratio` | Upvotes / (upvotes + downvotes) |
| `is_weekend` | Whether posted on weekend |
| `is_night` | Whether posted at night |
| `emoticon_total` | Sum of all emoticon counts |
| `sensitive_total` | Sum of sensitive feature flags |

---

## 🤖 Models

| Model | Accuracy | F1 Macro | F1 Weighted |
|-------|----------|----------|-------------|
| **LightGBM** ⭐ | **0.9115** | **0.8159** | **0.9138** |
| Logistic Regression | 0.9013 | 0.7703 | 0.8990 |
| XGBoost | 0.8979 | 0.7468 | 0.8943 |

**Best Model:** LightGBM with custom inverse-frequency class weights

---

## 🔁 Pipeline

```python
Pipeline([
    ('preprocessor', ColumnTransformer([
        ('tfidf_word', TfidfVectorizer(max_features=12000), 'comment'),
        ('tfidf_char', TfidfVectorizer(analyzer='char'), 'comment'),
        ('scaler',     StandardScaler(), num_cols)
    ])),
    ('model', LGBMClassifier(class_weight=class_weights))
])
```

---

## 📈 Evaluation

- **Primary metric:** F1 Macro (handles class imbalance)
- **Split:** 90/10 train-validation with stratification
- **Baseline issue:** Default models predicted majority class only
- **Fix:** Class weights + character-level TF-IDF + hyperparameter tuning

---

## 💡 Key Insights

### From the Data
- **Class imbalance is severe** — Label 0 is 21× more frequent than Label 3, making accuracy a misleading metric. F1 Macro is the right choice here
- **Label 1 has the longest comments** (avg 336 chars) and highest upvote engagement, suggesting these are detailed, well-received comments
- **Label 3 has the shortest comments** and lowest engagement — hardest class to classify with only 5,500 samples
- **73% of demographic columns** (`race`, `religion`, `gender`) are missing — filling with `"unknown"` preserved useful signal rather than dropping the columns entirely
- **emoticon_1 is the most used** emoticon at 14.6%, and emoticon usage varies meaningfully across labels
- **No strong linear correlation** between text features and upvotes — non-linear models like LightGBM are necessary

### From the Models
- **Accuracy is deceptive** — XGBoost had 89.7% accuracy but only 0.7468 F1 Macro because it barely predicted Label 3
- **Class weighting had the biggest single impact** on minority class recall — without it, Label 3 F1 was near zero
- **Character-level TF-IDF significantly improved results** over word-level alone by capturing partial word patterns and morphological features
- **LightGBM outperforms XGBoost** on sparse TF-IDF matrices — faster training and better handling of high-dimensional features
- **Feature engineering added value** beyond raw text — `vote_ratio`, `log_upvote`, `is_night`, and `emoticon_total` all contributed meaningful signal
- **Hyperparameter tuning mattered** — RandomizedSearchCV on LR improved F1 Macro by ~2%, and manual tuning of LightGBM's `num_leaves` and `subsample` reduced overfitting

---

## 🛠️ Tech Stack

- Python 3.12
- pandas, numpy
- scikit-learn
- LightGBM, XGBoost
- matplotlib, seaborn
- wordcloud

---

## 🚀 How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/23f2003897/comment-category-prediction.git
   cd comment-category-prediction
   ```

2. Install dependencies
   ```bash
   pip install pandas numpy scikit-learn lightgbm xgboost matplotlib seaborn wordcloud
   ```

3. Open the notebook
   ```bash
   jupyter notebook notebook-t12026.ipynb
   ```

4. Run all cells top to bottom

---

## 👤 Author

**Khusbu Pandey**  
Kaggle: [23f2003897](https://kaggle.com/code/khusbupandey/23f2003897-notebook-t12026)
