# HotelSentinel
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-orange.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-green.svg)
![Deployment](https://img.shields.io/badge/Deployment-Streamlit-purple.svg)

## Overview
**HotelSentinel** is an NLP-powered sentiment intelligence system that analyzes hotel reviews to classify customer sentiment and uncover the key factors influencing guest experience.

The project transforms unstructured textual feedback into actionable insights, enabling hotel managers to understand what drives positive and negative reviews and improve service quality and brand perception.

---
 
## Business Problem
Online reviews strongly influence hotel bookings and brand reputation.  
However, manually analyzing thousands of reviews is inefficient and subjective.

**Objective:**
- Classify hotel reviews into sentiment categories (positive, neutral, negative)
- Identify experience drivers that impact customer satisfaction
- Provide an interactive system for real-time sentiment prediction

---

## Dataset Description
- **Records:** ~20,000 hotel reviews  
- **Features:**
  - `Review` (text)
  - `Rating` (1–5 scale)

The dataset contains **no missing values or duplicates**, making it well-suited for NLP-based sentiment analysis.

---

## Methodology & System Design

### 1. Text Preprocessing
Applied standard NLP preprocessing techniques:
- Lowercasing
- Removal of special characters
- Tokenization
- Lemmatization

Additionally, **polarity and subjectivity** scores were computed to better understand sentiment distribution.

---

### 2. Exploratory Data Analysis (EDA)
Key analyses included:
- Rating distribution (5-star reviews dominant, 1-star least frequent)
- Review length distribution
- Sentiment polarity distribution
- Sentiment labeling strategy:
  - Ratings 1–2 → Negative
  - Rating 3 → Neutral
  - Ratings 4–5 → Positive
- Word clouds for each sentiment class
- Bi-gram analysis for top positive, negative, and neutral terms
- Review purpose distribution (business, family, solo travel)

EDA revealed a **highly imbalanced sentiment distribution**, with positive reviews forming the majority.

---

### 3. Feature Engineering
- Converted text into numerical features using **TF-IDF Vectorization**
- Addressed class imbalance using **SMOTE**
- Ensured balanced representation across sentiment classes before model training

---

### 4. Model Development & Evaluation
Trained and evaluated multiple classifiers:
- Logistic Regression
- Multinomial Naive Bayes
- Decision Tree
- Random Forest
- XGBoost

Models were compared using:
- Precision
- Recall
- F1-score
- Support per class

**Logistic Regression** provided the best balance of:
- Performance
- Interpretability
- Stability on high-dimensional TF-IDF features

---

### 5. Model Optimization
- Hyperparameter tuning performed using **GridSearchCV**
- Finalized Logistic Regression model
- Trained model and TF-IDF vectorizer saved using `joblib`

---

## Results & Insights
- Positive reviews dominated the dataset (~74%)
- SMOTE significantly improved recall for minority classes
- Logistic Regression achieved strong performance across all sentiment categories
- Key review terms strongly correlated with service quality, cleanliness, and comfort

The model effectively supports **experience intelligence**, not just sentiment prediction.

---

## Deployment
The finalized model was deployed using **Streamlit**, allowing users to:
- Enter a hotel review
- Receive predicted sentiment
- View confidence scores
- Visualize word cloud highlights

Deployment logic is implemented in:
Hotel_Rating_App.py

Saved artifacts:
final_logreg_model.joblib
tfidf_vectorizer.joblib


---

## Tech Stack
- **Language:** Python  
- **NLP:** TF-IDF, Lemmatization  
- **Modeling:** Scikit-learn  
- **Imbalance Handling:** SMOTE  
- **Visualization:** Matplotlib, Seaborn, WordCloud  
- **Deployment:** Streamlit  

---

## Repository Structure
```
hotel-sentinel/
│
├── Hotel_Rating_Classification_EDA.ipynb
├── Hotel_Rating_Classification_Preprocessing_and_Model_Building.ipynb
├── Hotel_Rating_App.py
├── final_logreg_model.joblib
├── tfidf_vectorizer.joblib
├── hotel_image.png
├── requirements.txt
└── README.md
```

---

## Key Engineering Decisions
- Prioritized NLP preprocessing quality over model complexity
- Addressed class imbalance before training
- Selected Logistic Regression for interpretability and robustness
- Deployed full pipeline for real-world usability

---

## Learnings
- Handling sentiment imbalance in real-world text data
- Importance of precision/recall in NLP classification
- Feature representation impact on model performance
- Translating NLP outputs into business insights

---

## Future Improvements
- Incorporate transformer-based models (BERT)
- Add aspect-based sentiment analysis
- Support multilingual reviews
- Improve explainability with SHAP for text features

---

## License
This project is licensed under the **MIT License**.

---

## Author
**Kumaran Elumalai**  
AI / ML Engineer | Data Scientist  

🔗 GitHub: https://github.com/Kumaran-Elumalai  
🔗 LinkedIn: https://linkedin.com/in/kumaran-elumalai
