---
license: apache-2.0
title: 💧 AI Farm Advisor
sdk: streamlit
emoji: 🌍
colorFrom: blue
colorTo: green
pinned: false
short_description: AI-powered crop and fertilizer recommendation system
sdk_version: 1.43.2
---
# **🌱 AI Farm Advisor**

## **📌 Introduction**
AI-powered system that predicts the optimal **crop** and **fertilizer** based on soil conditions, weather patterns, and nutrient levels using a **Stacking Ensemble** of 12 ML models. Includes an **AI chatbot** powered by Groq LLaMA for farm advisory.

## **📂 Datasets**

### Crop Recommendation — `Datasets/Crop_recommendation.csv` — 2,200 balanced records
Source: [Kaggle — Atharva Ingle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
| Feature | Description |
|---|---|
| N | Nitrogen content in soil |
| P | Phosphorus content in soil |
| K | Potassium content in soil |
| temperature | Air temperature (°C) |
| humidity | Relative humidity (%) |
| ph | Soil pH level |
| rainfall | Rainfall (mm) |
| **label** | **Target**: 22 crop types (rice, wheat, maize, etc.) |

### Fertilizer Recommendation — `Datasets/Fertilizer Prediction.csv`
Source: [Kaggle — Sanket Gondaliya](https://www.kaggle.com/datasets/sanketgondaliya/fertilizer)
| Feature | Description |
|---|---|
| Temperature | Air temperature (°C) |
| Humidity | Relative humidity (%) |
| Moisture | Soil moisture (%) |
| Soil Type | Sandy / Loamy / Black / Red / Clayey |
| Crop Type | Maize / Wheat / Paddy / Cotton / etc. |
| Nitrogen | Nitrogen level (N) |
| Potassium | Potassium level (K) |
| Phosphorous | Phosphorous level (P) |
| **Fertilizer Name** | **Target**: Urea / DAP / 28-28 / 14-35-14 / etc. |

## **📊 Methodology**
1. **EDA** — distribution plots, box plots, correlation heatmap
2. **Preprocessing** — Label Encoding, 70/30 train-test split
3. **Base Models** — 12 algorithms: LogisticRegression, GaussianNB, SVC, KNN, DecisionTree, ExtraTree, RandomForest, Bagging, GradientBoosting, AdaBoost, CatBoost, LightGBM
4. **Ensemble** — Stacking Classifier (5-fold CV) with Logistic Regression as meta-model
5. **AI Chatbot** — Groq LLaMA 3.3 70B for farm advisory

## **🚀 Technologies**
| Category | Tools |
|---|---|
| ML | Scikit-Learn, CatBoost, LightGBM |
| Data | Pandas, NumPy |
| Visualization | Plotly |
| AI Chatbot | Groq (LLaMA 3.3 70B) |
| Deployment | Streamlit |

## **📥 Setup**
```bash
git clone https://github.com/Shoaib954/fertilizer.git
cd fertilizer
pip install -r requirements.txt

# Train models
python train_model.py

# Run app
streamlit run app.py
```

## **📁 Project Structure**
```
fertilizer/
├── Datasets/
│   ├── Crop_recommendation.csv
│   └── Fertilizer Prediction.csv
├── Models/
│   ├── Crop_Recommendation_model.pkl
│   └── Fertilizer_Recommendation_model.pkl
├── Images/
│   └── shell.webp
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

## **📧 Contact**
📌 **Author:** Shoaib — [GitHub](https://github.com/Shoaib954)
