import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, BaggingClassifier,
    GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
)
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

os.makedirs('Models', exist_ok=True)


def get_base_models():
    models = {
        'LogisticRegression':         LogisticRegression(max_iter=1000, random_state=42),
        'GaussianNB':                 GaussianNB(),
        'SVC':                        SVC(probability=True, random_state=42),
        'KNeighborsClassifier':       KNeighborsClassifier(n_neighbors=5),
        'DecisionTreeClassifier':     DecisionTreeClassifier(random_state=42),
        'ExtraTreeClassifier':        ExtraTreeClassifier(random_state=42),
        'RandomForestClassifier':     RandomForestClassifier(n_estimators=100, random_state=42),
        'BaggingClassifier':          BaggingClassifier(random_state=42),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
        'AdaBoostClassifier':         AdaBoostClassifier(random_state=42),
    }
    if HAS_CATBOOST:
        models['CatBoostClassifier'] = CatBoostClassifier(verbose=0, random_state=42)
    if HAS_LGBM:
        models['LGBMClassifier'] = LGBMClassifier(verbose=-1, random_state=42)
    return models


def train_and_save(X_train, X_test, y_train, y_test, label_encoders, out_path, name):
    print(f"\n{'='*50}")
    print(f"Training {name}")
    print('='*50)

    base_models = get_base_models()
    for mname, model in base_models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"  {mname}: {acc:.4f}")

    meta = LogisticRegression(max_iter=1000, random_state=42)
    stacking = StackingClassifier(estimators=list(base_models.items()), final_estimator=meta, cv=5)
    stacking.fit(X_train, y_train)

    y_pred = stacking.predict(X_test)
    print(f"\nStacking Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    bundle = {'base_models': base_models, 'meta_model': stacking.final_estimator_}
    bundle.update(label_encoders)
    bundle['feature_names'] = list(X_train.columns)

    joblib.dump(bundle, out_path)
    print(f"Saved: {out_path}")


# ── IRRIGATION (Kaggle Crop Dataset features → irrigation schedule) ──────────
print("\nLoading Irrigation dataset (from Kaggle Crop features)...")
df_irr = pd.read_csv('Datasets/Irrigation_recommendation.csv')
print(f"Shape: {df_irr.shape}")
print(df_irr['irrigation_schedule'].value_counts())

le_irr = LabelEncoder().fit(df_irr['irrigation_schedule'])
X_irr  = df_irr.drop('irrigation_schedule', axis=1)
y_irr  = le_irr.transform(df_irr['irrigation_schedule'])

X_tr, X_te, y_tr, y_te = train_test_split(X_irr, y_irr, test_size=0.3, random_state=42, stratify=y_irr)
train_and_save(X_tr, X_te, y_tr, y_te,
               {'le_target': le_irr},
               'Models/Irrigation_Recommendation_model.pkl', 'Irrigation')


# ── FERTILIZER (Kaggle: Sanket Gondaliya) ────────────────────────────────────
print("\nLoading Fertilizer dataset (Kaggle: Sanket Gondaliya)...")
df_fert = pd.read_csv('Datasets/Fertilizer Prediction.csv')
df_fert.columns = df_fert.columns.str.strip()
df_fert = df_fert.rename(columns={'Temparature': 'Temperature', 'Fertilizer Name': 'Fertilizer'})
print(f"Shape: {df_fert.shape}")
print(df_fert['Fertilizer'].value_counts())

le_soil_fert = LabelEncoder().fit(df_fert['Soil Type'])
le_crop_fert = LabelEncoder().fit(df_fert['Crop Type'])
le_fert      = LabelEncoder().fit(df_fert['Fertilizer'])

X_fert = df_fert.drop('Fertilizer', axis=1).copy()
X_fert['Soil Type'] = le_soil_fert.transform(X_fert['Soil Type'])
X_fert['Crop Type'] = le_crop_fert.transform(X_fert['Crop Type'])
y_fert = le_fert.transform(df_fert['Fertilizer'])

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_fert, y_fert, test_size=0.3, random_state=42, stratify=y_fert)
train_and_save(X_tr2, X_te2, y_tr2, y_te2,
               {'le_soil': le_soil_fert, 'le_crop': le_crop_fert, 'le_target': le_fert},
               'Models/Fertilizer_Recommendation_model.pkl', 'Fertilizer')

print("\nAll models trained and saved!")
