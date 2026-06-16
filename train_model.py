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
os.makedirs('Datasets', exist_ok=True)


def get_base_models():
    models = {
        'LogisticRegression':         LogisticRegression(solver='liblinear', max_iter=1000, random_state=42),
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

    meta = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
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


# ── IRRIGATION ──────────────────────────────────────────────────────────────
df_irr = pd.read_csv('Datasets/Irrigation_recommendation.csv')

le_soil_irr = LabelEncoder().fit(df_irr['soil_type'])
le_crop_irr = LabelEncoder().fit(df_irr['crop_type'])
le_schedule = LabelEncoder().fit(df_irr['irrigation_schedule'])

X_irr = df_irr.drop('irrigation_schedule', axis=1).copy()
X_irr['soil_type'] = le_soil_irr.transform(X_irr['soil_type'])
X_irr['crop_type'] = le_crop_irr.transform(X_irr['crop_type'])
y_irr = le_schedule.transform(df_irr['irrigation_schedule'])

X_tr, X_te, y_tr, y_te = train_test_split(X_irr, y_irr, test_size=0.3, random_state=42, stratify=y_irr)
train_and_save(X_tr, X_te, y_tr, y_te,
               {'le_soil': le_soil_irr, 'le_crop': le_crop_irr, 'le_target': le_schedule},
               'Models/Irrigation_Recommendation_model.pkl', 'Irrigation')


# ── FERTILIZER DATASET GENERATION ───────────────────────────────────────────
print("\nGenerating fertilizer dataset...")
np.random.seed(42)
n = 2000
soil_types = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
crop_types = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy',
              'Barley', 'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']

# Fertilizer rules based on NPK levels
fertilizers = ['Urea', 'DAP', 'MOP', '14-35-14', '28-28', '17-17-17', '20-20', '10-26-26', 'TSP']

per = n // len(fertilizers)
dfs = []
for i, fert in enumerate(fertilizers):
    # Each fertilizer has a distinct N-P-K signature
    base_n = [150, 50, 20, 80, 120, 80, 100, 30, 10][i]
    base_p = [0, 120, 10, 140, 80, 80, 80, 130, 160][i]
    base_k = [0, 0, 200, 80, 0, 80, 80, 130, 0][i]
    dfs.append(pd.DataFrame({
        'Temperature': np.random.randint(15, 42, per),
        'Humidity':    np.random.randint(30, 90, per),
        'Moisture':    np.random.randint(20, 80, per),
        'Soil Type':   np.random.choice(soil_types, per),
        'Crop Type':   np.random.choice(crop_types, per),
        'Nitrogen':    np.clip(np.random.normal(base_n, 20, per), 0, 300).astype(int),
        'Potassium':   np.clip(np.random.normal(base_k, 20, per), 0, 300).astype(int),
        'Phosphorous': np.clip(np.random.normal(base_p, 20, per), 0, 300).astype(int),
        'Fertilizer':  [fert] * per
    }))

# Fill remaining rows
remainder = n - per * len(fertilizers)
if remainder > 0:
    dfs.append(dfs[0].iloc[:remainder].copy())

df_fert = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
df_fert.to_csv('Datasets/Fertilizer_recommendation.csv', index=False)
print(f"Fertilizer dataset saved: {df_fert.shape}")
print(df_fert['Fertilizer'].value_counts())


# ── FERTILIZER MODEL ─────────────────────────────────────────────────────────
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
