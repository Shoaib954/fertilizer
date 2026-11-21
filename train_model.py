import warnings
warnings.filterwarnings('ignore')

import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, BaggingClassifier,
    GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import os

os.makedirs('Models', exist_ok=True)

def get_base_models():
    return {
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
        'XGBClassifier':              XGBClassifier(verbosity=0, random_state=42, eval_metric='mlogloss'),
        'LGBMClassifier':             LGBMClassifier(verbose=-1, random_state=42),
    }

# ── Irrigation ────────────────────────────────────────────────────────────────
print("Training Irrigation models...")
df_irr = pd.read_csv('Datasets/Irrigation_recommendation.csv')
le_irr = LabelEncoder().fit(df_irr['irrigation_schedule'])
X_irr = df_irr.drop('irrigation_schedule', axis=1)
y_irr = le_irr.transform(df_irr['irrigation_schedule'])
X_tr, X_te, y_tr, y_te = train_test_split(X_irr, y_irr, test_size=0.3, random_state=42, stratify=y_irr)

irr_base = get_base_models()
for name, m in irr_base.items():
    print(f"  Fitting {name}...")
    m.fit(X_tr, y_tr)

print("  Fitting Irrigation StackingClassifier...")
irr_stack = StackingClassifier(
    estimators=list(irr_base.items()),
    final_estimator=LogisticRegression(max_iter=1000, random_state=42), cv=5
)
irr_stack.fit(X_tr, y_tr)

# ── Irrigation Metrics ───────────────────────────────────────────────────────
print("Computing Irrigation metrics...")
irr_metrics = {}
for name, m in irr_base.items():
    irr_metrics[name] = accuracy_score(y_te, m.predict(X_te))
irr_metrics['StackingClassifier'] = accuracy_score(y_te, irr_stack.predict(X_te))

y_pred_irr = irr_stack.predict(X_te)
irr_cm   = confusion_matrix(y_te, y_pred_irr)
irr_cr   = classification_report(y_te, y_pred_irr, target_names=le_irr.classes_, output_dict=True)

# ── Fertilizer ────────────────────────────────────────────────────────────────
print("Training Fertilizer models...")
df_fert = pd.read_csv('Datasets/fertilizer_recommendation.csv')

cat_cols = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
            'Irrigation_Type', 'Previous_Crop', 'Region']
encoders = {}
for col in cat_cols:
    le = LabelEncoder().fit(df_fert[col])
    encoders[col] = le
    df_fert[col] = le.transform(df_fert[col])

le_fert = LabelEncoder().fit(df_fert['Recommended_Fertilizer'])
X_fert = df_fert.drop('Recommended_Fertilizer', axis=1)
y_fert = le_fert.transform(df_fert['Recommended_Fertilizer'])
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_fert, y_fert, test_size=0.3, random_state=42, stratify=y_fert)

fert_base = get_base_models()
for name, m in fert_base.items():
    print(f"  Fitting {name}...")
    m.fit(X_tr2, y_tr2)

print("  Fitting Fertilizer StackingClassifier...")
fert_stack = StackingClassifier(
    estimators=list(fert_base.items()),
    final_estimator=LogisticRegression(max_iter=1000, random_state=42), cv=5
)
fert_stack.fit(X_tr2, y_tr2)

# ── Fertilizer Metrics ───────────────────────────────────────────────────────
print("Computing Fertilizer metrics...")
fert_metrics = {}
for name, m in fert_base.items():
    fert_metrics[name] = accuracy_score(y_te2, m.predict(X_te2))
fert_metrics['StackingClassifier'] = accuracy_score(y_te2, fert_stack.predict(X_te2))

y_pred_fert = fert_stack.predict(X_te2)
fert_cm  = confusion_matrix(y_te2, y_pred_fert)
fert_cr  = classification_report(y_te2, y_pred_fert, target_names=le_fert.classes_, output_dict=True)

# ── Save ──────────────────────────────────────────────────────────────────────
print("Saving models...")
with open('Models/irrigation_models.pkl', 'wb') as f:
    pickle.dump((irr_base, irr_stack, le_irr), f)

with open('Models/fertilizer_models.pkl', 'wb') as f:
    pickle.dump((fert_base, fert_stack, le_fert, encoders), f)

with open('Models/metrics.pkl', 'wb') as f:
    pickle.dump({
        'irr':  {'accuracies': irr_metrics,  'cm': irr_cm,  'cr': irr_cr,  'classes': le_irr.classes_},
        'fert': {'accuracies': fert_metrics, 'cm': fert_cm, 'cr': fert_cr, 'classes': le_fert.classes_},
    }, f)

print("Done! Models saved to Models/")
