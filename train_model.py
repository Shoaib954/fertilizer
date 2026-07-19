import warnings
warnings.filterwarnings('ignore')
import os, joblib, pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

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
        'CatBoostClassifier':         CatBoostClassifier(verbose=0, random_state=42),
        'LGBMClassifier':             LGBMClassifier(verbose=-1, random_state=42),
    }

# ── Irrigation ────────────────────────────────────────────────────────────────
print("Training Irrigation model...")
df_irr = pd.read_csv('Datasets/Irrigation_recommendation.csv')
le_irr = LabelEncoder().fit(df_irr['irrigation_schedule'])
X_irr  = df_irr.drop('irrigation_schedule', axis=1)
y_irr  = le_irr.transform(df_irr['irrigation_schedule'])
X_tr, X_te, y_tr, y_te = train_test_split(X_irr, y_irr, test_size=0.3, random_state=42, stratify=y_irr)

irr_base = get_base_models()
for name, m in irr_base.items():
    m.fit(X_tr, y_tr)
    print(f"  {name}: {m.score(X_te, y_te)*100:.2f}%")

irr_stack = StackingClassifier(estimators=list(irr_base.items()),
                               final_estimator=LogisticRegression(max_iter=1000, random_state=42), cv=5)
irr_stack.fit(X_tr, y_tr)
print(f"  Stacking: {irr_stack.score(X_te, y_te)*100:.2f}%")
joblib.dump({'base_models': irr_base, 'meta_model': irr_stack, 'le_target': le_irr}, 'Models/Irrigation_Recommendation_model.pkl')
print("Saved: Models/Irrigation_Recommendation_model.pkl")

# ── Fertilizer ────────────────────────────────────────────────────────────────
print("\nTraining Fertilizer model...")
df_fert = pd.read_csv('Datasets/Fertilizer Prediction.csv')
df_fert.columns = df_fert.columns.str.strip()
df_fert = df_fert.rename(columns={'Temparature': 'Temperature', 'Fertilizer Name': 'Fertilizer'})
le_soil = LabelEncoder().fit(df_fert['Soil Type'])
le_crop = LabelEncoder().fit(df_fert['Crop Type'])
le_fert = LabelEncoder().fit(df_fert['Fertilizer'])
X_fert  = df_fert.drop('Fertilizer', axis=1).copy()
X_fert['Soil Type'] = le_soil.transform(X_fert['Soil Type'])
X_fert['Crop Type'] = le_crop.transform(X_fert['Crop Type'])
y_fert  = le_fert.transform(df_fert['Fertilizer'])
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_fert, y_fert, test_size=0.3, random_state=42, stratify=y_fert)

fert_base = get_base_models()
for name, m in fert_base.items():
    m.fit(X_tr2, y_tr2)
    print(f"  {name}: {m.score(X_te2, y_te2)*100:.2f}%")

fert_stack = StackingClassifier(estimators=list(fert_base.items()),
                                final_estimator=LogisticRegression(max_iter=1000, random_state=42), cv=5)
fert_stack.fit(X_tr2, y_tr2)
print(f"  Stacking: {fert_stack.score(X_te2, y_te2)*100:.2f}%")
joblib.dump({'base_models': fert_base, 'meta_model': fert_stack,
             'le_soil': le_soil, 'le_crop': le_crop, 'le_target': le_fert}, 'Models/Fertilizer_Recommendation_model.pkl')
print("Saved: Models/Fertilizer_Recommendation_model.pkl")
