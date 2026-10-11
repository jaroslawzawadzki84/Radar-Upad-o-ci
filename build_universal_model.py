import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import joblib
import json

print("Start: Budowa Uniwersalnego Modelu (Horyzont 1-6 lat)")

# Sztywno wchodzimy do folderu processed!
data_dir = Path("V4FinBench_data_load-main/processed")
files = list(data_dir.glob("company_years_h*.parquet"))

if not files:
    print("Nie znaleziono plików w folderze processed. Szukam w głównym...")
    data_dir = Path("V4FinBench_data_load-main")
    files = list(data_dir.glob("company_years_h*.parquet"))

dfs = []
for f in files:
    h_str = f.stem.split('_')[-1]
    try:
        h_val = int(h_str.replace('h', ''))
    except ValueError:
        continue
        
    print(f"Wczytuję {f.name}...")
    
    try:
        # Główna próba wczytania (wymuszenie pyarrow)
        df = pd.read_parquet(f, engine="pyarrow")
    except Exception as e:
        print(f"Blad przy pyarrow: {e}, probuje fastparquet...")
        df = pd.read_parquet(f, engine="fastparquet")
        
    df['horizon'] = h_val
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
print(f"Polaczono {len(dfs)} horyzontow. Razem wierszy: {len(df_all)}")

# 2. Usuwanie przecieków (Data Leakage) i zbędnych kolumn
DROP_COLS = ['Insolvency_flag', 'num', 'emis_id', 'company', 'industry', 'link', 'operational_status']
df_all = df_all.drop(columns=[c for c in DROP_COLS if c in df_all.columns])

# 3. Podział Train/Test na podstawie roku
SPLIT_YEAR = 2016
train = df_all[df_all['year'] <= SPLIT_YEAR]
test = df_all[df_all['year'] > SPLIT_YEAR]

X_train = train.drop(columns=['main_label', 'year'])
y_train = train['main_label']
X_test = test.drop(columns=['main_label', 'year'])
y_test = test['main_label']

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 4. Hybrydowy downsampling
print("Downsampling zbioru treningowego...")
train_concat = pd.concat([X_train, y_train], axis=1)
bankrupts = train_concat[train_concat['main_label'] == 1]
healthy = train_concat[train_concat['main_label'] == 0]

healthy_sampled = healthy.sample(n=len(bankrupts) * 10, random_state=42)
train_balanced = pd.concat([bankrupts, healthy_sampled]).sample(frac=1, random_state=42)

X_train_bal = train_balanced.drop(columns=['main_label'])
y_train_bal = train_balanced['main_label']
print(f"Po balansie Train: {X_train_bal.shape}")

# 5. Ustalanie TOP cech
print("Szukanie najwazniejszych cech (Feature Selection)...")
xgb_base = xgb.XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.05, 
    subsample=0.6, colsample_bytree=0.8, random_state=42, 
    eval_metric='logloss', use_label_encoder=False
)

xgb_base.fit(X_train_bal, y_train_bal)
importances = pd.Series(xgb_base.feature_importances_, index=X_train_bal.columns)
top_features_sorted = importances.sort_values(ascending=False).index.tolist()

if 'horizon' in top_features_sorted:
    top_features_sorted.remove('horizon')

best_subset_size = 15
final_features = top_features_sorted[:best_subset_size] + ['horizon']
print(f"Ostatecznie wybrano {len(final_features)} najwazniejszych cech (w tym 'horizon').")

# 6. Trenowanie ostatecznego modelu + Kalibracja
print("Trenowanie docelowego, uniwersalnego modelu...")
xgb_final = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.6, colsample_bytree=0.8, random_state=42, eval_metric='logloss')

print("Kalibracja izotoniczna...")
calibrated_model = CalibratedClassifierCV(xgb_final, method='isotonic', cv=3)
calibrated_model.fit(X_train_bal[final_features], y_train_bal)

auc_final = roc_auc_score(y_test, calibrated_model.predict_proba(X_test[final_features])[:, 1])
print(f"Ostateczne AUC na teście: {auc_final:.4f}")

# 7. Zapis do plików pod aplikację Streamlit
joblib.dump(calibrated_model, 'universal_model_calibrated.pkl')
with open('app_features.json', 'w') as f:
    json.dump(final_features, f)

print("SUKCES! Zapisano 'universal_model_calibrated.pkl' oraz 'app_features.json'.")
