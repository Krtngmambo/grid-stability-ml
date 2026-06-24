import pandas as pd, joblib, os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print(">> Membaca data...", flush=True)
os.makedirs("model", exist_ok=True)

df = pd.read_csv("data/grid_stability.csv").drop(columns=["stab"])
X = df.drop(columns=["stabf"])
y = (df["stabf"] == "stable").astype(int)
print(">> Data OK:", X.shape, flush=True)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

print(">> Melatih Baseline (LogReg)...", flush=True)
baseline = LogisticRegression(max_iter=1000).fit(X_tr_s, y_tr)

print(">> Melatih Random Forest (tunggu ~1 menit, JANGAN Ctrl+C)...", flush=True)
rf = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    {"n_estimators": [200], "max_depth": [None, 20]},
    cv=3, scoring="f1", verbose=2).fit(X_tr_s, y_tr)

print(">> Melatih Neural Network (MLP)...", flush=True)
mlp = GridSearchCV(
    MLPClassifier(max_iter=300, random_state=42),
    {"hidden_layer_sizes": [(64, 32)], "alpha": [1e-4, 1e-3]},
    cv=3, scoring="f1", verbose=2).fit(X_tr_s, y_tr)

for name, m in [("Baseline (LogReg)", baseline),
                ("Random Forest", rf),
                ("Neural Net (MLP)", mlp)]:
    pred = m.predict(X_te_s)
    print("\n==============", name, "==============")
    print(classification_report(y_te, pred))
    print("Confusion matrix:\n", confusion_matrix(y_te, pred))
    print("ROC-AUC:", round(roc_auc_score(y_te, m.predict_proba(X_te_s)[:, 1]), 4))

joblib.dump(rf.best_estimator_, "model/grid_stability_rf.joblib")
joblib.dump(scaler, "model/scaler.joblib")
print("\n✅ Model & scaler tersimpan di folder model/", flush=True)