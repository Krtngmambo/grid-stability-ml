import joblib

m = joblib.load("model/grid_stability_rf.joblib")
joblib.dump(m, "model/grid_stability_rf.joblib", compress=3)
print("Selesai dikompres.")