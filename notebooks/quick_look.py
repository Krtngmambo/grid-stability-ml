import pandas as pd

df = pd.read_csv("data/grid_stability.csv")
print("Jumlah baris & kolom:", df.shape)
print("\n5 baris pertama:")
print(df.head())
print("\nJumlah tiap kelas:")
print(df["stabf"].value_counts())
print("\nAda data kosong?", df.isnull().sum().sum(), "sel kosong")