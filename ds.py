
import pandas as pd

df = pd.read_csv("quality_dataset.csv")

print(df["quality"].value_counts(dropna=False))