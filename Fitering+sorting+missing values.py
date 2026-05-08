import seaborn as sns

df = sns.load_dataset("titanic")

print("Missing Values:")
print(df.isnull().sum())

df = df.fillna(df.mean(numeric_only=True))

filtered = df[df["age"] > df["age"].mean()]
print("Filtered Data:")
print(filtered.head())

sorted_df = df.sort_values("age", ascending=False)
print("Sorted Data:")
print(sorted_df.head())