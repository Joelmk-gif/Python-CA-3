import seaborn as sns

df = sns.load_dataset("tips")

print("Correlation Matrix:")
print(df.corr(numeric_only=True))

avg = df["total_bill"].mean()
count = 0

for v in df["total_bill"]:
    if v > avg:
        count += 1

print("Values above average:", count)