import seaborn as sns
from sklearn.decomposition import FactorAnalysis, PCA

df = sns.load_dataset("iris")

X = df.drop("species", axis=1)

fa = FactorAnalysis(n_components=2)
fa.fit(X)

print("Factor Components:")
print(fa.components_)

p = PCA(n_components=2)
p.fit(X)

print("PCA Explained Variance Ratio:")
print(p.explained_variance_ratio_)