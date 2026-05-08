import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

df = sns.load_dataset("iris")

# Decision Tree
X = df.drop("species", axis=1)
y = df["species"]

dt = DecisionTreeClassifier().fit(X, y)
print("Decision Tree Accuracy:", dt.score(X, y))

# K-Means
data = df[["sepal_length", "sepal_width"]]

k = KMeans(n_clusters=3, n_init=10).fit(data)
print("Cluster Centers:")
print(k.cluster_centers_)