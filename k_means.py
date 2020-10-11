import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans

sns.set(
    context="notebook", palette="Spectral",
    style='darkgrid', font_scale=1.5, color_codes=True
)

###################
#  Load the JSON  #
###################
with open('data.json') as json_file:
    data = json.load(json_file)

#########################
#  Process & Transform  #
#########################

csv_columns = [
    "model", "Cores", "GBRAM", "batchSize", "learningRate",
    "learningRateDecay", "status", "accuracy", "responsetime"
]

csv_values = ""
for model_name in data:
    # split by comma
    replications = data[model_name]
    for experiment in replications["0"]:  # TODO add other replications
        parameters = experiment.split(',')
        row = ""
        for parameter in parameters:
            row += parameter.split('-')[0].strip()+","
        row = model_name+","+row
        csv_values += row.strip()

# Process, reshape and turn into a dataframe
data = np.array(csv_values.split(','))
data = data[:len(data)-1]
data = data.reshape(64, len(csv_columns))
df = pd.DataFrame(data, columns=csv_columns)

# Encode the non continuous/discrete columns to in integer number
categories = ['succeeded', 'failed']
le = preprocessing.LabelEncoder()
le.fit(categories)
arr = le.transform(['succeeded', 'failed'])
df['status'] = df['status'].apply((lambda x: le.transform([x])[0]))
print(df.head(10))
print(df.describe())

######################
#  Start of K-Means  #
######################
X = df.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8]].values

# Using the elbow method to find the optimal number of clusters.
# WCSS (Within Cluster Sum of Squares), WCSS is defined as the
# sum of the squared distance between each member of
# the cluster and its centroid.
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    # inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
sns.lineplot(range(1, 11), wcss, marker='o', color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

plt.figure(figsize=(15, 7))
sns.scatterplot(
    X[y_kmeans == 0, 0], X[y_kmeans == 0, 1],
    color='yellow', label='Cluster 1', s=50
)

sns.scatterplot(
    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
    color='red', label='Centroids', s=300, marker=','
)

plt.grid(False)
plt.title('Clusters of Jobs')
plt.xlabel('Number of Cores')
plt.ylabel('Responese Time')
plt.legend()
plt.show()
