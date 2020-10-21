import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
# Stdout options
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)

sns.set(
    context="notebook", palette="Spectral",
    style='darkgrid', font_scale=1.5, color_codes=True
)

###################
#  Load the JSON  #
###################
with open('ndata.json') as json_file:
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
    replications = data[model_name]
    for i in range(3):
        for experiment in replications[""+str(i)]:
            parameters = experiment.split(',')
            row = ""
            for parameter in parameters:
                row += parameter.split('-')[0].strip()+","
            row = model_name+","+row
            csv_values += row.strip()

# Process, reshape and turn into a dataframe
data = np.array(csv_values.split(','))
data = data[:len(data)-1]
data = data.reshape(192, len(csv_columns))
df = pd.DataFrame(data, columns=csv_columns)

# Convert these 2 columns to a numeric value
df['accuracy'] = pd.to_numeric(df['accuracy'])
df['responsetime'] = pd.to_numeric(df['responsetime'])

# Transform succeeded to normal (Nick dataset)
df['status'] = df['responsetime'].apply(
    lambda x: "normal" if x < 250.0 else "slow"
)

# Transform succeeded to normal (Mostafa dataset)
# df['status'] = df['responsetime'].apply(
#     lambda x: "normal" if x < 60.0 else "slow"
# )

###################################################################
# Encode the non continuous/discrete columns to in integer number #
# and transform integer-string values to integer                  #
###################################################################
df.to_csv('data_n.csv', float_format='%.6f', index=False)
print(df)

categories_status = ['normal', 'slow']
le = preprocessing.LabelEncoder()
le.fit(categories_status)
arr = le.transform(categories_status)
df['status'] = df['status'].apply((lambda x: le.transform([x])[0]))

categories_model = ['lenet5', 'bi-rnn']
le = preprocessing.LabelEncoder()
le.fit(categories_model)
arr = le.transform(categories_model)
df['model'] = df['model'].apply((lambda x: le.transform([x])[0]))

#############
#  Group By #
#############
# Group by these columns and calculate the the mean of
# the accuracy and the responseTime
# df = df.groupby([
#     'model', 'Cores', 'GBRAM', 'batchSize',
#     'learningRate', 'learningRateDecay'
# ], as_index=False).mean()

# df.to_csv('data_summed.csv', float_format='%.6f', )

######################
#  Start of K-Means  #
######################
# Take the last accuracy and response time column
X = df.iloc[:, [7, 8]].values

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

n_clusters = 2
km = KMeans(init='k-means++', n_clusters=n_clusters)
km_clustering = km.fit(X)
sns.pairplot(
    df.assign(hue=km_clustering.labels_), hue='hue', palette=["C0", "C1"]
)
plt.show()

# TODO
# plot scatter plot of GBRAM to Accuracy
# plot scatter plot of GBRAM respone
