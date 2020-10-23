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
# for model_name in data:
replications = data['lenet5']
for i in range(3):
    for experiment in replications[""+str(i)]:
        parameters = experiment.split(',')
        row = ""
        for parameter in parameters:
            row += parameter.split('-')[0].strip()+","
        row = 'lenet5'+","+row
        csv_values += row.strip()

data = np.array(csv_values.split(','))
data = data[:len(data)-1]

data = data.reshape(96, len(csv_columns))

df = pd.DataFrame(data, columns=csv_columns)

df['accuracy'] = pd.to_numeric(df['accuracy'])
df['responsetime'] = pd.to_numeric(df['responsetime'])
df['Cores'] = pd.to_numeric(df['Cores'])
df['learningRate'] = pd.to_numeric(df['learningRate'])

ax = sns.boxplot(data=df, y='responsetime',x='learningRate')
ax.set(ylabel='accuracy (%)')
ax.set(xlabel='Learning rate')
plt.subplots_adjust(bottom=0.15) 
plt.subplots_adjust(left=0.15) 

plt.show()
exit()
