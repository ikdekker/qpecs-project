import pandas as pd
import numpy as np
import scipy.stats as stats
import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols

import matplotlib.pyplot as plt
from pyDOE2 import *


# parameter definition
cores = [4,8]
learning_rate = [0.001,0.002]
learning_decay = [1,2]
memory = [4,8]

factors = dict({
        'ram': [1, 8],
        'cores': [1, 4],
        'batch_size': [64, 512],
        'learning_rate': [0.001, 0.1],
        'learning_rate_decay': [0.0001, 0.001]
    })



sign_table_binary = ff2n(len(factors))
# replace -1 with 0 to look up actual values in factors dict
sign_table = [[list(factors.values())[i][0] if x == -1 else list(factors.values())[i][1] for i,x in enumerate(sign_row)] for sign_row in sign_table_binary]
#sign_table = [list(map(lambda (i,x): factors[i][0] if x == -1 else factors[i][1], sign_row)) for sign_row in sign_table]

# response time should always be a factor of params
# in our 2^k factorial (with n factors): 2^n
# we have 5 now, so 2^5=32

import json

with open('data.json') as f:
  data = json.load(f)

def json_parse_to_dict(data, model):

  json_normalized = np.r_[list(data[model].values())].flatten()
  data = {}

  # first do a loop to average out the replications
#  replication_values = {}
#  for config_string in json_normalized:
#    setting = config_string.split(",")
#    setting_info = {}
#    for s in setting:
#      setting_info[s.split("-")[1].strip()] = s.split("-")[0].strip()
#    data_key = "{}|{}|{}|{}|{}".format(setting_info['GBRAM'],setting_info['Cores'],setting_info['batchSize'],setting_info['learningRate'],setting_info['learningRateDecay'])
#    if not data_key in replication_values:
#      replication_values[data_key] = []
#    replication_values[data_key].append(setting_info['responsetime'])

#  averages = {}
#  for k, v in replication_values.items():
#    averages[k] = np.average(list(map(float, v)))

  for config_string in json_normalized:
    setting = config_string.split(",")
    setting_info = {}
    for s in setting:
      setting_info[s.split("-")[1].strip()] = s.split("-")[0].strip()
    data_key = "{}|{}|{}|{}|{}".format(setting_info['GBRAM'],setting_info['Cores'],setting_info['batchSize'],setting_info['learningRate'],setting_info['learningRateDecay'])
    if not data_key in data:
      data[data_key] = []
    data[data_key].append(setting_info['responsetime'])

  return data
data = json_parse_to_dict(data,'bi-rnn')


data_names = {}
for sign_row in sign_table:
  for i,x in enumerate(sign_row):
    factor = list(factors.keys())[i]
    if not factor in data_names:
      data_names[factor] = []
    data_names[factor].append(x)

#print(data_names)

response_times = []

reps = 1

for r in range(0, reps):
  for config in sign_table:
    conf_str = "|".join(list(map(str, config)))
    if conf_str in data and len(data[conf_str]) >= (r+1):
      response_times.append(float(data[conf_str][r]))
    else:
      response_times.append(8888888) #maybe use average for missing cases?

print(response_times)
#response_times = range(1,33) # needs to be grabbed from a loop over factors that look up in the json data

d = {'Response': response_times}
d2 = {**data_names, **d}

df = pd.DataFrame(data=d2)
# Gettin summary statistics
print(rp.summary_cont(df.groupby( list(factors.keys()) ))['Response'])
#print(rp.summary_cont(df.groupby( 'ram' ))['Response'])
