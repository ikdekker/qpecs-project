import pandas as pd
import numpy as np
import scipy.stats as stats
import researchpy as rp
import statsmodels.api as sm
import json
from statsmodels.formula.api import ols

import matplotlib.pyplot as plt
from pyDOE2 import *


# parameter definition
#cores = [4,8]
#learning_rate = [0.001,0.002]
#learning_decay = [1,2]
#memory = [4,8]

factors = dict({
        'ram': [1, 8],
        'cores': [1, 4],
        'batch_size': [64, 512],
        'learning_rate': [0.001, 0.1],
        'learning_rate_decay': [0.0001, 0.001]
    })



sign_table_binary = ff2n(len(factors))
sign_table = [[list(factors.values())[i][0] if x == -1 else list(factors.values())[i][1] for i,x in enumerate(sign_row)] for sign_row in sign_table_binary]

with open('data.json') as f:
  data_json = json.load(f)



def json_parse_to_dict(data, model):
  """
  creates dictionaries, with support for replication, based on a string
  with config settings and a response time. Only supports one model.
  """
  json_normalized = np.r_[list(data[model].values())].flatten()
  data = {}
  # Extract response times and add them to a list in dictionary based on their configs
  for config_string in json_normalized:
    setting = config_string.split(",")
    setting_info = {}
    for s in setting:
      setting_info[s.split("-")[1].strip()] = s.split("-")[0].strip()
#            "4-Cores, 8-GBRAM, 512-batchSize, 0.1-learningRate, 0.0001-learningRateDecay, succeeded-status, 0.479600012302-accuracy, 60.41239890500037-responsetime",
    data_key = "{}|{}|{}|{}|{}".format(setting_info['GBRAM'],setting_info['Cores'],setting_info['batchSize'],setting_info['learningRate'],setting_info['learningRateDecay'])
    if not data_key in data:
      data[data_key] = []
    data[data_key].append(setting_info['responsetime'])

  return data
data = json_parse_to_dict(data_json,'bi-rnn')

data_names = {}

observations = len(np.r_[list(data_json['bi-rnn'].values())].flatten())
# the amount of replications that have been done (extracted from factors / observations)
reps = int(observations/2**len(factors))


# Create a sign table and fill it with the configuations that we used (found in the factors dict)
for sign_row in sign_table:
  for i,x in enumerate(sign_row):
    factor = list(factors.keys())[i]
    if not factor in data_names:
      data_names[factor] = []
    data_names[factor].append(x)

#print(data_names)

response_times = []


#print(data)
# Build the response_times array in the order the anova analyser expects
for r in range(0, reps):
  for config in sign_table:
    conf_str = "|".join(list(map(str, config)))
    if conf_str in data and len(data[conf_str]) >= (r+1):
      response_times.append(float(data[conf_str][r]))
    else:
      response_times.append(8888888) #maybe use average for missing cases?


# Final data
data_names_repeated = {}
for a, (b,c) in enumerate(data_names.items()):
  data_names_repeated[b] = np.tile(c,reps)

d = {'Response': response_times}
d2 = {**data_names_repeated, **d}


df = pd.DataFrame(data=d2)
# Gettin summary statistics
# todo: get better outputs, based on theory
#print(rp.summary_cont(df.groupby( list(factors.keys()) ))['Response'])
#print(rp.summary_cont(df.groupby( 'ram' ))['Response'])

model = ols('Response ~ C(ram)*C(cores)*C(batch_size)*C(learning_rate)*C(learning_rate_decay)', df).fit()

# Seeing if the overall model is significant
print(f"Overall model F({model.df_model: .0f},{model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4f}")
print(model.summary())
print("==============================")
print(sm.stats.anova_lm(model, typ= 2))
