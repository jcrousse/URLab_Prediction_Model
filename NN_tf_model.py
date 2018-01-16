import pickle
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
import time
import pandas as pd
from nn_graphs import nn_2_layers
from nn_model import nn_model

data = pickle.load(open("data.p", "rb"))

# NN training parameters
epochs = 10
learning_rate = 1e-3
logging = 100
dropout_prop = 0.5
n_splits_cv = 5
LAYERS = [10, 10] # Number of units per layer

# prepare dataset. Y= target, X = features only.
m = data.shape[0]

Y = data['Open_flg_pct'].values.reshape(m, 1)

X = data.reset_index()  # Integer index rather than dates for tensorflow
X = X.drop(['index', 'is_open', 'Date', 'minutes_to_next_event',
            'Open_flg_pct', 'day', 'month', 'open_pct', 'weekday'], axis=1)


mymodel = nn_model(nn_2_layers)
mymodel.train(X, Y, 2)

y = mymodel.predict(X,Y)
pred_df =pd.DataFrame((np.round_(1/(1+np.exp(-y)))).astype(int))
pred_df.columns=["prediction"]
df1 = pred_df
df2 = data.copy()
df2['obs_date'] = df2.index
df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)

result_df = pd.concat( [df1, df2], axis=1)

print(pred_df)