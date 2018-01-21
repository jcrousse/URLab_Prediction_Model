import pickle
import numpy as np
import pandas as pd
from nn_graphs import nn_2_layers
from nn_model import NNModelTTS, NNModelKF

data = pickle.load(open("data.p", "rb"))
data_dict = pickle.load(open("data_dict.p", "rb"))

# NN training parameters. We use default logging, dropout prob and CV split
epochs = 1000
learning_rate = 1e-3
layers = [10, 10]  # Number of units per layer
model_name = 'model_180121'

# prepare dataset. Y= target, X = features only.
m = data.shape[0]

Y = data[data_dict['target']].values.reshape(m, 1)
X = data[data_dict['features']].values


# myModel = NNModel(nn_2_layers)
# myModel = myModel.train(X, Y, epochs=epochs, model_id='model_1', layers=layers, learning_rate=learning_rate)
myModel = NNModelTTS(graph_creator=nn_2_layers)
myModel = myModel\
    .set_hyperparameters(n_features=X.shape[1],  epochs=epochs, model_id=model_name,
                         layers=layers, learning_rate=learning_rate)\
    .train(X, Y)\

predictions = myModel.predict(X)
indices = myModel.get_train_test_indices()

# re-organise data for d3 visualisaion

predictions.columns = ['binary_pred', 'logit_pred']
indices.columns = ['train_set_flg']

predictions['date'] = data.index
indices['date'] = data.index
predictions = predictions.set_index('date')
indices = indices.set_index('date')

result_df = pd.concat([data, predictions, indices], axis=1)

print(result_df)