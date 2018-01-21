import pickle
import numpy as np
import pandas as pd
from nn_graphs import nn_2_layers
from nn_model import NNModelTTS, NNModelKF

data = pickle.load(open("data.p", "rb"))
data_dict = pickle.load(open("data_dict.p", "rb"))

# NN training parameters. We use default logging, dropout prob and CV split
epochs = 10
learning_rate = 1e-3
layers = [10, 10]  # Number of units per layer

# prepare dataset. Y= target, X = features only.
m = data.shape[0]

Y = data[data_dict['target']].values.reshape(m, 1)
X = data[data_dict['features']].values


# myModel = NNModel(nn_2_layers)
#myModel = myModel.train(X, Y, epochs=epochs, model_id='model_1', layers=layers, learning_rate=learning_rate)
myModel = NNModelTTS(graph_creator=nn_2_layers)
myModel = myModel.set_hyperparameters(n_features=X.shape[1],  epochs=epochs, model_id='model_1', layers=layers, learning_rate=learning_rate).train(X,Y)
y = myModel.predict(X, model_id='model_1')

prediction_binary = pd.DataFrame((np.round_(1/(1+np.exp(-y)))).astype(int), index=data.index, columns=["prediction"])

result_df = pd.concat([data, prediction_binary], axis=1)

print(result_df)