import tensorflow as tf
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


class NNModel:
    def __init__(self, **kwargs):  # graph is a function to generate a NN graph
        self.graph_creator = kwargs.get('graph_creator')
        self.graph = tf.Graph()

        # TF graph elements, to be initialised by the graph_creator function:
        self.x = self.y_ = self.loss = self.train_step = self.keep_prob = self.y_logits = None
        # Hyperparameters
        self.model_id = self.epochs = self.logging = self.dropout_keep = None

    def set_hyperparameters(self, **kwargs):
        if 'model_id' not in kwargs:
            raise TypeError("missing argument model_id in function set_hyperparameters")
        if 'layers' not in kwargs:
            raise TypeError("missing argument layers in function set_hyperparameters")
        if 'n_features' not in kwargs:
            raise TypeError("missing argument n_features in function set_hyperparameters")

        learning_rate = kwargs.get('learning_rate', 1e-3)
        self.x, self.y_, self.loss, self.train_step, self.keep_prob, self.y_logits \
            = self.graph_creator(self.graph, kwargs.get('n_features'),  kwargs.get('layers'), learning_rate)

        self.model_id = 'model_id'
        self.epochs = kwargs.get('epochs', 1000)
        self.logging = kwargs.get('logging', 100)
        self.dropout_keep = kwargs.get('dropout_keep', 0.5)

        return self

    def train(self, x, y):
        pass

    def training(self,  train_set, test_set, train_label, test_label):
        err_log = []
        with tf.Session(graph=self.graph) as sess:

            # Start Training
            sess.run(tf.global_variables_initializer())
            start = time.time()
            print('Starting Training...')
            for i in range(self.epochs):

                if i % self.logging == 0:
                    err = self.loss.eval(feed_dict={self.x: train_set, self.y_: train_label, self.keep_prob: 1.0})
                    test_err = self.loss.eval(feed_dict={self.x: test_set, self.y_: test_label, self.keep_prob: 1.0})
                    print('Step {0}, Train Error: {1: .2f} | Test Error: {2: .2f}'.format(i, err, test_err))

                self.train_step.run(feed_dict={self.x: train_set,
                                               self.y_: train_label, self.keep_prob: self.dropout_keep})
            err = self.loss.eval(feed_dict={self.x: train_set, self.y_: train_label, self.keep_prob: 1.0})
            test_err = self.loss.eval(feed_dict={self.x: test_set, self.y_: test_label, self.keep_prob: 1.0})

            print('\nTraining Finished, Training Error: {0: .2f} '.format(err))
            print('Validation Error: {}'.format(test_err))
            end = time.time()
            print("Elapsed time :{}\n".format(end - start))
            err_log.append(test_err)

            saver = tf.train.Saver()
            saver.save(sess, self.generate_model_path())

        final_err = np.array(err_log).mean()
        print('Final validation score: {}'.format(final_err))

    def generate_model_path(self):
        return 'model/model{0}.ckpt'.format(self.model_id)

    def predict(self, x):
        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.generate_model_path())
            y = sess.run(self.y_logits, feed_dict={self.x: x, self.keep_prob: 1.0}).flatten()

            col_log_name = 'logit_pred_' + self.generate_model_path()
            col_bin_name = 'binary_pred_' + self.generate_model_path()

            preds = {col_log_name: y, col_bin_name: (np.round_(1 / (1 + np.exp(-y)))).astype(int)}
            return pd.DataFrame(preds)

    def get_train_test_indices(self):
        pass

    def train_test_indices_df(self, train_index, test_index):
        # return dataframe with value 1 for train index, 0 for test

        array = np.sort(np.concatenate([train_index, test_index]))
        index = pd.Index(array)
        col_name = self.generate_model_path() + '_train_flg'
        train_test_df = pd.DataFrame(0, index, columns=[col_name])
        train_index = pd.Index(train_index)
        train_test_df.loc[train_index, [col_name]] = 1
        return train_test_df


class NNModelKF(NNModel):
    def __init__(self, **kwargs):
        super(NNModelKF, self).__init__(**kwargs)
        self.id_kf = 0  # id for currently active fold
        self.n_splits_cv = 5
        # indices:
        self.train_index = [[]]
        self.test_index = [[]]

    def set_hyperparameters(self, **kwargs):
        self.n_splits_cv = kwargs.get('n_splits_cv', 5)
        super(NNModelKF, self).set_hyperparameters(**kwargs)
        return self

    def train(self, x, y):
        # We use K-fold cross validation to experiment and get a feel of the variance around predictive power
        kf = KFold(n_splits=self.n_splits_cv, shuffle=True, random_state=1)
        kf.get_n_splits(x)
        X = np.array(x)
        Y = np.array(y)

        for id_kf, (train_index, test_index) in enumerate(kf.split(X)):
            self.train_index.append(train_index)
            self.test_index.append(test_index)
            train_set, test_set = X[train_index], X[test_index]
            train_label, test_label = Y[train_index], Y[test_index]
            self.id_kf = id_kf
            self.training(train_set, test_set, train_label, test_label)

        print('K folds finished')
        return self

    # by default prediction on KF model retrieves model trained corresponding to self.id_kf value,
    #  which is the last model trained. This can be changed with the below function
    def select_predict_kf(self, id_kf):
        self.id_kf = id_kf

    def generate_model_path(self):
        return 'model/model{0}_{1}.ckpt'.format(self.model_id, self.id_kf)  # now contains fold ID as well

    def get_train_test_indices(self):
        return self.train_test_indices_df(self.train_index[self.id_kf], self.test_index[self.id_kf])


class NNModelTTS(NNModel):

    def __init__(self, **kwargs):
        super(NNModelTTS, self).__init__(**kwargs)
        self.test_size = 0.2
        # indices:
        self.train_index = self.test_index = None

    def set_hyperparameters(self, **kwargs):
        self.test_size = kwargs.get('test_size', 0.2)
        super(NNModelTTS, self).set_hyperparameters(**kwargs)

        return self

    def train(self, x, y):

        seq_index = np.arange(x.shape[0])

        train_index, test_index = train_test_split(seq_index, test_size=self.test_size)
        self.training(x[train_index], x[test_index], y[train_index], y[test_index])
        print('Training finished')

        self.train_index = train_index
        self.test_index = test_index

        return self

    def get_train_test_indices(self):
        return self.train_test_indices_df(self.train_index, self.test_index)
