import tensorflow as tf
import time
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

#todo: Split "set hyperparameters" and "train"
#todo figure out how to return train/test indices
class NNModel:
    def __init__(self, **kwargs):  # graph is a function to generate a NN graph
        self.graph_creator = kwargs.get('graph_creator')
        self.graph = tf.Graph()

        # TF graph elements, to be initialised by the graph_creator function:
        self.x = self.y_ = self.loss = self.train_step = self.keep_prob = self.y_logits = None
        # Hyperparameters
        self.model_id = self.epochs = self.logging = self.dropout_keep = None

    def set_hyperparameters(self, **kwargs):
        if 'model_id' not in  kwargs:
            raise TypeError("missing argument model_id in function set_hyperparameters")
        if 'layers' not in  kwargs:
            raise TypeError("missing argument layers in function set_hyperparameters")
        if 'n_features' not in  kwargs:
            raise TypeError("missing argument n_features in function set_hyperparameters")

        learning_rate = kwargs.get('learning_rate', 1e-3)
        self.x, self.y_, self.loss, self.train_step, self.keep_prob, self.y_logits \
            = self.graph_creator(self.graph,kwargs.get('n_features'),  kwargs.get('layers'), learning_rate)

        self.model_id = 'model_id'
        self.epochs = kwargs.get('epochs', 1000)
        self.logging = kwargs.get('logging', 100)
        self.dropout_keep = kwargs.get('dropout_keep', 0.5)

        return self

    def train(self, x, y):

        train_set, test_set, train_label, test_label = train_test_split(x, y, test_size=self.test_size)
        self.training(train_set, test_set, train_label, test_label)
        print('Training finished')

        return self

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

    def predict(self, x, model_id):
        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.generate_model_path())
            return sess.run(self.y_logits, feed_dict={self.x: x, self.keep_prob: 1.0})


class NNModelKF(NNModel):
    def __init__(self, **kwargs):
        super(NNModelKF, self).__init__(**kwargs)
        self.id_kf = 0  # id for currently active fold
        self.n_splits_cv = 5

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
        return 'model/model{0}_{1}.ckpt'.format(self.model_id, self.id_kf) # now contains fold ID as well


class NNModelTTS(NNModel):

    def __init__(self, **kwargs):
        super(NNModelTTS, self).__init__(**kwargs)
        self.test_size = 0.2

    def set_hyperparameters(self, **kwargs):
        self.test_size = kwargs.get('test_size', 0.2)
        super(NNModelTTS, self).set_hyperparameters(**kwargs)
        return self
