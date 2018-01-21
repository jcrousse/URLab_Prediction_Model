import tensorflow as tf
import time
import numpy as np
from sklearn.model_selection import KFold

# TODO: inherited class for K_fold from default train/test class
# by default prediction on KF model retrieves the first model trained. Can be changed with function KF_select model

class nn_model:
    def __init__(self, graph_creator):  # graph is a function to generate a NN graph
        self.graph_creator = graph_creator
        self.graph = tf.Graph()

        # TF graph elements, to be initialised by the graph_creator function:
        self.x = self.y_ = self.loss = self.train_step = self.keep_prob = self.y_logits = None

        self.id_kf = 0 # for KF_only

    def train(self, x, y, model_id, n_splits_cv=5, epochs=1000, logging=100, dropout_keep=0.5):

        # We use K-fold cross validation to experiment and get a feel of the variance around predictive power
        kf = KFold(n_splits=n_splits_cv, shuffle=True, random_state=1)
        kf.get_n_splits(x)
        X = np.array(x)
        Y = np.array(y)

        self.x, self.y_, self.loss, self.train_step, self.keep_prob, self.y_logits \
            = self.graph_creator(self.graph, X.shape[1], [10, 10], 0.1)

        for id_kf, (train_index, test_index) in enumerate(kf.split(X)):
            train_set, test_set = X[train_index], X[test_index]
            train_label, test_label = Y[train_index], Y[test_index]
            self.id_kf = id_kf
            self.training(train_set, test_set, train_label, test_label, logging, model_id, epochs, dropout_keep)

        return self

    def training(self,  train_set, test_set, train_label, test_label, logging, model_id, epochs, dropout_keep):
        err_log = []
        with tf.Session(graph=self.graph) as sess:

            # Start Training
            sess.run(tf.global_variables_initializer())
            start = time.time()
            print('Starting Training...')
            for i in range(epochs):

                if i % logging == 0:
                    err = self.loss.eval(feed_dict={self.x: train_set, self.y_: train_label, self.keep_prob: 1.0})
                    test_err = self.loss.eval(feed_dict={self.x: test_set, self.y_: test_label, self.keep_prob: 1.0})
                    print('Step {0}, Train Error: {1: .2f} | Test Error: {2: .2f}'.format(i, err, test_err))

                self.train_step.run(feed_dict={self.x: train_set,
                                               self.y_: train_label, self.keep_prob: dropout_keep})
            err = self.loss.eval(feed_dict={self.x: train_set, self.y_: train_label, self.keep_prob: 1.0})
            test_err = self.loss.eval(feed_dict={self.x: test_set, self.y_: test_label, self.keep_prob: 1.0})

            print('\nTraining Finished, Training Error: {0: .2f} '.format(err))
            print('Validation Error: {}'.format(test_err))
            end = time.time()
            print("Elapsed time :{}\n".format(end - start))
            err_log.append(test_err)

            saver = tf.train.Saver()
            saver.save(sess, self.generate_model_path(model_id))

        final_err = np.array(err_log).mean()
        print('K folds finished')
        print('Final validation score: {}'.format(final_err))

    def generate_model_path(self, model_id):
        return 'model/model{0}_{1}.ckpt'.format(model_id, self.id_kf) # to be modified for CV version

    def select_predict_kf(self, id_kf):
        self.id_kf = id_kf

    def predict(self, x, model_id):
        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.generate_model_path(model_id))
            return sess.run(self.y_logits, feed_dict={self.x: x, self.keep_prob: 1.0})


