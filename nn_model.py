import tensorflow as tf
import time
import numpy as np
from sklearn.model_selection import KFold

# TODO: clean up the function calls and large umber of arguments.
# TODO: inherited class for K_fold from default train/test class
# TODO: clean up those WTFs below

class nn_model:
    def __init__(self, graph_creator):  # graph is a function to generate a NN graph
        self.graph_creator = graph_creator
        self.graph = tf.Graph()


    def train(self, X, Y, n_splits_cv):  #use dict instead of many arguments?
        epochs = 10
        logging = 100
        dropout_prop = 0.5

        # We use K-fold cross validation to experiment and get a feel of the variance around predictive power
        kf = KFold(n_splits=n_splits_cv, shuffle=True, random_state=1)
        kf.get_n_splits(X)
        X = np.array(X)
        Y = np.array(Y)


        self.x, self.y_, self.loss, self.train_step, self.keep_prob, self.y_logits = self.graph_creator(self.graph, X.shape[1], [10,10], 0.1)

        err_log = []
        with tf.Session(graph=self.graph) as sess:
            for train_index, test_index in kf.split(X):
                train_set, test_set = X[train_index], X[test_index]
                train_label, test_label = Y[train_index], Y[test_index]

                n_feat = train_set.shape[1]

                # Start Training
                sess.run(tf.global_variables_initializer())
                start = time.time()
                print('Starting Training...')
                for i in range(epochs):

                    if i % logging == 0:
                        err = self.loss.eval(feed_dict={self.x: train_set, self.y_: train_label, self.keep_prob: 1.0})
                        test_err = self.loss.eval(feed_dict={self.x: test_set, self.y_: test_label, self.keep_prob: 1.0})
                        print('Step {0}, Train Error: {1: .2f} | Test Error: {2: .2f}'.format(i, err, test_err))

                    self.train_step.run(feed_dict={self.x: train_set, self.y_: train_label, self.keep_prob: dropout_prop}) #WhyTF fo I have keep prob AND dropout_prob???
                err = self.loss.eval(feed_dict={self.x: train_set, self.y_: train_label, self.keep_prob: 1.0})
                test_err = self.loss.eval(feed_dict={self.x: test_set, self.y_: test_label, self.keep_prob: 1.0})

                print('\nTraining Finished, Training Error: {0: .2f} '.format(err))
                print('Validation Error: {}'.format(test_err))
                end = time.time()
                print("Elapsed time :{}\n".format(end - start))

                err_log.append(test_err)

            saver = tf.train.Saver()
            saver.save(sess, 'model/model.ckpt')

        final_err = np.array(err_log).mean()
        print('K folds finished')
        print('Final validation score: {}'.format(final_err))

    def predict(self, X, Y):
        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, 'model/model.ckpt')
            return sess.run(self.y_logits, feed_dict={self.x: X, self.y_: Y, self.keep_prob: 1.0})  # WhyTF do I have an Y for prediction ?


