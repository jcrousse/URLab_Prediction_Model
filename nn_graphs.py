import tensorflow as tf


def nn_2_layers(graph, n_feat, layers, learning_rate):
    with graph.as_default():
        x = tf.placeholder(tf.float32, shape=[None, n_feat])
        y_ = tf.placeholder(tf.float32, shape=[None, 1])
        keep_prob = tf.placeholder(tf.float32)

        # First Layer
        W1 = tf.Variable(tf.truncated_normal([n_feat, layers[0]], stddev=0.1))
        b1 = tf.Variable(tf.constant(0.1, shape=[layers[0]]))

        h1 = tf.nn.relu(tf.matmul(x,W1) + b1)

        # Second Layer
        W2 = tf.Variable(tf.truncated_normal([layers[0], layers[1]], stddev=0.1))
        b2 = tf.Variable(tf.constant(0.1, shape=[layers[1]]))

        h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

        # Dropout
        h2_drop = tf.nn.dropout(h2, keep_prob)

        # Output Layer
        W3 = tf.Variable(tf.truncated_normal([layers[1], 1], stddev=0.1))
        b3 = tf.Variable(tf.constant(0.1, shape=[1]))

        y_logits = tf.matmul(h2_drop,W3) + b3

        # Loss function and optimizer
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_, logits=y_logits)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return x, y_, loss, train_step, keep_prob, y_logits
