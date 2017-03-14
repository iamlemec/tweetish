# tweet model

import tensorflow as tf
import numpy as np

import config
import model

dtype = tf.float32

class Model(object):
    def __init__(self):
        batch_size = config.batch_size
        num_steps = config.num_steps
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size

        # create data placeholders
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps], name='input')
        self.output_data = tf.placeholder(tf.int32, [batch_size, num_steps], name='output')
        self.weights = tf.placeholder(tf.float32, [batch_size, num_steps], name='weights')

        # create dummy feed vars for non-training
        # self.dummy_feed = {
            # self.input_data: np.zeros([config.batch_size, config.num_steps], dtype=np.int32),
            # self.output_data: np.zeros([config.batch_size, config.num_steps], dtype=np.int32),
            # self.weights: np.ones([config.batch_size, config.num_steps], dtype=np.int32)
        # }

        # dynamic meta parameters
        self.kp = tf.Variable(1.0, trainable=False)
        self.lr = tf.Variable(0.0, trainable=False)

        # construct embedding to reduce vocabulary dimension
        embedding = tf.get_variable("embedding", [vocab_size, hidden_size], dtype=dtype)
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        inputs = tf.nn.dropout(inputs, keep_prob=self.kp)

        # construct basic LSTM cell that is building block for RNN
        def lstm_cell():
            lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
            return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.kp)
        layers = [lstm_cell() for _ in range(config.num_layers)]
        cell = tf.contrib.rnn.MultiRNNCell(layers, state_is_tuple=True)

        # unroll RNN to appropriate hidden_size
        outputs = []
        state = cell.zero_state(batch_size, dtype)
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        # construct decoder to match sequence data (or predict)
        output = tf.reshape(tf.stack(outputs, 1), [-1, hidden_size])
        softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size], dtype=dtype)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=dtype)
        logits0 = tf.reshape(tf.matmul(output, softmax_w) + softmax_b,
            [batch_size, num_steps, vocab_size])
        self.logits = logits = tf.nn.dropout(logits0, keep_prob=self.kp)

        # calculate loss (inaccuracy of predictions)
        loss = tf.contrib.seq2seq.sequence_loss(logits, self.output_data, self.weights)
        self.cost = cost = tf.reduce_sum(loss) / batch_size

        # gradient descent voodoo
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step()
        )

        # updating meta-parameters
        self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self.new_kp = tf.placeholder(tf.float32, shape=[], name="new_keep_prob")

        self.lr_update = tf.assign(self.lr, self.new_lr)
        self.kp_update = tf.assign(self.kp, self.new_kp)

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})

    def assign_kp(self, session, kp_value):
        session.run(self.kp_update, feed_dict={self.new_kp: kp_value})

