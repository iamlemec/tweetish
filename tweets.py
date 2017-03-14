# tweet analyzer

import os
import time

import numpy as np
import tensorflow as tf

import config
import data
import model

flags = tf.flags
flags.DEFINE_string("data_path", None, "Where the training/test data is stored.")
flags.DEFINE_string("load_path", None, "Model input directory.")
flags.DEFINE_string("save_path", None, "Model output directory.")
flags.DEFINE_string("train", False, "Train model first.")
flags.DEFINE_string("simul", True, "Simulate a batch of tweets.")
FLAGS = flags.FLAGS

def run_epoch(session, model, feed, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0

    fetches = { "cost": model.cost }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(feed.epoch_size):
        inp, out, wgt = feed.get_batch()
        feed_dict = {model.input_data: inp, model.output_data: out, model.weights: wgt}
        values = session.run(fetches, feed_dict=feed_dict)
        cost = values['cost']

        costs += cost
        iters += config.num_steps

        if verbose and step % (feed.epoch_size // 5) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" % (
                step * 1.0 / feed.epoch_size,
                np.exp(costs / iters),
                iters * config.batch_size / (time.time() - start_time)
            ))

    return np.exp(costs / iters)

def sample(logs, temp=1.0):
    elogs = np.exp((logs-np.max(logs, axis=-1, keepdims=True))/temp)
    probs = elogs/np.sum(elogs, axis=-1, keepdims=True)
    probs /= 1.00001*np.sum(probs, axis=-1, keepdims=True)
    return [np.argmax(np.random.multinomial(1, pr, 1)) for pr in probs]

def simulate_batch(session, model, feed, temp=0.5):
    pids = [feed.vocab[p] for p in data.PRIM]
    prime = np.random.choice(pids, replace=True, size=config.batch_size)
    batch = np.array([([p]+(config.num_steps-1)*[0]) for p in prime], dtype=np.int32)

    dummy_output = np.zeros([config.batch_size, config.num_steps], dtype=np.int32)
    dummy_weights = np.ones([config.batch_size, config.num_steps], dtype=np.int32)

    feed_dict = {
        model.input_data: batch,
        model.output_data: dummy_output,
        model.weights: dummy_weights
    }

    for t in range(1, config.num_steps):
        logits = session.run(model.logits, feed_dict=feed_dict)
        logits = logits.reshape([config.batch_size, config.num_steps, config.vocab_size])
        # print(np.argmax(logits[:,t,:], axis=-1))
        batch[:,t] = sample(logits[:,t,:], temp=temp)

    iend = lambda row: np.nonzero(row==1)[0][0] if 1 in row else None
    trim = [row[:iend(row)] for row in batch]
    return [' '.join([feed.ivocab[i] for i in row]) for row in trim]

def main(_):
    d = data.Feeder(data_path=FLAGS.data_path)

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = model.Model()

            if FLAGS.train:
                tf.summary.scalar("Training Loss", m.cost)
                tf.summary.scalar("Learning Rate", m.lr)

        sv = tf.train.Supervisor()
        with sv.managed_session() as session:
            if FLAGS.load_path:
                ckpt = tf.train.latest_checkpoint(FLAGS.load_path)
                if ckpt is not None:
                    print("Loading saved parameters")
                    sv.saver.restore(session, ckpt)
                else:
                    print("Initializing fresh parameters")

            if FLAGS.train:
                m.assign_kp(session, config.keep_prob)
                for i in range(config.max_epoch):
                    lr_decay = config.lr_decay ** max(i + 1 - config.start_decay, 0.0)
                    m.assign_lr(session, config.learning_rate * lr_decay)
                    print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))

                    train_perplexity = run_epoch(session, m, d, eval_op=m.train_op, verbose=True)
                    print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                m.assign_kp(session, 1.0)

                if FLAGS.save_path:
                    print("Saving model to %s." % FLAGS.save_path)
                    sv.saver.save(session, os.path.join(FLAGS.save_path, 'model'), global_step=sv.global_step)

            # simulate some draws
            if FLAGS.simul:
                sim = simulate_batch(session, m, d, temp=0.5)
                print('\n'.join(sim))

if __name__ == "__main__":
    tf.app.run()

