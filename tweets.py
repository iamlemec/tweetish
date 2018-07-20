# tweet analyzer

import os
import time

import numpy as np
import tensorflow as tf

import config
import data
import model

def run_epoch(session, model, feed, summ_op=None, verbose=False):
    ops = {'train_op': model.train_op, 'cost': model.cost}
    if summ_op is not None:
        ops['summ_op'] = summ_op

    costs = 0.0
    for step in range(feed.n_batch):
        inp, out, wgt = feed.get_batch()
        feed_dict = {model.input_data: inp, model.output_data: out, model.weights: wgt}
        values = session.run(ops, feed_dict=feed_dict)
        costs += values['cost']

    costs = np.exp(costs/feed.n_batch)
    summ = values.get('summ_op', None)

    return summ, costs

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
        batch[:,t] = sample(logits[:,t-1,:], temp=temp)

    iend = lambda row: np.nonzero(row==1)[0][0] if 1 in row else None
    trim = [row[:iend(row)] for row in batch]
    return [' '.join([feed.ivocab[i] for i in row]) for row in trim]

def main(_):
    d = data.Feeder(data_path=FLAGS.data_path)

    with tf.Graph().as_default() as graph:
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        # construct model
        with tf.name_scope("Tweets"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = model.Model()

            # summaries for tensorboard
            if FLAGS.summ_path:
                if FLAGS.train:
                    tf.summary.scalar("Training Loss", m.cost)
                merged = tf.summary.merge_all()
                writer = tf.summary.FileWriter(FLAGS.summ_path, graph)
            else:
                merged = None

        sv = tf.train.Supervisor()
        with sv.managed_session() as session:
            # load model if needed
            if FLAGS.load_path:
                ckpt = tf.train.latest_checkpoint(FLAGS.load_path)
                if ckpt is not None:
                    print("Loading saved parameters")
                    sv.saver.restore(session, ckpt)
                else:
                    print("Initializing fresh parameters")

            # do training
            if FLAGS.train:
                # impose dropout for training then undo
                m.assign_drop(session, config.drop)
                for i in range(FLAGS.epoch):
                    summary, perplexity = run_epoch(session, m, d, summ_op=merged, verbose=True)
                    if summary:
                        writer.add_summary(summary, i)
                    print("Epoch: %d, Perplexity: %.3f" % (i, perplexity))
                m.assign_drop(session, 0.0)

                # save model params if needed
                if FLAGS.save_path:
                    print("Saving model to %s." % FLAGS.save_path)
                    if not os.path.isdir(FLAGS.save_path):
                        os.mkdir(FLAGS.save_path)
                    sv.saver.save(session, os.path.join(FLAGS.save_path, 'model'), global_step=sv.global_step)

            # simulate some draws
            if FLAGS.simul:
                sim = simulate_batch(session, m, d, temp=FLAGS.temp)
                print('\n'.join(sim))

if __name__ == "__main__":
    flags = tf.flags
    flags.DEFINE_string("data_path", None, "Where the training/test data is stored.")
    flags.DEFINE_string("load_path", None, "Model input directory.")
    flags.DEFINE_string("save_path", None, "Model output directory.")
    flags.DEFINE_string("summ_path", None, "Training log directory.")
    flags.DEFINE_bool("train", False, "Train model first.")
    flags.DEFINE_bool("simul", True, "Simulate a batch of tweets.")
    flags.DEFINE_integer("epoch", 10, "Number of epochs to run.")
    flags.DEFINE_float("temp", 1.0, "Temperature of simulation.")
    FLAGS = flags.FLAGS
    tf.app.run()

