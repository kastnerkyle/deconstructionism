from __future__ import print_function
import os
import argparse
import numpy as np
import tensorflow as tf
from collections import namedtuple

from utils import next_experiment_path
from batch_generator import BatchGenerator

import logging
import shutil
from tfdllib import get_logger
from tfdllib import Linear
from tfdllib import LSTMCell
from tfdllib import GaussianAttentionCell
from tfdllib import BernoulliAndCorrelatedGMMCost
from tfdllib import scan

tf.set_random_seed(2899)
# TODO: add help info
parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', dest='seq_len', default=256, type=int)
parser.add_argument('--batch_size', dest='batch_size', default=64, type=int)
parser.add_argument('--epochs', dest='epochs', default=8, type=int)
parser.add_argument('--window_mixtures', dest='window_mixtures', default=10, type=int)
parser.add_argument('--output_mixtures', dest='output_mixtures', default=20, type=int)
parser.add_argument('--lstm_layers', dest='lstm_layers', default=3, type=int)
parser.add_argument('--units_per_layer', dest='units', default=400, type=int)
parser.add_argument('--restore', dest='restore', default=None, type=str)
args = parser.parse_args()

epsilon = 1e-8

h_dim = args.units
forward_init = "truncated_normal"
rnn_init = "truncated_normal"
random_state = np.random.RandomState(1442)
output_mixtures = args.output_mixtures
window_mixtures = args.window_mixtures
num_units = args.units


def mixture(inputs, input_size, num_mixtures, bias, init="truncated_normal"):
    forward_init = init
    e = Linear([inputs], [input_size], 1, random_state=random_state,
               init=forward_init, name="mdn_e")
    pi = Linear([inputs], [input_size], num_mixtures, random_state=random_state,
                init=forward_init, name="mdn_pi")
    mu1 = Linear([inputs], [input_size], num_mixtures, random_state=random_state,
                 init=forward_init, name="mdn_mu1")
    mu2 = Linear([inputs], [input_size], num_mixtures, random_state=random_state,
                 init=forward_init, name="mdn_mu2")
    std1 = Linear([inputs], [input_size], num_mixtures, random_state=random_state,
                  init=forward_init, name="mdn_std1")
    std2 = Linear([inputs], [input_size], num_mixtures, random_state=random_state,
                  init=forward_init, name="mdn_std2")
    rho = Linear([inputs], [input_size], num_mixtures, random_state=random_state,
                 init=forward_init, name="mdn_rho")
    return tf.nn.sigmoid(e), \
           tf.nn.softmax(pi * (1. + bias), dim=-1), \
           mu1, mu2, \
           tf.exp(std1 - bias), tf.exp(std2 - bias), \
           tf.nn.tanh(rho)


def create_graph(num_letters, batch_size,
                 num_units=400, lstm_layers=3,
                 window_mixtures=10, output_mixtures=20):
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(2899)

        coordinates = tf.placeholder(tf.float32, shape=[None, batch_size, 3])
        coordinates_mask = tf.placeholder(tf.float32, shape=[None, batch_size])

        sequence = tf.placeholder(tf.float32, shape=[None, batch_size, num_letters])
        sequence_mask = tf.placeholder(tf.float32, shape=[None, batch_size])

        bias = tf.placeholder_with_default(tf.zeros(shape=[]), shape=[])
        att_w_init = tf.placeholder(tf.float32, shape=[batch_size, num_letters])
        att_k_init = tf.placeholder(tf.float32, shape=[batch_size, window_mixtures])
        att_h_init = tf.placeholder(tf.float32, shape=[batch_size, num_units])
        att_c_init = tf.placeholder(tf.float32, shape=[batch_size, num_units])
        h1_init = tf.placeholder(tf.float32, shape=[batch_size, num_units])
        c1_init = tf.placeholder(tf.float32, shape=[batch_size, num_units])
        h2_init = tf.placeholder(tf.float32, shape=[batch_size, num_units])
        c2_init = tf.placeholder(tf.float32, shape=[batch_size, num_units])

        def create_model(generate=None):
            in_coordinates = coordinates[:-1, :, :]
            in_coordinates_mask = coordinates_mask[:-1]
            out_coordinates = coordinates[1:, :, :]
            out_coordinates_mask = coordinates_mask[1:]

            def step(inp_t, inp_mask_t,
                     att_w_tm1, att_k_tm1, att_h_tm1, att_c_tm1,
                     h1_tm1, c1_tm1, h2_tm1, c2_tm1):

                o = GaussianAttentionCell([inp_t], [3],
                                          (att_h_tm1, att_c_tm1),
                                          att_k_tm1,
                                          sequence,
                                          num_letters,
                                          num_units,
                                          att_w_tm1,
                                          input_mask=inp_mask_t,
                                          conditioning_mask=sequence_mask,
                                          attention_scale = 1. / 25.,
                                          name="att",
                                          random_state=random_state,
                                          init=rnn_init)
                att_w_t, att_k_t, att_phi_t, s = o
                att_h_t = s[0]
                att_c_t = s[1]

                output, s = LSTMCell([inp_t, att_w_t, att_h_t],
                                     [3, num_letters, num_units],
                                     h1_tm1, c1_tm1, num_units,
                                     input_mask=inp_mask_t,
                                     random_state=random_state,
                                     name="rnn1", init=rnn_init)
                h1_t = s[0]
                c1_t = s[1]

                output, s = LSTMCell([inp_t, att_w_t, h1_t],
                                     [3, num_letters, num_units],
                                     h2_tm1, c2_tm1, num_units,
                                     input_mask=inp_mask_t,
                                     random_state=random_state,
                                     name="rnn2", init=rnn_init)
                h2_t = s[0]
                c2_t = s[1]
                return output, att_w_t, att_k_t, att_phi_t, att_h_t, att_c_t, h1_t, c1_t, h2_t, c2_t

            r = scan(step,
                     [in_coordinates, in_coordinates_mask],
                     [None, att_w_init, att_k_init, None, att_h_init, att_c_init,
                      h1_init, c1_init, h2_init, c2_init])
            output = r[0]
            att_w = r[1]
            att_k = r[2]
            att_phi = r[3]
            att_h = r[4]
            att_c = r[5]
            h1 = r[6]
            c1 = r[7]
            h2 = r[8]
            c2 = r[9]

            #output = tf.reshape(output, [-1, num_units])
            mo = mixture(output, num_units, output_mixtures, bias)
            e, pi, mu1, mu2, std1, std2, rho = mo

            #coords = tf.reshape(out_coordinates, [-1, 3])
            #xs, ys, es = tf.unstack(tf.expand_dims(coords, axis=2), axis=1)

            xs = out_coordinates[..., 0][..., None]
            ys = out_coordinates[..., 1][..., None]
            es = out_coordinates[..., 2][..., None]

            cc = BernoulliAndCorrelatedGMMCost(e, pi,
                                               [mu1, mu2],
                                               [std1, std2],
                                               rho,
                                               es,
                                               [xs, ys],
                                               name="cost")
            # mask + reduce_mean, slightly unstable
            #cc = in_coordinates_mask * cc
            #loss = tf.reduce_mean(cc)
            # mask + true weighted, better (flat) but also unstable
            #loss = tf.reduce_sum(cc / (tf.reduce_sum(in_coordinates_mask)))
            # no mask on loss - 0s become a form of biasing / noise?
            loss = tf.reduce_mean(cc)

            # save params for easier model loading and prediction
            for param in [('coordinates', coordinates),
                          ('in_coordinates', in_coordinates),
                          ('out_coordinates', out_coordinates),
                          ('coordinates_mask', coordinates_mask),
                          ('in_coordinates_mask', in_coordinates_mask),
                          ('out_coordinates_mask', out_coordinates_mask),
                          ('sequence', sequence),
                          ('sequence_mask', sequence_mask),
                          ('bias', bias),
                          ('e', e), ('pi', pi),
                          ('mu1', mu1), ('mu2', mu2),
                          ('std1', std1), ('std2', std2),
                          ('rho', rho),
                          ('att_w_init', att_w_init),
                          ('att_k_init', att_k_init),
                          ('att_h_init', att_h_init),
                          ('att_c_init', att_c_init),
                          ('h1_init', h1_init),
                          ('c1_init', c1_init),
                          ('h2_init', h2_init),
                          ('c2_init', c2_init),
                          ('att_w', att_w),
                          ('att_k', att_k),
                          ('att_phi', att_phi),
                          ('att_h', att_h),
                          ('att_c', att_c),
                          ('h1', h1),
                          ('c1', c1),
                          ('h2', h2),
                          ('c2', c2)]:
                tf.add_to_collection(*param)

            with tf.name_scope('training'):
                steps = tf.Variable(0.)
                learning_rate = tf.train.exponential_decay(0.001, steps, staircase=True,
                                                           decay_steps=10000, decay_rate=0.5)

                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, use_locking=True)
                grad, var = zip(*optimizer.compute_gradients(loss))
                grad, _ = tf.clip_by_global_norm(grad, 3.)
                train_step = optimizer.apply_gradients(zip(grad, var), global_step=steps)

            with tf.name_scope('summary'):
                # TODO: add more summaries
                summary = tf.summary.merge([
                    tf.summary.scalar('loss', loss)
                ])

            things_names = ["coordinates",
                            "coordinates_mask",
                            "sequence",
                            "sequence_mask",
                            "att_w_init",
                            "att_k_init",
                            "att_h_init",
                            "att_c_init",
                            "h1_init",
                            "c1_init",
                            "h2_init",
                            "c2_init",
                            "att_w",
                            "att_k",
                            "att_phi",
                            "att_h",
                            "att_c",
                            "h1",
                            "c1",
                            "h2",
                            "c2",
                            "loss",
                            "train_step",
                            "learning_rate",
                            "summary"]
            things_tf = [coordinates,
                         coordinates_mask,
                         sequence,
                         sequence_mask,
                         att_w_init,
                         att_k_init,
                         att_h_init,
                         att_c_init,
                         h1_init,
                         c1_init,
                         h2_init,
                         c2_init,
                         att_w,
                         att_k,
                         att_phi,
                         att_h,
                         att_c,
                         h1,
                         c1,
                         h2,
                         c2,
                         loss,
                         train_step,
                         learning_rate,
                         summary]
            return namedtuple('Model', things_names)(*things_tf)

        train_model = create_model(generate=None)
        _ = create_model(generate=True)  # just to create ops for generation

    return graph, train_model


def make_mask(arr):
    mask = np.ones_like(arr[:, :, 0])
    last_step = arr.shape[0] * arr[0, :, 0]
    for mbi in range(arr.shape[1]):
        for step in range(arr.shape[0]):
            if arr[step:, mbi].min() == 0. and arr[step:, mbi].max() == 0.:
                last_step[mbi] = step
                mask[step:, mbi] = 0.
                break
    return mask


def main():
    restore_model = args.restore
    seq_len = args.seq_len
    batch_size = args.batch_size
    num_epoch = args.epochs
    batches_per_epoch = 1000

    batch_generator = BatchGenerator(batch_size, seq_len, 2177)
    g, vs = create_graph(batch_generator.num_letters, batch_size,
                         num_units=args.units, lstm_layers=args.lstm_layers,
                         window_mixtures=args.window_mixtures,
                         output_mixtures=args.output_mixtures)

    with tf.Session(graph=g) as sess:
        model_saver = tf.train.Saver(max_to_keep=2)
        if restore_model:
            model_file = tf.train.latest_checkpoint(os.path.join(restore_model, 'models'))
            experiment_path = restore_model
            epoch = int(model_file.split('-')[-1]) + 1
            model_saver.restore(sess, model_file)
        else:
            sess.run(tf.global_variables_initializer())
            experiment_path = next_experiment_path()
            epoch = 0

        logger = get_logger()
        fh = logging.FileHandler(os.path.join(experiment_path, "experiment_run.log"))
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

        logger.info(" ")
        logger.info("Using experiment path {}".format(experiment_path))
        logger.info(" ")
        shutil.copy2(os.getcwd() + "/" + __file__, experiment_path)
        shutil.copy2(os.getcwd() + "/" + "tfdllib.py", experiment_path)

        for k, v in args.__dict__.items():
            logger.info("argparse argument {} had value {}".format(k, v))

        logger.info(" ")
        logger.info("Model information")
        for t_var in tf.trainable_variables():
            logger.info(t_var)
        logger.info(" ")

        summary_writer = tf.summary.FileWriter(experiment_path, graph=g, flush_secs=10)
        summary_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START),
                                       global_step=epoch * batches_per_epoch)

        logger.info(" ")

        num_letters = batch_generator.num_letters
        att_w_init_np = np.zeros((batch_size, num_letters))
        att_k_init_np = np.zeros((batch_size, window_mixtures))
        att_h_init_np = np.zeros((batch_size, num_units))
        att_c_init_np = np.zeros((batch_size, num_units))
        h1_init_np = np.zeros((batch_size, num_units))
        c1_init_np = np.zeros((batch_size, num_units))
        h2_init_np = np.zeros((batch_size, num_units))
        c2_init_np = np.zeros((batch_size, num_units))
        for e in range(epoch, num_epoch):
            logger.info("Epoch {}".format(e))
            for b in range(1, batches_per_epoch + 1):
                coords, seq, reset, needed = batch_generator.next_batch2()
                coords_mask = make_mask(coords)
                seq_mask = make_mask(seq)

                if needed:
                    att_w_init *= reset
                    att_k_init *= reset
                    att_h_init *= reset
                    att_c_init *= reset
                    h1_init *= reset
                    c1_init *= reset
                    h2_init *= reset
                    c2_init *= reset

                feed = {vs.coordinates: coords,
                        vs.coordinates_mask: coords_mask,
                        vs.sequence: seq,
                        vs.sequence_mask: seq_mask,
                        vs.att_w_init: att_w_init_np,
                        vs.att_k_init: att_k_init_np,
                        vs.att_h_init: att_h_init_np,
                        vs.att_c_init: att_c_init_np,
                        vs.h1_init: h1_init_np,
                        vs.c1_init: c1_init_np,
                        vs.h2_init: h2_init_np,
                        vs.c2_init: c2_init_np}
                outs = [vs.att_w, vs.att_k, vs.att_phi,
                        vs.att_h, vs.att_c,
                        vs.h1, vs.c1, vs.h2, vs.c2,
                        vs.loss, vs.summary, vs.train_step]
                r = sess.run(outs, feed_dict=feed)
                att_w_np = r[0]
                att_k_np = r[1]
                att_phi_np = r[2]
                att_h_np = r[3]
                att_c_np = r[5]
                h1_np = r[5]
                c1_np = r[6]
                h2_np = r[7]
                c2_np = r[8]
                l = r[-3]
                s = r[-2]
                _ = r[-1]

                # set next inits
                att_w_init = att_w_np[-1]
                att_k_init = att_k_np[-1]
                att_h_init = att_h_np[-1]
                att_c_init = att_c_np[-1]
                h1_init = h1_np[-1]
                c1_init = c1_np[-1]
                h2_init = h2_np[-1]
                c2_init = c2_np[-1]
                summary_writer.add_summary(s, global_step=e * batches_per_epoch + b)
                print('\r[{:5d}/{:5d}] loss = {}'.format(b, batches_per_epoch, l), end='')
            logger.info("\n[{:5d}/{:5d}] loss = {}".format(b, batches_per_epoch, l))
            logger.info(" ")

            model_saver.save(sess, os.path.join(experiment_path, 'models', 'model'),
                             global_step=e)


if __name__ == '__main__':
    main()
