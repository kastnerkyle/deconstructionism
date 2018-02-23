from __future__ import print_function
import os
import pickle
import argparse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from matplotlib import animation
#import seaborn
from collections import namedtuple
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model', dest='model_path', type=str, default=os.path.join('pretrained', 'model-29'))
parser.add_argument('--text', dest='text', type=str, default=None)
parser.add_argument('--bias', dest='bias', type=float, default=1.)
parser.add_argument('--force', dest='force', action='store_true', default=False)
parser.add_argument('--noinfo', dest='info', action='store_false', default=True)
parser.add_argument('--save', dest='save', type=str, default=None)
parser.add_argument('--seed', dest='seed', type=int, default=1999)
parser.add_argument('--stop_scale', dest='stop_scale', type=float, default=7.5)
parser.add_argument('--color', dest='color', type=str, default=None)
args = parser.parse_args()

random_state = np.random.RandomState(args.seed)

def sample(e, mu1, mu2, std1, std2, rho):

    cov = np.array([[std1 * std1, std1 * std2 * rho],
                    [std1 * std2 * rho, std2 * std2]])
    mean = np.array([mu1, mu2])

    x, y = random_state.multivariate_normal(mean, cov)
    end = random_state.binomial(1, e)
    return np.array([x, y, end])


def split_strokes(points):
    points = np.array(points)
    strokes = []
    b = 0
    for e in range(len(points)):
        if points[e, 2] == 1.:
            strokes += [points[b: e + 1, :2].copy()]
            b = e + 1
    strokes += [points[b: e + 1, :2].copy()]
    return strokes


def cumsum(points):
    sums = np.cumsum(points[:, :2], axis=0)
    return np.concatenate([sums, points[:, 2:]], axis=1)


def sample_text(sess, args_text, translation):
    fields = ['in_coordinates',
              'in_coordinates_mask',
              'sequence',
              'sequence_mask',
              'bias',
              'e', 'pi', 'mu1', 'mu2', 'std1', 'std2', 'rho',
              'att_w_init', 'att_w',
              'att_k_init', 'att_k',
              'att_h_init', 'att_h',
              'att_c_init', 'att_c',
              'h1_init', 'h1',
              'c1_init', 'c1',
              'h2_init', 'h2',
              'c2_init', 'c2',
              'att_phi']
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )

    text = np.array([translation.get(c, 0) for c in args_text])

    num_letters = len(translation)
    num_units = 400
    window_mixtures = 10
    output_mixtures = 20
    batch_size = 64
    choose_i = 0
    bc = np.ones((batch_size, 1))
    coord = np.array([0., 0., 1.])
    coord = coord[None] * bc
    coord = coord[None]
    coord_mask = 0. * coord[:, :, 0] + 1.
    coords = [coord]

    sequence = np.eye(len(translation), dtype=np.float32)[text]
    sequence = np.expand_dims(np.concatenate([sequence, np.zeros((1, len(translation)))]), axis=0)
    bs = np.ones((batch_size, 1, 1))
    sequence = bs * sequence
    sequence = sequence.transpose(1, 0, 2)
    sequence_mask = 0. * sequence[:, :, 0] + 1.

    att_w_init_np = np.zeros((batch_size, num_letters))
    att_k_init_np = np.zeros((batch_size, window_mixtures))
    att_h_init_np = np.zeros((batch_size, num_units))
    att_c_init_np = np.zeros((batch_size, num_units))
    h1_init_np = np.zeros((batch_size, num_units))
    c1_init_np = np.zeros((batch_size, num_units))
    h2_init_np = np.zeros((batch_size, num_units))
    c2_init_np = np.zeros((batch_size, num_units))

    phi_data, window_data, kappa_data, stroke_data = [], [], [], []
    sequence_len = len(args_text)
    for s in range(1, 60 * sequence_len + 1):
        print('\r[{:5d}] sampling... {}'.format(s, 'synthesis'), end='')
        feed = {vs.in_coordinates: coord, #np.array(coords)[:, 0],
                vs.in_coordinates_mask: coord_mask,
                vs.sequence: sequence,
                vs.sequence_mask: sequence_mask,
                vs.bias: args.bias,
                vs.att_w_init: att_w_init_np, #*0
                vs.att_k_init: att_k_init_np, #*0
                vs.att_h_init: att_h_init_np, #*0
                vs.att_c_init: att_c_init_np, #*0
                vs.h1_init: h1_init_np, #*0
                vs.c1_init: c1_init_np, #*0
                vs.h2_init: h2_init_np, #*0
                vs.c2_init: c2_init_np} #*0

        o = sess.run([vs.e, vs.pi, vs.mu1, vs.mu2,
                     vs.std1, vs.std2, vs.rho,
                     vs.att_w, vs.att_k, vs.att_phi,
                     vs.att_h, vs.att_c,
                     vs.h1, vs.c1,
                     vs.h2, vs.c2],
                     feed_dict=feed)
        e = o[0]
        pi = o[1]
        mu1 = o[2]
        mu2 = o[3]
        std1 = o[4]
        std2 = o[5]
        rho = o[6]
        att_w = o[7]
        att_k = o[8]
        att_phi = o[9]
        att_h = o[10]
        att_c = o[11]
        h1 = o[12]
        c1 = o[13]
        h2 = o[14]
        c2 = o[15]

        def _2d(a):
            if len(a.shape) > 2:
                if a.shape[0] == 1:
                    return a[0]
                else:
                    raise ValueError("array is >2D, and dim 0 has more than 1 element! shape {}".format(a.shape))
            else:
                return a

        e = _2d(e)
        pi = _2d(pi)
        mu1 = _2d(mu1)
        mu2 = _2d(mu2)
        std1 = _2d(std1)
        std2 = _2d(std2)
        rho = _2d(rho)

        kappa = att_k
        window = att_w
        phi = att_phi

        # Synthesis mode
        phi_data += [phi[-1, choose_i, :]]
        window_data += [window[-1, choose_i, :]]
        kappa_data += [kappa[-1, choose_i, :]]
        # ---
        g = random_state.choice(np.arange(pi.shape[1]), p=pi[choose_i])
        coord = sample(e[choose_i, 0],
                       mu1[choose_i, g], mu2[choose_i, g],
                       std1[choose_i, g], std2[choose_i, g],
                       rho[choose_i, g])

        coord = coord[None] * bc
        coord = coord[None]
        coords += [coord]
        stroke_data += [[mu1[choose_i, g], mu2[choose_i, g],
                         std1[choose_i, g], std2[choose_i, g], rho[choose_i, g], coord[-1, choose_i, 2]]]

        att_w_init_np = att_w[-1]
        att_k_init_np = att_k[-1]
        att_h_init_np = att_h[-1]
        att_c_init_np = att_c[-1]
        h1_init_np = h1[-1]
        c1_init_np = c1[-1]
        h2_init_np = h2[-1]
        c2_init_np = c2[-1]

        thresh = phi[-1, choose_i, -1] > args.stop_scale * np.max(phi[-1, choose_i, :-1])
        #thresh = mu1.mean() > 1.1 * len(text)
        #thresh = finish[0, 0] > 0.9

        if not args.force and thresh:
            print('\nFinished sampling!\n')
            break

    # becomes (len, 1, batch_size, 3)
    coords = np.array(coords)[:, 0, choose_i]
    coords[-1, 2] = 1.
    return phi_data, window_data, kappa_data, stroke_data, coords


def main():
    with open(os.path.join('data', 'translation.pkl'), 'rb') as file:
        translation = pickle.load(file)
    rev_translation = {v: k for k, v in translation.items()}
    charset = [rev_translation[i] for i in range(len(rev_translation))]
    charset[0] = ''

    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph(args.model_path + '.meta')
        saver.restore(sess, args.model_path)

        if args.text is not None:
            args_text = args.text
        else:
            raise ValueError("Must pass --text argument")

        phi_data, window_data, kappa_data, stroke_data, coords = sample_text(sess, args_text, translation)

        strokes = np.array(stroke_data)
        epsilon = 1e-8
        strokes[:, :2] = np.cumsum(strokes[:, :2], axis=0)
        minx, maxx = np.min(strokes[:, 0]), np.max(strokes[:, 0])
        miny, maxy = np.min(strokes[:, 1]), np.max(strokes[:, 1])

        if args.info:
            delta = abs(maxx - minx) / 400.
            x = np.arange(minx, maxx, delta)
            y = np.arange(miny, maxy, delta)
            x_grid, y_grid = np.meshgrid(x, y)
            z_grid = np.zeros_like(x_grid)
            for i in range(strokes.shape[0]):
                # what
                cov = np.array([[strokes[i, 2], 0.],
                                [0., strokes[i, 3]]])
                gauss = mlab.bivariate_normal(x_grid, y_grid, mux=strokes[i, 0], muy=strokes[i, 1],
                                              sigmax=cov[0, 0], sigmay=cov[1, 1],
                                              sigmaxy=strokes[i, 4] * cov[0, 0] * cov[1, 1])
                # needs to be rho * sigmax * sigmay
                z_grid += gauss * np.power(strokes[i, 2] + strokes[i, 3], 0.4) / (np.max(gauss) + epsilon)

            for f in os.listdir("."):
                if "plot" in f and f.endswith(".png"):
                    print("Removing old plot {}".format(f))
                    os.remove(f)

            t = int(time.time())
            new = "plot_{}".format(hash(t) % 10 ** 5) + "_{}.png"
            plt.figure()
            plt.imshow(z_grid, interpolation="bilinear", cmap=cm.jet)
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.axis("off")
            new_d = new.format("density")
            plt.title("Density")
            print("Saving to {}".format(new_d))
            plt.savefig(new_d)
            plt.close()

            plt.figure()
            for stroke in split_strokes(cumsum(np.array(coords))):
                if args.color is not None:
                    plt.plot(stroke[:, 0], -stroke[:, 1], color=args.color)
                else:
                    plt.plot(stroke[:, 0], -stroke[:, 1])
            plt.title(args.text)
            plt.axes().set_aspect('equal')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.axis("off")

            new_h = new.format("handwriting")
            print("Saving to {}".format(new_h))
            plt.savefig(new_h)
            plt.close()

            plt.figure()
            phi_img = np.vstack(phi_data).T[::-1, :]
            plt.imshow(phi_img, interpolation='nearest', aspect='auto', cmap=cm.jet)
            plt.yticks(np.arange(0, len(args_text) + 1))
            plt.axes().set_yticklabels(list(' ' + args_text[::-1]), rotation='vertical', fontsize=8)
            plt.grid(False)
            plt.title('Phi')
            new_p = new.format("phi")
            print("Saving to {}".format(new_p))
            plt.savefig(new_p)
            plt.close()

            plt.figure()
            window_img = np.vstack(window_data).T
            plt.imshow(window_img, interpolation='nearest', aspect='auto', cmap=cm.jet)
            plt.yticks(np.arange(0, len(charset)))
            plt.axes().set_yticklabels(list(charset), rotation='vertical', fontsize=8)
            plt.grid(False)
            plt.title('Window')
            new_w = new.format("window")
            print("Saving to {}".format(new_w))
            plt.savefig(new_w)
            plt.close()
        else:
            fig, ax = plt.subplots(1, 1)
            for stroke in split_strokes(cumsum(np.array(coords))):
                if args.color is not None:
                    plt.plot(stroke[:, 0], -stroke[:, 1], color=args.color)
                else:
                    plt.plot(stroke[:, 0], -stroke[:, 1])
            ax.set_title(args.text)
            ax.set_aspect('equal')

            for f in os.listdir("."):
                if "gen_plot" in f and f.endswith(".png"):
                    print("Removing old plot {}".format(f))
                    os.remove(f)

            t = int(time.time())
            new = "gen_plot_{}.png".format(hash(t) % 10 ** 5)
            print("Saving to {}".format(new))
            plt.savefig(new)


if __name__ == '__main__':
    main()
    print("")
