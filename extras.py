from __future__ import print_function
# Author: Kyle Kastner
# License: BSD 3-clause
# Thanks to Jose (@sotelo) for tons of guidance and debug help
# Credit also to Junyoung (@jych) and Shawn (@shawntan) for help/utility funcs
# Strangeness in init could be from onehots, via @igul222. Ty init for one hot layer as N(0, 1) just as in embedding
# since oh.dot(w) is basically an embedding
import os
import re
import tarfile
from collections import Counter, OrderedDict
from bs4 import BeautifulSoup as Soup
import sys
import pickle
import numpy as np
import fnmatch
from scipy import linalg
from functools import wraps
import exceptions
from pthbldr import pe


class base_iterator(object):
    def __init__(self, list_of_containers, minibatch_size,
                 axis,
                 start_index=0,
                 stop_index=np.inf,
                 make_mask=False,
                 one_hot_class_size=None):
        self.list_of_containers = list_of_containers
        self.minibatch_size = minibatch_size
        self.make_mask = make_mask
        self.start_index = start_index
        self.stop_index = stop_index
        self.slice_start_ = start_index
        self.axis = axis
        if axis not in [0, 1]:
            raise ValueError("Unknown sample_axis setting %i" % axis)
        self.one_hot_class_size = one_hot_class_size
        if one_hot_class_size is not None:
            assert len(self.one_hot_class_size) == len(list_of_containers)

    def reset(self):
        self.slice_start_ = self.start_index

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        self.slice_end_ = self.slice_start_ + self.minibatch_size
        if self.slice_end_ > self.stop_index:
            # TODO: Think about boundary issues with weird shaped last mb
            self.reset()
            raise StopIteration("Stop index reached")
        ind = slice(self.slice_start_, self.slice_end_)
        self.slice_start_ = self.slice_end_
        if self.make_mask is False:
            res = self._slice_without_masks(ind)
            if not all([self.minibatch_size in r.shape for r in res]):
                # TODO: Check that things are even
                self.reset()
                raise StopIteration("Partial slice returned, end of iteration")
            return res
        else:
            res = self._slice_with_masks(ind)
            # TODO: Check that things are even
            if not all([self.minibatch_size in r.shape for r in res]):
                self.reset()
                raise StopIteration("Partial slice returned, end of iteration")
            return res

    def _slice_without_masks(self, ind):
        raise AttributeError("Subclass base_iterator and override this method")

    def _slice_with_masks(self, ind):
        raise AttributeError("Subclass base_iterator and override this method")


class list_iterator(base_iterator):
    def _slice_without_masks(self, ind, return_shapes=False):
        sliced_c = [np.asarray(c[ind]) for c in self.list_of_containers]
        # object arrays
        shapes = [[sci.shape for sci in sc] for sc in sliced_c]

        if min([len(i) for i in sliced_c]) < self.minibatch_size:
            self.reset()
            raise StopIteration("Invalid length slice")
        self.is_thin = [False for lc in self.list_of_containers]
        for n in range(len(sliced_c)):
            sc = sliced_c[n]
            if self.one_hot_class_size is not None:
                convert_it = self.one_hot_class_size[n]
                if convert_it is not None:
                    raise ValueError("One hot conversion not implemented")
            if not isinstance(sc, np.ndarray) or sc.dtype == np.object:
                maxlen = max([len(i) for i in sc])
                # Assume they at least have the same internal dtype
                if len(sc[0].shape) > 1:
                    total_shape = (maxlen, sc[0].shape[1])
                elif len(sc[0].shape) == 1:
                    total_shape = (maxlen, 1)
                else:
                    raise ValueError("Unhandled array size in list")
                if self.axis == 0:
                    raise ValueError("Unsupported axis of iteration")
                    new_sc = np.zeros((len(sc), total_shape[0],
                                       total_shape[1]))
                    new_sc = new_sc.squeeze().astype(sc[0].dtype)
                else:
                    new_sc = np.zeros((total_shape[0], len(sc),
                                       total_shape[1]))
                    new_sc = new_sc.astype(sc[0].dtype)
                    for m, sc_i in enumerate(sc):
                        if len(sc_i.shape) == 1:
                            self.is_thin[n] = True
                            # if the array is 1D still broadcast fill
                            sc_i = sc_i[:, None]
                        new_sc[:len(sc_i), m, :] = sc_i
                if self.is_thin[n]:
                    sliced_c[n] = new_sc[..., 0]
                else:
                    sliced_c[n] = new_sc
        if not return_shapes:
            return sliced_c
        else:
            return sliced_c, shapes

    def _slice_with_masks(self, ind):
        cs, cs_shapes = self._slice_without_masks(ind, return_shapes=True)
        if self.axis == 0:
            ms = [np.zeros_like(c[:, 0]) for c in cs]
        elif self.axis == 1:
            ms = [np.zeros_like(c)
                  if self.is_thin[n] else np.zeros_like(c[:, :, 0])
                  for n, c in enumerate(cs)]
        for ni, csi in enumerate(cs):
            for ii in range(len(cs_shapes[ni])):
                if cs_shapes[ni][ii][0] < 1:
                    raise AttributeError("Minibatch has invalid content size {}".format(cs_shapes[ni][ii][0]))
                assert cs_shapes[ni][ii]
                ms[ni][:cs_shapes[ni][ii][0], ii] = 1.
        assert len(cs) == len(ms)
        return [i for sublist in list(zip(cs, ms)) for i in sublist]


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    labels_shape = labels_dense.shape
    labels_dense = labels_dense.reshape([-1])
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    labels_one_hot = labels_one_hot.reshape(labels_shape+(num_classes,))
    return labels_one_hot


def tokenize_ind(phrase, vocabulary):
    phrase = phrase + " "
    vocabulary_size = len(vocabulary.keys())
    phrase = [vocabulary[char_] for char_ in phrase]
    phrase = np.array(phrase, dtype='int32').ravel()
    phrase = dense_to_one_hot(phrase, vocabulary_size)
    return phrase


# https://mrcoles.com/blog/3-decorator-examples-and-awesome-python/
def rsync_fetch(fetch_func, machine_to_fetch_from, *args, **kwargs):
    """
    be sure not to call it as
    rsync_fetch(fetch_func, machine_name)
    not
    rsync_fetch(fetch_func(), machine_name)
    """
    try:
        r = fetch_func(*args, **kwargs)
    except Exception as e:
        if isinstance(e, IOError):
            full_path = e.filename
            filedir = str(os.sep).join(full_path.split(os.sep)[:-1])
            if not os.path.exists(filedir):
                if filedir[-1] != "/":
                    fd = filedir + "/"
                else:
                    fd = filedir
                os.makedirs(fd)

            if filedir[-1] != "/":
                fd = filedir + "/"
            else:
                fd = filedir

            if not os.path.exists(full_path):
                sdir = str(machine_to_fetch_from) + ":" + fd
                cmd = "rsync -avhp --progress %s %s" % (sdir, fd)
                pe(cmd, shell=True)
        else:
            print("unknown error {}".format(e))
        r = fetch_func(*args, **kwargs)
    return r



def plot_lines_iamondb_example(X, title="", save_name=None):
    import matplotlib.pyplot as plt
    f, ax = plt.subplots()
    x = np.cumsum(X[:, 1])
    y = np.cumsum(X[:, 2])

    size_x = x.max() - x.min()
    size_y = y.max() - y.min()

    f.set_size_inches(5 * size_x / size_y, 5)
    cuts = np.where(X[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=1.5)
        start = cut_value + 1
    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_title(title)

    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.close()


def implot(arr, title="", cmap="gray", save_name=None):
    import matplotlib.pyplot as plt
    f, ax = plt.subplots()
    ax.matshow(arr, cmap=cmap)
    plt.axis("off")

    def autoaspect(x_range, y_range):
        """
        The aspect to make a plot square with ax.set_aspect in Matplotlib
        """
        mx = max(x_range, y_range)
        mn = min(x_range, y_range)
        if x_range <= y_range:
            return mx / float(mn)
        else:
            return mn / float(mx)

    x1 = arr.shape[0]
    y1 = arr.shape[1]
    asp = autoaspect(x1, y1)
    ax.set_aspect(asp)
    plt.title(title)
    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name)


def check_fetch_iamondb():
    """ Check for IAMONDB data

        This dataset cannot be downloaded automatically!
    """
    #partial_path = get_dataset_dir("iamondb")
    partial_path = os.sep + "Tmp" + os.sep + "kastner" + os.sep + "iamondb"
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    combined_data_path = os.path.join(partial_path, "original-xml-part.tar.gz")
    untarred_data_path = os.path.join(partial_path, "original")
    if not os.path.exists(combined_data_path):
        files = "original-xml-part.tar.gz"
        url = "http://www.iam.unibe.ch/fki/databases/"
        url += "iam-on-line-handwriting-database/"
        url += "download-the-iam-on-line-handwriting-database"
        err = "Path %s does not exist!" % combined_data_path
        err += " Download the %s files from %s" % (files, url)
        err += " and place them in the directory %s" % partial_path
        print("WARNING: {}".format(err))
    return partial_path

"""
- all points:
>> [[x1, y1, e1], ..., [xn, yn, en]]
- indexed values
>> [h1, ... hn]
"""


def distance(p1, p2, axis=None):
    return np.sqrt(np.sum(np.square(p1 - p2), axis=axis))


def clear_middle(pts):
    to_remove = set()
    for i in range(1, len(pts) - 1):
        p1, p2, p3 = pts[i - 1: i + 2, :2]
        dist = distance(p1, p2) + distance(p2, p3)
        if dist > 1500:
            to_remove.add(i)
    npts = []
    for i in range(len(pts)):
        if i not in to_remove:
            npts += [pts[i]]
    return np.array(npts)


def separate(pts):
    seps = []
    for i in range(0, len(pts) - 1):
        if distance(pts[i], pts[i+1]) > 600:
            seps += [i + 1]
    return [pts[b:e] for b, e in zip([0] + seps, seps + [len(pts)])]


def iamondb_extract(partial_path):
    """
    Lightly modified from https://github.com/Grzego/handwriting-generation/blob/master/preprocess.py
    """
    data = []
    charset = set()

    file_no = 0
    pth = os.path.join(partial_path, "original")
    for root, dirs, files in os.walk(pth):
        for file in files:
            file_name, extension = os.path.splitext(file)
            if extension == '.xml':
                file_no += 1
                print('[{:5d}] File {} -- '.format(file_no, os.path.join(root, file)), end='')
                xml = ElementTree.parse(os.path.join(root, file)).getroot()
                transcription = xml.findall('Transcription')
                if not transcription:
                    print('skipped')
                    continue
                #texts = [html.unescape(s.get('text')) for s in transcription[0].findall('TextLine')]
                texts = [HTMLParser.HTMLParser().unescape(s.get('text')) for s in transcription[0].findall('TextLine')]
                points = [s.findall('Point') for s in xml.findall('StrokeSet')[0].findall('Stroke')]
                strokes = []
                mid_points = []
                for ps in points:
                    pts = np.array([[int(p.get('x')), int(p.get('y')), 0] for p in ps])
                    pts[-1, 2] = 1

                    pts = clear_middle(pts)
                    if len(pts) == 0:
                        continue

                    seps = separate(pts)
                    for pss in seps:
                        if len(seps) > 1 and len(pss) == 1:
                            continue
                        pss[-1, 2] = 1

                        xmax, ymax = max(pss, key=lambda x: x[0])[0], max(pss, key=lambda x: x[1])[1]
                        xmin, ymin = min(pss, key=lambda x: x[0])[0], min(pss, key=lambda x: x[1])[1]

                        strokes += [pss]
                        mid_points += [[(xmax + xmin) / 2., (ymax + ymin) / 2.]]
                distances = [-(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))
                             for p1, p2 in zip(mid_points, mid_points[1:])]
                splits = sorted(np.argsort(distances)[:len(texts) - 1] + 1)
                lines = []
                for b, e in zip([0] + splits, splits + [len(strokes)]):
                    lines += [[p for pts in strokes[b:e] for p in pts]]
                print('lines = {:4d}; texts = {:4d}'.format(len(lines), len(texts)))
                charset |= set(''.join(texts))
                data += [(texts, lines)]
    print('data = {}; charset = ({}) {}'.format(len(data), len(charset), ''.join(sorted(charset))))

    translation = {'<NULL>': 0}
    for c in ''.join(sorted(charset)):
        translation[c] = len(translation)

    def translate(txt):
        return list(map(lambda x: translation[x], txt))

    dataset = []
    labels = []
    for texts, lines in data:
        for text, line in zip(texts, lines):
            line = np.array(line, dtype=np.float32)
            line[:, 0] = line[:, 0] - np.min(line[:, 0])
            line[:, 1] = line[:, 1] - np.mean(line[:, 1])

            dataset += [line]
            labels += [translate(text)]

    whole_data = np.concatenate(dataset, axis=0)
    std_y = np.std(whole_data[:, 1])
    norm_data = []
    for line in dataset:
        line[:, :2] /= std_y
        norm_data += [line]
    dataset = norm_data

    print('datset = {}; labels = {}'.format(len(dataset), len(labels)))

    save_path = os.path.join(partial_path, 'preprocessed_data')
    try:
        os.makedirs(save_path)
    except FileExistsError:
        pass
    np.save(os.path.join(save_path, 'dataset'), np.array(dataset))
    np.save(os.path.join(save_path, 'labels'), np.array(labels))
    with open(os.path.join(save_path, 'translation.pkl'), 'wb') as file:
        pickle.dump(translation, file)
    print("Preprocessing finished and cached at {}".format(save_path))


def fetch_iamondb():
    partial_path = check_fetch_iamondb()
    combined_data_path = os.path.join(partial_path, "original-xml-part.tar.gz")
    untarred_data_path = os.path.join(partial_path, "original")
    if not os.path.exists(untarred_data_path):
        print("Now untarring {}".format(combined_data_path))
        tar = tarfile.open(combined_data_path, "r:gz")
        tar.extractall(partial_path)
        tar.close()

    saved_dataset_path = os.path.join(partial_path, 'preprocessed_data')
    if not os.path.exists(saved_dataset_path):
        iamondb_extract(partial_path)

    dataset_path = os.path.join(saved_dataset_path, "dataset.npy")
    labels_path = os.path.join(saved_dataset_path, "labels.npy")
    translation_path = os.path.join(saved_dataset_path, "translation.pkl")

    dataset = np.load(dataset_path)
    dataset = [np.array(d) for d in dataset]
    temp = []
    for d in dataset:
        # dataset stores actual pen points, but we will train on differences between consecutive points
        offs = d[1:, :2] - d[:-1, :2]
        ends = d[1:, 2]
        temp += [np.concatenate([[[0., 0., 1.]], np.concatenate([offs, ends[:, None]], axis=1)], axis=0)]
    # because lines are of different length, we store them in python array (not numpy)
    dataset = temp
    labels = np.load(labels_path)
    labels = [np.array(l) for l in labels]
    with open(translation_path, 'rb') as f:
        translation = pickle.load(f)
    # be sure of consisten ordering
    new_translation = OrderedDict()
    for k in sorted(translation.keys()):
        new_translation[k] = translation[k]
    translation = new_translation
    dataset_storage = {}
    dataset_storage["data"] = dataset
    dataset_storage["target"] = labels
    inverse_translation = {v: k for k, v in translation.items()}
    dataset_storage["target_phrases"] = ["".join([inverse_translation[ci] for ci in labels[i]]) for i in range(len(labels))]
    dataset_storage["vocabulary_size"] = len(translation)
    dataset_storage["vocabulary"] = sorted(translation.keys())
    return dataset_storage
