from __future__ import print_function
import tensorflow as tf
import numpy as np
import uuid
from scipy import linalg
from scipy.stats import truncnorm
from scipy.misc import factorial
import tensorflow as tf
import shutil
import socket
import os
import re
import copy
import sys
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import logging
from collections import OrderedDict

logging.basicConfig(level=logging.INFO,
                    format='%(message)s')
logger = logging.getLogger(__name__)

string_f = StringIO()
ch = logging.StreamHandler(string_f)
# Automatically put the HTML break characters on there for html logger
formatter = logging.Formatter('%(message)s<br>')
ch.setFormatter(formatter)
logger.addHandler(ch)

def get_logger():
    return logger

sys.setrecursionlimit(40000)

# Storage of internal shared
_lib_shared_params = OrderedDict()


def _get_name():
    return str(uuid.uuid4())


def _get_shared(name):
    if name in _lib_shared_params.keys():
        logger.info("Found name %s in shared parameters" % name)
        return _lib_shared_params[name]
    else:
        raise NameError("Name not found in shared params!")


def _set_shared(name, variable):
    if name in _lib_shared_params.keys():
        raise ValueError("Trying to set key %s which already exists!" % name)
    _lib_shared_params[name] = variable


def get_params_dict():
    return _lib_shared_params

weight_norm_default = False
def get_weight_norm_default():
    return weight_norm_default

strict_mode_default = False
def get_strict_mode_default():
    return strict_mode_default


def print_network(params_dict):
    logger.info("=====================")
    logger.info("Model Summary")
    logger.info("format: {name} {shape}, {parameter_count}")
    logger.info("---------------------")
    for k, v in params_dict.items():
        #strip_name = "_".join(k.split("_")[1:])
        strip_name = k
        shp = tuple(shape(v))
        k_count = np.prod(shp) / float(1E3)
        logger.info("{} {}, {}K".format(strip_name, shp, k_count))
    params = params_dict.values()
    n_params = sum([np.prod(shape(p)) for p in params])
    logger.info("---------------------")
    logger.info(" ")
    logger.info("Total: {}M".format(n_params / float(1E6)))
    logger.info("=====================")


def shape(x):
    r = x.get_shape().as_list()
    r = [ri if ri != None else -1 for ri in r]

    #if len([ri for ri in r if ri == -1]) > 1:
    #    raise ValueError("Too many None shapes in shape dim {}, should only 1 -1 dim at most".format(r))
    return r


def ndim(x):
    return len(shape(x))


def sigmoid(x):
    return tf.sigmoid(x)


def tanh(x):
    return tf.tanh(x)


def np_zeros(shape):
    """
    Builds a numpy variable filled with zeros
    Parameters
    ----------
    shape, tuple of ints
        shape of zeros to initialize
    Returns
    -------
    initialized_zeros, array-like
        Array-like of zeros the same size as shape parameter
    """
    return np.zeros(shape).astype("float32")


def np_normal(shape, random_state, scale=0.01):
    """
    Builds a numpy variable filled with normal random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 0.01)
        default of 0.01 results in normal random values with variance 0.01
    Returns
    -------
    initialized_normal, array-like
        Array-like of normal random values the same size as shape parameter
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        shp = shape
    return (scale * random_state.randn(*shp)).astype("float32")


def np_truncated_normal(shape, random_state, scale=0.075):
    """
    Builds a numpy variable filled with truncated normal random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 0.075)
        default of 0.075
    Returns
    -------
    initialized_normal, array-like
        Array-like of truncated normal random values the same size as shape parameter
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        shp = shape

    sigma = scale
    lower = -2 * sigma
    upper = 2 * sigma
    mu = 0
    N = np.prod(shp)
    samples = truncnorm.rvs(
              (lower - mu) / float(sigma), (upper - mu) / float(sigma),
              loc=mu, scale=sigma, size=N, random_state=random_state)
    return samples.reshape(shp).astype("float32")


def np_tanh_fan_normal(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 1.)
        default of 1. results in normal random values
        with sqrt(2 / (fan in + fan out)) scale
    Returns
    -------
    initialized_fan, array-like
        Array-like of random values the same size as shape parameter
    References
    ----------
    Understanding the difficulty of training deep feedforward neural networks
        X. Glorot, Y. Bengio
    """
    # The . after the 2 is critical! shape has dtype int...
    if type(shape[0]) is tuple:
        kern_sum = np.prod(shape[0]) + np.prod(shape[1])
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        kern_sum = np.sum(shape)
        shp = shape
    var = scale * np.sqrt(2. / kern_sum)
    return var * random_state.randn(*shp).astype("float32")


def np_variance_scaled_uniform(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 1.)
        default of 1. results in uniform random values
        with 1 * sqrt(1 / (n_dims)) scale
    Returns
    -------
    initialized_scaled, array-like
        Array-like of random values the same size as shape parameter
    References
    ----------
    Efficient Backprop
        Y. LeCun, L. Bottou, G. Orr, K. Muller
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
        kern_sum = np.prod(shape[0])
    else:
        shp = shape
        kern_sum = shape[0]
    #  Make sure bounds aren't the same
    bound = scale * np.sqrt(3. / float(kern_sum))  # sqrt(3) for std of uniform
    return random_state.uniform(low=-bound, high=bound, size=shp).astype(
        "float32")


def np_glorot_uniform(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 1.)
        default of 1. results in uniform random values
        with 1. * sqrt(6 / (n_in + n_out)) scale
    Returns
    -------
    initialized_scaled, array-like
        Array-like of random values the same size as shape parameter
    """
    shp = shape
    kern_sum = sum(shp)
    bound = scale * np.sqrt(6. / float(kern_sum))
    return random_state.uniform(low=-bound, high=bound, size=shp).astype(
        "float32")


def np_ortho(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with orthonormal random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 1.)
        default of 1. results in orthonormal random values sacled by 1.
    Returns
    -------
    initialized_ortho, array-like
        Array-like of random values the same size as shape parameter
    References
    ----------
    Exact solutions to the nonlinear dynamics of learning in deep linear
    neural networks
        A. Saxe, J. McClelland, S. Ganguli
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
        flat_shp = (shp[0], np.prd(shp[1:]))
    else:
        shp = shape
        flat_shp = shape
    g = random_state.randn(*flat_shp)
    U, S, VT = linalg.svd(g, full_matrices=False)
    res = U if U.shape == flat_shp else VT  # pick one with the correct shape
    res = res.reshape(shp)
    return (scale * res).astype("float32")


def make_numpy_biases(bias_dims):
    return [np_zeros((dim,)) for dim in bias_dims]


def make_numpy_weights(in_dim, out_dims, random_state, init=None,
                       scale="default"):
    """
    Will return as many things as are in the list of out_dims
    You *must* get a list back, even for 1 element retuTrue
    blah, = make_weights(...)
    or
    [blah] = make_weights(...)
    """
    ff = [None] * len(out_dims)
    fs = [scale] * len(out_dims)
    for i, out_dim in enumerate(out_dims):
        if init is None:
            if in_dim == out_dim:
                ff[i] = np_ortho
                fs[i] = 1.
            else:
                ff[i] = np_variance_scaled_uniform
                fs[i] = 1.
        elif init == "ortho":
            if in_dim != out_dim:
                raise ValueError("Unable to use ortho init for non-square matrices!")
            ff[i] = np_ortho
            fs[i] = 1.
        elif init == "glorot_uniform":
            ff[i] = np_glorot_uniform
        elif init == "normal":
            ff[i] = np_normal
            fs[i] = 0.01
        elif init == "truncated_normal":
            ff[i] = np_truncated_normal
            fs[i] = 0.075
        elif init == "embedding_normal":
            ff[i] = np_truncated_normal
            fs[i] = 1. / np.sqrt(in_dim)
        else:
            raise ValueError("Unknown init type %s" % init)

    ws = []
    for i, out_dim in enumerate(out_dims):
        if fs[i] == "default":
            ws.append(ff[i]((in_dim, out_dim), random_state))
        else:
            ws.append(ff[i]((in_dim, out_dim), random_state, scale=fs[i]))
    return ws


def dot(a, b):
    # Generalized dot for nd sequences, assumes last axis is projection
    # b must be rank 2
    a_tup = shape(a)
    b_tup = shape(b)
    if len(a_tup) == 2 and len(b_tup) == 2:
        return tf.matmul(a, b)
    elif len(a_tup) == 3 and len(b_tup) == 2:
        # more generic, supports multiple -1 axes
        return tf.einsum("ijk,kl->ijl", a, b)
        #a_i = tf.reshape(a, [-1, a_tup[-1]])
        #a_n = tf.matmul(a_i, b)
        #a_nf = tf.reshape(a_n, list(a_tup[:-1]) + [b_tup[-1]])
        #return a_nf
    else:
        raise ValueError("Shapes for arguments to dot() are {} and {}, not supported!".format(a_tup, b_tup))


def scan(fn, sequences, outputs_info):
    nonepos = [n for n, o in enumerate(outputs_info) if o is None]
    nonnone = [o for o in outputs_info if o is not None]
    sequences_and_nonnone = sequences + nonnone
    sliced = [s[0] for s in sequences] + nonnone
    inf_ret = fn(*sliced)
    if len(outputs_info) < len(inf_ret):
        raise ValueError("More outputs from `fn` than elements in outputs_info. Expected {} outs, given outputs_info of length {}, but `fn` returns {}. Pass None in outputs_info for returns which don't accumulate".format(len(outputs_info), len(outputs_info), len(inf_ret)))
    initializers = []
    for n in range(len(outputs_info)):
        if outputs_info[n] is not None:
            initializers.append(outputs_info[n])
        else:
            initializers.append(0. * inf_ret[n])
    def wrapwrap(nonepos, initializers):
        type_class = "list" if isinstance(initializers, list) else "tuple"
        def fnwrap(accs, inps):
            inps_then_accs = inps + [a for n, a in enumerate(accs) if n not in nonepos]
            fn_rets = fn(*inps_then_accs)
            return [fr for fr in fn_rets]
        return fnwrap
    this_fn = wrapwrap(nonepos, initializers)
    r = tf.scan(this_fn, sequences, initializers)
    return r


def Embedding(indices, n_symbols, output_dim, random_state=None,
              init="gaussian",
              strict=None, name=None):
    """
    Last dimension of indices tensor must be 1!!!!
    """
    if name is None:
        name = _get_name()

    if random_state is None:
        raise ValueError("Must pass random_state argument to Embedding")

    name_w = name + "_embedding_w"

    if strict is None:
        strict = get_strict_mode_default()

    if strict:
        cur_defs = get_params_dict()
        if name_w in cur_defs:
            raise ValueError("Name {} already created in params dict!".format(name_w))

    if init != "gaussian":
        raise ValueError("Currently unsupported init type {}".format(init))

    try:
        vectors = _get_shared(name_w)
    except NameError:
        logger.info("Linear layer {} initialized using init {}".format(name, init))
        vectors_weight = random_state.randn(n_symbols, output_dim).astype("float32")
        vectors = tf.Variable(vectors_weight, trainable=True)
        _set_shared(name_w, vectors)

    ii = tf.cast(indices, "int32")
    shp = shape(ii)
    nd = ndim(ii)
    if shp[-1] != 1:
        if nd < 3:
            logger.info("Embedding input should have last dimension 1, inferring dimension to 1, from shape {} to {}".format(shp, tuple(list(shp) + [1])))
            ii = tf.expand_dims(ii, axis=-1)
        else:
            raise ValueError("Embedding layer input must have last dimension 1 for input size > 3D, got {}".format(shp))

    shp = shape(ii)
    nd = len(shp)
    lu = tf.nn.embedding_lookup(vectors, ii)
    if nd == 3:
        lu = lu[:, :, 0]
    elif nd == 2:
        lu = lu[:, 0]
    else:
        raise ValueError("Input dimension not handled, Embedding input shape {} results in shape {}".format(shp, shape(lu)))
    return lu


def Linear(list_of_inputs, list_of_input_dims, output_dim, random_state=None,
           name=None, init=None, scale="default", biases=True, bias_offset=0.,
           strict=None):
    if random_state is None:
        raise ValueError("Must pass random_state to Linear")
    nd = ndim(list_of_inputs[0])
    input_var = tf.concat(list_of_inputs, axis=nd - 1)
    input_dim = sum(list_of_input_dims)
    if init is None or type(init) is str:
        logger.info("Linear layer {} initialized using init {}".format(name, init))
        weight_values, = make_numpy_weights(input_dim, [output_dim],
                                            random_state=random_state,
                                            init=init, scale=scale)
    else:
        # rely on announcement from parent class
        weight_values=init[0]

    if name is None:
        name = _get_name()

    name_w = name + "_linear_w"
    name_b = name + "_linear_b"
    name_out = name + "_linear_out"

    if strict is None:
        strict = get_strict_mode_default()

    if strict:
        cur_defs = get_params_dict()
        if name_w in cur_defs:
            raise ValueError("Name {} already created in params dict!".format(name_w))

        if name_b in cur_defs:
            raise ValueError("Name {} already created in params dict!".format(name_b))

    try:
        weight = _get_shared(name_w)
    except NameError:
        weight = tf.Variable(weight_values, trainable=True, name=name_w)
        _set_shared(name_w, weight)

    out = dot(input_var, weight)

    if biases:
        if (init is None) or (type(init) is str):
            b, = make_numpy_biases([output_dim])
        else:
            b = init[1]
        b = b + bias_offset
        try:
            biases = _get_shared(name_b)
        except NameError:
            biases = tf.Variable(b, trainable=True, name=name_b)
            _set_shared(name_b, biases)
        out = out + biases
    out = tf.identity(out, name=name_out)
    return out


def SimpleRNNCell(list_of_inputs, list_of_input_dims, previous_hidden,
                  num_units, output_dim, random_state=None,
                  name=None, init=None, scale="default", strict=None):
    # output is the thing to use in following layers, state is a tuple that contains things to feed into the next call
    if random_state is None:
        raise ValueError("Must pass random_state")

    if name is None:
        name = _get_name()
    hidden_dim = num_units
    inp_to_h = Linear(list_of_inputs, list_of_input_dims, hidden_dim, random_state=random_state,
                      name=name + "_simple_rnn_inp_to_h",
                      init=init, strict=strict)
    h_to_h = Linear([previous_hidden], [hidden_dim], hidden_dim, random_state=random_state,
                    name=name + "_simple_rnn_h_to_h", biases=False,
                    init=init, strict=strict)
    h = tf.nn.tanh(inp_to_h + h_to_h)
    h_to_out = Linear([h], [hidden_dim], output_dim, random_state=random_state,
                      name=name + "_simple_rnn_h_to_out",
                      init=init, strict=strict)
    return h_to_out, (h,)


def LSTMCell(list_of_inputs, list_of_input_dims,
             previous_hidden, previous_cell,
             num_units,
             output_dim=None,
             input_mask=None,
             random_state=None,
             name=None, init=None, scale="default",
             forget_bias=1., strict=None):
    # output is the thing to use in following layers, state is a tuple that feeds into the next call
    if random_state is None:
        raise ValueError("Must pass random_state")

    if name is None:
        name = _get_name()

    input_dim = sum(list_of_input_dims)
    hidden_dim = 4 * num_units

    if init is None or init == "truncated_normal":
        inp_init = "truncated_normal"
        h_init = "truncated_normal"
        out_init = "truncated_normal"
    elif init == "glorot_uniform":
        inp_init = "glorot_uniform"
        h_init = "glorot_uniform"
        out_init = "glorot_uniform"
    elif init == "normal":
        inp_init = "normal"
        h_init = "normal"
        out_init = "normal"
    else:
        raise ValueError("Unknown init argument {}".format(init))

    comb_w_np, = make_numpy_weights(input_dim + num_units, [hidden_dim],
                                    random_state=random_state,
                                    init=inp_init)
    comb_b_np, = make_numpy_biases([hidden_dim])

    logger.info("LSTMCell {} input to hidden initialized using init {}".format(name, inp_init))
    logger.info("LSTMCell {} hidden to hidden initialized using init {}".format(name, h_init))
    lstm_proj = Linear(list_of_inputs + [previous_hidden], list_of_input_dims + [hidden_dim],
                       hidden_dim,
                       random_state=random_state,
                       name=name + "_lstm_proj",
                       init=(comb_w_np, comb_b_np), strict=strict)

    i, j, f, o = tf.split(lstm_proj, 4, axis=-1)

    c = tf.sigmoid(f + forget_bias) * previous_cell + tf.sigmoid(i) * tf.tanh(j)
    if input_mask is not None:
        c = input_mask[:, None] * c + (1. - input_mask[:, None]) * previous_cell

    h = tf.sigmoid(o) * tf.tanh(c)
    if input_mask is not None:
        h = input_mask[:, None] * h + (1. - input_mask[:, None]) * h

    if output_dim is not None:
        h_to_out_w_np, = make_numpy_weights(num_units, [output_dim],
                                            random_state=random_state,
                                            init=out_init)
        h_to_out_b_np, = make_numpy_biases([output_dim])
        h_to_out = Linear([h], [num_units], output_dim, random_state=random_state,
                          name=name + "_lstm_h_to_out",
                          init=(h_to_out_w_np, h_to_out_b_np), strict=strict)
        final_out = h_to_out
        logger.info("LSTMCell {} hidden to output initialized using init {}".format(name, out_init))
    else:
        final_out = h
    return final_out, (h, c)


def GaussianAttentionCell(list_of_step_inputs, list_of_step_input_dims,
                          previous_state_list,
                          previous_attention_position,
                          full_conditioning_tensor,
                          full_conditioning_tensor_dim,
                          num_units,
                          previous_attention_weight,
                          att_dim=10,
                          attention_scale=1.,
                          cell_type="lstm",
                          name=None,
                          input_mask=None,
                          conditioning_mask=None,
                          random_state=None, strict=None, init=None):
    #returns w_t, k_t, phi_t, state
    # where state is the state tuple retruned by the inner cell_type

    if name is None:
        name = _get_name()
    name = name + "_gaussian_attention"

    check = any([len(shape(si)) != 2 for si in list_of_step_inputs])
    if check:
        raise ValueError("Unable to support step_input with n_dims != 2")

    if init is None or init == "truncated_normal":
        rnn_init = "truncated_normal"
        forward_init = "truncated_normal"
    else:
        raise ValueError("init != None not supported")

    if cell_type == "gru":
        raise ValueError("NYI")
    elif cell_type == "lstm":
        att_rnn_out, state = LSTMCell(list_of_step_inputs + [previous_attention_weight],
                                      list_of_step_input_dims + [full_conditioning_tensor_dim],
                                      previous_state_list[0], previous_state_list[1],
                                      num_units,
                                      input_mask=input_mask,
                                      random_state=random_state,
                                      name=name + "_gauss_att_lstm",
                                      init=rnn_init)
    else:
        raise ValueError("Unsupported cell_type %s" % cell_type)

    ret = Linear(
        list_of_inputs=[att_rnn_out], list_of_input_dims=[num_units],
        output_dim=3 * att_dim, name=name + "_group",
        random_state=random_state,
        strict=strict, init=forward_init)
    a_t = ret[:, :att_dim]
    b_t = ret[:, att_dim:2 * att_dim]
    k_t = ret[:, 2 * att_dim:]

    k_tm1 = previous_attention_position
    cond_dim = full_conditioning_tensor_dim
    ctx = full_conditioning_tensor
    ctx_mask = conditioning_mask

    """
    ctx = Linear(
        list_of_inputs=[full_conditioning_tensor],
        list_of_input_dims=[full_conditioning_tensor_dim],
        output_dim=next_proj_dim, name=name + "_proj_ctx",
        weight_norm=weight_norm,
        random_state=random_state,
        strict=strict, init=ctx_forward_init)
    """

    a_t = tf.exp(a_t)
    b_t = tf.exp(b_t)
    a_t = tf.identity(a_t, name=name + "_a_scale")
    b_t = tf.identity(b_t, name=name + "_b_scale")
    step_size = attention_scale * tf.exp(k_t)
    k_t = k_tm1 + step_size
    k_t = tf.identity(k_t, name=name + "_position")

    # tf.shape and tensor.shape are not the same...
    u = tf.cast(tf.range(0., limit=tf.shape(full_conditioning_tensor)[0], delta=1.), dtype=tf.float32)
    u = tf.expand_dims(tf.expand_dims(u, axis=0), axis=0)

    def calc_phi(lk_t, la_t, lb_t, lu):
        la_t = tf.expand_dims(la_t, axis=2)
        lb_t = tf.expand_dims(lb_t, axis=2)
        lk_t = tf.expand_dims(lk_t, axis=2)
        phi = tf.exp(-tf.square(lk_t - lu) * lb_t) * la_t
        phi = tf.reduce_sum(phi, axis=1, keep_dims=True)
        return phi

    phi_t = calc_phi(k_t, a_t, b_t, u)
    phi_t = tf.identity(phi_t, name=name + "_phi")

    """
        # Notes from pytorch tests
        # sanity check shapes for proper equivalent to np.dot
        aaaa = np.random.randn(50, 1, 46)
        bbbb = np.random.randn(50, 46, 400)
        r = np.matmul(aaaa, bbbb)
        # r has shape ms, 1, embed_dim
        # since aaaa and bbbb are > 2d, treated as stack of matrices, matrix dims on last 2 axes
        # this means 50, 1, 46 x 50, 46, 400 is 50 reps of 1, 46 x 46, 400
        # leaving shape 50, 1, 400
        # equivalent to dot for 1 matrix is is (aaaa[0][:, :, None] * bbbb[0][None, :, :]).sum(axis=-2)
        # so for all 50, (aaaa[:, :, :, None] * bbbb[:, None, :, :]).sum(axis=-2)
        # ((aaaa[:, :, :, None] * bbbb[:, None, :, :]).sum(axis=-2) == r).all()
        _a = Variable(th.FloatTensor(aaaa))
        _b = Variable(th.FloatTensor(bbbb))
        e_a = _a[:, :, :, None].expand(_a.size(0), _a.size(1), _a.size(2), _b.size(2))
        e_b = _b[:, None, :, :].expand(_b.size(0), _a.size(1), _b.size(1), _b.size(2))
        # In [17]: np.sum(((e_a * e_b).sum(dim=-2)[:, :, 0].data.numpy() - r) ** 2)
        # Out[17]: 1.6481219193765024e-08
        # equivalent to comb = th.matmul(phi, c), for backwards compat
        e_phi = phi[:, :, :, None].expand(phi.size(0), phi.size(1), phi.size(2), c.size(2))
        e_c = c[:, None, :, :].expand(c.size(0), phi.size(1), c.size(1), c.size(2))
        comb = (e_phi * e_c).sum(dim=-2)[:, :, 0]
        # comb has shape minibatch_size, 1, embed_size
        # w_t has shape minibatch_size, embed_size
        w_t = comb[:, 0, :]
    """
    if conditioning_mask is not None:
        w_t_pre = phi_t * tf.transpose(ctx, (1, 2, 0))
        w_t_masked = w_t_pre * (tf.transpose(ctx_mask, (1, 0))[:, None])
        w_t = tf.reduce_sum(w_t_masked, axis=-1)[:, None]
    else:
        w_t = tf.matmul(phi_t, tf.transpose(ctx, (1, 0, 2)))
    phi_t = phi_t[:, 0]
    w_t = w_t[:, 0]
    w_t = tf.identity(w_t, name=name + "_post_weighting")
    return w_t, k_t, phi_t, state


def LogitBernoulliAndCorrelatedLogitGMM(
        list_of_inputs, list_of_input_dims, output_dim=2, name=None, n_components=10,
        random_state=None, strict=None, init=None):
    """
    returns logit_bernoulli, logit_coeffs, mus, logit_sigmas, corr
    """
    assert n_components >= 1
    if name is None:
        name = _get_name()
    else:
        name = name + "_logit_bernoulli_and_correlated_logit_gaussian_mixture"


    def _reshape(l, d=n_components):
        if d == 1:
            shp = shape(l)
            t = tf.reshape(l, shp[:-1] + [1, shp[-1]])
            return t
        if len(shape(l)) == 2:
            t = tf.reshape(l, (-1, output_dim, d))
        elif len(shape(l)) == 3:
            shp = shape(l)
            t = tf.reshape(l, (-1, shp[1], output_dim, d))
        else:
            raise ValueError("input ndim not supported for gaussian "
                             "mixture layer")
        return t

    if output_dim != 2:
        raise ValueError("General calculation for GMM not yet implemented")

    mus = Linear(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        output_dim=n_components * output_dim, name=name + "_mus_pre",
        random_state=random_state,
        strict=strict, init=init)
    mus = _reshape(mus)
    mus = tf.identity(mus, name=name + "_mus")

    logit_sigmas = Linear(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        output_dim=n_components * output_dim, name=name + "_logit_sigmas_pre",
        random_state=random_state,
        strict=strict, init=init)
    logit_sigmas = _reshape(logit_sigmas)
    logit_sigmas = tf.identity(logit_sigmas, name=name + "_logit_sigmas")

    """
    coeffs = Linear(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        output_dim=n_components, name=name + "_coeffs_pre",
        weight_norm=weight_norm, random_state=random_state,
        strict=strict, init=init)
    coeffs = tf.nn.softmax(coeffs)
    coeffs = _reshape(coeffs, 1)
    coeffs = tf.identity(coeffs, name=name + "_coeffs")
    """
    logit_coeffs = Linear(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        output_dim=n_components, name=name + "_logit_coeffs_pre",
        random_state=random_state,
        strict=strict, init=init)
    logit_coeffs = _reshape(logit_coeffs, 1)
    logit_coeffs = tf.identity(logit_coeffs, name=name + "_logit_coeffs")

    calc_corr = int(factorial(output_dim ** 2 // 2 - 1))
    corrs = Linear(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        output_dim=n_components * calc_corr, name=name + "_corrs_pre",
        random_state=random_state,
        strict=strict, init=init)
    corrs = tf.tanh(corrs)
    corrs = _reshape(corrs, calc_corr)
    corrs = tf.identity(corrs, name + "_corrs")

    logit_bernoullis = Linear(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        output_dim=1, name=name + "_logit_bernoullis_pre",
        random_state=random_state,
        strict=strict, init=init)
    logit_bernoullis = tf.identity(logit_bernoullis, name + "_logit_bernoullis")
    return logit_bernoullis, logit_coeffs, mus, logit_sigmas, corrs


# from A d B
# https://github.com/adbrebs/handwriting/blob/master/model.py
def _logsumexp(inputs, axis=-1):
    max_i = tf.reduce_max(inputs, axis=axis)
    z = tf.log(tf.reduce_sum(tf.exp(inputs - max_i[..., None]), axis=axis)) + max_i
    return z


def LogitBernoulliAndCorrelatedLogitGMMCost(
    logit_bernoulli_values, logit_coeff_values, mu_values, logit_sigma_values, corr_values,
    true_values, name=None):
    """
    Logit bernoulli combined with correlated gaussian mixture model negative log
    likelihood compared to true_values.

    This is typically paired with LogitBernoulliAndCorrelatedLogitGMM

    Based on implementation from Junyoung Chung.

    Parameters
    ----------
    logit_bernoulli_values : tensor, shape
        The predicted values out of some layer, normallu a linear layer
    logit_coeff_values : tensor, shape
        The predicted values out of some layer, normally a linear layer
    mu_values : tensor, shape
        The predicted values out of some layer, normally a linear layer
    logit_sigma_values : tensor, shape
        The predicted values out of some layer, normally a linear layer
    true_values : tensor, shape[:-1]
        Ground truth values. Must be the same shape as mu_values.shape[:-1].
    Returns
    -------
    nll : tensor, shape predicted_values.shape[1:]
        The cost per sample, or per sample per step if 3D
    References
    ----------
    [1] University of Utah Lectures
        http://www.cs.utah.edu/~piyush/teaching/gmm.pdf
    [2] Statlect.com
        http://www.statlect.com/normal_distribution_maximum_likelihood.htm
    """
    if name == None:
        name = _get_name()
    else:
        name = name

    tv = true_values
    if len(shape(tv)) == 3:
        true_values = tf.expand_dims(tv, axis=2)
    elif len(shape(tv)) == 2:
        true_values = tf.expand_dims(tv, axis=1)
    else:
        raise ValueError("shape of labels not currently supported {}".format(shape(tv)))

    def _subslice(arr, idx):
        if len(shape(arr)) == 3:
            return arr[:, idx]
        elif len(shape(arr)) == 4:
            return arr[:, :, idx]
        raise ValueError("Unsupported ndim {}".format(shape(arr)))

    mu_values = tf.identity(mu_values, name=name + "_mus")
    mu_1 = _subslice(mu_values, 0)
    mu_2 = _subslice(mu_values, 1)
    corr_values = _subslice(corr_values, 0)
    corr_values = tf.identity(corr_values, name=name + "_corrs")

    sigma_values = tf.exp(logit_sigma_values) + 1E-12
    sigma_values = tf.identity(sigma_values, name=name + "_sigmas")

    sigma_1 = _subslice(sigma_values, 0)
    sigma_2 = _subslice(sigma_values, 1)

    bernoulli_values = tf.nn.sigmoid(logit_bernoulli_values)
    bernoulli_values = tf.identity(bernoulli_values, name=name + "_bernoullis")

    logit_coeff_values = _subslice(logit_coeff_values, 0)
    coeff_values = tf.nn.softmax(logit_coeff_values, dim=-1)
    coeff_values = tf.identity(coeff_values, name=name + "_coeffs")

    """
    logit_sigma_1 = _subslice(logit_sigma_values, 0)
    logit_sigma_2 = _subslice(logit_sigma_values, 1)
    logit_coeff_values = _subslice(logit_coeff_values, 0)
    """

    true_0 = true_values[..., 0]
    true_1 = true_values[..., 1]
    true_2 = true_values[..., 2]

    # don't be clever
    buff = (1. - tf.square(corr_values)) + 1E-6
    x_term = (true_1 - mu_1) / sigma_1
    y_term = (true_2 - mu_2) / sigma_2

    Z = tf.square(x_term) + tf.square(y_term) - 2. * corr_values * x_term * y_term
    N = 1. / (2. * np.pi * sigma_1 * sigma_2 * tf.sqrt(buff)) * tf.exp(-Z / (2. * buff))
    ep = tf.reduce_sum(true_0 * bernoulli_values + (1. - true_0) * (1. - bernoulli_values), axis=-1)
    rp = tf.reduce_sum(coeff_values * N, axis=-1)
    nll = -tf.log(rp + 1E-8) - tf.log(ep + 1E-8)

    """
    ll_b = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=true_0, logits=logit_bernoulli_values), axis=-1)
    ll_b = tf.identity(ll_b, name=name + "_binary_ll")

    buff = 1 - corr_values ** 2 + 1E-8
    inner1 = (0.5 * tf.log(buff) +
              logit_sigma_1 + logit_sigma_2 + tf.log(2 * np.pi))

    z1 = ((true_1 - mu_1) ** 2) / tf.exp(2 * logit_sigma_1)
    z2 = ((true_2 - mu_2) ** 2) / tf.exp(2 * logit_sigma_2)
    zr = (2 * corr_values * (true_1 - mu_1) * (true_2 - mu_2)) / (
        tf.exp(logit_sigma_1 + logit_sigma_2))
    z = z1 + z2 - zr

    inner2 = .5 * (1. / buff)
    ll_g = -(inner1 + z * inner2)
    ll_g = tf.identity(ll_g, name=name + "_gaussian_ll")

    ll_sm = tf.nn.log_softmax(logit_coeff_values, dim=-1)
    ll_sm = tf.identity(ll_sm, name=name + "_coeff_ll")

    nllp1 = -_logsumexp(ll_g + ll_sm,
                        axis=len(shape(logit_coeff_values)) - 1)
    nllp1 = tf.identity(nllp1, name=name + "_gmm_nll")

    nllp2 = - ll_b
    nllp2 = tf.identity(nllp2, name=name + "_b_nll")

    nll = nllp1 + nllp2
    nll = tf.identity(nll, name=name + "_full_nll")
    """
    return nll


def BernoulliAndCorrelatedGMMCost(
    bernoulli_values, coeff_values, mu_values_list, sigma_values_list,
    corr_values, true_values_bernoulli, true_values_coord_list, name=None):
    """
    Logit bernoulli combined with correlated gaussian mixture model negative log
    likelihood compared to true_values.

    This is typically paired with LogitBernoulliAndCorrelatedLogitGMM

    Based on implementation from Junyoung Chung.

    Parameters
    ----------
    bernoulli_values : tensor, shape
        The predicted values out of some layer, normally a sigmoid layer
    coeff_values : tensor, shape
        The predicted values out of some layer, normally a softmax layer
    mu_values_list: tensor, shape
        The predicted values out of some layer, normally a linear layer
    sigma_values_list: tensor, shape
        list of predicted values out of some layer, normally an exp or softplus layer
    corr_values: tensor, shape
    true_values_bernoulli : tensor, shape[:-1]
        Ground truth values. Must be the same shape as mu_values.shape[:-1],
        assumes the bernoulli true values are on the first entry ([:, :, 0])
    true_values_coords_list :
    Returns
    -------
    nll : tensor, shape predicted_values.shape[1:]
        The cost per sample, or per sample per step if 3D
    References
    ----------
    [1] University of Utah Lectures
        http://www.cs.utah.edu/~piyush/teaching/gmm.pdf
    [2] Statlect.com
        http://www.statlect.com/normal_distribution_maximum_likelihood.htm
    """
    if name == None:
        name = _get_name()
    else:
        name = name

    error_msg = "Dimension of variable {} not supported, got {}. Must be 2"
    if len(shape(true_values_bernoulli)) != 2:
        raise ValueError(error_msg.format("true_values_bernoulli", len(shape(true_values_bernoulli))))
    elif any([len(shape(tvc)) != 2 for tvc in true_values_coord_list]):
        raise ValueError(error_msg.format("true_values_coord_list", [len(shape(true_values_coord_list[0])), len(shape(truce_values_coord_list[1]))]))
    elif len(shape(bernoulli_values)) != 2:
        raise ValueError(error_msg.format("bernoulli_values", len(shape(bernoulli_values))))
    elif len(shape(coeff_values)) != 2:
        raise ValueError(error_msg.format("coeff_values", len(shape(coeff_values))))
    elif any([len(shape(m)) != 2 for m in mu_values_list]):
        raise ValueError(error_msg.format("mu_values", [len(shape(mu_values[0])), len(shape(mu_values_list[1]))]))
    elif any([len(shape(s)) != 2 for s in sigma_values_list]):
        raise ValueError(error_msg.format("sigma_values", [len(shape(sigma_values[0])), len(shape(sigma_values[1]))]))
    elif len(shape(corr_values)) != 2:
        raise ValueError(error_msg.format("corr_values", len(shape(corr_values))))

    mu_1 = mu_values_list[0]
    mu_1 = tf.identity(mu_1, name=name + "_mu_1")
    mu_2 = mu_values_list[1]
    mu_2 = tf.identity(mu_2, name=name + "_mu_2")

    corr_values = tf.identity(corr_values, name=name + "_corrs")

    sigma_1 = sigma_values_list[0]
    sigma_1 = tf.identity(sigma_1, name=name + "_sigma_1")
    sigma_2 = sigma_values_list[1]
    sigma_2 = tf.identity(sigma_2, name=name + "_sigma_2")

    bernoulli_values = tf.identity(bernoulli_values, name=name + "_bernoullis")
    coeff_values = tf.identity(coeff_values, name=name + "_coeffs")

    true_0 = true_values_bernoulli
    true_1 = true_values_coord_list[0]
    true_2 = true_values_coord_list[1]

    # don't be clever
    buff = (1. - tf.square(corr_values)) + 1E-6
    x_term = (true_1 - mu_1) / sigma_1
    y_term = (true_2 - mu_2) / sigma_2

    Z = tf.square(x_term) + tf.square(y_term) - 2. * corr_values * x_term * y_term
    N = 1. / (2. * np.pi * sigma_1 * sigma_2 * tf.sqrt(buff)) * tf.exp(-Z / (2. * buff))
    ep = true_0 * bernoulli_values + (1. - true_0) * (1. - bernoulli_values)
    rp = tf.reduce_sum(coeff_values * N, axis=-1)
    nll = -tf.log(rp + 1E-8) - tf.log(ep + 1E-8)
    return nll
