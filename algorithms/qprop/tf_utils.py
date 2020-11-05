import tensorflow as tf
import numpy as np

EPS = 1e-8  # avoid zero denominator


def gaussian_log_likelihood(x, mu, log_std):  # No EPS in baseline
    logp_per = -0.5 * (tf.square((x - mu)/(tf.exp(log_std))) + 2*log_std + np.log(2*np.pi))  # logp of per element
    return tf.reduce_sum(logp_per, axis=1)


def sample_average_kl(mu0, log_std0, mu1, log_std1):
    """
    Compute sample averaged KL divergence of between batches of diagonal gaussian distributed RVs
    :param mu0: mean_0
    :param log_std0: variance_0
    :param mu1: mean_1
    :param log_std1: variance_1
    :return: tensor mean KL divergence
    """
    var0, var1 = tf.exp(2 * log_std0), tf.exp(2 * log_std1)
    kl = tf.reduce_sum(0.5 * ((tf.square(mu1 - mu0) + var0) / (var1 + EPS) - 1) + log_std1 - log_std0,
                       axis=1)
    return tf.reduce_mean(kl)


def flat_concat(x_list):
    return tf.concat([tf.reshape(x, [-1]) for x in x_list], axis=0)


def flat_grad(f, param_list):
    return flat_concat(tf.gradients(ys=f, xs=param_list))


def hessian_vector_product(f, params_list):
    # Hx = grad(g^T*x)
    g = flat_grad(f, params_list)
    x = tf.placeholder(tf.float32, shape=g.shape)
    return x, flat_grad(tf.reduce_sum(g * x), params_list)


def assign_vars_from_flat(x, vars_list):
    """
    Get the operations for assigning x to vars_list
    :param x: a flattened vector
    :param vars_list: list of tensorflow variables to be assigned
    :return: operations for assigning the parameters
    """
    splitted_x = tf.split(x, [get_size(var) for var in vars_list])
    new_vars_list = [tf.reshape(new_var, var.shape) for var, new_var in zip(vars_list, splitted_x)]
    # Create an op that groups param assign operations
    return tf.group([tf.assign(var, new_var) for var, new_var in zip(vars_list, new_vars_list)])


def get_size(var):
    return int(np.prod(var.shape.as_list()))