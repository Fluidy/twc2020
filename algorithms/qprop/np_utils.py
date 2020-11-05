import numpy as np


def conjugate_gradient_alg(Hx, g, iters=10, residual_tol=1e-10):
    """
    Conjugate gradient algorithm
    :param Hx: the function to compute the value of Hx, input: batch of data
    :param g: gradient of pi loss
    :param iters: number of iterations
    :param residual_tol: residual tolerance
    :return: x = H^(-1)*g
    """

    x = np.zeros_like(g)
    r = g.copy()
    p = r.copy()
    r_dot_old = np.dot(r, r)
    for _ in range(iters):
        z = Hx(p)
        alpha = r_dot_old / (np.dot(p, z))
        x += alpha * p
        r -= alpha * z
        r_dot_new = np.dot(r, r)
        p = r + (r_dot_new / r_dot_old) * p
        r_dot_old = r_dot_new
        if r_dot_old < residual_tol:
            break
    return x


def iter_batches(arrays, *, num_batches=None, batch_size=None, shuffle=True, include_final_partial_batch=True):
    """
    Mini-batch generator
    """
    assert (num_batches is None) != (batch_size is None), 'Provide num_batches or batch_size, but not both'
    arrays = tuple(map(np.asarray, arrays))
    n = arrays[0].shape[0]
    assert all(a.shape[0] == n for a in arrays[1:])
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    sections = np.arange(0, n, batch_size)[1:] if num_batches is None else num_batches
    for batch_idx in np.array_split(indices, sections):
        if include_final_partial_batch or len(batch_idx) == batch_size:
            yield tuple(a[batch_idx] for a in arrays)


