#!/usr/bin/env python3
# vim: sw=2
import os
import numpy as np
import better_exchook
better_exchook.install()
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from termcolor import colored
from ref_transduce import forward_pass
from ref_transduce import transduce as transduce_ref

NEG_INF = -float("inf")
def logsumexp(*args):  # summation in linear space -> LSE in log-space
    """
    Stable log sum exp.
    """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = np.log(sum(np.exp(a - a_max)
                    for a in args))
    return a_max + lsp

def log_softmax(acts, axis=None):
    assert axis is not None
    a = acts - np.max(acts, axis=axis, keepdims=True)  # normalize for stability
    probs = np.sum(np.exp(a), axis=axis, keepdims=True)
    log_probs = a - np.log(probs)
    return log_probs

def print_alpha_matrix(a):
    assert len(a.shape) == 2
    print("Alpha matrix: (%d, %d)" % tuple(a.shape))
    print(a)


### PYTHON IMPLEMENTATION ###
def numpy_forward(log_probs, labels, blank_index, debug=False):
    n_time, n_target, _ = log_probs.shape
    alpha = np.zeros((n_time, n_target))  # 1 in log-space
    print("U=%d, T=%d, V=%d" % (n_target, n_time, n_vocab))
    for t in range(1, n_time):  # first row
        alpha[t, 0] = alpha[t-1, 0] + log_probs[t-1, 0, blank_index]
    for u in range(1, n_target):  # first column
        alpha[0, u] = alpha[0, u-1] + log_probs[0, u-1, labels[u-1]]
        
    for t in range(1,n_time):
        for u in range(1,n_target):
            print("t=%02d, u=%02d: " % (t, u))
            a = alpha[t-1, u] + log_probs[t-1, u, blank_index]
            b = alpha[t, u-1] + log_probs[t, u-1, labels[u-1]]
            alpha[t,u] = logsumexp(a,b)  # addition in linear-space -> LSE in log-space
    if debug:
        print_alpha_matrix(alpha)
    log_posterior = alpha[-1,-1] + log_probs[-1,-1, blank_index]
    return alpha, log_posterior

### TENSORFLOW IMPLEMENTATION ###
def tf_forward(log_probs_input, labels_input, blank_index, debug=False, sess=None):
    n_time, n_target, n_vocab = log_probs_input.shape
    assert len(labels_input.shape) == 1
    # labels = tf.cast(tf.convert_to_tensor(labels_input), tf.int32)
    # log_probs = tf.convert_to_tensor(log_probs_input)
    labels = tf.compat.v1.placeholder(tf.int32, [None])
    log_probs = tf.compat.v1.placeholder(tf.float32, [n_time, n_target, n_vocab])

    if debug:
        print("U=%d, T=%d, V=%d" % (n_target, n_time, n_vocab))
    # we actually only need alpha[0,0]==0 (in log-space, which is ==1 in linear-space)
    alpha = tf.zeros((n_time, n_target))  # in log-space


    # precompute first col (blank)
    update_col = tf.cumsum(log_probs[:, 0, blank_index], exclusive=True)
    # stitch together the alpha matrix by concat [precomputed_row ; old_rows]
    alpha = tf.concat(values=[tf.expand_dims(update_col, axis=1), alpha[:,1:]], axis=1)
    if debug:
        print("after precomputing first col")
        print_alpha_matrix(alpha)
        print()

    # precompute first row (y)
    idxs_w_labels = tf.stack([tf.tile([0], [n_target-1]), tf.range(n_target-1), labels], axis=-1)
    log_probs_y = tf.gather_nd(log_probs, idxs_w_labels)
    update_row = tf.concat([[0], tf.cumsum(log_probs_y, exclusive=False)], axis=0)
    alpha = tf.concat([tf.expand_dims(update_row, axis=0), alpha[1:,:]], axis=0)
    if debug:
        print("row", update_row)
        print("after precomputing first row")
        print_alpha_matrix(alpha)

    def cond(n, alpha):
        return tf.less(n, tf.reduce_max([n_time, n_target]))
    
    def body_forward(n, alpha):
        # alpha(t-1,u) + logprobs(t-1, u)
        # alpha_blank      + lp_blank
    
        if debug:
            print("Iteration n=%d" % n)
        # we index a diagonal, starting at the t=0, u=1
        # for n=1: [[0,1]]
        # for n=2: [[0,2], [1,1]]
        # for n=3: [[0,3], [1,2], [2,1]]
        idxs = tf.stack([tf.range(0,n), n-tf.range(n)], axis=-1)
        idxs = tf.cond(n > n_time-1, lambda: idxs[:n_time-1], lambda: idxs)
        if debug:
            print("Idxs(blank):", idxs)
        
        # a_blank: all the elements one behind in time-dimension
        #idxs = tf.print(idxs, [idxs], "idxs")
        alpha_blank = tf.gather_nd(alpha, idxs)  # (N+1,)
        # we select the log-probs for blank from 2d tensor (T, U)
        # we can reuse the same index tensor.
        lp_blank = tf.gather_nd(log_probs[:,:,blank_index], idxs)
        if debug:
            print("alpha_blank", alpha_blank)
            print("lp_blank", lp_blank)    
        
        # 
        # alpha(t,u-1) + logprobs(t, u-1)
        # alpha_y      + lp_y
        # we index a diagonal, starting at t=1, u=0
        # for n=1: [[1,0]]
        # for n=2: [[1,1], [2,0]]
        # for n=3: [[1,2], [2,1], [3,0]]
        # plus we append the labels indices
        idxs = tf.stack([tf.range(n)+1, n-tf.range(n)-1], axis=-1)
        idxs = tf.cond(n > n_time-1, lambda: idxs[:n_time-1], lambda: idxs)
        if debug:
            print("Idxs(y):", idxs)
        
        # for the labels, we need:
        # for n=1: [labels[0]]
        # for n=2: [labels[1], labels[0]]
        # for n=3: [labels[2], labels[1], labels[0]]
        rev_labels = tf.cast(tf.reverse(labels[:n], axis=[0]), dtype=tf.int32)
        idxs_w_labels = tf.stack([tf.range(n)+1, n-tf.range(n)-1, rev_labels], axis=-1)
        idxs_w_labels = tf.cond(n > n_time-1, lambda: idxs_w_labels[:n_time-1], lambda: idxs_w_labels)
        if debug:
            print("Idxs(y_labels):", idxs_w_labels)
        alpha_y = tf.gather_nd(alpha, idxs)  # (N-1,)
        lp_y = tf.gather_nd(log_probs, idxs_w_labels)
        if debug:
            print("alpha_y", alpha_y)
            print("lp_y", lp_y)
    
        # for the new diagonal (alphas to update)
        if debug:
            print("n=%d: LSE(" % n, ["%.3f" % v.numpy()[0] for v in [alpha_blank,lp_blank, alpha_y,lp_y]], ")")
        red_op = tf.stack([alpha_blank+lp_blank, alpha_y+lp_y], axis=0)
        new_alphas = tf.math.reduce_logsumexp(red_op, axis=0)
        
        if debug:
            print("new_alphas", new_alphas)
        
        # diagonal to update,
        # n=1: [[1,1]]
        # n=2: [[1,2], [2,1]]
        idxs = tf.stack([tf.range(n)+1, n-tf.range(n)], axis=-1)
        idxs = tf.cond(n > n_time-1, lambda: idxs[:n_time-1], lambda: idxs)
        if debug:
            print("Idxs(update):", idxs)
        alpha = tf.tensor_scatter_nd_update(alpha, idxs, new_alphas)
        
        if debug:
            print()
            print_alpha_matrix(alpha)
            print("\n")
        
        n += 1
        return [n, alpha]
    
    
    n = tf.constant(1)  # we compute the first row +col beforehand
    final_n, final_alpha = tf.while_loop(cond, body_forward, [n, alpha], name="rnnt")
    # p(y|x) = alpha(T,U) * blank(T,U)  (--> in log-space)
    ll_tf = final_alpha[n_time-1, n_target-1] + log_probs[n_time-1, n_target-1, blank_index]

    init_tf = tf.compat.v1.initializers.global_variables()
    gradients_tf = tf.gradients(xs=log_probs, ys=[-ll_tf])
    sess.run(init_tf)
    alpha_tf, n, ll_tf, gradients_tf = sess.run([final_alpha, final_n, ll_tf, gradients_tf],
            feed_dict={log_probs: log_probs_input,
                labels: labels_input})
    return alpha_tf, ll_tf, gradients_tf


# TODO:
# test with random activations
# test with other labels
# test with other lengths (input/label)
if __name__ == '__main__':
    sess = tf.compat.v1.Session()
    n_time = 2
    n_target = 3
    n_vocab = 5
    blank_index = 0
    acts = np.array([0.1, 0.6, 0.1, 0.1, 0.1, 0.1,
                     0.1, 0.6, 0.1, 0.1, 0.1, 0.1,
                     0.2, 0.8, 0.1, 0.1, 0.6, 0.1,
                     0.1, 0.1, 0.1, 0.1, 0.2, 0.1,
                     0.1, 0.7, 0.1, 0.2, 0.1, 0.1], dtype=np.float32).reshape(n_time, n_target, n_vocab)

    labels = np.array([1,2])
    log_probs = log_softmax(acts, axis=2)  # along vocabulary

    alpha_ref, ll_ref = forward_pass(log_probs, labels, blank=blank_index)
    ll_ref, grads_ref = transduce_ref(log_probs, labels, blank=blank_index)
    ll_ref = -ll_ref
    print(colored("Reference", "red"), "implementation:")
    print("log posterior:", ll_ref)
    print_alpha_matrix(alpha_ref)
    print(grads_ref)

    print("\n")

    alpha_np, ll_np = numpy_forward(log_probs, labels, blank_index=blank_index, debug=True)
    print(colored("NumPy", "red"), "implementation:")
    print("log posterior:", ll_np)
    print_alpha_matrix(alpha_np)
    print()
    np.testing.assert_allclose(alpha_np, alpha_ref)
    print("numpy vs ref: alpha matrices", colored("MATCH", "green"))
    np.testing.assert_almost_equal(ll_np, ll_ref, decimal=6)
    print("numpy vs ref: log posterior", colored("MATCH", "green"))

    print("\n")


    with sess.as_default():
        alpha_tf, ll_tf, gradients_tf = tf_forward(log_probs, labels, blank_index=blank_index, debug=False, sess=sess)
    print(colored("TensorFlow", "red"), "implementation:")
    print("log posterior:", ll_tf)
    print_alpha_matrix(alpha_tf)
    print(gradients_tf[0])

    np.testing.assert_allclose(alpha_tf, alpha_ref)
    print("TF vs ref: alpha matrices", colored("MATCH", "green"))
    np.testing.assert_almost_equal(ll_tf, ll_ref, decimal=6)
    print("TF vs ref: log posterior", colored("MATCH", "green"))

