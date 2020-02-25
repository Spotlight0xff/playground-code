#!/usr/bin/env python3
# vim: sw=2
"""
Implementation of the RNN-T loss in pure TF,
plus comparisons against reference implementations.
"""
import os
import sys
import numpy as np
import better_exchook
import tensorflow as tf
from termcolor import colored
from ref_transduce import forward_pass, transduce_batch
from ref_transduce import transduce as transduce_ref

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# better_exchook.install()
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
  """computes log(softmax(x, axis)) in a numerical stable way."""
  assert axis is not None
  a = acts - np.max(acts, axis=axis, keepdims=True)  # normalize for stability
  probs = np.sum(np.exp(a), axis=axis, keepdims=True)
  log_probs = a - np.log(probs)
  return log_probs


def py_print_iteration_info(msg, var, n, debug=True):
  """adds a tf.print op to the graph while ensuring it will run (when the output is used)."""
  if not debug:
    return var
  var_print = tf.print("n=", n, "\t", msg, tf.shape(var), var,
                       summarize=-1, output_stream=sys.stdout)
  with tf.control_dependencies([var_print]):
    var = tf.identity(var)
  return var


def numpy_forward(log_probs, labels, blank_index, debug=False):
  """Forward calculation of the RNN-T loss."""
  n_time, n_target, n_vocab = log_probs.shape
  alpha = np.zeros((n_time, n_target))  # 1 in log-space
  print("a = alpha[t-1, u] + log_probs[t - 1, u, blank_index]")
  print("b = alpha[t, u - 1] + log_probs[t, u - 1, labels[u - 1]]")
  print("alpha[t,u] = LSE(a,b)")
  if debug:
    print("U=%d, T=%d, V=%d" % (n_target, n_time, n_vocab))
  for t in range(1, n_time):  # first row
    alpha[t, 0] = alpha[t - 1, 0] + log_probs[t - 1, 0, blank_index]
    print('t=%2d u= 0: alpha[%d, 0] + log_probs[%d, 0, %d] = %.3f + %.3f = %.3f' % (
      t, t-1, t-1, blank_index, alpha[t - 1, 0], log_probs[t - 1, 0, blank_index], alpha[t, 0]))
  for u in range(1, n_target):  # first column
    alpha[0, u] = alpha[0, u - 1] + log_probs[0, u - 1, labels[u - 1]]
    print('t= 0 u=%2d: alpha[0, %d] + log_probs[0, %d, labels[%d]=%d] = %.3f + %.3f = %.3f' % (
      u, u-1, u-1, u-1, labels[u - 1], alpha[0, u - 1], log_probs[0, u - 1, labels[u - 1]], alpha[0, u]))

  for t in range(1, n_time):
    for u in range(1, n_target):
      a = alpha[t - 1, u] + log_probs[t - 1, u, blank_index]
      b = alpha[t, u - 1] + log_probs[t, u - 1, labels[u - 1]]
      alpha[t, u] = elem = logsumexp(a, b)  # addition in linear-space -> LSE in log-space
      print('t=%2d u=%2d: LSE(%.3f + %.3f, %.3f +  %.3f) = LSE(%.3f, %.3f) = %.3f' % (t, u,
                                                                                      alpha[t - 1, u],
                                                                                      log_probs[t - 1, u, blank_index],
                                                                                      alpha[t, u - 1],
                                                                                      log_probs[
                                                                                        t, u - 1, labels[u - 1]],
                                                                                      a, b, elem))

  if debug:
    assert len(alpha.shape) == 2
    print("Alpha matrix: (%d, %d)" % tuple(alpha.shape))
    print(alpha)
  log_posterior = alpha[n_time-1, n_target-1] + log_probs[n_time-1, n_target-1, blank_index]
  if debug:
    print("log-posterior = alpha[%d, %d] + log_probs[%d, %d, %d] = %.3f + %.3f = %.4f" %
          (n_time-1, n_target-1, n_time-1, n_target-1, blank_index,
           alpha[n_time-1, n_target-1],
           log_probs[n_time - 1, n_target - 1, blank_index],
           log_posterior
           ))
  return alpha, log_posterior


def tf_forward_old(log_probs_input, labels_input, blank_index, debug=False, sess=None):
  """Pure TF implementation of the RNN-T loss."""
  n_time, n_target, n_vocab = log_probs_input.shape
  assert len(labels_input.shape) == 1
  labels = tf.compat.v1.placeholder(tf.int32, [None])
  log_probs = tf.compat.v1.placeholder(tf.float32, [n_time, n_target, n_vocab])

  if debug:
    print("U=%d, T=%d, V=%d" % (n_target, n_time, n_vocab))
  # we actually only need alpha[0,0]==0 (in log-space, which is ==1 in linear-space)
  alpha = tf.zeros((n_time, n_target))  # in log-space

  # precompute first col (blank)
  update_col = tf.cumsum(log_probs[:, 0, blank_index], exclusive=True)
  # stitch together the alpha matrix by concat [precomputed_row ; old_rows]
  alpha = tf.concat(values=[tf.expand_dims(update_col, axis=1), alpha[:, 1:]], axis=1)

  # precompute first row (y)
  idxs_w_labels = tf.stack([tf.tile([0], [n_target - 1]), tf.range(n_target - 1), labels], axis=-1)
  log_probs_y = tf.gather_nd(log_probs, idxs_w_labels)
  update_row = tf.concat([[0], tf.cumsum(log_probs_y, exclusive=False)], axis=0)
  alpha = tf.concat([tf.expand_dims(update_row, axis=0), alpha[1:, :]], axis=0)
  alpha = py_print_iteration_info("after precomputation, alpha:\n", alpha, -1, debug=debug)

  def cond(n, alpha):
    return tf.less(n, tf.reduce_max([n_time + 1, n_target]))

  def body_forward(n, alpha):
    # alpha(t-1,u) + logprobs(t-1, u)
    # alpha_blank      + lp_blank

    # we index a diagonal, starting at the t=0, u=1
    # for n=1: [[0,1]]
    # for n=2: [[0,2], [1,1]]
    # for n=3: [[0,3], [1,2], [2,1]]
    idxs = tf.stack([tf.range(0, n), n - tf.range(n)], axis=-1)
    idxs = tf.cond(n > n_time - 1, lambda: idxs[:n_time - 1], lambda: idxs)
    idxs = tf.cond(n > n_target - 1, lambda: idxs[n - n_target + 1:], lambda: idxs)
    idxs = py_print_iteration_info("Idxs(blank)", idxs, n, debug=debug)

    alpha_blank = tf.gather_nd(alpha, idxs)  # (N+1,)
    # we select the log-probs for blank from 2d tensor (T, U)
    # we can reuse the same index tensor.
    lp_blank = tf.gather_nd(log_probs[:, :, blank_index], idxs)

    #
    # alpha(t,u-1) + logprobs(t, u-1)
    # alpha_y      + lp_y
    # we index a diagonal, starting at t=1, u=0
    # for n=1: [[1,0]]
    # for n=2: [[1,1], [2,0]]
    # for n=3: [[1,2], [2,1], [3,0]]
    # plus we append the labels indices
    idxs = tf.stack([tf.range(n) + 1, n - tf.range(n) - 1], axis=-1)
    idxs = tf.cond(n > n_time - 1, lambda: idxs[:n_time - 1], lambda: idxs)
    idxs = tf.cond(n > n_target - 1, lambda: idxs[n - n_target + 1:], lambda: idxs)
    idxs = py_print_iteration_info("Idxs(y)", idxs, n, debug=debug)

    # for the labels, we need:
    # for n=1: [labels[0]]
    # for n=2: [labels[1], labels[0]]
    # for n=3: [labels[2], labels[1], labels[0]]
    rev_labels = tf.cast(tf.reverse(labels[:n], axis=[0]), dtype=tf.int32)

    # n=2: in the case where n > U, we have to cut of the reversed label sequence
    rev_labels = tf.cond(n > n_time - 1, lambda: rev_labels[:n_time - 1], lambda: rev_labels)
    rev_labels = tf.cond(n > n_target, lambda: rev_labels[:n - n_target], lambda: rev_labels)
    rev_labels = py_print_iteration_info("rev_labels", rev_labels, n, debug=debug)

    # idxs_w_labels = tf.stack([tf.range(n)+1, n-tf.range(n)-1, rev_labels], axis=-1)
    idxs_w_labels = tf.stack(tf.unstack(idxs, axis=-1) + [rev_labels], axis=-1)
    # idxs_w_labels = tf.cond(n > n_time-1, lambda: idxs_w_labels[:n_time-1], lambda: idxs_w_labels)
    # idxs_w_labels = tf.cond(n > n_target-1, lambda: idxs_w_labels[n - n_target + 1:], lambda: idxs_w_labels)
    idxs_w_labels = py_print_iteration_info("Idxs(w/labels)", idxs_w_labels, n, debug=debug)
    alpha_y = tf.gather_nd(alpha, idxs)  # (N-1,)
    lp_y = tf.gather_nd(log_probs, idxs_w_labels)

    # for the new diagonal (alphas to update)
    # if debug:
    #    print("n=%d: LSE(" % n, ["%.3f" % v.numpy()[0] for v in [alpha_blank,lp_blank, alpha_y,lp_y]], ")")
    red_op = tf.stack([alpha_blank + lp_blank, alpha_y + lp_y], axis=0)
    new_alphas = tf.math.reduce_logsumexp(red_op, axis=0)

    # diagonal to update,
    # n=1: [[1,1]]
    # n=2: [[1,2], [2,1]]
    idxs = tf.stack([tf.range(n) + 1, n - tf.range(n)], axis=-1)
    idxs = tf.cond(n > n_time - 1, lambda: idxs[:n_time - 1], lambda: idxs)
    idxs = tf.cond(n > n_target - 1, lambda: idxs[n - n_target + 1:], lambda: idxs)
    idxs = py_print_iteration_info("Idxs(update)", idxs, n, debug=debug)

    alpha = tf.tensor_scatter_nd_update(alpha, idxs, new_alphas)

    n += 1
    return [n, alpha]

  n = tf.constant(1)  # we compute the first row +col beforehand
  final_n, final_alpha = tf.while_loop(cond, body_forward, [n, alpha], name="rnnt")
  # p(y|x) = alpha(T,U) * blank(T,U)  (--> in log-space)
  ll_tf = final_alpha[n_time - 1, n_target - 1] + log_probs[n_time - 1, n_target - 1, blank_index]

  init_tf = tf.compat.v1.initializers.global_variables()
  gradients_tf = tf.gradients(xs=log_probs, ys=[-ll_tf])

  sess.run(init_tf)
  alpha_tf, n, ll_tf, gradients_tf = sess.run([final_alpha, final_n, ll_tf, gradients_tf],
                                              feed_dict={log_probs: log_probs_input,
                                                         labels: labels_input})
  return alpha_tf, ll_tf, gradients_tf


def tf_forward(log_probs_input, labels_input, blank_index, debug=False, sess=None):
  """Pure TF implementation of the RNN-T loss."""
  n_time, n_target, n_vocab = log_probs_input.shape
  assert len(labels_input.shape) == 1
  labels = tf.compat.v1.placeholder(tf.int32, [None])
  log_probs = tf.compat.v1.placeholder(tf.float32, [n_time, n_target, n_vocab])

  if debug:
    print("U=%d, T=%d, V=%d" % (n_target, n_time, n_vocab))
  # we actually only need alpha[0,0]==0 (in log-space, which is ==1 in linear-space)
  # alpha = tf.zeros((n_time, n_target))  # in log-space

  # precompute first col (blank)
  precomputed_col = tf.concat([[0], tf.cumsum(log_probs[:, 0, blank_index], exclusive=False)[:-1]], axis=0)
  precomputed_col = py_print_iteration_info("precomputed_col", precomputed_col, 0, debug=debug)
  # stitch together the alpha matrix by concat [precomputed_row ; old_rows]
  # alpha = tf.concat(values=[tf.expand_dims(update_col, axis=1), alpha[:,1:]], axis=1)

  # precompute first row (y)
  idxs_w_labels = tf.stack([tf.tile([0], [n_target - 1]), tf.range(n_target - 1), labels], axis=-1)
  log_probs_y = tf.gather_nd(log_probs, idxs_w_labels)
  # precomputed_row = tf.cumsum(log_probs_y, exclusive=False)
  precomputed_row = tf.concat([[0], tf.cumsum(log_probs_y, exclusive=False)], axis=0)
  precomputed_row = py_print_iteration_info("precomputed_row", precomputed_row, 0, debug=debug)

  # alpha = tf.concat([tf.expand_dims(update_row, axis=0), alpha[1:,:]], axis=0)
  # alpha = py_print_iteration_info("after precomputation, alpha:\n", alpha, -1, debug=debug)

  num_diagonals = n_time + n_target - 2
  alpha_ta = tf.TensorArray(
    dtype=tf.float32,
    clear_after_read=False,  # TODO, check, correct?
    size=num_diagonals,
    dynamic_size=True,
    infer_shape=False,
    element_shape=(None,),
    name="alpha_diagonals",
  )
  alpha_ta = alpha_ta.write(0, [precomputed_row[1], precomputed_col[1]])
  # we start iteration at n=1, so for n=0 we have two (precomputed) items as the diagonal

  def cond(n, *args):
    """We run the loop until all elements are covered by diagonals.
    Minus two, because we start at n=1 and we have precomputed values already.
    """
    return tf.less(n, num_diagonals)

  def body_forward(n, alpha_ta):
    """body of the while_loop, loops over the diagonals of the alpha-matrix."""
    # alpha(t-1,u) + logprobs(t-1, u)
    # alpha_blank      + lp_blank

    # we index a diagonal, starting at the t=0, u=1
    # for n=1: [[0,1]]
    # for n=2: [[0,2], [1,1]]
    # for n=3: [[0,3], [1,2], [2,1]]
    idxs = tf.stack([tf.range(0, n), n - tf.range(n)], axis=-1)
    idxs = tf.cond(n > n_time - 1, lambda: idxs[:n_time - 1], lambda: idxs)
    idxs = tf.cond(n > n_target - 1, lambda: idxs[n - n_target + 1:], lambda: idxs)
    idxs = py_print_iteration_info("Idxs(blank)", idxs, n, debug=debug)

    prev_diagonal = alpha_ta.read(n - 1)
    # prev_diagonal = tf.cond(n > n_target-1,
    #                        lambda: prev_diagonal,
    #                        lambda: tf.concat([[precomputed_row[n-1]], prev_diagonal], axis=-1))
    prev_diagonal = py_print_iteration_info("prev_diagonal", prev_diagonal, n, debug=debug)

    # we add the precomputed element in-front, but only when N <= U
    alpha_blank = prev_diagonal[:-1]
    #alpha_blank = tf.cond(n > n_target-1,
    #                      lambda: prev_diagonal,
    #                      lambda: tf.concat([[precomputed_row[n-1]], prev_diagonal], axis=-1))
    alpha_blank = py_print_iteration_info("alpha(blank)", alpha_blank, n, debug=debug)
    # alpha_blank = tf.gather_nd(alpha, idxs)  # (N+1,)
    # we select the log-probs for blank from 2d tensor (T, U)
    # we can reuse the same index tensor.
    # TODO: this will be inefficient during backprop, change/determine impact?
    lp_blank = tf.gather_nd(log_probs[:, :, blank_index], idxs)

    #
    # alpha(t,u-1) + logprobs(t, u-1)
    # alpha_y      + lp_y
    # we index a diagonal, starting at t=1, u=0
    # for n=1: [[1,0]]
    # for n=2: [[1,1], [2,0]]
    # for n=3: [[1,2], [2,1], [3,0]]
    # plus we append the labels indices
    idxs = tf.stack([tf.range(1, n+1), n - tf.range(n)-1], axis=-1)
    idxs = tf.cond(n > n_time - 1, lambda: idxs[:n_time - 1], lambda: idxs)
    idxs = tf.cond(n > n_target - 1, lambda: idxs[n - n_target + 1:], lambda: idxs)
    idxs = py_print_iteration_info("Idxs(y)", idxs, n, debug=debug)

    # for the labels, we need:
    # for n=1: [labels[0]]
    # for n=2: [labels[1], labels[0]]
    # for n=3: [labels[2], labels[1], labels[0]]
    # rev_labels = tf.cast(tf.reverse(labels[:n], axis=[0]), dtype=tf.int32)

    # phase (a): diagonals are "growing"
    phase_a_end = tf.cond(tf.greater(n_target, n_time),
                          lambda: n_time-1,
                          lambda: tf.add(n_target, -1))
    # phase (b): diagonals are "stagnant" (maybe phase_a_end==phase_b_end)
    phase_b_end = tf.cond(tf.greater(n_target, n_time),
                          lambda: n_target - 1,
                          lambda: n_time - 1)
    slice_limits = tf.case([
      (  # phase (a), both T>U and U>T
        n <= phase_a_end,
        lambda: (0, n)
      ), (  # phase (b), T >= U
        tf.logical_and(tf.logical_and(n <= phase_b_end, n_time >= n_target), n > phase_a_end),
        lambda: (0, n_target-1)  # full target axis, for U=3 we have labels=[1,2]
      ), (  # phase (b), U > T
        tf.logical_and(tf.logical_and(n <= phase_b_end, n_target > n_time), n > phase_a_end),
        lambda: (n-n_time+1, n)
      ), (  # phase (c), T >= U
        tf.logical_and(n > phase_b_end, n_time >= n_target),
        lambda: (n-n_time+1, n_target-1)
      ), (  # phase (c), U > T
        tf.logical_and(n > phase_b_end, n_target > n_time),
        lambda: (n-n_time+1, n_target-1)
      )
    ], exclusive=True)
    slice_limits = py_print_iteration_info("slice_limits", slice_limits, n, debug=debug)
    rev_labels = tf.cast(tf.reverse(labels[slice_limits[0]:slice_limits[1]], axis=[0]), dtype=tf.int32)
    rev_labels = py_print_iteration_info("rev_labels", rev_labels, n, debug=debug)

    idxs_w_labels = tf.stack([tf.range(n) + 1, n - tf.range(n) - 1, rev_labels], axis=-1)
    idxs_w_labels = tf.stack(tf.unstack(idxs, axis=-1) + [rev_labels], axis=-1)
    idxs_w_labels = tf.cond(n > n_time - 1, lambda: idxs_w_labels[:n_time - 1], lambda: idxs_w_labels)
    # idxs_w_labels = tf.cond(n > n_target - 1, lambda: idxs_w_labels[n - n_target + 1:], lambda: idxs_w_labels)
    idxs_w_labels = py_print_iteration_info("Idxs(w/labels)", idxs_w_labels, n, debug=debug)
    # alpha_y = tf.gather_nd(alpha, idxs)  # (N-1,)

    # TODO: fix for batched case
    # we add the precomputed element last, but only when N <= U
    alpha_y = prev_diagonal[1:]
    alpha_y = py_print_iteration_info("alpha(y)", alpha_y, n, debug=debug)

    # we need the first few items, then concat -neg-inf to
    # not perform the addition for the last item (first column)
    lp_y = tf.gather_nd(log_probs, idxs_w_labels)

    # for the new diagonal (alphas to update)
    # if debug:
    #    print("n=%d: LSE(" % n, ["%.3f" % v.numpy()[0] for v in [alpha_blank,lp_blank, alpha_y,lp_y]], ")")
    red_op = tf.stack([alpha_blank + lp_blank, alpha_y + lp_y], axis=0)
    # NOTE: this does not include the precomputed values (first row/column)
    red_op = py_print_iteration_info("red-op", red_op, n, debug=debug)
    new_diagonal = tf.math.reduce_logsumexp(red_op, axis=0)

    # add the precomputed values
    new_diagonal = tf.cond(n < n_target-1,
                           lambda: tf.concat([[precomputed_row[n + 1]], new_diagonal], axis=-1),
                           lambda: new_diagonal
                           )
    new_diagonal = tf.cond(n < n_time-1,
                           lambda: tf.concat([new_diagonal, [precomputed_col[n+1]]], axis=-1),
                           lambda: new_diagonal
                           )
    new_diagonal = py_print_iteration_info("new_diagonal", new_diagonal, n, debug=debug)

    # diagonal to update,
    # n=1: [[1,1]]
    # n=2: [[1,2], [2,1]]
    # idxs = tf.stack([tf.range(n)+1, n-tf.range(n)], axis=-1)
    # idxs = tf.cond(n > n_time-1, lambda: idxs[:n_time-1], lambda: idxs)
    # idxs = tf.cond(n > n_target - 1, lambda: idxs[n - n_target + 1:], lambda: idxs)
    # idxs = py_print_iteration_info("Idxs(update)", idxs, n, debug=debug)
    return [n + 1, alpha_ta.write(n, new_diagonal)]

  n = tf.constant(1)
  final_n, alpha_out_ta = tf.while_loop(cond, body_forward, [n, alpha_ta],
                                        parallel_iterations=1,  # need this due to the iterative computation using TAs
                                        name="rnnt")

  # stitch together the alpha matrix and compute the gradients manually.

  # p(y|x) = alpha(T,U) * blank(T,U)  (--> in log-space)
  # ll_tf = final_alpha[n_time-1, n_target-1] + log_probs[n_time-1, n_target-1, blank_index]
  ll_tf = alpha_out_ta.read(alpha_out_ta.size()-1)[0] + log_probs[n_time-1, n_target-1, blank_index]
  # this is not needed for the final implementation,
  # but for debugging we can stitch together the alpha matrix from the diagonals
  def build_alpha_matrix(alpha_diagonals):
    alpha_matrix = tf.zeros((n_time, n_target))
    return alpha_matrix
  final_alpha = build_alpha_matrix(alpha_out_ta)
  init_tf = tf.compat.v1.initializers.global_variables()
  gradients_tf = tf.gradients(xs=log_probs, ys=[-ll_tf])
  # gradients_tf = tf.zeros_like(log_probs)  # TODO
  # ll_tf = tf.constant(0)  # TODO DEBUG
  sess.run(init_tf)
  alpha_tf, n, ll_tf, gradients_tf = sess.run([final_alpha, final_n, ll_tf, gradients_tf],
                                              feed_dict={log_probs: log_probs_input,
                                                         labels: labels_input})
  return alpha_tf, ll_tf, gradients_tf


def tf_forward_batched(log_probs_input, labels_input, input_lengths, label_lengths, blank_index, debug=False,
                       sess=None, timing=False):
  """
  Computes the batched forward pass of the RNN-T model.
  TODO: We also compute the gradients of the backpropagation step.
  B: batch, T: time, U:target/labels, V: vocabulary

  :param tf.Tensor log_probs_input: (B, T, U, V) log-probabilities
  :param tf.Tensor labels_input: (B, V) labels
  :param input_lengths: (B,) length of input frames
  :param label_lengths: (B,) length of labels
  :param int blank_index: index of the blank symbol in the vocabulary
  :param bool debug: enable verbose logging
  :param tf.Session sess:
  :return:
  """
  """Pure TF implementation of the RNN-T loss."""
  from TFUtil import expand_dims_unbroadcast
  n_batch, max_time, max_target, n_vocab = log_probs_input.shape
  assert len(labels_input.shape) == 2  # (B, U)
  labels = tf.compat.v1.placeholder(tf.int32, [None, None])
  log_probs = tf.compat.v1.placeholder(tf.float32, [None, max_time, max_target, n_vocab])
  input_lens = tf.tile([max_time], [n_batch])
  label_lens = tf.tile([max_target-1], [n_batch])
  if debug:
    print("B=%d, T=%d, U=%d, V=%d" % (n_batch, max_time, max_target, n_vocab))
    labels = py_print_iteration_info("labels", labels, 0, debug=debug)

  # precompute first col (blank)
  # (B, 1) ; (B, T-1)
  precomputed_col = tf.concat([tf.zeros((n_batch, 1)),
                               tf.cumsum(log_probs[:, :, 0, 0], exclusive=False, axis=1)[:, :-1]], axis=1)
  precomputed_col = py_print_iteration_info("precomputed_col", precomputed_col, 0, debug=debug)

  # precompute first row (y)
  # TODO: check when labels length differ
  a = expand_dims_unbroadcast(tf.range(n_batch), axis=1, dim=max_target - 1)  # (B,U-1)
  b = expand_dims_unbroadcast(tf.range(max_target - 1), axis=0, dim=n_batch)  # (B, U-1)
  c = labels  # (B, U-1)
  indices_w_labels = tf.stack([a, b, c], axis=-1)  # (B, U-1, 3)
  # log_probs[:,0,:,:]: (B, U, V)
  log_probs_y = tf.gather_nd(log_probs[:, 0, :, :], indices_w_labels)  # (B, U-1)
  precomputed_row = tf.concat([
    tf.tile([[0.]], [n_batch, 1]),
    tf.cumsum(log_probs_y, exclusive=False, axis=1)
  ], axis=1)  # (B, U)

  # 3 = 1 for precomputation, 1 for starting at n=1, 1 for label lengths+1==U
  num_diagonals = max_time + max_target - 3
  alpha_ta = tf.TensorArray(
    dtype=tf.float32,
    clear_after_read=False,  # TODO, check, correct?
    size=num_diagonals,
    dynamic_size=False,
    infer_shape=False,
    element_shape=(None, None,),
    name="alpha_diagonals",
  )
  # (B,1)
  alpha_ta = alpha_ta.write(0, tf.stack([precomputed_row[:, 1], precomputed_col[:, 1]], axis=-1))
  # we start iteration at n=1, so for n=0 we have two (precomputed) items as the diagonal

  def select_diagonal_batched(n, input_lens=None, label_lens=None):
    """
    Helper function to index various diagonals in a 2D matrix, which can be non-square.
    One diagonal starts from the top-right and goes down to the bottom-left.
    `n=1` indices (with start_row=0, start_col=0):
    [[0,0]]
    `n`=2:
    [[0,1], [1,0]]
    `n`=3:
    [[0,2], [1,1], [2,0]]

    :param n: specifies the diagonal to select
    :param tf.Tensor input_lens:
    :param tf.Tensor label_lens:
    :return: (B, N') tensor of indices
    :rtype: tf.Tensor
    """

    def build_mask(start, slice_len, n_max, n_batch):
      """
      Builds a binary mask for [start,start+len)
      :param n_max: scalar
      :param n_batch: scalar
      :param start:  (B,)
      :param slice_len:  (B,)
      :return:
      """
      range_mat = tf.expand_dims(tf.tile([0], [n_batch]), axis=1) + tf.expand_dims(tf.range(n_max), axis=0)
      mask = tf.where(tf.logical_and(tf.greater_equal(range_mat, tf.expand_dims(start, axis=1)),  # (B, 1)
                                     tf.less(range_mat,  tf.expand_dims(start + slice_len, axis=1))),
                      tf.ones_like(range_mat),
                      tf.zeros_like(range_mat)
                      )  # (B, N)
      mask = tf.cast(mask, tf.bool)
      return mask

    from TFUtil import expand_dims_unbroadcast
    n_tiled = tf.tile([n], [n_batch])  # (B,)
    input_lens = py_print_iteration_info("input_lens", input_lens, n, debug=debug)
    diff_t_u = tf.abs(input_lens - label_lens)  # (B,)
    min_t_u = tf.minimum(input_lens, label_lens+1)  # (B,)

    batch_idxs = expand_dims_unbroadcast(tf.range(n_batch), 1, n)  # (B, N)
    batch_idxs = tf.reshape(batch_idxs, (-1,))  # (B*N,)
    indices = tf.stack([
      batch_idxs,
      tf.tile(tf.range(0, n), [n_batch]),
      tf.tile(n - tf.range(n) - 1, [n_batch]),
    ], axis=-1)  # (N*B, 3)

    # reshape, so that we have for each batch each item in the diag
    indices = tf.reshape(indices, [n_batch, n, 3])  # (B, N, 3)

    # mask for phase (b)
    cond_b = tf.logical_and(
      tf.greater_equal(n_tiled, min_t_u),
      tf.less(n_tiled, min_t_u + diff_t_u))
    idxs_len_b = tf.where(cond_b, min_t_u, n_tiled)
    idxs_start_b = tf.where(tf.logical_and(cond_b, tf.greater(input_lens, label_lens)),
                            n_tiled - (label_lens+1),  # T > U
                            tf.zeros_like(n_tiled),  # T < U
                            )
    # set the correct start when we are in phase (b), otherwise no mask
    idxs_start_b = tf.where(cond_b,
                            idxs_start_b,
                            tf.zeros_like(n_tiled))

    idxs_mask_b = build_mask(idxs_start_b, idxs_len_b, n_max=n, n_batch=n_batch)
    idxs_mask_b = py_print_iteration_info("mask_b", idxs_mask_b, n, debug=debug)

    # mask for phase (c)
    cond_c = tf.greater_equal(n_tiled, min_t_u + diff_t_u)
    idxs_len_c = tf.where(cond_c,
                          input_lens + (label_lens) - n_tiled,  # phase (c)
                          n_tiled)  # default-case
    idxs_len_c = py_print_iteration_info("idxs_len_c", idxs_len_c, n, debug=debug)
    idxs_start_c = tf.where(tf.logical_and(cond_c, tf.greater(label_lens,  input_lens)),
                            n_tiled - label_lens,  # U > T
                            n_tiled - min_t_u)  # T > U

    idxs_start_c = tf.where(cond_c,
                            idxs_start_c,
                            tf.zeros_like(n_tiled))  # (B,)
    idxs_start_c = py_print_iteration_info("idxs_start_c", idxs_start_c, n, debug=debug)
    idxs_mask_c = build_mask(idxs_start_c, idxs_len_c, n_max=n, n_batch=n_batch)
    idxs_mask_c = py_print_iteration_info("mask_c", idxs_mask_c, n, debug=debug)

    indices = py_print_iteration_info("indices pre-mask", indices, n, debug=debug)
    mask = tf.logical_and(idxs_mask_b, idxs_mask_c)
    mask = py_print_iteration_info("mask", mask, n, debug=debug)
    indices_masked = tf.boolean_mask(indices, mask)

    indices_masked = tf.reshape(indices_masked, [n_batch, -1, 3])
    indices_masked = py_print_iteration_info("indices post-mask", indices_masked, n, debug=debug)
    return indices_masked

  def cond(n, *args):
    """We run the loop until all elements are covered by diagonals.
    Minus two, because we start at n=1 and we have precomputed values already.
    """
    return tf.less(n, num_diagonals)

  def body_forward(n, alpha_ta):
    """body of the while_loop, loops over the diagonals of the alpha-tensor."""
    # alpha(t-1,u) + logprobs(t-1, u)
    # alpha_blank      + lp_blank

    diag_idxs = select_diagonal_batched(n + 1, input_lens, label_lens)  # (B, N', 3)

    # we index a diagonal for each , starting at the t=0, u=1
    # for n=1: [[0,1]]
    # for n=2: [[0,2], [1,1]]
    # for n=3: [[0,3], [1,2], [2,1]]
    idxs_blank = diag_idxs[:, :-1, :]  # (B, N', 3)
    idxs_blank = tf.reshape(idxs_blank, (-1, 3))  # (B*N', 3)
    idxs_blank = py_print_iteration_info("Idxs(blank)", idxs_blank, n, debug=debug)

    prev_diagonal = alpha_ta.read(n - 1)  # (B, N+2)
    prev_diagonal = py_print_iteration_info("prev_diagonal", prev_diagonal, n, debug=debug)

    # we add the precomputed element in-front, but only when N <= U
    alpha_blank = prev_diagonal[:, :-1]  # (B, N+1)
    alpha_blank = py_print_iteration_info("alpha(blank)", alpha_blank, n, debug=debug)
    # we select the log-probs for blank from 2d tensor (T, U)
    # we can reuse the same index tensor.
    # TODO: this may be inefficient during backprop, change?
    lp_blank = tf.gather_nd(log_probs[:, :, :, blank_index], idxs_blank)
    lp_blank = tf.reshape(lp_blank, (n_batch, -1))
    lp_blank = py_print_iteration_info("lp(blank)", lp_blank, n, debug=debug)


    #
    # alpha(t,u-1) + logprobs(t, u-1)
    # alpha_y      + lp_y
    # we index a diagonal, starting at t=1, u=0
    # for n=1: [[1,0]]
    # for n=2: [[1,1], [2,0]]
    # for n=3: [[1,2], [2,1], [3,0]]
    # plus we append the labels indices

    idxs_y = diag_idxs[:, 1:, :]  # (B, N', 3)
    idxs_labels = tf.stack([idxs_y[..., 0], idxs_y[..., 2]], axis=-1)  # (B, N', 2)
    idxs_labels = py_print_iteration_info("Idxs(labels)", idxs_labels, n, debug=debug)  # (B*N', 3)
    idxs_y = tf.reshape(idxs_y, (-1, 3))  # (B*N', 3)
    idxs_labels = tf.reshape(idxs_labels, (-1, 2))  # (B*N', 2)
    rev_labels = tf.gather_nd(labels, idxs_labels)  # (B*N',)
    rev_labels = py_print_iteration_info("rev_labels", rev_labels, n, debug=debug)
    idxs_y = py_print_iteration_info("Idxs(y)", idxs_y, n, debug=debug)  # (B*N', 3)


    # (B*N', 3) ; (B*N', 1) -> (B*N', 4)
    idxs_w_labels = tf.stack(tf.unstack(idxs_y, axis=-1) + [rev_labels], axis=-1)
    idxs_w_labels = py_print_iteration_info("Idxs(w/labels)", idxs_w_labels, n, debug=debug)

    # we add the precomputed element last, but only when N <= U
    alpha_y = prev_diagonal[:, 1:]  # (B, N)
    alpha_y = py_print_iteration_info("alpha(y)", alpha_y, n, debug=debug)

    # we need the first few items, then concat -neg-inf to
    # not perform the addition for the last item (first column)
    lp_y = tf.gather_nd(log_probs, idxs_w_labels)  # (B*N',)
    lp_y = tf.reshape(lp_y, [n_batch, -1])
    lp_y = py_print_iteration_info("lp(y)", lp_y, n, debug=debug)


    # alpha_blank: (B, N)
    # lp_blank: (B, N)
    # alpha_y: (B, N)
    # lp_y: (B, N)
    red_op = tf.stack([alpha_blank + lp_blank, alpha_y + lp_y], axis=0)  # (2, B, N)
    # NOTE: this does not include the precomputed values (first row/column)
    red_op = py_print_iteration_info("red-op", red_op, n, debug=debug)
    new_diagonal = tf.math.reduce_logsumexp(red_op, axis=0)  # (B, N+1)
    new_diagonal = py_print_iteration_info("computed", new_diagonal, n, debug=debug)

    # add the precomputed values
    new_diagonal = tf.cond(tf.less(n, max_target-1),
                           lambda: tf.concat([tf.expand_dims(precomputed_row[:, n + 1], axis=1), new_diagonal], axis=1),
                           lambda: new_diagonal
                           )
    new_diagonal = tf.cond(tf.less(n, max_time-1),
                           lambda: tf.concat([new_diagonal, tf.expand_dims(precomputed_col[:, n + 1], axis=1)], axis=1),
                           lambda: new_diagonal)
    #

    new_diagonal = py_print_iteration_info("new_diagonal", new_diagonal, n, debug=debug)

    # diagonal to update,
    # n=1: [[1,1]]
    # n=2: [[1,2], [2,1]]
    # idxs = tf.stack([tf.range(n)+1, n-tf.range(n)], axis=-1)
    # idxs = tf.cond(n > n_time-1, lambda: idxs[:n_time-1], lambda: idxs)
    # idxs = tf.cond(n > n_target - 1, lambda: idxs[n - n_target + 1:], lambda: idxs)
    # idxs = py_print_iteration_info("Idxs(update)", idxs, n, debug=debug)
    return [n + 1, alpha_ta.write(n, new_diagonal)]

  n = tf.constant(1)
  final_n, alpha_out_ta = tf.while_loop(cond, body_forward, [n, alpha_ta],
                                        parallel_iterations=1,  # need this due to the iterative computation using TAs
                                        name="rnnt")

  # stitch together the alpha matrix and compute the gradients manually.

  # p(y|x) = alpha(T,U) * blank(T,U)  (--> in log-space)
  # ll_tf = final_alpha[n_time-1, n_target-1] + log_probs[n_time-1, n_target-1, blank_index]

  # (B,): batch index -> diagonal index
  diag_idxs = tf.constant(input_lengths + label_lengths - 2)  # (B,)
  #diag_idxs = py_print_iteration_info("diag_idxs", diag_idxs, n, debug=debug)
  # diag_idxs = tf.map_fn(lambda i: input_lengths[i] + label_lengths[i] - 2, tf.range(n_batch))
  res_ta = tf.TensorArray(
    dtype=tf.float32,
    clear_after_read=False,  # TODO, check, correct?
    size=n_batch,
    dynamic_size=False,
    infer_shape=False,
    element_shape=(),
    name="alpha_diagonals",
  )
  within_diag_idx = tf.where(
    tf.greater_equal(tf.tile([max_target], [n_batch]), tf.tile([max_time], [n_batch])),
                             max_time - tf.constant(input_lengths),  # U >= T
                             max_target - tf.constant(label_lengths)-1,  # U < T
                             )  # (B,)
  # within_diag_idx = tf.constant([4, 4, 4, 3, 5, 2, 5, 3, 5, 5, 0])
  def ta_read_body(i, res_ta):
    ta_item = alpha_out_ta.read(diag_idxs[i])[i]

    # TODO: check logic
    var_print = tf.print("FINAL", "\t", "i=", i, "T=", tf.constant(max_time), "U=", tf.constant(max_target),
                         "T'=", tf.constant(input_lengths)[i],
                         "U'=", tf.constant(label_lengths)[i],
                         "diag_idx=", diag_idxs[i],
                         "within_diag_idx=", within_diag_idx[i],
                         "size(alpha)", tf.shape(ta_item), ta_item,
                         summarize=-1, output_stream=sys.stdout)

    with tf.control_dependencies([var_print]):
      ta_item = tf.identity(ta_item)
    # TODO: we may have some final value in the middle of the diagonal
    return i+1, res_ta.write(i, ta_item[within_diag_idx[i]])  # TODO this may be wrong, check logic.

  i, a_ta = tf.while_loop(
    lambda i, res_ta: tf.less(i, n_batch),
    ta_read_body, (tf.constant(0, tf.int32), res_ta)
  )
  a = a_ta.stack()
  indices = tf.stack([
    tf.range(n_batch),
    input_lengths-1,
    label_lengths,
    tf.tile([blank_index], [n_batch]),
  ], axis=-1)  # (N, 3)
  indices = py_print_iteration_info("indices", indices, n, debug=debug)
  b = tf.gather_nd(log_probs, indices)
  a = py_print_iteration_info("a", a, n, debug=debug)
  b = py_print_iteration_info("b", b, n, debug=debug)
  ll_tf = a + b
  # this is not needed for the final implementation,
  # but for debugging we can stitch together the alpha matrix from the diagonals
  def build_alpha_matrix(alpha_diagonals):
    alpha_matrix = tf.zeros((n_batch, max_time, max_target))
    return alpha_matrix
  final_alpha = build_alpha_matrix(alpha_out_ta)
  init_tf = tf.compat.v1.initializers.global_variables()
  gradients_tf = tf.gradients(xs=log_probs, ys=[-ll_tf])[0]

  sess.run(init_tf)

  tf_run_opts = {}
  if timing:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    tf_run_opts = {'options': run_options, 'run_metadata': run_metadata}
  alpha_tf, n, ll_tf, gradients_tf = sess.run([final_alpha, final_n, ll_tf, gradients_tf],
                                              feed_dict={log_probs: log_probs_input,
                                                         labels: labels_input}, **tf_run_opts)
  if timing:
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    from tensorflow.python.client import timeline
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline_tf_ímpl.json', 'w') as f:
      f.write(ctf)
  return alpha_tf, ll_tf, gradients_tf


def tf_forward_shifted(log_probs_np, labels_np, input_lengths_np=None, label_lengths_np=None, blank_index=0, debug=False,
                       sess=None, timing=False):
  """
  Computes the batched forward pass of the RNN-T model.
  B: batch, T: time, U:target/labels, V: vocabulary

  :param np.ndarray log_probs_np: (B, T, U, V) log-probabilities
  :param np.ndarray labels_np: (B, V) labels
  :param np.ndarray input_lengths_np: (B,) length of input frames
  :param np.ndarray label_lengths_np: (B,) length of labels
  :param int blank_index: index of the blank symbol in the vocabulary
  :param bool debug: enable verbose logging
  :param tf.Session sess:
  :return:
  """
  """Pure TF implementation of the RNN-T loss."""
  n_batch, max_time, max_target, n_vocab = log_probs_np.shape
  if debug:
    print("B=%d, T=%d, U=%d, V=%d" % (n_batch, max_time, max_target, n_vocab))
  assert len(labels_np.shape) == 2  # (B, U)

  labels = tf.compat.v1.placeholder(tf.int32, [None, None])
  log_probs = tf.compat.v1.placeholder(tf.float32, [None, max_time, max_target, n_vocab])
  input_lengths = tf.compat.v1.placeholder(tf.int32, [None])
  label_lengths = tf.compat.v1.placeholder(tf.int32, [None])

  def shift_logprobs(mat, axis, axis_to_expand):
    """
    Shifts the log-probs per-batch row-wise.

    :param mat: (B, U, T, V)
    :param axis:
    :param axis_to_expand:
    :return: (B, T+U+1, U, V)
    """
    # mat: (B, T, U, V)
    # axis_to_expand: usually U
    # axis: usually T
    # batch-axis has to be first
    dim_axis = tf.shape(mat)[axis]  # T
    n_batch = tf.shape(mat)[0]
    n_vocab = tf.shape(mat)[-1]
    shifts = tf.expand_dims(tf.cast(tf.range(dim_axis), tf.float32), axis=1)  # (T,1)
    shifts = shifts[tf.newaxis, :, :, tf.newaxis]
    shifts = tf.tile(shifts, [n_batch, 1, 1, n_vocab])
    pads = tf.zeros((n_batch, dim_axis, dim_axis, n_vocab), dtype=tf.float32)
    # (B, T, 1, V) ; (B, T, U, V) ; (B, T, T, V)
    # -> (B, T, U+T+1, V)
    a_ranged = tf.concat([shifts, mat, pads], axis=axis_to_expand)  # (T, U+1)
    U = tf.shape(mat)[axis_to_expand]
    T = dim_axis

    def fn(x):  # x: (B, U+T+1, V)
      shift = tf.cast(x[0][0][0], tf.int32)  # (B,)
      # 1:U+1 is the original data, in front: shift as wanted, back: padding for shape
      n = tf.pad(x[:, 1:U + 1, :], [[0, 0],  # B
                                    [shift, T + 1 - shift],  # U+T+1
                                    [0, 0]  # V
                                    ])
      return n

    t = tf.map_fn(fn, elems=tf.transpose(a_ranged, [1, 0, 2, 3]))
    t = tf.transpose(t, [1, 0, 2, 3])
    # TODO: maybe cut off the last dim (t[:,:,:-1,:], only padding?)s
    return t

  log_probs_tr = tf.transpose(log_probs, [0, 2, 1, 3])  # (B, T, U, V) -> (B, U, T, V)
  log_probs_shifted = shift_logprobs(log_probs_tr, axis=1, axis_to_expand=2)  # (B, U+T+1, U, V)


  def shift_matrix_2d(mat, n_time):
    mat = tf.convert_to_tensor(mat)
    shape = tf.shape(mat)
    mat = tf.expand_dims(mat, axis=-1)  # (B, U, 1)
    mat = tf.tile(mat, [1, 1, n_time])  # (B, U, T)
    # batch, rows
    idxs_b, idxs_rows, idxs_cols = tf.meshgrid(
        tf.range(shape[0]),  # (B,)
        tf.range(shape[1]),  # (U,)
        tf.range(n_time),    # (T,)
        indexing='ij')
    shifts = tf.range(shape[1])  # (T,)
    # (B, U, T) + (1, U, 1)
    idxs_cols = idxs_cols + shifts[tf.newaxis, :, tf.newaxis]
    idxs = tf.stack([idxs_b, idxs_rows, idxs_cols], axis=-1)

    # (B, U, T+U)
    new_shape = [shape[0], shape[1], shape[1] + n_time]
    # idxs: (B, U, U+T, 3)
    scat_mat = tf.scatter_nd(indices=idxs, updates=mat,
                             shape=new_shape)
    return scat_mat

  labels_in = py_print_iteration_info("labels", labels, 0, debug=debug)
  labels_shifted_mat = shift_matrix_2d(labels_in, n_time=max_time)  # (B, U, T+U)
  labels_shifted_mat = tf.transpose(labels_shifted_mat, (2, 0, 1))
  labels_shifted_mat = py_print_iteration_info("labels_shifted_mat", labels_shifted_mat, 0, debug=debug)

  # 2 = 1 for starting at n=1, 1 for label lengths+1==U
  num_diagonals = max_time + max_target - 2

  labels_shifted_ta = tf.TensorArray(
    dtype=tf.int32,
    clear_after_read=False,
    size=max_time + max_target,
    dynamic_size=True,
    infer_shape=False,
    element_shape=(n_batch, max_target-1),  # (B,)
    name="labels_shifted",
  )
  labels_shifted_ta = labels_shifted_ta.unstack(labels_shifted_mat)

  log_probs_ta = tf.TensorArray(
    dtype=tf.float32,
    clear_after_read=False,
    size=max_time + max_target + 1,
    dynamic_size=False,
    infer_shape=False,
    element_shape=(None, None, None),
    name="log_probs_shifted",
  )
  # (B, U+T+1, U, V) -> [(B, U, V)] * (U+T+1)
  log_probs_ta = log_probs_ta.unstack(tf.transpose(log_probs_shifted, [2, 0, 1, 3]))

  alpha_ta = tf.TensorArray(
    dtype=tf.float32,
    clear_after_read=False,  # TODO, check, correct?
    size=num_diagonals,
    dynamic_size=False,
    infer_shape=False,
    element_shape=(n_batch, None,),  # (B, n)
    name="alpha_diagonals",
  )
  alpha_ta = alpha_ta.write(1, tf.zeros((n_batch, 2)))

  def cond(n, *args):
    """We run the loop until all elements are covered by diagonals.
    Minus two, because we start at n=1 and we have precomputed values already.
    """
    return tf.less(n, num_diagonals)

  def body_forward(n, alpha_ta):
    """body of the while_loop, loops over the diagonals of the alpha-tensor."""
    # alpha(t-1,u) + logprobs(t-1, u)
    # alpha_blank      + lp_blank

    prev_max_diag_len = tf.minimum(max_target-1, n-1)
    prev_diagonal = alpha_ta.read(n-1)[:, :prev_max_diag_len]  # (B, n-1)
    prev_diagonal = py_print_iteration_info("prev_diagonal", prev_diagonal, n, debug=debug)
    lp_diagonal = log_probs_ta.read(n-2)[:, :prev_max_diag_len, :]  # (B, U|n, V)
    lp_diagonal = py_print_iteration_info("lp_diagonal", lp_diagonal, n, debug=debug)

    alpha_blank = prev_diagonal  # (B, N)
    alpha_blank = tf.concat([alpha_blank, tf.tile([[tf.constant(NEG_INF)]], [n_batch, 1])], axis=1)
    alpha_blank = py_print_iteration_info("alpha(blank)", alpha_blank, n, debug=debug)

    # (B, U, V) -> (B, U)
    lp_blank = lp_diagonal[:, :, blank_index]  # (B, U)
    lp_blank = tf.concat([lp_blank, tf.tile([[tf.constant(NEG_INF)]], [n_batch, 1])], axis=1)
    lp_blank = py_print_iteration_info("lp(blank)", lp_blank, n, debug=debug)

    # (B,N-1) ; (B,1) ->  (B, N)
    alpha_y = prev_diagonal
    alpha_y = tf.concat([tf.tile([[tf.constant(NEG_INF)]], [n_batch, 1]), alpha_y], axis=1)
    alpha_y = py_print_iteration_info("alpha(y)", alpha_y, n, debug=debug)

    labels_maxlen = tf.minimum(max_target-1, n-1)
    labels_shifted = labels_shifted_ta.read(n-1)
    labels_shifted = labels_shifted[:, :labels_maxlen]  # (B,U)
    labels_shifted = py_print_iteration_info("labels_shifted", labels_shifted, n, debug=debug)
    B, R = tf.meshgrid(
      tf.range(tf.shape(labels_shifted)[0]),
      tf.range(tf.shape(labels_shifted)[1]),
      indexing='ij'
    )
    lp_y_idxs = tf.stack([B, R, labels_shifted], axis=-1)  # (B, V, 3)
    lp_y_idxs = py_print_iteration_info("lp_y_idxs", lp_y_idxs, n, debug=debug)
    lp_y = tf.gather_nd(lp_diagonal[:, :, :], lp_y_idxs)  # (B, U)
    # (B, U) ; (B, 1) -> (B, U+1)
    lp_y = tf.concat([tf.tile([[tf.constant(NEG_INF)]], [n_batch, 1]), lp_y], axis=1)
    lp_y = py_print_iteration_info("lp(y)", lp_y, n, debug=debug)


    # all should have shape (B, n)
    blank = alpha_blank + lp_blank
    y = alpha_y + lp_y
    red_op = tf.stack([blank, y], axis=0)  # (2, B, N)
    red_op = py_print_iteration_info("red-op", red_op, n, debug=debug)
    new_diagonal = tf.math.reduce_logsumexp(red_op, axis=0)  # (B, N)

    new_diagonal = new_diagonal[:, :n]
    new_diagonal = py_print_iteration_info("new_diagonal", new_diagonal, n, debug=debug)
    return [n + 1, alpha_ta.write(n, new_diagonal)]

  n = tf.constant(2)
  final_n, alpha_out_ta = tf.while_loop(cond, body_forward, [n, alpha_ta],
                                        parallel_iterations=1,  # need this due to the iterative computation using TAs
                                        name="rnnt")

  # stitch together the alpha matrix and compute the gradients manually.

  # p(y|x) = alpha(T,U) * blank(T,U)  (--> in log-space)
  # ll_tf = final_alpha[n_time-1, n_target-1] + log_probs[n_time-1, n_target-1, blank_index]

  # (B,): batch index -> diagonal index
  diag_idxs = input_lengths + label_lengths  # (B,)
  res_ta = tf.TensorArray(
    dtype=tf.float32,
    clear_after_read=False,  # TODO, check, correct?
    size=n_batch,
    dynamic_size=False,
    infer_shape=False,
    element_shape=(),
    name="alpha_diagonals",
  )
  # (B,): batch index -> index within diagonal
  # We need to handle the U>T case for each example.
  within_diag_idx = tf.minimum(input_lengths, label_lengths)
  within_diag_idx = tf.where(tf.greater(label_lengths, input_lengths),
                             label_lengths,
                             within_diag_idx)

  def ta_read_body(i, res_ta):
    ta_item = alpha_out_ta.read(diag_idxs[i])[i]

    var_print = tf.print("FINAL", "\t", "i=", i, "T=", tf.constant(max_time), "U=", tf.constant(max_target),
                         "T'=", input_lengths[i],
                         "U'=", label_lengths[i],
                         "diag_idx=", diag_idxs[i],
                         "within_diag_idx=", within_diag_idx[i],
                         "size(alpha)", tf.shape(ta_item), ta_item,
                         summarize=-1, output_stream=sys.stdout)

    with tf.control_dependencies([var_print]):
      ta_item = tf.identity(ta_item)
    return i+1, res_ta.write(i, ta_item[within_diag_idx[i]])

  i, a_ta = tf.while_loop(
    lambda i, res_ta: i < n_batch,
    ta_read_body, (tf.constant(0, tf.int32), res_ta)
  )
  indices = tf.stack([
    tf.range(n_batch),
    input_lengths-1,
    label_lengths,
    tf.tile([blank_index], [n_batch]),
  ], axis=-1)  # (N, 3)
  ll_tf = a_ta.stack() + tf.gather_nd(log_probs, indices)

  # this is not needed for the final implementation,
  # but for debugging we can stitch together the alpha matrix from the diagonals
  def build_alpha_matrix(alpha_diagonals):
    alpha_matrix = tf.zeros((n_batch, max_time, max_target))
    return alpha_matrix
  final_alpha = build_alpha_matrix(alpha_out_ta)
  init_tf = tf.compat.v1.initializers.global_variables()
  gradients_tf = tf.gradients(xs=log_probs, ys=[-ll_tf])[0]

  sess.run(init_tf)

  tf_run_opts = {}
  if timing:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    tf_run_opts = {'options': run_options, 'run_metadata': run_metadata}
  alpha_tf, n, ll_tf, gradients_tf = sess.run([final_alpha, final_n, ll_tf, gradients_tf],
                                              feed_dict={log_probs: log_probs_np,
                                                         labels: labels_np,
                                                         input_lengths: input_lengths_np,
                                                         label_lengths: label_lengths_np}, **tf_run_opts)
  if timing:
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    from tensorflow.python.client import timeline
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline_tf_impl.json', 'w') as f:
      f.write(ctf)
  return alpha_tf, ll_tf, gradients_tf


def test_impl(name, acts, labels, blank_index, with_warprnnt=True):
  """
  runs the different implementations on the same data, comparing them.

  :param name: test name
  :param np.ndarray acts: (B, T, U, V)
  :param np.ndarray labels: (B, U)
  :param int blank_index:
  :param bool with_warprnnt: whether to check also WarpRNN-T implementation
  :return:
  """
  assert len(acts.shape) == 3  # (T, U, V)
  n_time, n_target, n_vocab = acts.shape
  log_probs = log_softmax(acts, axis=2)  # along vocabulary

  alpha_ref, ll_ref = forward_pass(log_probs, labels, blank=blank_index)
  ll_ref, grads_ref = transduce_ref(log_probs, labels, blank=blank_index)
  ll_ref = -ll_ref

  print("Test", colored("%s" % name, "yellow"))

  if with_warprnnt:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "returnn"))
    from extern.HawkAaronWarpTransducer import rnnt_loss
    with sess.as_default():
      input_lengths_t = tf.constant([n_time], dtype=tf.int32)  # (1,)
      label_lengths_t = tf.constant([len(labels)], dtype=tf.int32)  # (1,)
      log_probs_t = tf.placeholder(dtype=tf.float32, shape=(1, n_time, n_target, n_vocab))  # # (1, T, U, V)
      labels_t = tf.placeholder(dtype=tf.int32, shape=(1, len(labels),))
      costs_warprnnt_tf = rnnt_loss(log_probs_t, labels_t,
                                    input_lengths_t, label_lengths_t, blank_label=blank_index)
      grads_warprnnt_tf = tf.gradients(costs_warprnnt_tf, [log_probs_t])[0]
      costs_warprnnt, grads_warprnnt = sess.run([costs_warprnnt_tf, grads_warprnnt_tf],
                                                feed_dict={
                                                  log_probs_t: log_probs.reshape(1, n_time, n_target, n_vocab),
                                                  labels_t: labels.reshape(1, len(labels)),
                                                })
      grads_warprnnt = grads_warprnnt[0]
      costs_warprnnt = -costs_warprnnt

      print(colored("Warp Reference", "red"),
            "implementation: log-posterior=%.4f, |alpha|=......, |grads|=%.4f" % (
            costs_warprnnt, np.linalg.norm(grads_warprnnt)))

  print(colored("Reference     ", "red"),
        "implementation: log-posterior=%.4f, |alpha|=%.4f, |grads|=%.4f" % (
        ll_ref, np.linalg.norm(alpha_ref), np.linalg.norm(grads_ref)))

  alpha_np, ll_np = numpy_forward(log_probs, labels, blank_index=blank_index, debug=False)
  print(colored("NumPy         ", "red"),
        "implementation: log-posterior=%.4f, |alpha|=%.4f" % (ll_np, np.linalg.norm(alpha_np)))

  with sess.as_default():
    alpha_tf, ll_tf, grads_tf = tf_forward_shifted(log_probs, labels, blank_index=blank_index, debug=True, sess=sess)
  print(colored("TensorFlow    ", "red"),
        "implementation: log-posterior=%.4f, |alpha|=%.4f, |grads|=%.4f" % (
        ll_tf, np.linalg.norm(alpha_tf), np.linalg.norm(grads_tf[0])))

  # Do all the tests (numpy vs ref vs TF), for score, alpha, and grads (TF/Ref)
  np.testing.assert_allclose(alpha_np, alpha_ref)
  print("numpy vs ref: alpha matrices", colored("MATCH", "green"))
  np.testing.assert_almost_equal(ll_np, ll_ref, decimal=6)
  print("numpy vs ref: log posterior", colored("MATCH", "green"))
  print()
  # We don't have an alpha-matrix anymore (instead there are diagonals)
  # np.testing.assert_allclose(alpha_tf, alpha_ref)
  # print("TF vs ref: alpha matrices", colored("MATCH", "green"))
  np.testing.assert_almost_equal(ll_tf, ll_ref, decimal=5)
  print("TF vs ref: log posterior", colored("MATCH", "green"))
  np.testing.assert_almost_equal(grads_tf[0], grads_ref, decimal=5)
  print("TF vs ref: gradients", colored("MATCH", "green"))
  if with_warprnnt:
    np.testing.assert_almost_equal(ll_tf, costs_warprnnt, decimal=5)
    print("TF vs Warp: log posterior", colored("MATCH", "green"))
    np.testing.assert_almost_equal(grads_tf[0], grads_warprnnt, decimal=5)
    print("TF vs Warp: gradients", colored("MATCH", "green"))
  print()


def test_small():
  """Small test, copied from
    https://github.com/awni/transducer/blob/master/ref_transduce.py
  """
  blank_index = 0
  n_time = 2
  n_target = 3
  n_vocab = 5
  acts = np.array([0.1, 0.6, 0.1, 0.1, 0.1, 0.1,
                   0.1, 0.6, 0.1, 0.1, 0.1, 0.1,
                   0.2, 0.8, 0.1, 0.1, 0.6, 0.1,
                   0.1, 0.1, 0.1, 0.1, 0.2, 0.1,
                   0.1, 0.7, 0.1, 0.2, 0.1, 0.1], dtype=np.float32).reshape(n_time, n_target, n_vocab)

  labels = np.array([1, 2])
  test_impl("test_small", acts, labels, blank_index)


def test_size_t_greater_u():
  """Tests for case when T > 2*U"""
  blank_index = 0
  n_time = 8
  n_target = 3
  n_vocab = 5
  acts = np.random.random_sample((n_time, n_target, n_vocab))
  labels = np.array([1, 2])
  test_impl("test_size: T>U", acts, labels, blank_index)


def test_size_u_greater_t():
  """Tests for case when U >> T"""
  blank_index = 0
  n_time = 3
  n_target = 8
  n_vocab = 5
  acts = np.random.random_sample((n_time, n_target, n_vocab))
  labels = np.random.randint(1, n_vocab-1, (n_target-1,))
  test_impl("test_size: U>T", acts, labels, blank_index)


def test_size_t_equal_u():
  """Tests for case when T == U"""
  blank_index = 0
  n_time = 7
  n_target = 7
  n_vocab = 5
  acts = np.random.random_sample((n_time, n_target, n_vocab))
  labels = np.random.randint(1, n_vocab-1, (n_target-1,))
  test_impl("test_size: U==T", acts, labels, blank_index)


def test_sizes():
  """Tests for case when T == U"""
  blank_index = 0
  n_vocab = 5


  # DEBUG: error case
  #n_time = 4; n_target=5
  #acts = np.random.random_sample((n_time, n_target, n_vocab))
  #labels = np.random.randint(1, n_vocab - 1, (n_target - 1,))
  #test_impl(f"test_size: T={n_time}, U={n_target}", acts, labels, blank_index)
  #return


  for n_time in range(2, 12):
    for n_target in range(2, 12):
      acts = np.random.random_sample((n_time, n_target, n_vocab))
      labels = np.random.randint(1, n_vocab-1, (n_target-1,))
      test_impl("test_sizes: T=%d, U=%d" % (n_time, n_target), acts, labels, blank_index)


def test_random():
  """Tests when there are some zeros: 0 -> -inf in log-space."""
  blank_index = 0
  n_time = 2
  n_target = 3
  n_vocab = 5
  acts = np.random.standard_normal((n_time, n_target, n_vocab))
  labels = np.array([1, 2])
  test_impl("test_random", acts, labels, blank_index)


def test_batched():
  """https://github.com/awni/transducer/blob/master/test.py
  check only first in mini batch.
  """
  name = "batched"
  print("Test", colored("%s" % name, "yellow"))
  acts = np.array(
    [
      [[[0.06535690384862791, 0.7875301411923206, 0.08159176605666074],
        [0.5297155426466327, 0.7506749639230854, 0.7541348379087998],
        [0.6097641124736383, 0.8681404965673826, 0.6225318186056529]],

       [[0.6685222872103057, 0.8580392805336061, 0.16453892311765583],
        [0.989779515236694, 0.944298460961015, 0.6031678586829663],
        [0.9467833543605416, 0.666202507295747, 0.28688179752461884]],

       [[0.09418426230195986, 0.3666735970751962, 0.736168049462793],
        [0.1666804425271342, 0.7141542198635192, 0.3993997272216727],
        [0.5359823524146038, 0.29182076440286386, 0.6126422611507932]],

       [[0.3242405528768486, 0.8007644367291621, 0.5241057606558068],
        [0.779194617063042, 0.18331417220174862, 0.113745182072432],
        [0.24022162381327106, 0.3394695622533106, 0.1341595066017014]]],

      [[[0.5055615569388828, 0.051597282072282646, 0.6402903936686337],
        [0.43073311517251, 0.8294731834714112, 0.1774668847323424],
        [0.3207001991262245, 0.04288308912457006, 0.30280282975568984]],

       [[0.6751777088333762, 0.569537369330242, 0.5584738347504452],
        [0.08313242153985256, 0.06016544344162322, 0.10795752845152584],
        [0.7486153608562472, 0.943918041459349, 0.4863558118797222]],

       [[0.4181986264486809, 0.6524078485043804, 0.024242983423721887],
        [0.13458171554507403, 0.3663418070512402, 0.2958297395361563],
        [0.9236695822497084, 0.6899291482654177, 0.7418981733448822]],

       [[0.25000547599982104, 0.6034295486281007, 0.9872887878887768],
        [0.5926057265215715, 0.8846724004467684, 0.5434495396894328],
        [0.6607698886038497, 0.3771277082495921, 0.3580209022231813]]]])

  #expected_costs = np.array([4.2806528590890736, 3.9384369822503591])
  # expected_grads = np.array([
  #   [[[-0.4322264564338117, -0.5677735435661883, 0.0],
  #     [-0.36565009313836844, 0.0, -0.20212345042782007],
  #     [-0.20212345042782007, 0.0, 0.0]],
  #
  #    [[-0.16521672442463506, -0.2670097320091765, 0.0],
  #     [-0.3943653886107811, 0.0, -0.2382944365367636],
  #     [-0.44041788696458367, 0.0, 0.0]],
  #
  #    [[-0.052129794015740985, -0.11308693040889405, 0.0],
  #     [-0.18313786985332664, 0.0, -0.3243144491663483],
  #     [-0.7647323361309323, 0.0, 0.0]],
  #
  #    [[0.0, -0.052129794015740985, 0.0],
  #     [0.0, 0.0, -0.23526766386906767],
  #     [-1.0, 0.0, 0.0]]],
  #
  #   [[[-0.7161424128232795, -0.2838575871767207, 0.0],
  #     [-0.18382932237365335, -0.10002826480306751, 0.0],
  #     [-0.10002826480306751, 0.0, 0.0]],
  #
  #    [[-0.41121794618117213, -0.3049244666421072, 0.0],
  #     [-0.3295759402552584, -0.15917784876050195, 0.0],
  #     [-0.2592061135635692, 0.0, 0.0]],
  #
  #    [[-0.11607642141651396, -0.29514152476465827, 0.0],
  #     [-0.2865333615432337, -0.3381841034766833, 0.0],
  #     [-0.5973902170402529, 0.0, 0.0]],
  #
  #    [[0.0, -0.11607642141651396, 0.0],
  #     [0.0, -0.4026097829597475, 0.0],
  #     [-1.0, 0.0, 0.0]]]])
  n_batch = 4
  max_input = 80
  max_target = 20
  n_vocab = 50
  np.random.seed(42)
  labels = np.random.randint(1, n_vocab, (n_batch, max_target-1))
  input_lengths = np.random.randint(1, max_input, (n_batch,), dtype=np.int32)
  label_lengths = np.random.randint(1, max_target - 1, (n_batch,), dtype=np.int32)
  acts = np.random.normal(0, 1, (n_batch, max_input, max_target, n_vocab))

  log_probs = log_softmax(acts, axis=3)  # along vocabulary for (B, T, U, V)
  # alpha_ref, ll_ref = forward_pass(log_probs, labels, blank=0)
  # Note: The transduce_batch implementation is wrong, for varied-sized inputs/labels!!!
  #       it doesn't handle the varying sizes (for log-likelihood computation)
  #ll_ref, grads_ref = transduce_batch(log_probs, labels, blank=0)
  #ll_ref = -np.array(ll_ref)
  #print(colored("Reference ", "red"),
  #      "implementation: log-posterior=%r, |alpha|=%.4f" % (ll_ref, 0))
  timing = False

  if True:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "returnn"))
    from extern.HawkAaronWarpTransducer import rnnt_loss
    with sess.as_default():
      input_lengths_t = tf.constant(input_lengths, dtype=tf.int32)  # (B, T)
      label_lengths_t = tf.constant(label_lengths, dtype=tf.int32)  # (1,)
      log_probs_t = tf.constant(log_probs, dtype=tf.float32)  # (B, T, U, V)
      labels_t = tf.constant(labels, dtype=tf.int32)
      costs_warprnnt_tf = rnnt_loss(log_probs_t, labels_t,
                                    input_lengths_t, label_lengths_t, blank_label=0)
      grads_warprnnt_tf = tf.gradients(costs_warprnnt_tf, [log_probs_t])[0]
      tf_run_opts = {}
      if timing:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        tf_run_opts = {'options': run_options, 'run_metadata': run_metadata}
      costs_warprnnt, grads_warprnnt = sess.run([costs_warprnnt_tf, grads_warprnnt_tf],
                                                feed_dict={
                                                  log_probs_t: log_probs,
                                                  labels_t: labels,
                                                }, **tf_run_opts)

      # Create the Timeline object, and write it to a json
      if timing:
        from tensorflow.python.client import timeline
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline_warprnnt.json', 'w') as f:
          f.write(ctf)


      print(colored("Warp Reference", "red"),
            "implementation: log-posterior=%r, |alpha|=......, |grads|=%.4f" % (
            costs_warprnnt, np.linalg.norm(grads_warprnnt)))
  # alpha_np, ll_np = numpy_forward(log_probs[0], labels[0], blank_index=0, debug=False)
  # print(colored("NumPy     ", "red"),
  #      "implementation: log-posterior=%.4f, |alpha|=%.4f" % (ll_np, np.linalg.norm(alpha_np)))

  with sess.as_default():
    alpha_tf, costs_tf, grads_tf = tf_forward_shifted(log_probs, labels,
                                                      input_lengths_np=input_lengths,
                                                      label_lengths_np=label_lengths,
                                                      blank_index=0, debug=False, sess=sess,
                                                      timing=timing)
  grads_tf = grads_tf
  print(colored("TensorFlow", "red"),
        "implementation: log-posterior=%r, |alpha|=%.4f" % (costs_tf, np.linalg.norm(alpha_tf)))

  # Check against expected values
  # np.testing.assert_allclose(alpha_tf, alpha_ref)
  # print("TF vs ref: alpha matrices", colored("MATCH", "green"))
  np.testing.assert_almost_equal(costs_tf, -costs_warprnnt, decimal=6)
  print("TF vs Warp RNN-T: log posterior", colored("MATCH", "green"))
  np.testing.assert_almost_equal(grads_tf, grads_warprnnt, decimal=4)
  print("TF vs Warp RNN-T: gradients", colored("MATCH", "green"))


if __name__ == '__main__':
  sess = tf.Session()
  import better_exchook
  better_exchook.install()

  #test_small()
  #test_size_u_greater_t()
  #test_size_t_greater_u()
  #test_size_t_equal_u()
  #test_random()
  test_batched()
  #test_sizes()
