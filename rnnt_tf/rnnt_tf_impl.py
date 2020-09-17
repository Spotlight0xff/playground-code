#!/usr/bin/env python3
# vim: sw=2
"""
Implementation of the RNN-T loss in pure TF,
plus comparisons against reference implementations.
"""
import os
import sys

import numpy as np
from termcolor import colored
from ref_transduce import transduce as transduce_ref
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "returnn"))
from returnn.extern.HawkAaronWarpTransducer import warp_rnnt_loss
from returnn.tf.compat import v1 as tf
NEG_INF = -float("inf")


class ComputationResult(object):
  """Represents the information about the computation result,
  for a comparison across the implementations."""
  def __init__(self, name, costs=None, alphas=None, alignments=None, grads=None):
    self.name = name
    self.costs = costs
    self.alphas = alphas
    self.alignments = alignments
    self.grads = grads

  def __repr__(self):
    cost = sum(self.costs) if isinstance(self.costs, (list, np.ndarray)) else self.costs
    if self.grads is None:
      grads_msg = "n/a"
    else:
      grads = self.grads
      if isinstance(grads, np.ndarray):
        if len(grads.shape) == 4:
          *grads, = grads
      grads_msg = "%.4f" % sum([np.linalg.norm(grad) for grad in grads])
    ret = "%20s implementation: log-posterior=%.3f, |grads|=%s" % (colored("%20s" % self.name, "red"), cost, grads_msg)
    return ret


def log_sum_exp(*args):  # summation in linear space -> LSE in log-space
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


def numpy_forward(log_probs, labels, blank_index, debug=False, with_alignment=False,
                  semiring_add=log_sum_exp) -> ComputationResult:
  """Forward calculation of the RNN-T loss."""
  n_time, n_target, n_vocab = log_probs.shape
  alpha = np.zeros((n_time, n_target))  # 1 in log-space
  if debug:
    print("Numpy RNN-T loss computation, with-alignment: %r" % with_alignment)
    print("a = alpha[t-1, u] + log_probs[t - 1, u, blank_index]")
    print("b = alpha[t, u - 1] + log_probs[t, u - 1, labels[u - 1]]")
    print("alpha[t,u] = LSE(a,b)")
  # backtrace matrix for (U+T) -> U, storing in each cell (from_t, from_u, symbol)
  if with_alignment:
    bt_mat = np.zeros((n_time, n_target, 3), dtype=np.int32)

  if debug:
    print("T=%d, U=%d, V=%d" % (n_time, n_target, n_vocab))
    print("labels:", labels, "blank-index:", blank_index)
  for t in range(1, n_time):  # first row
    alpha[t, 0] = alpha[t - 1, 0] + log_probs[t - 1, 0, blank_index]
    if with_alignment:
      bt_mat[t, 0] = (t-1, 0, blank_index)  # noqa  can only be blank
    if debug:
      print('t=%2d u= 0: alpha[%d, 0] + log_probs[%d, 0, %d] = %.3f + %.3f = %.3f' % (
        t, t-1, t-1, blank_index, alpha[t - 1, 0], log_probs[t - 1, 0, blank_index], alpha[t, 0]))
  for u in range(1, n_target):  # first column
    alpha[0, u] = alpha[0, u - 1] + log_probs[0, u - 1, labels[u - 1]]
    if with_alignment:
      bt_mat[0, u] = (0, u-1, labels[u-1])  # can only be emit
    if debug:
      print('t= 0 u=%2d: alpha[0, %d] + log_probs[0, %d, labels[%d]=%d] = %.3f + %.3f = %.3f' % (
        u, u-1, u-1, u-1, labels[u - 1], alpha[0, u - 1], log_probs[0, u - 1, labels[u - 1]], alpha[0, u]))

  for t in range(1, n_time):
    for u in range(1, n_target):
      a = alpha[t - 1, u] + log_probs[t - 1, u, blank_index]
      b = alpha[t, u - 1] + log_probs[t, u - 1, labels[u - 1]]
      alpha[t, u] = elem = semiring_add(a, b)
      if with_alignment:
        from_idx = np.argmax([a, b])
        if from_idx == 0:  # blank
          bt_mat[t, u] = (t-1, u, blank_index)
        elif from_idx == 1:  # emit
          bt_mat[t, u] = (t, u-1, labels[u-1])
      if debug:
        print("t=%2d u=%2d: "
              "LSE(%.3f + %.3f, %.3f +  %.3f)"
              "= LSE(%.3f, %.3f) = %.3f" % (t, u,
                                            alpha[t - 1, u], log_probs[t - 1, u, blank_index],
                                            alpha[t, u - 1], log_probs[t, u - 1, labels[u - 1]],
                                            a, b, elem))

  if debug:
    assert len(alpha.shape) == 2
    print("Alpha matrix: (%d, %d)" % tuple(alpha.shape))
    print(alpha)
  alignment = None
  if with_alignment:
    alignment = backtrack_alignment_rnnt_numpy(bt_mat, n_time=n_time, n_target=n_target, debug=debug)
  log_posterior = alpha[n_time-1, n_target-1] + log_probs[n_time-1, n_target-1, blank_index]
  if debug:
    print("log-posterior = alpha[%d, %d] + log_probs[%d, %d, %d] = %.3f + %.3f = %.4f" %
          (n_time-1, n_target-1, n_time-1, n_target-1, blank_index,
           alpha[n_time-1, n_target-1],
           log_probs[n_time - 1, n_target - 1, blank_index],
           log_posterior
           ))
  return ComputationResult("numpy forward", costs=log_posterior, alphas=alpha, alignments=alignment, grads=None)


def backtrack_alignment_rnnt_numpy(bt_mat, n_time, n_target, debug=False):
  """Computes the alignment from the backtracking matrix.
  We do this in a batched fashion so we can compare/copy this directly to TF.

  :param np.ndarray bt_mat: backtracking matrix (T, U+1, 3)
  :param int n_time: scalar
  :param int n_target: scalar
  :param bool debug:

  :return alignment of form (B, T+U) -> [V]
  :rtype np.ndarray
  """
  assert len(bt_mat.shape) == 3
  assert bt_mat.shape == (n_time, n_target, 3)
  # (B, T+U+1) -> [V]
  init_t, init_u = n_time-1, n_target-1
  idx = bt_mat[init_t, init_u]  # (from_t, from_u, symbol)
  alignments = [idx[2]]
  if debug:
    print("bt-mat:", bt_mat.shape)
    print("bt-mat[%r, %r] = initial-idx = %r" % (init_t, init_u, idx))
  t = n_time - 1
  u = n_target - 1
  for _ in reversed(range(1, n_time + n_target-1)):  # every path is T+U long.
    assert idx.shape == (3,)
    t = idx[0]  # time
    u = idx[1]  # target
    idx = bt_mat[t, u]
    alignments.insert(0, idx[2])
  assert t == u == 0  # noqa
  return np.array(alignments)


def backtrack_alignment_rnnt_numpy_batched(bt_mat, input_lens, label_lens, blank_index, debug=False):
  """Computes the alignment from the backtracking matrix.
  We do this in a batched fashion so we can compare/copy this directly to TF.

  :param np.ndarray bt_mat: backtracking matrix (T+U+1, B, U, 3)
  :param np.ndarray input_lens: (B,)
  :param np.ndarray label_lens: (B,)
  :param int blank_index:
  :param bool debug:

  :return alignment of form (B, T+U) -> [V]
  :rtype np.ndarray
  """
  assert len(input_lens) == len(label_lens)
  max_path, n_batch, max_target, track_dim = bt_mat.shape
  assert track_dim == 2  # (from_u, label)
  # (B, T+U+1) -> [V]
  # alignments[:, 0] should be blank, because in RNN-T we initially consume a frame, then we can emit.
  alignments = np.ones((n_batch, max_path), dtype=np.int32) * blank_index
  if debug:
    print("#diagonals:", max_path, "B=%r" % n_batch, "U=%r" % max_target, "D=%r" % track_dim)
    print("bt-mat:", bt_mat.shape)
  idx = np.zeros((n_batch, 2), dtype=np.int32)
  assert bt_mat.shape[0] == max_path
  for s in reversed(range(1, max_path)):  # every path is T+U long.
    init_u = np.where(np.greater_equal(s, input_lens+label_lens-1),
                      label_lens,  # if we are at the end of some path (or behind)
                      idx[:, 0]  # if we are within a path, continue backtracking.
                      )
    assert init_u.shape == (n_batch,)
    idx = bt_mat[s, np.arange(n_batch), init_u]
    assert idx.shape == (n_batch, 2), init_u.shape
    write_align = np.where(s <= input_lens+label_lens, idx[:, 1], -1)
    if debug:
      print("s=%r, init-u=%r, idx=%r, align=%r" % (s, init_u[0], idx[0].tolist(), write_align[0].tolist()))
      print("s=%r, bt_mat:" % s, bt_mat[s])
    alignments[:, s] = write_align  # [B]
  return alignments


def numpy_forward_shifted_batched(log_probs, labels, blank_index, input_lens, label_lens, debug=False,
                                  with_alignment=False):
  """Forward calculation of the RNN-T loss using the same diagonal strategy."""
  n_batch, max_time, max_target, n_vocab = log_probs.shape  # (B, T, U+1, V)
  assert labels.shape == (n_batch, max_target-1)  # (B, U)
  if debug:
    print("U=%d, T=%d, V=%d" % (max_target, max_time, n_vocab))
    print("log-probs: (B=%d, T=%d, U=%d, V=%d)" % (n_batch, max_time, max_target, n_vocab))
    print("labels: (B=%d, U-1=%d)" % (n_batch, labels.shape[1]))
  num_diagonals = max_time + max_target

  # backtrack matrix, (s, i, u) -> [from_u, symbol/blank]
  bt_mat = np.zeros((max_time+max_target-1, n_batch, max_target, 2), dtype=np.int32)
  # bt_mat: (T+U+1, B, U, 2), same form as `shifted_logprobs`.
  # then simply bt_mat[n] =

  with tf.Session().as_default():
    tf_shifted = tf_shift_logprobs(tf.cast(tf.transpose(log_probs, [0, 2, 1, 3]), dtype=tf.float32), axis=1)
    shifted_logprobs = sess.run(tf_shifted)
    assert shifted_logprobs.shape == (n_batch, max_target, max_time+max_target, n_vocab)
    shifted_logprobs = np.transpose(shifted_logprobs, (0, 2, 1, 3))  # (B, U, U+T+1, V) -> (B, U+T+1, U, V)
  assert shifted_logprobs.shape == (n_batch, num_diagonals, max_target, n_vocab)
  if debug:
    print("log-probs")
    for n in range(2, max_time+max_target):
      print("lp(blank) for n=%d" % n, shifted_logprobs[0, n - 2, ..., 0])

  def print_debug(n_, *args):
    """Some basic debug information printing."""
    if debug:
      print("[n=%2d]" % n_, *args)
  # alpha diagonals
  alphas = [[], np.zeros((n_batch, 1))]

  for n in range(2, max_time+max_target):
    # actually previous one.
    lp_diagonal = shifted_logprobs[:, n-2, :n-1]  # (B, n-1, V)
    print_debug(n, "lp_diagonal", lp_diagonal)

    prev_diagonal = alphas[n-1][:, :n]
    print_debug(n, "prev_diagonal", prev_diagonal)

    alpha_blank = prev_diagonal  # (B, N)
    alpha_blank = np.concatenate([alpha_blank, np.tile([[NEG_INF]], [n_batch, 1])], axis=1)

    # (B, U, V) -> (B, U)
    lp_blank = lp_diagonal[:, :, blank_index]  # (B, U)
    lp_blank = np.concatenate([lp_blank, np.tile([[NEG_INF]], [n_batch, 1])], axis=1)

    alpha_y = prev_diagonal
    alpha_y = np.concatenate([np.tile([[NEG_INF]], [n_batch, 1]), alpha_y], axis=1)

    # NOTE:
    # We compute the diagonals from bottom-left to top-right.
    # However, we cut off the diagonals in the top-right corner,
    # as soon as we can make sure there are no interesting values.
    # this happens when n > U.
    cut_off = max_target
    if n > cut_off:  # phase (c), cut off top-right
      alpha_y = alpha_y[:, :cut_off]
      lp_blank = lp_blank[:, :cut_off]
      alpha_blank = alpha_blank[:, :cut_off]

    labels_maxlen = min(max_target - 1, n - 1)
    labels_shifted = labels[:, :labels_maxlen]
    print_debug(n, "labels_shifted", labels_shifted)
    batchs_idxs, rows_idxs = np.meshgrid(
      np.arange(np.shape(labels_shifted)[0]),  # B
      np.arange(np.shape(labels_shifted)[1]),  # U-1
      indexing='ij'
    )
    lp_y = lp_diagonal[batchs_idxs, rows_idxs, labels_shifted]
    lp_y = np.concatenate([np.tile([[NEG_INF]], [n_batch, 1]), lp_y], axis=1)

    # all should have shape (B, n)
    print_debug(n, colored("lp_blank", "green"), lp_blank)
    print_debug(n, colored("alpha_blank", "green"), alpha_blank)
    blank = alpha_blank + lp_blank
    print_debug(n, colored("blank", "green"), blank)

    print_debug(n, "labels_shifted", labels_shifted)
    print_debug(n, colored("lp_y", "red"), lp_y)
    print_debug(n, colored("alpha_y", "red"), alpha_y)
    y = alpha_y + lp_y
    print_debug(n, colored("y", "red"), y)
    new_diagonal = np.logaddexp(blank, y)  # (B, N)
    new_diagonal = new_diagonal[:, :n]
    print_debug(n, "new_diagonal", new_diagonal)
    alphas.append(new_diagonal)  # s.t. alphas[n] == new_diagonal

    # for alignment generation
    argmax_idx = np.argmax([blank, y], axis=0)[:, :n]  # (B, n|U) -> [2], 0: blank, 1: emit
    print_debug(n, colored("argmax-idx", "cyan"), argmax_idx)
    max_len_diag = min(n, max_target)
    # assert argmax_idx.shape[1] == max_len
    u_ranged = np.tile(np.arange(max_len_diag)[None], [n_batch, 1])  # [B, n|U]
    print_debug(n, colored("u_ranged", "cyan"), u_ranged)
    u_ranged_1 = u_ranged - 1  # for emit, u-1 for every item in the diagonal
    blank_tiled = np.tile([[blank_index]], [n_batch, 1])  # [B, 1]

    print_debug(n, colored("labels_shifted", "cyan"), labels_shifted)
    stack_blank_sel = np.stack([u_ranged, np.tile(blank_tiled, [1, max_len_diag])], axis=-1)  # [B, n|U, 2]
    print_debug(n, colored("stack_blank_sel", "cyan"), stack_blank_sel)

    labels_emit_sel = labels[np.arange(n_batch)[:, np.newaxis], np.maximum(0, np.arange(max_len_diag)[np.newaxis, :]-1)]
    stack_emit_sel = np.stack([u_ranged_1, labels_emit_sel], axis=-1)
    print_debug(n, colored("stack_emit_sel", "cyan"), stack_emit_sel)

    # based on the argmax indices, we either store for:
    # blank: (u,   blank-idx)
    # emit : (u-1, labels[u-1])
    best_sel = np.where((argmax_idx == 0)[..., None],
                        stack_blank_sel,  # blank
                        stack_emit_sel   # emit
                        )
    print_debug(n, colored("best_sel", "cyan"), best_sel)
    bt_mat[n-1, :, :n] = best_sel  # (B, U|n, 2)

    if debug:
      print("\n")

  list_nll = []
  diag_idxs = input_lens + label_lens  # (B,)

  # (B,): batch index -> index within diagonal
  # We need to handle the U>T case for each example.
  within_diag_idx = label_lens
  for i in range(n_batch):
    ta_item = alphas[diag_idxs[i]]  # (B, N)

    a = ta_item[i, within_diag_idx[i]]
    b = log_probs[i, input_lens[i]-1, label_lens[i], blank_index]
    if debug:
      print("FINAL i=%d, diag_idx=%d, within_diag_idx=%d" % (i, diag_idxs[i], within_diag_idx[i]))
      print("FINAL i=%d" % i, "a=%.3f + b=%.3f" % (a, b))
    nll = a + b
    list_nll.append(nll)
  alignments = None
  if with_alignment:
    alignments = backtrack_alignment_rnnt_numpy_batched(bt_mat, input_lens, label_lens,
                                                        blank_index=blank_index, debug=debug)
  return ComputationResult("NumPy Batched", costs=np.array(list_nll), alignments=alignments)


def backtrack_alignment_tf(bt_ta, input_lengths, label_lengths, blank_index):
  """Computes the alignment from the backtracking matrix.
  :param tf.TensorArray bt_ta: [T+U] * (B, U, 2)
  :param tf.Tensor input_lengths: (B,)
  :param tf.Tensor label_lengths: (B,)
  :param int blank_index:
  """
  max_path = bt_ta.size()
  n_batch = tf.shape(input_lengths)[0]
  alignments_ta = tf.TensorArray(
    dtype=tf.int32,
    size=max_path,
    dynamic_size=False,
    infer_shape=False,
    element_shape=(None,),  # [V]
    name="alignments"
  )
  initial_idx = tf.zeros((n_batch, 2), dtype=tf.int32)

  def body(s, alignments, idx):
    """Runs over s=[max_path-1,..., 1] (along the alignment)
    :param int s: path index
    :param tf.TensorArray alignments: [T+U] * (V,)
    :param tf.Tensor idx: (B, 2) -> [from_u, symbol/blank]
    """
    backtrack = bt_ta.read(s)  # (B, U, 2)

    init_u = tf.where(tf.greater_equal(s, input_lengths + label_lengths - 1),
                      label_lengths,  # if we are at the end of some path (or behind)
                      idx[:, 0]  # if we are within a path, continue backtracking.
                      )
    backtrack_indices = tf.stack([tf.range(n_batch), init_u], axis=-1)  # (B, 2)
    idx = tf.gather_nd(backtrack, backtrack_indices)
    align_write = tf.where(
      tf.less_equal(s, input_lengths + label_lengths),
      idx[:, 1],  # within alignment
      tf.ones((n_batch,), dtype=tf.int32) * blank_index)  # outside, assume blank
    alignments = alignments.write(s, align_write)
    return s-1, alignments, idx
  init_s = max_path-1
  final_s, final_alignments_ta, final_idx = tf.while_loop(lambda s, *args: tf.greater_equal(s, 1),
                                                          body, (init_s, alignments_ta, initial_idx))
  final_alignments_ta = final_alignments_ta.write(0, tf.tile([blank_index], [n_batch]))
  return tf.transpose(final_alignments_ta.stack())


def tf_shift_logprobs(mat, axis):
  """
  Shifts the log-probs per-batch row-wise.

  :param mat: (B, U, T, V)
  :param axis:
  :return: (B, T+U+1, U, V)
  """
  # mat: (B, T, U, V)
  # axis_to_expand: usually U
  # axis: usually T
  # batch-axis has to be first
  max_time = tf.shape(mat)[axis]  # T

  def fn(args):  # x: (B, U, V)
    """Computes the shift per diagonal and pads accordingly."""
    x, shift = args
    padded = tf.pad(x, [[0, 0],  # B
                        [shift, max_time - shift],  # U+T+1
                        [0, 0]  # V
                        ], constant_values=0)
    return padded, shift

  elems0 = tf.transpose(mat, [1, 0, 2, 3])  # [T, B, U, V]
  elems1 = tf.range(max_time)  # [T]
  t, _ = tf.map_fn(fn, elems=(elems0, elems1))  # T* [B, T+U+1, V]
  t = tf.transpose(t, [1, 0, 2, 3])  # [B, T, U+1, V]
  return t


def rnnt_loss(log_probs, labels, input_lengths=None, label_lengths=None,
              blank_index=0, debug=False, with_alignment=False):
  """
  Computes the batched forward pass of the RNN-T model.
  B: batch, T: time, U:target/labels, V: vocabulary

  :param tf.Tensor log_probs: (B, T, U+1, V) log-probabilities
  :param tf.Tensor labels: (B, U) -> [V] labels
  :param tf.Tensor input_lengths: (B,) length of input frames
  :param tf.Tensor label_lengths: (B,) length of labels
  :param int blank_index: index of the blank symbol in the vocabulary
  :param bool debug: enable verbose logging
  :param bool with_alignment: whether to generate the alignments or not.
  :return:
  with_alignment=True -> (costs, alignments)
                =False -> costs
  """
  """Pure TF implementation of the RNN-T loss."""
  shape = tf.shape(log_probs)
  n_batch = shape[0]     # B
  max_time = shape[1]    # T
  max_target = shape[2]  # U

  log_probs_tr = tf.transpose(log_probs, [0, 2, 1, 3])  # (B, T, U, V) -> (B, U, T, V)
  log_probs_shifted = tf_shift_logprobs(log_probs_tr, axis=1)  # (B, U+T+1, U, V)

  num_diagonals = max_time + max_target

  labels = py_print_iteration_info("labels", labels, 0, debug=debug)

  log_probs_ta = tf.TensorArray(
    dtype=tf.float32,
    clear_after_read=False,
    size=num_diagonals,
    dynamic_size=False,
    infer_shape=False,
    element_shape=(None, None, None),  # (B, U, V)
    name="log_probs_shifted",
  )
  # (B, U+T+1, U, V) -> [(B, U, V)] * (U+T+1)
  log_probs_ta = log_probs_ta.unstack(tf.transpose(log_probs_shifted, [2, 0, 1, 3]))

  init_alpha_ta = tf.TensorArray(
    dtype=tf.float32,
    clear_after_read=False,
    size=num_diagonals,
    dynamic_size=False,
    infer_shape=False,
    element_shape=(None, None,),  # (B, n)
    name="alpha_diagonals",
  )
  init_alpha_ta = init_alpha_ta.write(1, tf.zeros((n_batch, 1)))

  if with_alignment:
    # backtrack matrix, for each diagonal and target -> [from_u, symbol/blank]
    init_backtrack_ta = tf.TensorArray(
      dtype=tf.int32,
      size=num_diagonals-1,
      dynamic_size=False,
      infer_shape=False,
      element_shape=(None, None, 2),
      name="backtrack"
    )
  else:
    init_backtrack_ta = None

  def cond(n, *_):
    """We run the loop until all elements are covered by diagonals.
    """
    return tf.less(n, num_diagonals)

  def body_forward(n, alpha_ta, *args):
    """body of the while_loop, loops over the diagonals of the alpha-tensor."""
    # alpha(t-1,u) + logprobs(t-1, u)
    # alpha_blank      + lp_blank

    lp_diagonal = log_probs_ta.read(n-2)[:, :n-1, :]  # (B, U|n, V)
    lp_diagonal = py_print_iteration_info("lp_diagonal", lp_diagonal, n, debug=debug)

    prev_diagonal = alpha_ta.read(n-1)[:, :n]  # (B, n-1)
    prev_diagonal = py_print_iteration_info("prev_diagonal", prev_diagonal, n, debug=debug)

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

    labels_max_len = tf.minimum(max_target-1, n-1)
    labels_shifted = labels[:, :labels_max_len]  # (B, U-1|n-1)
    labels_shifted = py_print_iteration_info("labels_shifted", labels_shifted, n, debug=debug)
    batchs, rows = tf.meshgrid(
      tf.range(n_batch),
      tf.range(labels_max_len),
      indexing='ij'
    )
    lp_y_indices = tf.stack([batchs, rows, labels_shifted], axis=-1)  # (B, U-1|n-1, 3)
    lp_y_indices = py_print_iteration_info("lp_y_indices", lp_y_indices, n, debug=debug)
    lp_y = tf.gather_nd(lp_diagonal[:, :, :], lp_y_indices)  # (B, U)
    # (B, U) ; (B, 1) -> (B, U+1)
    lp_y = tf.concat([tf.tile([[tf.constant(NEG_INF)]], [n_batch, 1]), lp_y], axis=1)
    lp_y = py_print_iteration_info("lp(y)", lp_y, n, debug=debug)

    cut_off = max_target
    alpha_y = tf.cond(tf.greater(n, max_target),
                      lambda: alpha_y[:, :cut_off],
                      lambda: alpha_y)
    lp_blank = tf.cond(tf.greater(n, max_target),
                       lambda: lp_blank[:, :cut_off],
                       lambda: lp_blank)
    alpha_blank = tf.cond(tf.greater(n, max_target),
                          lambda: alpha_blank[:, :cut_off],
                          lambda: alpha_blank)

    # all should have shape (B, n)
    blank = alpha_blank + lp_blank
    y = alpha_y + lp_y
    red_op = tf.stack([blank, y], axis=0)  # (2, B, N)
    red_op = py_print_iteration_info("red-op", red_op, n, debug=debug)
    new_diagonal = tf.math.reduce_logsumexp(red_op, axis=0)  # (B, N)

    new_diagonal = new_diagonal[:, :n]
    new_diagonal = py_print_iteration_info("new_diagonal", new_diagonal, n, debug=debug)

    if with_alignment:
      backtrack_ta = args[0]
      argmax_idx = tf.argmax([blank, y], axis=0)
      max_len_diag = tf.minimum(n, max_target)
      u_ranged = tf.tile(tf.range(max_len_diag)[None], [n_batch, 1])  # (B, n|U)
      blank_tiled = tf.tile([[blank_index]], [n_batch, 1])

      stack_blank_sel = tf.stack([u_ranged, tf.tile(blank_tiled, [1, max_len_diag])], axis=-1)

      b, r = tf.meshgrid(
        tf.range(n_batch),
        tf.maximum(0, tf.range(max_len_diag) - 1),
        indexing='ij'
      )
      labels_indices = tf.stack([b, r], axis=-1)
      labels_emit_sel = tf.gather_nd(labels, labels_indices)  # (B, n)  labels[u-1]
      stack_emit_sel = tf.stack([u_ranged-1, labels_emit_sel], axis=-1)
      best_sel = tf.where(tf.tile(tf.equal(argmax_idx, 0)[..., None], [1, 1, 2]),
                          stack_blank_sel,  # blank
                          stack_emit_sel  # emit
                          )
      backtrack_ta = backtrack_ta.write(n-1, best_sel)
    else:
      backtrack_ta = None
    return [n + 1, alpha_ta.write(n, new_diagonal)] + ([backtrack_ta] if with_alignment else [])

  init_n = tf.constant(2)
  if with_alignment:
    final_n, alpha_out_ta, backtrack_out_ta = tf.while_loop(cond, body_forward,
                                                            [init_n, init_alpha_ta, init_backtrack_ta],
                                                            parallel_iterations=1,  # iterative computation
                                                            name="rnnt")
  else:
    final_n, alpha_out_ta = tf.while_loop(cond, body_forward, [init_n, init_alpha_ta],
                                          parallel_iterations=1,  # iterative computation
                                          name="rnnt")
    backtrack_out_ta = None

  # p(y|x) = alpha(T,U) * blank(T,U)  (--> in log-space)

  # (B,): batch index -> diagonal index
  diag_idxs = input_lengths + label_lengths  # (B,)

  # (B,): batch index -> index within diagonal
  within_diag_idx = label_lengths

  res_ta = tf.TensorArray(
    dtype=tf.float32,
    clear_after_read=True,
    size=n_batch,
    dynamic_size=False,
    infer_shape=False,
    element_shape=(),
    name="alpha_diagonals",
  )

  def ta_read_body(i, res_loop_ta):
    """Reads from the alpha-diagonals TensorArray. We need this because of the inconsistent shapes in the TA."""
    ta_item = alpha_out_ta.read(diag_idxs[i])[i]
    return i+1, res_loop_ta.write(i, ta_item[within_diag_idx[i]])

  final_i, a_ta = tf.while_loop(
    lambda i, _: i < n_batch,
    ta_read_body, (tf.constant(0, tf.int32), res_ta)
  )
  indices = tf.stack([
    tf.range(n_batch),
    input_lengths-1,  # noqa T-1
    label_lengths,    # U-1
    tf.tile([blank_index], [n_batch]),
  ], axis=-1)  # (B, 3)
  ll_tf = a_ta.stack() + tf.gather_nd(log_probs, indices)

  if with_alignment:
    assert backtrack_out_ta is not None
    alignments = backtrack_alignment_tf(backtrack_out_ta, input_lengths, label_lengths, blank_index)
    return ll_tf, alignments
  else:
    return ll_tf


def wrap_tf_rnnt(log_probs, labels, input_lengths=None, label_lengths=None,
                 blank_index=0, debug=False, with_alignment=False) -> ComputationResult:
  """Wraps the TF RNN-T implementation, so that we can compare it against the other implementations."""
  with sess.as_default():
    with tf.name_scope("rnnt_loss"):
      labels_ph = tf.compat.v1.placeholder(tf.int32, [None, None])
      log_probs_ph = tf.compat.v1.placeholder(tf.float32, [None, None, None, None])
      input_lengths_ph = tf.compat.v1.placeholder(tf.int32, [None])
      label_lengths_ph = tf.compat.v1.placeholder(tf.int32, [None])

      res_ph = rnnt_loss(log_probs_ph, labels_ph, input_lengths=input_lengths_ph, label_lengths=label_lengths_ph,
                         blank_index=blank_index, debug=debug, with_alignment=with_alignment)
      if with_alignment:
        costs_ph, alignments_ph = res_ph
      else:
        costs_ph = res_ph
      grads_ph = tf.gradients(xs=log_probs_ph, ys=[-costs_ph])[0]
      fetches = [costs_ph, grads_ph]
      if with_alignment:
        fetches += [alignments_ph]
      fetches_result = sess.run(fetches,
                                feed_dict={log_probs_ph: log_probs,
                                           labels_ph: labels,
                                           input_lengths_ph: input_lengths,
                                           label_lengths_ph: label_lengths})
      if with_alignment:
        costs_tf, grads_tf, alignments_tf = fetches_result
      else:
        costs_tf, grads_tf = fetches_result
        alignments_tf = None
      return ComputationResult("Pure Tensorflow", costs=costs_tf, grads=grads_tf,
                               alignments=alignments_tf)


def wrap_ref_rnnt(log_probs, labels, input_lens, label_lens, blank_index) -> ComputationResult:
  """Wraps the transducer-ref implementation in our format for comparison."""
  list_ll = []
  list_grads = []
  n_batch = log_probs.shape[0]
  for i in range(n_batch):
    log_probs_i = log_probs[i, :input_lens[i], :label_lens[i]+1, :]
    labels_i = labels[i, :label_lens[i]]
    cost_ref, grads_ref = transduce_ref(log_probs_i, labels_i,
                                        blank=blank_index)
    list_ll.append(-cost_ref)
    list_grads.append(grads_ref)
  costs_ref = np.stack(list_ll, axis=0)
  return ComputationResult("Reference", costs=costs_ref, grads=list_grads)


def wrap_numpy_rnnt(log_probs, labels, input_lens, label_lens, blank_index, debug=False,
                    with_alignment=False) -> ComputationResult:
  """Wraps the numpy debug implementation in our format for comparison."""
  list_costs = []
  list_alignments = []
  assert len(log_probs.shape) == 4
  n_batch = log_probs.shape[0]
  for i in range(n_batch):
    log_probs_i = log_probs[i, :input_lens[i], :label_lens[i]+1, :]
    labels_i = labels[i, :label_lens[i]]
    result = numpy_forward(log_probs_i, labels_i,
                           blank_index=blank_index, debug=debug, with_alignment=with_alignment)
    list_costs.append(result.costs)
    list_alignments.append(result.alignments)
  costs = np.stack(list_costs, axis=0)
  alignments = list_alignments if with_alignment else None
  return ComputationResult("Numpy", costs=costs, alignments=alignments)


def wrap_warp_rnnt(log_probs, labels, input_lens, label_lens, blank_index) -> ComputationResult:
  """Warps the external CUDA Warp-RNN-T implementation."""
  with sess.as_default():
    input_lengths_t = tf.constant(input_lens, dtype=tf.int32)  # (B,)
    label_lengths_t = tf.constant(label_lens, dtype=tf.int32)  # (B,)
    log_probs_t = tf.constant(log_probs, dtype=tf.float32)  # (B, T, U, V)
    labels_t = tf.constant(labels, dtype=tf.int32)
    costs_warp_rnnt_tf = warp_rnnt_loss(log_probs_t, labels_t,
                                        input_lengths_t, label_lengths_t, blank_label=blank_index)
    grads_warp_rnnt_tf = tf.gradients(costs_warp_rnnt_tf, [log_probs_t])[0]
    costs_warp_rnnt, grads_warp_rnnt = sess.run([costs_warp_rnnt_tf, grads_warp_rnnt_tf])
    return ComputationResult("Warp Tensorflow", costs=-costs_warp_rnnt, grads=grads_warp_rnnt)


def test_impl(name, acts, labels, blank_index, input_lens=None, label_lens=None,
              debug=False, with_alignment=False):
  """
  runs the different implementations on the same data, comparing them.

  :param str name: test name
  :param np.ndarray acts: (B, T, U, V) or (T, U, V)
  :param np.ndarray labels: (B, U) or (U,)
  :param int blank_index:
  :param np.ndarray label_lens: (B,)|None
  :param np.ndarray input_lens: (B,)|None
  :param bool debug:
  :param bool with_alignment:
  :return: costs, grads
  """
  if len(acts.shape) == 3:  # single -> batched
    assert len(labels.shape) == 1
    acts = np.expand_dims(acts, axis=0)  # (B, T, U, V)
    labels = np.expand_dims(labels, axis=0)  # (B, U)
    n_batch, n_time, n_labels, n_vocab = acts.shape
    assert input_lens is None
    assert label_lens is None
    input_lens = np.array([n_time])  # (B,)=(1,)
    label_lens = np.array([n_labels - 1])
  assert len(acts.shape) == 4  # (B, T, U, V)
  assert input_lens is not None
  assert label_lens is not None
  n_batch, n_time, n_target, n_vocab = acts.shape
  log_probs = log_softmax(acts, axis=-1)  # along vocabulary

  print("Test", colored("%s" % name, "yellow"))

  result_ref = wrap_ref_rnnt(log_probs, labels, input_lens, label_lens, blank_index)
  print(result_ref)

  result_numpy = wrap_numpy_rnnt(log_probs, labels, input_lens, label_lens, blank_index, debug, with_alignment)
  print(result_numpy)

  result_numpy_batched = numpy_forward_shifted_batched(log_probs, labels, blank_index=blank_index,
                                                       input_lens=input_lens, label_lens=label_lens,
                                                       debug=debug, with_alignment=with_alignment)
  print(result_numpy_batched)

  result_warp = wrap_warp_rnnt(log_probs, labels, input_lens, label_lens, blank_index)
  print(result_warp)

  result_tf = wrap_tf_rnnt(log_probs, labels, input_lens, label_lens, blank_index, debug, with_alignment)
  print(result_tf)

  # Do all the tests (ref vs TF), for score, and grads (TF/Ref)
  # We don't have an alpha-matrix anymore (instead there are diagonals)
  np.testing.assert_almost_equal(result_warp.costs, result_ref.costs, decimal=5)
  print("Warp vs Reference: log posterior", colored("MATCH", "green"))

  for i in range(n_batch):
    np.testing.assert_almost_equal(result_warp.grads[i, :input_lens[i], :label_lens[i]+1], result_ref.grads[i],
                                   decimal=5)
  print("Warp vs Reference: gradients    ", colored("MATCH", "green"))

  np.testing.assert_almost_equal(result_tf.costs, result_warp.costs, decimal=5)
  print("TF vs Warp: log posterior       ", colored("MATCH", "green"))

  np.testing.assert_almost_equal(result_tf.grads, result_warp.grads, decimal=4)
  print("TF vs Warp: gradients           ", colored("MATCH", "green"))

  if with_alignment:
    for i in range(n_batch):
      alignment_tf = result_tf.alignments[i, :input_lens[i] + label_lens[i]]
      alignment_np_batched = result_numpy_batched.alignments[i, :input_lens[i] + label_lens[i]]
      if debug:
        print("alignments[%2d] NP  :" % i, result_numpy.alignments[i])
        print("alignments[%2d] NP-b:" % i, alignment_np_batched)
        print("alignments[%2d] TF-b:" % i, alignment_tf)
      non_blank_alignment = alignment_tf[alignment_tf != blank_index]
      np.testing.assert_equal(non_blank_alignment, labels[i][:label_lens[i]])
      np.testing.assert_equal(alignment_np_batched, result_numpy.alignments[i])
      np.testing.assert_almost_equal(alignment_tf,
                                     result_numpy.alignments[i],
                                     decimal=5)
    print("TF vs Numpy: alignments         ", colored("MATCH", "green"))

  print()
  return result_tf.costs, result_tf.grads


def test_alignment():
  """Small test, copied from
    https://github.com/awni/transducer/blob/master/ref_transduce.py
    Also checks alignment generation.
  """
  blank_index = 0
  n_time = 2
  n_target = 3
  n_vocab = 5
  acts = np.array([0.1, 0.5, 0.3, 0.2, 0.1, 0.2,
                   0.1, 0.4, 0.2, 0.2, 0.1, 0.1,
                   0.2, 0.4, 0.1, 0.3, 0.6, 0.3,
                   0.2, 0.1, 0.05, 0.1, 0.2, 0.1,
                   0.1, 0.7, 0.1, 0.2, 0.1, 0.4], dtype=np.float32).reshape((n_time, n_target, n_vocab))

  labels = np.array([1, 2])
  test_impl("test_alignment", acts, labels, blank_index, debug=False, with_alignment=True)


def compute_path_score_from_alignment(alignment, log_probs, labels, blank_index):
  """
  Computes the log-prob score along the given alignment.

  :param np.ndarray alignment: (T+U,) -> [V]
  :param np.ndarray log_probs: (T, U+1, V)
  :param np.ndarray labels: (U,) labels
  :param int blank_index:
  """
  t = u = 0
  elements = []
  for i, symbol in enumerate(alignment):
    if symbol == blank_index:
      t += 1
      elements.append(log_probs[t-1, u, blank_index])
    else:
      u += 1
      elements.append(log_probs[t, u-1, labels[u-1]])
  return sum(elements)


def test_alignment_score():
  """
  In addition to comparing the alignments themselves (Numpy vs TF),
  we also compute the score of the best path (per alignment) which be the same
  score as when we take the tropical semiring for the loss computation.
  :return:
  """
  n_time = 20
  n_target = 8
  n_vocab = 5
  blank_index = 0

  np.random.seed(7)
  acts = np.random.random((n_time, n_target+1, n_vocab))
  labels = np.random.randint(1, n_vocab, size=(n_target,))
  log_probs = log_softmax(acts, axis=-1)
  result = numpy_forward(log_probs, labels, blank_index=blank_index, debug=False, with_alignment=True)
  result_batched = numpy_forward_shifted_batched(log_probs[None], labels[None], blank_index=blank_index,
                                                 input_lens=np.array([n_time]), label_lens=np.array([n_target]),
                                                 debug=False, with_alignment=True)
  result_tf = wrap_tf_rnnt(log_probs[None], labels[None], blank_index=blank_index,
                           input_lengths=np.array([n_time]), label_lengths=np.array([n_target]),
                           debug=False, with_alignment=True)

  # print("Alignment (Numpy)        ", result.alignments, "(len %r)" % len(result.alignments))
  # print("Alignment (Batched Numpy)", result_batched.alignments[0], "(len %r)" % len(result_batched.alignments[0]))
  # print("Alignment (Tensorflow)   ", result_tf.alignments[0], "(len %r)" % len(result_tf.alignments[0]))
  np.testing.assert_equal(result_batched.alignments[0], result.alignments)
  np.testing.assert_equal(result_tf.alignments[0], result.alignments)
  nonzero_alignment = result.alignments[result.alignments != blank_index]
  np.testing.assert_equal(nonzero_alignment, labels)
  assert len(result.alignments[result.alignments == blank_index]) == n_time
  score = compute_path_score_from_alignment(result.alignments[1:], log_probs, labels, blank_index=blank_index)
  score_batched = compute_path_score_from_alignment(result_batched.alignments[0, 1:],
                                                    log_probs, labels, blank_index=blank_index)
  score_tf = compute_path_score_from_alignment(result_tf.alignments[0, 1:],
                                               log_probs, labels, blank_index=blank_index)
  np.testing.assert_equal(score_tf, score)
  np.testing.assert_equal(score_batched, score)

  # The path score using the tropical semiring should be the same as the
  # score computed from the alignment.
  result_tropical = numpy_forward(log_probs, labels, blank_index=blank_index, debug=False, with_alignment=False,
                                  semiring_add=max)  # Tropical Semiring, log-sum-exp -> max
  # TODO: it is fairly close, but not equal, check why!
  np.testing.assert_almost_equal(result_tropical.costs, score, decimal=0)


def generate_random_inputs(n_batch, n_time, n_target, n_vocab, seed=42):
  """Generate some random input for RNN-T tests."""
  np.random.seed(seed)
  acts = np.random.random((n_batch, n_time, n_target + 1, n_vocab))
  labels = np.random.randint(1, n_vocab, size=(n_batch, n_target,))
  label_lengths = np.random.randint(1, n_target, (n_batch,))
  input_lengths = label_lengths + np.random.randint(0, n_time-n_target, (n_batch,))
  return acts, labels, input_lengths, label_lengths


def test_alignment_batched():
  """Test the alignment generation for the batched case.
  This is especially important for seqs of different lengths,
  this case is not trivial for the backtracking (easy to make mistakes)."""
  blank_index = 0
  n_batch = 33
  acts, labels, input_lens, label_lens = generate_random_inputs(n_batch, n_time=20, n_target=8, n_vocab=5)
  log_probs = log_softmax(acts, axis=-1)
  result_batched = numpy_forward_shifted_batched(log_probs, labels, blank_index=blank_index,
                                                 input_lens=input_lens, label_lens=label_lens,
                                                 debug=False, with_alignment=True)
  result_singles = wrap_numpy_rnnt(log_probs, labels, input_lens, label_lens, blank_index=blank_index,
                                   with_alignment=True)
  result_tf = wrap_tf_rnnt(log_probs, labels, input_lens, label_lens, blank_index=blank_index,
                           with_alignment=True)
  for i in range(n_batch):
    max_path = input_lens[i] + label_lens[i]
    alignment_batched = result_batched.alignments[i, :max_path]
    alignment_single = result_singles.alignments[i]
    alignment_tf = result_tf.alignments[i, :max_path]
    np.testing.assert_allclose(result_batched.costs[i], result_singles.costs[i])
    np.testing.assert_allclose(result_tf.costs[i], result_singles.costs[i], rtol=1e-5)
    np.testing.assert_allclose(alignment_batched, alignment_single)
    np.testing.assert_allclose(alignment_tf, alignment_single)


def test_alignment_batched_tf():
  """Test the alignment generation for the batched case.
  This is especially important for seqs of different lengths,
  this case is not trivial for the backtracking (easy to make mistakes)."""
  blank_index = 0
  acts, labels, input_lens, label_lens = generate_random_inputs(n_batch=43, n_time=20, n_target=8, n_vocab=5)
  test_impl("test_alignment_batched_tf", acts, labels, blank_index,
            input_lens, label_lens, debug=False, with_alignment=True)


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

  for n_time in range(3, 12):
    for n_target in range(3, 12):
      acts = np.random.random_sample((n_time, n_target, n_vocab))
      labels = np.random.randint(1, n_vocab-1, (n_target-1,))
      test_impl("test_sizes: T=%d, U=%d" % (n_time, n_target), acts, labels, blank_index, with_alignment=True)


def test_blank_idx_nonzero():
  """Tests when the blank-idx is not 0."""
  n_time = 2
  n_target = 3
  n_vocab = 5
  acts = np.random.standard_normal((n_time, n_target, n_vocab))
  labels = np.array([1, 2])
  # blank_index = 2  # this test breaks when we have a blank_index which is within the labels (e.g. 1 or 2)
  for blank_index in range(3, n_vocab):
    test_impl("test_blank_idx (%d)" % blank_index, acts, labels, blank_index=blank_index)


def test_batched():
  """https://github.com/awni/transducer/blob/master/test.py
  check only first in mini batch.
  """
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

  expected_costs = np.array([4.2806528590890736, 3.9384369822503591])
  expected_grads = np.array([
    [[[-0.4322264564338117, -0.5677735435661883, 0.0],
      [-0.36565009313836844, 0.0, -0.20212345042782007],
      [-0.20212345042782007, 0.0, 0.0]],

     [[-0.16521672442463506, -0.2670097320091765, 0.0],
      [-0.3943653886107811, 0.0, -0.2382944365367636],
      [-0.44041788696458367, 0.0, 0.0]],

     [[-0.052129794015740985, -0.11308693040889405, 0.0],
      [-0.18313786985332664, 0.0, -0.3243144491663483],
      [-0.7647323361309323, 0.0, 0.0]],

     [[0.0, -0.052129794015740985, 0.0],
      [0.0, 0.0, -0.23526766386906767],
      [-1.0, 0.0, 0.0]]],

    [[[-0.7161424128232795, -0.2838575871767207, 0.0],
      [-0.18382932237365335, -0.10002826480306751, 0.0],
      [-0.10002826480306751, 0.0, 0.0]],

     [[-0.41121794618117213, -0.3049244666421072, 0.0],
      [-0.3295759402552584, -0.15917784876050195, 0.0],
      [-0.2592061135635692, 0.0, 0.0]],

     [[-0.11607642141651396, -0.29514152476465827, 0.0],
      [-0.2865333615432337, -0.3381841034766833, 0.0],
      [-0.5973902170402529, 0.0, 0.0]],

     [[0.0, -0.11607642141651396, 0.0],
      [0.0, -0.4026097829597475, 0.0],
      [-1.0, 0.0, 0.0]]]])
  n_batch, n_time, n_labels, n_vocab = acts.shape
  input_lengths = np.array([4]*n_batch)

  labels = np.array([[1, 2], [1, 1]])
  label_lengths = np.array([2] * n_batch)

  costs_tf, grads_tf = test_impl("batched", acts, labels, blank_index=0, input_lens=input_lengths,
                                 label_lens=label_lengths)
  np.testing.assert_almost_equal(costs_tf, -expected_costs, decimal=5)
  print("TF vs Warp: log posterior        ", colored("MATCH", "green"))
  np.testing.assert_almost_equal(grads_tf, expected_grads, decimal=4)
  print("TF vs Warp: gradients            ", colored("MATCH", "green"))


def test_batched_big():
  """
  Test to
  :return:
  """
  blank_index = 0
  acts, labels, input_lens, label_lens = generate_random_inputs(n_batch=33, n_time=20, n_target=8, n_vocab=5)
  test_impl("test_batched_big", acts, labels, blank_index, input_lens, label_lens)


if __name__ == '__main__':
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  sess = tf.Session()
  from returnn.tf.util.basic import setup_tf_thread_pools
  setup_tf_thread_pools(1, log_file=None, tf_session_opts={})
  import better_exchook
  better_exchook.install()

  test_alignment()
  test_alignment_score()
  test_alignment_batched()
  test_alignment_batched_tf()
  test_size_u_greater_t()
  test_size_t_greater_u()
  test_size_t_equal_u()
  test_blank_idx_nonzero()
  test_batched()
  test_batched_big()
  test_sizes()
