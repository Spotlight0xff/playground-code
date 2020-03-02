#!/usr/bin/env python3
# vim: sw=2
"""
Implementation of the RNA loss in pure TF,
plus comparisons against reference implementations.
This is very similar to RNN-T loss, but restricts
the paths to be strictly monotonic.

references:
  * recurrent neural aligner:
      https://pdfs.semanticscholar.org/7703/a2c5468ecbee5b62c048339a03358ed5fe19.pdf
"""
import os
import sys
import numpy as np
import tensorflow as tf
from termcolor import colored
from ref_transduce import transduce as transduce_ref
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "returnn"))
from ref_rna import forward_pass, analytical_gradient, backward_pass

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


def py_print_iteration_info(msg, var, n, *vars, debug=True):
  """adds a tf.print op to the graph while ensuring it will run (when the output is used)."""
  if not debug:
    return var
  var_print = tf.print("n=", n, "\t", msg, tf.shape(var), var, *vars,
                       summarize=-1, output_stream=sys.stdout)
  with tf.control_dependencies([var_print]):
    var = tf.identity(var)
  return var


def numpy_forward(log_probs, labels, blank_index, debug=False):
  """Forward calculation of the RNA loss."""
  n_time, n_target, n_vocab = log_probs.shape  # (T, U, V)
  alpha = np.zeros((n_time+1, n_target))  # 1 in log-space
  print("a = alpha[t-1, u - 1] + log_probs[t - 1, u - 1, labels[u - 1]]")
  print("b = alpha[t-1, u] + log_probs[t-1, u,  blank]")
  print("alpha[t,u] = LSE(a,b)")
  debug = True
  if debug:
    print("U=%d, T=%d, V=%d" % (n_target, n_time, n_vocab))
  for t in range(1, n_time):
    # blank - blank - blank - ...
    alpha[t, 0] = alpha[t - 1, 0] + log_probs[t - 1, 0, blank_index]
    print('t=%2d u= 0: alpha[%d, 0] + log_probs[%d, 0, %d] = %.3f + %.3f = %.3f' % (
      t, t-1, t-1, blank_index, alpha[t - 1, 0], log_probs[t - 1, 0, blank_index], alpha[t, 0]))

    # label - label - label - ...
    u = t
    if u < n_target:
      alpha[t, u] = alpha[t-1, u-1] + log_probs[t-1, u-1, labels[u-1]]

#  for u in range(1, n_target):  # first column
#    alpha[0, u] = alpha[0, u - 1] + log_probs[0, u - 1, labels[u - 1]]
#    print('t= 0 u=%2d: alpha[0, %d] + log_probs[0, %d, labels[%d]=%d] = %.3f + %.3f = %.3f' % (
#      u, u-1, u-1, u-1, labels[u - 1], alpha[0, u - 1], log_probs[0, u - 1, labels[u - 1]], alpha[0, u]))

  for t in range(1, n_time+1):
    for u in range(1, min(t, n_target)):
      skip = alpha[t - 1, u] + log_probs[t - 1, u, blank_index]
      emit = alpha[t - 1, u - 1] + log_probs[t - 1, u - 1, labels[u - 1]]
      alpha[t, u] = elem = logsumexp(skip, emit)  # addition in linear-space -> LSE in log-space
      print('t=%2d u=%2d: LSE(%.3f + %.3f, %.3f +  %.3f) = LSE(%.3f, %.3f) = %.3f' % (t, u,
                                                                                      alpha[t - 1, u],
                                                                                      log_probs[t - 1, u, blank_index],
                                                                                      alpha[t - 1, u - 1],
                                                                                      log_probs[t - 1, u - 1,
                                                                                                labels[u - 1]],
                                                                                      skip, emit, elem))

  if debug:
    assert len(alpha.shape) == 2
    print("Alpha matrix: (%d, %d)" % tuple(alpha.shape))
    print(alpha)
  nll = - alpha[n_time, n_target-1]
  if debug:
    print("negative log-likelihood = - alpha[%d, %d] = %.4f" %
          (n_time, n_target-1, nll))
  return alpha, nll


def numpy_forward_shifted_batched(log_probs, labels, blank_index, input_lens, label_lens, debug=False):
  """Forward calculation of the RNA loss using the same diagonal strategy."""
  n_batch, max_time, max_target, n_vocab = log_probs.shape  # (B, T, U, V)
  assert labels.shape == (n_batch, max_target-1)  # (B, U-1)
  if debug:
    print("U=%d, T=%d, V=%d" % (max_target, max_time, n_vocab))
    print("log-probs: (B=%d, T=%d, U=%d, V=%d)" % (n_batch, max_time, max_target, n_vocab))
    print("labels: (B=%d, U-1=%d)" % (n_batch, labels.shape[1]))
  num_diagonals = max_time + max_target

  def print_debug(n, *vars):
    """Some basic debug information printing."""
    if debug:
      print("[n=%2d]" % n, *vars)
  # alpha diagonals
  alphas = [[], np.zeros((n_batch, 1))]

  for n in range(2, max_time+2):
    # actually previous one.
    lp_diagonal = log_probs[:, n-2, :n-1]
    # lp_diagonal = shifted_logprobs[:, n-2, :n-1]  # (B, n-1, V)
    print_debug(n, "lp_diagonal", lp_diagonal)

    diag_maxlen = min(max_target, n)
    prev_diagonal = alphas[n-1][:, :diag_maxlen]
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

    if debug:
      print("\n")

  list_nll = []
  diag_idxs = input_lens + 1  # (B,)

  # (B,): batch index -> index within diagonal
  # We need to handle the U>T case for each example.
  within_diag_idx = label_lens
  for i in range(n_batch):
    ta_item = alphas[diag_idxs[i]]  # (B, N)

    a = ta_item[i, within_diag_idx[i]]
    # b = log_probs[i, input_lens[i]-1, label_lens[i], blank_index]
    if debug:
      print("FINAL i=%d, diag_idx=%d, within_diag_idx=%d" % (i, diag_idxs[i], within_diag_idx[i]))
      print("FINAL i=%d" % i, "NLL=%.3f" % (-a))
    nll = - a # + b
    list_nll.append(nll)
  return np.array(list_nll)  # (B,)


def tf_shift_logprobs(mat, axis, axis_to_expand):
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
  max_time = tf.shape(mat)[axis]  # T
  n_batch = tf.shape(mat)[0]
  max_target = tf.shape(mat)[axis_to_expand]
  n_vocab = tf.shape(mat)[-1]
  shifts = tf.expand_dims(tf.cast(tf.range(max_time), tf.float32), axis=1)  # (T,1)
  shifts = shifts[tf.newaxis, :, :, tf.newaxis]
  shifts = tf.tile(shifts, [n_batch, 1, 1, n_vocab])
  pads = tf.zeros((n_batch, max_time, max_time, n_vocab), dtype=tf.float32)
  # (B, T, 1, V) ; (B, T, U, V) ; (B, T, T, V)
  # -> (B, T, U+T+1, V)
  a_ranged = tf.concat([shifts, mat, pads], axis=axis_to_expand)  # (T, U+1)

  def fn(x):  # x: (B, U+T+1, V)
    """Computes the shift per diagonal and pads accordingly."""
    shift = tf.cast(x[0][0][0], tf.int32)  # (B,)
    # 1:U+1 is the original data, in front: shift as wanted, back: padding for shape
    n = tf.pad(x[:, 1:max_target + 1, :], [[0, 0],  # B
                                           [shift, max_time + 1 - shift],  # U+T+1
                                           [0, 0]  # V
                                           ], constant_values=0)
    return n

  t = tf.map_fn(fn, elems=tf.transpose(a_ranged, [1, 0, 2, 3]))
  t = tf.transpose(t, [1, 0, 2, 3])
  return t[:, :, :-1, :]


def tf_forward_shifted_rna(log_probs, labels, input_lengths=None, label_lengths=None, blank_index=0, debug=False):
  """
  Computes the batched forward pass of the RNA model.
  B: batch, T: time, U:target/labels, V: vocabulary

  :param tf.Tensor log_probs: (B, T, U, V) log-probabilities
  :param tf.Tensor labels: (B, U-1) -> [V] labels
  :param tf.Tensor input_lengths: (B,) length of input frames
  :param tf.Tensor label_lengths: (B,) length of labels
  :param int blank_index: index of the blank symbol in the vocabulary
  :param bool debug: enable verbose logging
  :return:
  """
  """Pure TF implementation of the RNA loss."""
  shape = tf.shape(log_probs)
  n_batch = shape[0]     # B
  max_time = shape[1]    # T
  max_target = shape[2]  # U

  num_diagonals = max_time + 2

  labels = py_print_iteration_info("labels", labels, 0, debug=debug)

  log_probs_ta = tf.TensorArray(
    dtype=tf.float32,
    clear_after_read=False,
    size=num_diagonals,
    dynamic_size=False,
    infer_shape=False,
    element_shape=(None, None, None),  # (B, U, V)
    name="log_probs",
  )
  # (B, T, U, V) -> [(B, U, V)] * (T)
  log_probs_ta = log_probs_ta.unstack(tf.transpose(log_probs, [1, 0, 2, 3]))

  alpha_ta = tf.TensorArray(
    dtype=tf.float32,
    clear_after_read=False,
    size=num_diagonals,
    dynamic_size=False,
    infer_shape=False,
    element_shape=(None, None,),  # (B, n)
    name="alpha_diagonals",
  )
  alpha_ta = alpha_ta.write(1, tf.zeros((n_batch, 1)))

  def cond(n, *args):
    """We run the loop until all elements are covered by diagonals.
    """
    return tf.less(n, num_diagonals)

  def body_forward(n, alpha_ta):
    """body of the while_loop, loops over the diagonals of the alpha-tensor."""
    # alpha(t-1,u-1) + logprobs(t-1, u-1)
    # alpha_blank      + lp_blank

    lp_diagonal = log_probs_ta.read(n-2)[:, :n-1, :]  # (B, U|n, V)
    lp_diagonal = py_print_iteration_info("lp_diagonal", lp_diagonal, n, debug=debug)

    diag_maxlen = tf.reduce_min([max_target, n])
    prev_diagonal = alpha_ta.read(n-1)[:, :diag_maxlen]  # (B, n-1)
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

    labels_maxlen = tf.minimum(max_target-1, n-1)
    labels_shifted = labels[:, :labels_maxlen]  # (B, U-1|n-1)
    labels_shifted = py_print_iteration_info("labels_shifted", labels_shifted, n, debug=debug)
    batchs, rows = tf.meshgrid(
      tf.range(n_batch),
      tf.range(labels_maxlen),
      indexing='ij'
    )
    lp_y_idxs = tf.stack([batchs, rows, labels_shifted], axis=-1)  # (B, U-1|n-1, 3)
    lp_y_idxs = py_print_iteration_info("lp_y_idxs", lp_y_idxs, n, debug=debug)
    lp_y = tf.gather_nd(lp_diagonal[:, :, :], lp_y_idxs)  # (B, U)
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
    return [n + 1, alpha_ta.write(n, new_diagonal)]

  n = tf.constant(2)
  final_n, alpha_out_ta = tf.while_loop(cond, body_forward, [n, alpha_ta],
                                        parallel_iterations=1,  # need this due to the iterative computation using TAs
                                        name="rna_loss")

  # p(y|x) = alpha(T,U) * blank(T,U)  (--> in log-space)
  # ll_tf = final_alpha[n_time-1, n_target-1]

  # (B,): batch index -> diagonal index
  diag_idxs = input_lengths +1   # (B,)

  # (B,): batch index -> index within diagonal
  within_diag_idx = label_lengths
  within_diag_idx = tf.where(tf.less_equal(label_lengths, input_lengths),
      within_diag_idx,  # everything ok, T>U
      tf.ones_like(within_diag_idx) * -1)  #  U > T, not possible in RNA

  res_ta = tf.TensorArray(
    dtype=tf.float32,
    clear_after_read=True,
    size=n_batch,
    dynamic_size=False,
    infer_shape=False,
    element_shape=(),
    name="alpha_diagonals",
  )
  tf_neg_inf = tf.constant(NEG_INF)

  def ta_read_body(i, res_loop_ta):
    """Reads from the alpha-diagonals TensorArray. We need this because of the inconsistent shapes in the TA."""
    ta_item = alpha_out_ta.read(diag_idxs[i])[i]
    elem = tf.cond(tf.equal(within_diag_idx[i], -1), lambda: tf_neg_inf, lambda: ta_item[within_diag_idx[i]])
    elem = py_print_iteration_info("FINAL", elem, i, "diag_idxs", diag_idxs, "within_diag_idx:",within_diag_idx, debug=debug)
    return i+1, res_loop_ta.write(i, elem)

  _, ll_ta = tf.while_loop(
    lambda i, res_ta: i < n_batch,
    ta_read_body, (tf.constant(0, tf.int32), res_ta)
  )
  return ll_ta.stack()


def test_impl(name, acts, labels, blank_index, input_lens=None, label_lens=None,
              timing=False, debug=False):
  """
  runs the different implementations on the same data, comparing them.

  :param name: test name
  :param np.ndarray acts: (B, T, U, V) or (T, U, V)
  :param np.ndarray labels: (B, U) or (U,)
  :param int blank_index:
  :param label_lens: (B,)|None
  :param input_lens: (B,)|None
  :param timing:
  :param debug:
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

  def print_results(name, cost, grads):
    """Prints the results of an implementation."""
    cost = np.sum(cost)
    if grads is None:
      grads_msg = "n/a"
    else:
      grads_msg = "%.3f" % np.linalg.norm(grads)
    print(colored("%20s" % name, "red"),
          "implementation: log-posterior=%.3f, |grads|=%s" % (
            cost, grads_msg))
  print("Test", colored("%s" % name, "yellow"))

  list_ll = []
  list_grads = []
  list_alphas = []
  for i in range(n_batch):
    log_probs_i = log_probs[i, :input_lens[i] + 1, :label_lens[i] + 1]
    labels_i = labels[i, :label_lens[i]]
    alphas, ll_forward = forward_pass(log_probs_i, labels_i, blank_index)
    betas, ll_backward = backward_pass(log_probs_i, labels_i, blank_index)
    assert np.allclose(ll_forward, ll_backward, atol=1e-12, rtol=1e-12),\
      "Log-likelihood from forward and backward pass mismatch."
    analytical_grads = analytical_gradient(log_probs_i, alphas, betas, labels_i, blank_index)
    list_ll.append(-ll_forward)
    list_alphas.append(alphas)
    list_grads.append(analytical_grads)

    # alpha_np, cost_np = numpy_forward(log_probs_i, labels_i, blank_index, debug)
  costs_ref = np.stack(list_ll, axis=0)
  print_results("Reference", costs_ref, list_grads)

  costs_np = numpy_forward_shifted_batched(log_probs, labels, blank_index=blank_index,
                                           input_lens=input_lens, label_lens=label_lens,
                                           debug=debug)
  np.testing.assert_almost_equal(costs_np, costs_ref, decimal=5)
  print_results("NumPy", costs_np, None)

  with sess.as_default():
    labels_ph = tf.compat.v1.placeholder(tf.int32, [None, None])
    log_probs_ph = tf.compat.v1.placeholder(tf.float32, [None, None, None, None])
    input_lengths_ph = tf.compat.v1.placeholder(tf.int32, [None])
    label_lengths_ph = tf.compat.v1.placeholder(tf.int32, [None])
    tf_run_opts = {}
    if timing:
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      tf_run_opts = {'options': run_options, 'run_metadata': run_metadata}

    costs_ph = tf_forward_shifted_rna(log_probs_ph, labels_ph,
                                      input_lengths=input_lengths_ph,
                                      label_lengths=label_lengths_ph,
                                      blank_index=blank_index, debug=debug)
    grads_ph = tf.gradients(xs=log_probs_ph, ys=[-costs_ph])[0]
    ll_tf, grads_tf = sess.run([costs_ph, grads_ph],
                               feed_dict={log_probs_ph: log_probs,
                                          labels_ph: labels,
                                          input_lengths_ph: input_lens,
                                          label_lengths_ph: label_lens}, **tf_run_opts)
    if timing:
      max_bytes = sess.run(tf.contrib.memory_stats.MaxBytesInUse())
      print("max bytes in use:", max_bytes)
      from tensorflow.python.client import timeline
      tl = timeline.Timeline(run_metadata.step_stats)
      ctf = tl.generate_chrome_trace_format()
      with open('timeline_tf_impl_%s.json' % name, 'w') as f:
        f.write(ctf)
    nll_tf = -ll_tf
  print_results("Tensorflow", nll_tf, grads_tf)

  # Do all the tests (ref vs TF), for score, and grads
  np.testing.assert_almost_equal(nll_tf, costs_ref, decimal=4)
  print("TF vs Reference: log posterior ", colored("MATCH", "green"))
  for i in range(n_batch):
    np.testing.assert_almost_equal(grads_tf[i], list_grads[i], decimal=4)
  print("TF vs Reference: gradients     ", colored("MATCH", "green"))
  print()
  return nll_tf, grads_tf


def test_small():
  """Small test, copied from
    https://github.com/awni/transducer/blob/master/ref_transduce.py
  """
  blank_index = 0
  vocab_size = 4
  input_len = 5
  output_len = 4
  acts = np.random.rand(input_len, output_len, vocab_size)
  labels = np.random.randint(1, vocab_size, output_len-1)
  test_impl("test_small", acts, labels, blank_index)


def test_size_t_greater_u():
  """Tests for case when T > 2*U"""
  blank_index = 0
  n_time = 20
  n_target = 3
  n_vocab = 5
  acts = np.random.random_sample((n_time, n_target, n_vocab))
  labels = np.array([1, 2])
  test_impl("test_size: T>U", acts, labels, blank_index)


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
    for n_target in range(3, n_time+2):
      acts = np.random.random_sample((n_time, n_target, n_vocab))
      labels = np.random.randint(1, n_vocab-1, (n_target-1,))
      test_impl("test_sizes: T=%d, U=%d" % (n_time, n_target), acts, labels, blank_index)


def test_blank_idx_nonzero():
  """Tests when the blank-idx is not 0."""
  n_time = 2
  n_target = 3
  n_vocab = 5
  acts = np.random.standard_normal((n_time, n_target, n_vocab))
  labels = np.array([1, 2])
  for blank_index in range(n_vocab):
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

  n_batch, n_time, n_labels, n_vocab = acts.shape
  input_lengths = np.array([4]*n_batch)

  labels = np.array([[1, 2], [1, 1]])
  label_lengths = np.array([2] * n_batch)

  test_impl("batched", acts, labels, blank_index=0, input_lens=input_lengths,
            label_lens=label_lengths, timing=False)


def test_big():
  """Big test, with timing.
  """
  blank_index = 0
  vocab_size = 50
  input_len = 200
  output_len = 40
  acts = np.random.rand(input_len, output_len, vocab_size)
  labels = np.random.randint(1, vocab_size, output_len-1)
  test_impl("test_big", acts, labels, blank_index, timing=True)


if __name__ == '__main__':
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  sess = tf.Session()
  import better_exchook
  better_exchook.install()

  np.random.seed(42)

  test_small()
  test_size_t_greater_u()
  test_size_t_equal_u()
  test_sizes()
  test_blank_idx_nonzero()
  test_batched()
  test_big()
