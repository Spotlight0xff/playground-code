#!/usr/bin/env python3
"""
Compares the weights from two similar/equal networks.
Uses the RETURNN framework.

See `returnn/tools/tf_inspect_checkpoint.py` and
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/inspect_checkpoint.py
"""
import os
import sys
import numpy as np
import seaborn

from tensorflow.python import pywrap_tensorflow
from argparse import ArgumentParser, Namespace
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QSizePolicy, QWidget, QListWidget, QHBoxLayout, QListWidgetItem


class PlotSelectWidget(QWidget):
    """Simple Widget which shows a plot and a selection list."""
    def __init__(self, args: Namespace):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'PyQt5 matplotlib'
        self.width = 640
        self.height = 400
        self.weights1 = TensorWeights(args.network1)
        self.weights2 = TensorWeights(args.network2)
        assert self.weights1.tensors.keys() == self.weights2.tensors.keys()
        assert self.weights1.num_params == self.weights2.num_params
        self.plot_canvas = None
        self.init_ui()

    def init_ui(self):
      """Initializes the window."""
      self.setWindowTitle(self.title)
      self.setGeometry(self.left, self.top, self.width, self.height)

      self.plot_canvas = PlotCanvas(self, width=5, height=4)
      self.plot_canvas.move(0, 0)
      # sp_left = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
      # sp_left.setHorizontalStretch(3)

      list_widget = QListWidget()
      list_widget.addItems(self.weights1.tensors.keys())
      list_widget.itemClicked.connect(self.item_clicked)
      # list_widget.setSizeAdjustPolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
      list_widget.setMaximumWidth(300)
      # sp_right = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
      # sp_right.setHorizontalStretch(0.2)
      # list_widget.setSizeAdjustPolicy(sp_right)

      window_layout = QHBoxLayout(self)
      window_layout.setStretch(0, 7)
      window_layout.setStretch(0, 3)
      window_layout.addWidget(self.plot_canvas)
      window_layout.addWidget(list_widget)
      self.setLayout(window_layout)
      self.show()

    def item_clicked(self, item: QListWidgetItem):
      """An item in the list is clicked."""
      tensor1 = self.weights1.tensors[item.text()]
      tensor2 = self.weights2.tensors[item.text()]
      title = "%s %r" % (item.text(), tensor1.shape)
      self.plot_canvas.compare_tensors(tensor1, tensor2, title)


class PlotCanvas(FigureCanvas):
    """Plots a plot into a Qt Canvas"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.tensor1 = None
        self.tensor2 = None
        self.plot()

    def compare_tensors(self, tensor1, tensor2, title):
        """
        :param np.ndarray tensor1:
        :param np.ndarray tensor2:
        :param str title:
        """
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.plot(title)

    def plot(self, title="Compare Weights"):
      """
      Plots histograms of `self.tensor1` and `self.tensor2`.
      :param title:
      """
      ax = self.figure.add_subplot(111)
      ax.cla()
      seaborn.set_style()

      if self.tensor1 is not None:
        red = seaborn.color_palette()[0]
        ax.hist(np.reshape(self.tensor1, (-1,)), bins=50, label="tensor1",
                alpha=0.7,
                color=red)
      if self.tensor2 is not None:
        blue = seaborn.color_palette()[2]
        ax.hist(np.reshape(self.tensor2, (-1,)), bins=50, label="tensor2",
                alpha=0.7, color=blue)
      ax.set_title(title)
      if self.tensor1 is not None:
        mu1 = float(np.mean(self.tensor1))
        mu2 = float(np.mean(self.tensor2))
        sigma1 = float(np.std(self.tensor1))
        sigma2 = float(np.std(self.tensor2))
        ax.legend(labels=["net1 ($\\mu=%.2f, \\sigma=%.2f$)" % (mu1, sigma1),
                          "net2 ($\\mu=%.2f, \\sigma=%.2f$)" % (mu2, sigma2)])
      self.draw()


class TensorWeights:
  """Represents the tensors from the network."""
  def __init__(self, filename):
    assert os.path.exists(filename + ".meta")
    self.checkpoint_path = filename
    self.tensors = {}
    self.num_params = 0
    self.parse_tensors()

  def parse_tensors(self):
    """Parses the tensor from the checkpoint."""
    try:
      reader = pywrap_tensorflow.NewCheckpointReader(self.checkpoint_path)
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in sorted(var_to_shape_map):
        tensor = reader.get_tensor(key)
        self.num_params += int(np.prod(tensor.shape))
        self.tensors[key] = tensor

    except Exception as e:  # pylint: disable=broad-except
      print("Error occurred: %r" % e)
      if "corrupted compressed block contents" in str(e):
        print("It's likely that your checkpoint file has been compressed "
              "with SNAPPY.")
      if ("Data loss" in str(e) and
              (any([e in self.checkpoint_path for e in [".index", ".meta", ".data"]]))):
        proposed_file = ".".join(self.checkpoint_path.split(".")[0:-1])
        v2_file_error_template = """
    It's likely that this is a V2 checkpoint and you need to provide the filename
    *prefix*.  Try removing the '.' and extension.  Try:
    inspect checkpoint --file_name = {}"""
        print(v2_file_error_template.format(proposed_file))


def main():
  """Simple tool to visualize network weights/differences."""
  parser = ArgumentParser()
  parser.add_argument("network1")
  parser.add_argument("network2")
  args = parser.parse_args()

  print("Loading weights from %s" % args.network1)
  weights1 = TensorWeights(args.network1)

  print("Loading weights from %s" % args.network2)
  weights2 = TensorWeights(args.network2)

  assert weights1.tensors.keys() == weights2.tensors.keys()
  assert weights1.num_params == weights2.num_params
  print("Both networks have the same tensor shapes.")
  print("Num params:", weights1.num_params)

  app = QApplication(sys.argv)
  select = PlotSelectWidget(args)
  app.exec()


if __name__ == "__main__":
  import better_exchook
  better_exchook.install()
  main()
