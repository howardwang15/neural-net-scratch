import numpy as np

class Layer:
  def __init__(self, name):
    self._name = name

  def forward(self, inputs):
    pass

  def backprop(self):
    pass

  def _validate(self, inputs):
    if not isinstance(inputs, np.ndarray):
      raise ValueError('Inputs must be of type numpy.ndarray')

    if inputs.dtype not in ['int32', 'int64', 'float32', 'float64']:
      raise ValueError('Inputs must be of type int32, int64, float32, float64')

    if len(inputs.shape) < 2:
      raise ValueError('Inputs must be of at least dimension 2 (including batch)')

  @property
  def name(self):
    return self._name
