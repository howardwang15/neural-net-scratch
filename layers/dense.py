from layers import Layer

import numpy as np

class Dense(Layer):
  def __init__(self, input_layer, dim, activation=None, **kwargs):
    super().__init__(**kwargs)
    self._dim = dim
    self._activation = activation
    self._weights = np.random.randn((input_layer.output_shape, dim))
    self._biases = np.random.randn((dim))
  
  @property
  def weights(self):
    return self._weights

  @property
  def biases(self):
    return self._bias