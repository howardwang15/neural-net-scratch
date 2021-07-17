import numpy as np

from .layer import Layer

class Input(Layer):
  def __init__(self, shape, **kwargs):
    super().__init__(**kwargs)
    self._shape = shape

  def forward(self, inputs):
    self._validate(inputs)
    return inputs

  def _validate(self, inputs: np.ndarray):
    super()._validate(inputs)

    # Make sure number of dimensions is consistent with what layer expects (plus batch dimension)
    if len(inputs.shape) != len(self._shape) + 1:
      raise ValueError('Number of dimensions must be consistent with input layer shape')

    # Make sure shape of input matches shape expected
    inputs_minus_batch_shape = inputs.shape[1:]
    for actual, expected in zip(inputs_minus_batch_shape, self._shape):
      if actual != expected:
        raise ValueError(f'Dimension {actual} does not match expected dimension of {expected}')

  @property
  def output_shape(self):
    return self._shape
