from .layer import Layer

class Input(Layer):
  def __init__(self, shape, **kwargs):
    super().__init__(**kwargs)
    self._shape = shape

  def forward(self, inputs):
      return inputs

  @property
  def output_shape(self):
    return self._shape