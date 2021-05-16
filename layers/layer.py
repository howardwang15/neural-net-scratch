class Layer:
  def __init__(self, name):
    self._name = name

  def forward(self, inputs):
    pass

  def backprop(self):
    pass

  @property
  def name(self):
    return self._name
