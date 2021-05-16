from core.layers import Layer

def test_base_layer_name():
  layer = Layer(name='layer')
  assert layer.name == 'layer'
