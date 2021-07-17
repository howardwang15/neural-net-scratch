from core.layers import Input

import pytest
import re
import numpy as np


def test_input_layer_name():
  # Test that `name` parameter gets constructed properly from base `Layer` class
  input_layer = Input(name='input_layer', shape=(256,))
  assert input_layer.name == 'input_layer'


def test_input_layer_shape():
  # Test that `shape` parameter gets constructed properly
  input_layer = Input(name='input_layer', shape=(256,))
  assert input_layer.output_shape == (256,)


def test_input_layer_forward_numpy_type_check():
  # Test that input argument to `forward` is a NumPy array
  input_layer = Input(name='input_layer', shape=(2,))
  input_data = [0, 1]
  with pytest.raises(ValueError, match='Inputs must be of type numpy.ndarray'):
    input_layer.forward(input_data)
  
  input_data = 5
  with pytest.raises(ValueError, match='Inputs must be of type numpy.ndarray'):
    input_layer.forward(input_data)


def test_input_layer_forward_numpy_inner_type_check():
  # Test that NumPy input argument to `forward` contains int32, int64, float32, float64 values 
  input_layer = Input(name='input_layer', shape=(2,))
  input_data = np.array(['a', 'b'])
  with pytest.raises(ValueError, match='Inputs must be of type int32, int64, float32, float64'):
    input_layer.forward(input_data)

  input_data = np.array([['e']])
  with pytest.raises(ValueError, match='Inputs must be of type int32, int64, float32, float64'):
    input_layer.forward(input_data)

  input_data = np.array([[5]], dtype=np.int16)
  with pytest.raises(ValueError, match='Inputs must be of type int32, int64, float32, float64'):
    input_layer.forward(input_data)


def test_input_layer_forward_shape_outer_check():
  # Test that number of dimensions of `Input` layer matches number of dimensions passed to `forward`
  input_layer = Input(name='input_layer', shape=(2,))
  input_data = np.array([0, 1])
  with pytest.raises(ValueError, match=re.escape('Inputs must be of at least dimension 2 (including batch)')):
    input_layer.forward(input_data)

  input_data = np.array([[[0, 1]]])
  with pytest.raises(ValueError, match='Number of dimensions must be consistent with input layer shape'):
    input_layer.forward(input_data)

  input_layer = Input(name='input_layer', shape=(2, 3))
  input_data = np.array([[1, 2]])
  with pytest.raises(ValueError, match='Number of dimensions must be consistent with input layer shape'):
    input_layer.forward(input_data) 


def test_input_layer_forward_shape_inner_check():
  input_layer = Input(name='input_layer', shape=(2,))
  input_data = np.array([[1, 2, 3]])
  with pytest.raises(ValueError, match=f'Dimension 3 does not match expected dimension of 2'):
    input_layer.forward(input_data)

def test_input_layer_forward_proper():
  input_layer = Input(name='input_layer', shape=(2,))
  input_data = np.array([[1, 2]])
  expected = np.array([[1, 2]])
  res = input_layer.forward(input_data)
  assert np.array_equal(res, expected)
