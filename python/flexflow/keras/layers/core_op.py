# Copyright 2020 Stanford University, Los Alamos National Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import flexflow.core as ff
from flexflow.core.flexflow_logger import fflogger
import random

from .base_op import _Op
from .input_layer import Input
from flexflow.keras.models.tensor import Tensor
from flexflow.keras.initializers import Zeros, GlorotUniform, RandomUniform, RandomNormal, DefaultInitializer, Initializer

class _DenseOp(_Op):
  __slots__ = ['in_channels']
  def __init__(self, layer):
    super(_DenseOp, self).__init__(layer) 
    self.in_channels = 0
    
  def verify_meta_data(self):
    assert self.input_shape != (0, 0), "input shape is wrong"
    assert self.output_shape != (0, 0), "output shape is wrong"
    assert self.in_channels != 0, " in channels is wrong"
    assert self.layer.out_channels != 0, " out channels is wrong"
    
  def get_summary(self):
    summary = "%s%s\t\t%s%s\n"%(self._get_summary_name(), self.output_shape, self.input_shape, self._get_summary_connected_to())
    return summary
    
  def __call__(self, input_tensor):
    return self._connect_layer_1_input_1_output(input_tensor)
    
  def _calculate_inout_shape(self, input_tensor):
    assert input_tensor.num_dims == 2, "[Dense]: shape of input tensor is wrong"
    input_b = input_tensor.batch_shape[0]
    in_dim = input_tensor.batch_shape[1]
    assert in_dim != 0, "wrong in_dim"
    if (self.in_channels != 0): # check if user input is correct
      assert self.in_channels == in_dim, "wrong input_w"
    self.output_shape = (input_b, self.layer.out_channels)
    self.input_shape = (input_b, in_dim)
    self.in_channels = in_dim
    fflogger.debug("dense input %s, output %s" %( str(self.input_shape), str(self.output_shape)))
    
  def _verify_inout_tensor_shape(self, input_tensor, output_tensor):
    assert input_tensor.num_dims == 2, "[Dense]: check input tensor dims"
    assert input_tensor.batch_shape[1] == self.input_shape[1]
    assert output_tensor.num_dims == 2, "[Dense]: check output tensor dims"
    assert output_tensor.batch_shape[1] == self.output_shape[1]
    
class _FlattenOp(_Op):
  __slots__ = []
  def __init__(self, layer):
    super(_FlattenOp, self).__init__(layer) 
    
  def verify_meta_data(self):
    assert self.input_shape != 0, "input shape is wrong"
    assert self.output_shape != (0, 0), "output shape is wrong"
    
  def get_summary(self):
    summary = "%s%s\t\t%s%s\n"%(self._get_summary_name(), self.output_shape, self.input_shape, self._get_summary_connected_to())
    return summary
    
  def __call__(self, input_tensor):
    return self._connect_layer_1_input_1_output(input_tensor)
    
  def _calculate_inout_shape(self, input_tensor):
    input_shape = input_tensor.batch_shape
    self.input_shape = input_shape
    flat_size = 1
    for i in range(1, len(input_shape)):
      flat_size *= input_shape[i]
    self.output_shape = (input_shape[0], flat_size)
    fflogger.debug("flat input %s, output %s" %( str(self.input_shape), str(self.output_shape)))
    
  def _verify_inout_tensor_shape(self, input_tensor, output_tensor):
    assert input_tensor.num_dims == len(self.input_shape), "[Flatten]: check input tensor dims"
    for i in range (1, input_tensor.num_dims):
      assert input_tensor.batch_shape[i] == self.input_shape[i]
    assert output_tensor.num_dims == 2, "[Flatten]: check output tensor dims"
    assert output_tensor.batch_shape[1] == self.output_shape[1]
    
class _EmbeddingOp(_Op):
  __slots__ = []
  def __init__(self,layer):      
    super(_EmbeddingOp, self).__init__(layer) 
      
  def verify_meta_data(self):
    pass
    
  def get_summary(self):
    summary = "%s%s\t\t%s%s\n"%(self._get_summary_name(), self.output_shape, self.input_shape, self._get_summary_connected_to())
    return summary
    
  def __call__(self, input_tensor):
    return self._connect_layer_1_input_1_output(input_tensor)
    
  def _calculate_inout_shape(self, input_tensor):
    assert input_tensor.num_dims == 2, "[Embedding]: shape of input tensor is wrong"
    input_b = input_tensor.batch_shape[0]
    in_dim = input_tensor.batch_shape[1]
    assert in_dim != 0, "wrong in_dim"
    assert self.layer.input_length == in_dim, "wrong input_w"
    self.output_shape = (input_b, self.layer.out_channels)
    self.input_shape = (input_b, self.input_length)
    fflogger.debug("embedding input %s, output %s" %( str(self.input_shape), str(self.output_shape)))
    
  def _verify_inout_tensor_shape(self, input_tensor, output_tensor):
    assert input_tensor.num_dims == 2, "[Embedding]: check input tensor dims"
    assert input_tensor.batch_shape[1] == self.input_shape[1]
    assert output_tensor.num_dims == 2, "[Embedding]: check output tensor dims"
    assert output_tensor.batch_shape[1] == self.output_shape[1]
    
class _ActivationOp(_Op):
  __slots__ = []
  def __init__(self, layer):  
    super(_ActivationOp, self).__init__(layer) 
      
  def verify_meta_data(self):
    pass
    
  def get_summary(self):
    summary = "%s%s\t\t%s%s\n"%(self._get_summary_name(), self.output_shape, self.input_shape, self._get_summary_connected_to())
    return summary
    
  def __call__(self, input_tensor):
    return self._connect_layer_1_input_1_output(input_tensor)
    
  def _calculate_inout_shape(self, input_tensor):
    self.input_shape = input_tensor.batch_shape
    self.output_shape = input_tensor.batch_shape
    
  def _verify_inout_tensor_shape(self, input_tensor, output_tensor):
    pass
    
class _DropoutOp(_Op):
  __slots__ = []
  def __init__(self, layer):
    super(_DropoutOp, self).__init__(layer) 
      
  def verify_meta_data(self):
    pass
    
  def get_summary(self):
    summary = "%s%s\t\t%s%s\n"%(self._get_summary_name(), self.output_shape, self.input_shape, self._get_summary_connected_to())
    return summary
    
  def __call__(self, input_tensor):
    return self._connect_layer_1_input_1_output(input_tensor)
    
  def _calculate_inout_shape(self, input_tensor):
    self.input_shape = input_tensor.batch_shape
    self.output_shape = input_tensor.batch_shape
    
  def _verify_inout_tensor_shape(self, input_tensor, output_tensor):
    pass