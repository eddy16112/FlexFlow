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

from .base_op import _Op
from flexflow.keras.models.tensor import Tensor

class Layer(object):
  __slots__ = ['_name', 'layer_type', 'op_list']
  def __init__(self, default_name, layer_type, **kwargs):
    name = default_name
    if 'name' in kwargs:
      name = kwargs["name"]
      
    self._name = name
    self.layer_type = layer_type
    self.op_list = []
    
  @property
  def name(self):
    return self._name
    
  @property
  def input(self):
    op = self.op_list[0]
    if (len(op.input_tensors) == 1):
      return op.input_tensors[0]
    else:
      return op.input_tensors 
      
  @property
  def output(self):
    op = self.op_list[0]
    if (len(op.output_tensors) == 1):
      return op.output_tensors[0]
    else:
      return op.output_tensors
    
  def _get_weights(self, ffmodel):
    assert self.op_list[0]._ffhandle != None, "handle is not set correctly"
    kernel_parameter = self.op_list[0]._ffhandle.get_weight_tensor()
    bias_parameter = self.op_list[0]._ffhandle.get_bias_tensor()
    kernel_array = kernel_parameter.get_weights(ffmodel)
    bias_array = bias_parameter.get_weights(ffmodel)
    return (kernel_array, bias_array)
    
  def _set_weights(self, ffmodel, kernel, bias):
    assert self.op_list[0]._ffhandle != None, "handle is not set correctly"
    kernel_parameter = self.op_list[0]._ffhandle.get_weight_tensor()
    bias_parameter = self.op_list[0]._ffhandle.get_bias_tensor()
    kernel_parameter.set_weights(ffmodel, kernel)
    bias_parameter.set_weights(ffmodel, bias)