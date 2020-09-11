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

from .base_op import _Op
from .input_layer import Input
from flexflow.keras.models.tensor import Tensor

class _MergeOp(_Op):
  def __init__(self, layer):
    super(_MergeOp, self).__init__(layer) 
  
  def verify_meta_data(self):
   pass
    
  def get_summary(self):
    summary = "%s%s%s\n"%(self._get_summary_name(), self.output_shape, self._get_summary_connected_to())
    return summary
    
  def _verify_inout_tensor_shape(self, input_tensors, output_tensor):
    assert self.__check_duplications(input_tensors) == False, "[Merge]: dunpicated input_tensors is not supported"
    for input_tensor in input_tensors:
      assert input_tensor.num_dims == len(self.input_shape), "[Merge]: check input tensor dims"
      for i in range (1, input_tensor.num_dims):
        if isinstance(self, _ConcatenateOp) and self.layer.axis == i:
          continue
        assert input_tensor.batch_shape[i] == self.input_shape[i]
    assert output_tensor.num_dims == len(self.output_shape), "[Merge]: check output tensor dims"
    for i in range (1, output_tensor.num_dims):
      assert output_tensor.batch_shape[i] == self.output_shape[i]
    
  def __check_duplications(self, input_tensors):
    if len(input_tensors) == len(set(input_tensors)):
      return False
    else:
      return True
    
class _ConcatenateOp(_MergeOp):
  __slots__ = []
  def __init__(self, layer):
    super(_ConcatenateOp, self).__init__(layer) 
    
  def __call__(self, input_tensors):
    return self._connect_layer_n_input_1_output(input_tensors)
    
  def _calculate_inout_shape(self, input_tensors):
    if (input_tensors[0].num_dims == 2):
      output_shape = [input_tensors[0].batch_shape[0], 0]
      for input_tensor in input_tensors:
        output_shape[self.layer.axis] += input_tensor.batch_shape[self.layer.axis]
      self.output_shape = (output_shape[0], output_shape[1])
    elif (input_tensors[0].num_dims == 4):
      output_shape = [input_tensors[0].batch_shape[0], 0, input_tensors[0].batch_shape[2], input_tensors[0].batch_shape[3]]
      for input_tensor in input_tensors:
        output_shape[self.layer.axis] += input_tensor.batch_shape[self.layer.axis]
      self.output_shape = (output_shape[0], output_shape[1], output_shape[2], output_shape[3])
    else:
      assert 0, "un-supported dims"
    fflogger.debug("concat output %s" %( str(self.output_shape)))
    self.input_shape = input_tensors[0].batch_shape
    
class _AddOp(_MergeOp):
  __slots__ = []
  def __init__(self, layer):
    super(_AddOp, self).__init__(layer) 
    
  def __call__(self, input_tensors):
    return self._connect_layer_n_input_1_output(input_tensors)
    
  def _calculate_inout_shape(self, input_tensors):    
    assert len(input_tensors) == 2, "check input_tensors"   
    self.input_shape = input_tensors[0].batch_shape
    self.output_shape = input_tensors[0].batch_shape
    fflogger.debug("add output %s" %( str(self.output_shape)))
    
class _SubtractOp(_MergeOp):
  __slots__ = []
  def __init__(self, layer):
    super(_SubtractOp, self).__init__(layer) 
    
  def __call__(self, input_tensors):
    return self._connect_layer_n_input_1_output(input_tensors)
    
  def _calculate_inout_shape(self, input_tensors): 
    assert len(input_tensors) == 2, "check input_tensors"   
    self.input_shape = input_tensors[0].batch_shape
    self.output_shape = input_tensors[0].batch_shape
    fflogger.debug("subtract output %s" %( str(self.output_shape)))
    
class _MultiplyOp(_MergeOp):
  __slots__ = []
  def __init__(self, layer):
    super(_MultiplyOp, self).__init__(layer) 
    
  def __call__(self, input_tensors):
    return self._connect_layer_n_input_1_output(input_tensors)
    
  def _calculate_inout_shape(self, input_tensors): 
    assert len(input_tensors) == 2, "check input_tensors"   
    self.input_shape = input_tensors[0].batch_shape
    self.output_shape = input_tensors[0].batch_shape
    fflogger.debug("multiply output %s" %( str(self.output_shape)))
