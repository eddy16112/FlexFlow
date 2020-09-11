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

from .base_layer import Layer
from .input_layer import Input
from .merge_op import _ConcatenateOp, _AddOp, _SubtractOp, _MultiplyOp
from flexflow.keras.models.tensor import Tensor

class _Merge(Layer):
  def __init__(self, default_name, layer_type):
    super(_Merge, self).__init__(default_name, layer_type) 
    
def concatenate(input_tensors, _axis=1):
  return Concatenate(axis=_axis)(input_tensors)
    
class Concatenate(_Merge):
  __slots__ = ['axis']
  def __init__(self, axis, **kwargs):
    super(Concatenate, self).__init__("concatenate", "Concatenate", **kwargs) 
    self.axis = axis
    
  def __call__(self, input_tensors):
    op = _ConcatenateOp(self)
    self.op_list.append(op)
    return op(input_tensors)

def add(input_tensors):
  return Add()(input_tensors)
    
class Add(_Merge):
  def __init__(self, **kwargs):
    super(Add, self).__init__("add", "Add", **kwargs) 
    
  def __call__(self, input_tensors):
    op = _AddOp(self)
    self.op_list.append(op)
    return op(input_tensors)
    
def subtract(input_tensors):
  return Subtract()(input_tensors)
    
class Subtract(_Merge):
  def __init__(self, **kwargs):
    super(Subtract, self).__init__("subtract", "Subtract", **kwargs) 
    
  def __call__(self, input_tensors):
    op = _SubtractOp(self)
    self.op_list.append(op)
    return op(input_tensors)

def multiply(input_tensors):
  return Multiply()(input_tensors)
    
class Multiply(_Merge):
  def __init__(self, **kwargs):
    super(Multiply, self).__init__("multiply", "Multiply", **kwargs) 
  
  def __call__(self, input_tensors):
    op = _MultiplyOp(self)
    self.op_list.append(op)
    return op(input_tensors)