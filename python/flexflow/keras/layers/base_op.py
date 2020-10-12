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

from flexflow.keras.models.tensor import Tensor

class _Op(object):
  __slots__ = ['ffhandle', 'initialized',\
               'op_id', 'layer', 'prev_ops', 'next_ops',\
               'input_tensors', 'output_tensors', \
               'input_shape', 'output_shape', 'nb_visited_prev_ops', 'has_visited']
  def __init__(self, layer):
    self.ffhandle = None
    self.initialized = False
    self.op_id = -1
    self.layer = layer
    self.prev_ops = []
    self.next_ops = []
    self.input_tensors = []
    self.output_tensors = []
    self.input_shape = None
    self.output_shape = None
    self.nb_visited_prev_ops = 0
    self.has_visited = False;
    
  def reset_connection(self):
    self.prev_ops.clear()
    self.next_ops.clear()
    self.input_tensors.clear()
    self.output_tensors.clear()
    self.nb_visited_prev_ops = 0
    self.initialized = False
    self.has_visited = False

  def set_batch_size(self, size):
    if self.input_shape != None:
      lst = list(self.input_shape)
      lst[0] = size
      self.input_shape = tuple(lst)
    lst = list(self.output_shape)
    lst[0] = size
    self.output_shape = tuple(lst)
    
  def _get_summary_name(self):
    str_name = "{0:25}".format(self.layer.name + " (" + self.layer.layer_type + ")")
    return str_name
    
  def _get_summary_connected_to(self):
    str_name = ""
    for op in self.prev_ops:
      str_name += "\t%s"%(op.layer.name)
    return str_name
    
  def _connect_layer_1_input_1_output(self, input_tensor):
    assert self.initialized == False, "[Layer]: layer is initialized, do not reuse the layer"
    self.initialized = True
    self._calculate_inout_shape(input_tensor)
    output_tensor = Tensor(batch_shape=self.output_shape, dtype=input_tensor.dtype, meta_only=True)
    self._verify_inout_tensor_shape(input_tensor, output_tensor)
    self.input_tensors.append(input_tensor)
    self.output_tensors.append(output_tensor)
    
    output_tensor.set_from_op(self)
    input_tensor.set_to_op(self)
    
    assert input_tensor.from_op != 0, "[Layer]: check input tensor"
    self.prev_ops.append(input_tensor.from_op)
    input_tensor.from_op.next_ops.append(self)

    return output_tensor
    
  def _connect_layer_n_input_1_output(self, input_tensors):
    assert self.initialized == False, "[Layer]: layer is initialized, do not reuse the layer"
    self.initialized = True
    self._calculate_inout_shape(input_tensors)
    output_tensor = Tensor(batch_shape=self.output_shape, dtype=input_tensors[0].dtype, meta_only=True) 
    self._verify_inout_tensor_shape(input_tensors, output_tensor)
    self.output_tensors.append(output_tensor)
    
    output_tensor.set_from_op(self)
    
    for tensor in input_tensors:
      self.input_tensors.append(tensor)
      tensor.set_to_op(self)

      assert tensor.from_op != 0, "check input tensor"
      self.prev_ops.append(tensor.from_op)
      tensor.from_op.next_ops.append(self)
    return output_tensor