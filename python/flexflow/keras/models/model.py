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

from .base_model import BaseModel
from .tensor import Tensor
from flexflow.keras.layers import Conv2D, MaxPooling2D, _FlattenOp, _DenseOp, _ActivationOp, Concatenate, Input, _InputOp

class Model(BaseModel):
  def __init__(self, inputs, outputs, name=None):
    super(Model, self).__init__(name)
    
    if (isinstance(inputs, list) == False):
       inputs = [inputs]
    
    self._input_tensors = inputs
    for input_tensor in inputs:
      self._input_ops.append(input_tensor.from_op)
    self._output_tensor = outputs
    
    self.__traverse_dag_dfs()
    fflogger.debug("nb_layers %d" %(self._nb_ops))
    
  def __call__(self, input_tensor):
    self._output_tensor = input_tensor
    for op in self._ops:
      self._output_tensor = op.layer(self._output_tensor)
    self._input_tensors = [input_tensor]
    return self._output_tensor
    
  def _add_op_metadata(self, op):
    self._ops.append(op)
    #assert layer.layer_id == -1, "layer id is inited"
    assert op.ffhandle == None, "layer handle is inited"
    op.op_id = self._nb_ops
    self._nb_ops += 1       

  def __traverse_dag_bfs(self):
    bfs_queue = []
    for input_op in self._input_ops:
      bfs_queue.append(input_op)
    while(len(bfs_queue) != 0):
      op = bfs_queue.pop(0)
      if (isinstance(op, _InputOp) == False):
       #fflogger.debug(layer)
        self._add_op_metadata(op)
      for child in op.next_ops:
        assert child not in bfs_queue, "already in the stack"
        if child.nb_visited_prev_ops == len(child.prev_ops)-1:
          if child.has_visited == False:
            child.has_visited = True
            bfs_queue.append(child)
        else:
          child.nb_visited_prev_ops += 1
    for op in self._ops:
      op.nb_visited_prev_ops = 0
      op.has_visited = False
    
  def __traverse_dag_dfs(self):    
    dfs_stack = []
    for input_op in reversed(self._input_ops):
      dfs_stack.append(input_op)
    while(len(dfs_stack) != 0):
      op = dfs_stack.pop()
      if (isinstance(op, _InputOp) == False):
        #fflogger.debug(layer)
        self._add_op_metadata(op)
      for child in reversed(op.next_ops):
        assert child not in dfs_stack, "already in the stack"
        if child.nb_visited_prev_ops == len(child.prev_ops)-1:
          if child.has_visited == False:
            child.has_visited = True
            dfs_stack.append(child)
        else:
          child.nb_visited_prev_ops += 1
    for op in self._ops:
      op.nb_visited_prev_ops = 0
      op.has_visited = False
