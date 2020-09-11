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
import math

from .base_layer import Layer
from .input_layer import Input
from .pool_op import _Pooling2DOp
from flexflow.keras.models.tensor import Tensor

class Pooling2D(Layer):
  __slots__ = ['in_channels', 'out_channels', 'kernel_size', 'stride', \
               'padding', 'pool_type']
  def __init__(self, pool_size, strides, padding="valid", default_name="pool2d", pool_type=ff.PoolType.POOL_MAX, layer_type="MaxPooling2D", **kwargs):
    super(Pooling2D, self).__init__(default_name, layer_type, **kwargs) 
    
    assert len(pool_size)==2, "wrong dim of pool_size"
    self.kernel_size = pool_size
    assert len(strides)==2, "wrong dim of strides"
    self.stride = strides
    if (padding == "valid"):
      self.padding = (0, 0)
    elif (padding == "same"):
      self.padding = "same"
    elif (isinstance(padding, list) or isinstance(padding, tuple)):
      assert len(padding)==2, "[Pooling2D]: wrong dim of padding"
      self.padding = tuple(padding)
    else:
      assert 0, "[Pooling2D]: check padding"
    self.pool_type = pool_type
    
  def __call__(self, input_tensor):
    op = _Pooling2DOp(self)
    self.op_list.append(op)
    return op(input_tensor)
    
class MaxPooling2D(Pooling2D):
  def __init__(self, pool_size=(2, 2), strides=None, padding="valid", data_format=None, **kwargs):
    if data_format == 'channels_last':
      assert 0, "data_format channels_last is not supported"
    super(MaxPooling2D, self).__init__(pool_size, strides, padding, "maxpool2d", ff.PoolType.POOL_MAX, "MaxPooling2D", **kwargs) 
    
class AveragePooling2D(Pooling2D):
  def __init__(self, pool_size=(2, 2), strides=None, padding="valid", data_format=None, **kwargs):
    if data_format == 'channels_last':
      assert 0, "data_format channels_last is not supported"
    super(AveragePooling2D, self).__init__(pool_size, strides, padding, "averagepool2d", ff.PoolType.POOL_AVG, "AveragePooling2D", **kwargs) 