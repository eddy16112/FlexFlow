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
from .convolutional_op import _Conv2DOp
from flexflow.keras.models.tensor import Tensor
from flexflow.keras.initializers import Zeros, GlorotUniform, RandomUniform, RandomNormal, DefaultInitializer, Initializer

class Conv2D(Layer):
  __slots__ = ['out_channels', 'kernel_size', 'stride', \
               'padding', 'activation', 'use_bias', 'kernel_initializer', \
               'bias_initializer']
  def __init__(self, 
               filters, 
               input_shape=None, 
               kernel_size=0, 
               strides=(1, 1), 
               padding="valid", 
               data_format=None, 
               dilation_rate=(1, 1),
               groups=1, 
               activation=None, 
               use_bias=True, 
               kernel_initializer='glorot_uniform', 
               bias_initializer='zeros', 
               kernel_regularizer=None, 
               bias_regularizer=None, 
               activity_regularizer=None, 
               kernel_constraint=None, 
               bias_constraint=None, 
               **kwargs):
    if data_format == 'channels_last':
      assert 0, "data_format channels_last is not supported"
    if dilation_rate != (1,1):
      assert 0, "dilation_rate is not supported"
    if groups != 1:
      assert 0, "groups is not supported"
    if kernel_regularizer != None:
      assert 0, "kernel_regularizer is not supported"
    if bias_regularizer != None:
      assert 0, "bias_regularizer is not supported"
    if activity_regularizer != None:
      assert 0, "activity_regularizer is not supported"
    if kernel_constraint != None:
      assert 0, "kernel_constraint is not supported"
    if bias_constraint != None:
      assert 0, "bias_constraint is not supported"
    
    super(Conv2D, self).__init__("conv2d", "Conv2D", **kwargs) 
    
    if kernel_initializer == "glorot_uniform":
      self.kernel_initializer = DefaultInitializer()
    elif isinstance(kernel_initializer, Initializer) == True:
      self.kernel_initializer = kernel_initializer
    else:
      assert 0, "[Dense]: unknown kernel_initializer"
      
    if bias_initializer == "zeros":
      self.bias_initializer = DefaultInitializer()
    elif isinstance(bias_initializer, Initializer) == True:
      self.bias_initializer = bias_initializer
    else:
      assert 0, "[Dense]: unknown bias_initializer"
    
    self.out_channels = filters
    assert len(kernel_size)==2, "wrong dim of kernel_size"
    self.kernel_size = kernel_size
    assert len(strides)==2, "wrong dim of stride"
    self.stride = strides
    if padding == "valid":
      self.padding = (0, 0)
    elif padding == "same":
      self.padding = "same"
    elif (isinstance(padding, list) or isinstance(padding, tuple)):
      assert len(padding)==2, "[Conv2D]: wrong dim of padding"
      self.padding = tuple(padding)
    else:
      assert 0, "[Conv2D]: check padding"
    if (activation == None):
      self.activation = ff.ActiMode.AC_MODE_NONE
    elif(activation =="relu"):
      self.activation = ff.ActiMode.AC_MODE_RELU
    else:
      assert 0, "activation is not supported"
    if input_shape != None:
      if len(input_shape) == 4:
        op = _Conv2DOp(self)
        op.in_channels = input_shape[1]
        op.input_shape = (input_shape[0], input_shape[1], input_shape[2], input_shape[3])
        self.op_list.append(op)
      elif len(input_shape) == 3:
        op = _Conv2DOp(self)
        op.in_channels = input_shape[0]
        op.input_shape = (0, input_shape[0], input_shape[1], input_shape[2])
        self.op_list.append(op)
    self.use_bias = use_bias
    
  def get_weights(self, ffmodel):
    return self._get_weights(ffmodel)
    
  def set_weights(self, ffmodel, kernel, bias):
    self._set_weights(ffmodel, kernel, bias)
    
  def __call__(self, input_tensor):
    if len(self.op_list) == 0:
      op = _Conv2DOp(self)
      self.op_list.append(op)
    else:
      if self.op_list[0].initialized == True:
        op = _Conv2DOp(self)
        self.op_list.append(op)
      else:
        op = self.op_list[0]
    return op(input_tensor)
