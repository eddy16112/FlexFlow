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

from .base_layer import Layer
from .input_layer import Input
from .core_op import _DenseOp, _FlattenOp, _EmbeddingOp, _ActivationOp, _DropoutOp
from flexflow.keras.models.tensor import Tensor
from flexflow.keras.initializers import Zeros, GlorotUniform, RandomUniform, RandomNormal, DefaultInitializer, Initializer

class Dense(Layer):
  __slots__ = ['out_channels', 'activation', 'use_bias', \
               'kernel_initializer', 'bias_initializer']
  def __init__(self, units, input_shape=(0,), 
               activation=None, use_bias=True,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
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
    
    super(Dense, self).__init__('dense', 'Dense', **kwargs) 
    
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
    
    self.out_channels = units
    self.use_bias = use_bias
    if (len(input_shape) == 2):
      op = _DenseOp(self)
      op.in_channels = input_shape[1]
      op.input_shape = (input_shape[0], input_shape[1])
      self.op_list.append(op)
    elif (len(input_shape) == 1):
      op = _DenseOp(self)
      op.in_channels = input_shape[0]
      op.input_shape = (0, input_shape[0])
      self.op_list.append(op)
    if (activation == None):
      self.activation = ff.ActiMode.AC_MODE_NONE
    elif(activation =="relu"):
      self.activation = ff.ActiMode.AC_MODE_RELU
    elif(activation =="sigmoid"):
      self.activation = ff.ActiMode.AC_MODE_SIGMOID
    else:
      assert 0, "activation is not supported"
    
  def get_weights(self, ffmodel):
    return self._get_weights(ffmodel)
    
  def set_weights(self, ffmodel, kernel, bias):
    self._set_weights(ffmodel, kernel, bias)
  
  def __call__(self, input_tensor):
    if len(self.op_list) == 0:
      op = _DenseOp(self)
      self.op_list.append(op)
    else:
      if self.op_list[0].initialized == True:
        op = _DenseOp(self)
        self.op_list.append(op)
      else:
        op = self.op_list[0]
    return op(input_tensor)
    
class Flatten(Layer):
  def __init__(self, data_format=None, **kwargs):
    if data_format != None:
      assert 0, "data_format is not supported"
    super(Flatten, self).__init__('flat', 'Flatten', **kwargs) 
    
  def __call__(self, input_tensor):    
    op = _FlattenOp(self)
    self.op_list.append(op)
    return op(input_tensor)
    
class Embedding(Layer):
  def __init__(self, 
               input_dim,
               output_dim,
               embeddings_initializer="uniform",
               embeddings_regularizer=None,
               activity_regularizer=None,
               embeddings_constraint=None,
               mask_zero=False,
               input_length=None,
               **kwargs):
    self.input_dim = input_dim
    self.out_channels = output_dim
    self.input_length = input_length
    
    if embeddings_initializer == "uniform":
      self.embeddings_initializer = RandomUniform(random.randint(0,1024), -0.05, 0.05)
      
    super(Embedding, self).__init__("embedding", "Embedding", **kwargs) 
    
  def __call__(self, input_tensor):
    op = _EmbeddingOp(self)
    self.op_list.append(op)
    return op(input_tensor)
    
class Activation(Layer):
  def __init__(self, activation=None, **kwargs):
    
    if (activation == 'softmax') or (activation == 'relu') or (activation == 'sigmoid') or (activation == 'tanh') or (activation == 'elu'):
      self.activation = activation
    else:
      assert 0, '[Activation]: unsupported activation'
      
    super(Activation, self).__init__(self.activation, 'Activation', **kwargs) 
    
  def __call__(self, input_tensor):
    op = _ActivationOp(self)
    self.op_list.append(op)
    return op(input_tensor)
    
class Dropout(Layer):
  def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
    if noise_shape != None:
      assert 0, "noise_shape is not supported"
    self.rate = rate
    self.noise_shape = noise_shape
    if seed == None:
      _seed = 0
    self.seed = _seed
      
    super(Dropout, self).__init__('dropout', 'Dropout', **kwargs) 
      
  def __call__(self, input_tensor):
    op = _DropoutOp(self)
    self.op_list.append(op)
    return op(input_tensor)