# Copyright 2017 Stanford University
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

ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
#LG_RT_DIR	?= legion/runtime
endif

# Flags for directing the runtime makefile what to include
DEBUG           ?= 1		# Include debugging symbols
MAX_DIM         ?= 4		# Maximum number of dimensions
OUTPUT_LEVEL    ?= LEVEL_DEBUG	# Compile time logging level
USE_CUDA        ?= 0		# Include CUDA support (requires CUDA)
USE_HIP         ?= 2
USE_GASNET      ?= 0		# Include GASNet support (requires GASNet)
USE_HDF         ?= 0		# Include HDF5 support (requires HDF5)
ALT_MAPPERS     ?= 0		# Include alternative mappers (not recommended)

# Put the binary file name here
OUTFILE		?= $(app)
# List all the application source files here
GEN_SRC		?= src/runtime/model.cc src/mapper/mapper.cc src/runtime/initializer.cc src/runtime/optimizer.cc\
		src/runtime/strategy.pb.cc src/runtime/strategy.cc $(app).cc
GEN_HIP_SRC	?= src/ops/conv_2d.cu src/runtime/model.cu src/ops/pool_2d.cu src/ops/batch_norm.cu src/ops/linear.cu  \
		src/ops/softmax.cu src/ops/concat.cu src/ops/flat.cu src/ops/embedding.cu src/ops/mse_loss.cu\
		src/runtime/initializer_kernel.cu src/runtime/optimizer_kernel.cu src/runtime/accessor_kernel.cu\
		src/runtime/cuda_helper.cu # .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	?= -Iinclude/ -I/home/wwu/app/protobuf/include -I/home/wwu/app/hipdnn/hipdnn/include
CC_FLAGS	?=
NVCC_FLAGS	?=
GASNET_FLAGS	?=
LD_FLAGS	?= -L$(CUDA_PATH)/lib64 -lcudnn -lcublas -lcublasLt -lcurand -L/home/wwu/app/protobuf/lib -lprotobuf -L/home/wwu/app/hipdnn/hipdnn/lib -lhipdnn -L$(HIP_PATH)/lib -lhipblas
# For Point and Rect typedefs
CC_FLAGS	+= -std=c++11
NVCC_FLAGS  += -std=c++11
###########################################################################
#
#   Don't change anything below here
#   
###########################################################################

include $(LG_RT_DIR)/runtime.mk

