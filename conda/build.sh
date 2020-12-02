# build flexflow
export CUDNN_HOME=/projects/opt/centos7/cuda/10.1
export CUDA_HOME=/projects/opt/centos7/cuda/10.1
export PROTOBUF_DIR=/home/wwu/app/protobuf-3.11.4
export FF_HOME=/home/wwu/FlexFlow
export LG_RT_DIR=/home/wwu/legion-cr/runtime
export LD_LIBRARY_PATH=/home/wwu/app/protobuf-3.11.4/lib:$LD_LIBRARY_PATH
export PYTHON_VERSION_MAJOR=3
export FF_ENABLE_DEBUG=1
export DEBUG=1
cd flexflow/python

make
