/* Copyright 2018 Stanford
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "model.h"
#include "cuda_helper.h"

Tensor FFModel::conv2d(std::string name,
                       Tensor input, int outChannels,
                       int kernelH, int kernelW,
                       int strideH, int strideW,
                       int paddingH, int paddingW, bool relu)
{
  printf("CP#1\n");
  assert(input.numDim == 4); /*NCHW*/
  int inChannels = input.adim[2];
  bool firstLayer = false;
  //if (input.region == inputImage.region)
  //  firstLayer = true;
  ParallelConfig pc;
  assert(config.find_parallel_config(name, pc));
  IndexSpaceT<4> task_is = IndexSpaceT<4>(get_or_create_task_is(pc));
  Conv2D *conv = new Conv2D(name, config, input, task_is,
                            inChannels, outChannels, kernelH, kernelW,
                            strideH, strideW, paddingH, paddingW,
                            relu, firstLayer);
  layers.push_back(conv);
  return conv->output;
}

/*
locals[0] = kernel
locals[1] = bias
*/
Conv2D::Conv2D(std::string _name, FFConfig _config,
               Tensor _input, IndexSpaceT<4> _task_is,
               int _in_channels, int _out_channels,
               int _kernel_h, int _kernel_w,
               int _stride_h, int _stride_w,
               int _padding_h, int _padding_w,
               bool _relu, bool _first_layer)
: Op(_name, _input), task_is(_task_is),
  in_channels(_in_channels), out_channels(_out_channels),
  kernel_h(_kernel_h), kernel_w(_kernel_w),
  stride_h(_stride_h), stride_w(_stride_w),
  padding_h(_padding_h), padding_w(_padding_w),
  relu(_relu), first_layer(_first_layer), profiling(_config.profiling)
{
  Context ctx = _config.lg_ctx;
  Runtime* runtime = _config.lg_hlr;
  Rect<4> part_rect = runtime->get_index_space_domain(ctx, task_is);
  num_replica = part_rect.volume();
  // Create output tensor
  int input_w = _input.adim[0];
  int input_h = _input.adim[1];
  int output_w = 1 + (input_w + 2 * padding_w - kernel_w) / stride_w;
  int output_h = 1 + (input_h + 2 * padding_h - kernel_h) / stride_h;
  int output_c = out_channels;
  int output_n = _input.adim[3];
  int num_par_w = part_rect.hi[0] - part_rect.lo[0] + 1;
  int num_par_h = part_rect.hi[1] - part_rect.lo[1] + 1;
  int num_par_c = part_rect.hi[2] - part_rect.lo[2] + 1;
  int num_par_n = part_rect.hi[3] - part_rect.lo[3] + 1;
 
  FieldSpace fs = _config.field_space;

  IndexSpaceT<4> output_is;
  {
    //const Legion::coord_t lo[4] = {0, 0, 0, 0};
    //const Legion::coord_t hi[4] = {output_w-1, output_h-1, output_c-1, output_n-1};
    Rect<4> output_rect(Point<4>(0, 0, 0, 0),
        Point<4>(output_w-1, output_h-1, output_c-1, output_n-1));
    output_is = runtime->create_index_space<4>(ctx, output_rect);
  }
  LogicalRegion output_lr = runtime->create_logical_region(ctx, output_is, fs);
  LogicalRegion output_grad_lr = runtime->create_logical_region(ctx, output_is, fs);
  int extent_w = (output_w + num_par_w - 1) / num_par_w;
  int extent_h = (output_h + num_par_h - 1) / num_par_h;
  int extent_c = output_c / num_par_c;
  int extent_n = output_n / num_par_n;
  assert(output_c % num_par_c == 0);
  assert(output_n % num_par_n == 0);
  Transform<4, 4, coord_t> transform;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      transform[i][j] = 0;
  transform[0][0] = extent_w;
  transform[1][1] = extent_h;
  transform[2][2] = extent_c;
  transform[3][3] = extent_n;
  IndexPartition output_ip;
  {
    //int lo[4] = {0, 0, 0, 0};
    //int hi[4] = {extent_w-1, extent_h-1, extent_c-1, extent_n-1};
    Rect<4> extent(Realm::Point<4>(0, 0, 0, 0),
        Realm::Point<4>(extent_w-1, extent_h-1, extent_c-1, extent_n-1));
    output_ip = runtime->create_partition_by_restriction(ctx, output_is, task_is, transform, extent);
    assert(runtime->is_index_partition_disjoint(ctx, output_ip));
    assert(runtime->is_index_partition_complete(ctx, output_ip));
  }
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, output_ip);
  LogicalPartition output_grad_lp =
    runtime->get_logical_partition(ctx, output_grad_lr, output_ip);

  int kernel_nc = num_replica * in_channels * out_channels;
  Rect<1, coord_t> kernel_rect(0, kernel_w * kernel_h * in_channels * out_channels - 1);
  Rect<1, coord_t> kernel_grad_rect(0, kernel_w * kernel_h * kernel_nc - 1);
  IndexSpaceT<1> kernel_is = runtime->create_index_space(ctx, kernel_rect);
  IndexSpaceT<1> kernel_grad_is = runtime->create_index_space(ctx, kernel_grad_rect);
  LogicalRegion kernel_lr = runtime->create_logical_region(ctx, kernel_is, fs);
  LogicalRegion kernel_grad_lr = runtime->create_logical_region(ctx, kernel_grad_is, fs);
  IndexPartition kernel_grad_ip =
    runtime->create_equal_partition(ctx, kernel_grad_is, task_is);
  LogicalPartition kernel_grad_lp =
    runtime->get_logical_partition(ctx, kernel_grad_lr, kernel_grad_ip);
  Tensor kernel_tensor;
  kernel_tensor.numDim = 0;
  kernel_tensor.region = kernel_lr;
  kernel_tensor.region_grad = kernel_grad_lr;
  kernel_tensor.part = LogicalPartition::NO_PART;
  kernel_tensor.part_grad = kernel_grad_lp;
  locals[0] = kernel_tensor;

  int bias_nc = num_replica * out_channels;
  Rect<1, coord_t> bias_grad_rect(0, bias_nc - 1);
  Rect<1, coord_t> bias_rect(0, out_channels - 1);
  IndexSpaceT<1> bias_is = runtime->create_index_space(ctx, bias_rect);
  IndexSpaceT<1> bias_grad_is = runtime->create_index_space(ctx, bias_grad_rect);
  LogicalRegion bias_lr = runtime->create_logical_region(ctx, bias_is, fs);
  LogicalRegion bias_grad_lr =
    runtime->create_logical_region(ctx, bias_grad_is, fs);
  IndexPartition bias_grad_ip =
    runtime->create_equal_partition(ctx, bias_grad_is, task_is);
  LogicalPartition bias_grad_lp =
    runtime->get_logical_partition(ctx, bias_grad_lr, bias_grad_ip);
  Tensor bias_tensor;
  bias_tensor.numDim = 0;
  bias_tensor.region = bias_lr;
  bias_tensor.region_grad = bias_grad_lr;
  bias_tensor.part = LogicalPartition::NO_PART;
  bias_tensor.part_grad = bias_grad_lp;
  locals[1] = bias_tensor;
  numLocals = 2;

  output.numDim = 4;
  output.adim[0] = output_w;
  output.adim[1] = output_h;
  output.adim[2] = out_channels;
  output.adim[3] = _input.adim[3];
  output.pdim[0] = extent_w;
  output.pdim[1] = extent_h;
  output.pdim[2] = extent_c;
  output.pdim[3] = extent_n;
  output.region = output_lr;
  output.part = output_lp;
  output.region_grad = output_grad_lr;
  output.part_grad = output_grad_lp;
  printf("Create conv layer: output(n=%d c=%d h=%d w=%d)\n",
         output.adim[3], output.adim[2], output.adim[1], output.adim[0]);

  // Compute partition bound for input
  Rect<4> input_part_rect =
    runtime->get_index_partition_color_space(ctx, inputs[0].part.get_index_partition());
  if (input_part_rect == part_rect) {
    input_lps[0] = _input.part;
  } else {
    printf("WARNING: input has a different partition!!!\n");
    IndexSpaceT<4> input_is = IndexSpaceT<4>(inputs[0].region.get_index_space());
    //extent_w = stride_w * (output.pdim[0]-1) + kernel_w - 2 * padding_w;
    //extent_h = stride_h * (output.pdim[1]-1) + kernel_h - 2 * padding_h;
    //extent_nc = inputs[0].adim[2] * inputs[0].adim[3] / num_par_n;
    extent_w = (inputs[0].adim[0] + num_par_w - 1) / num_par_w;
    extent_h = (inputs[0].adim[1] + num_par_h - 1) / num_par_h;
    extent_c = inputs[0].adim[2] / num_par_c;
    extent_n = inputs[0].adim[3] / num_par_n;
    assert(inputs[0].adim[2] % num_par_c == 0);
    assert(inputs[0].adim[3] % num_par_n == 0);
    //transform[0][0] = stride_w * output.pdim[0];
    //transform[1][1] = stride_h * output.pdim[1];
    //transform[2][2] = extent_nc;
    transform[0][0] = extent_w;
    transform[1][1] = extent_h;
    transform[2][2] = extent_c;
    transform[3][3] = extent_n;

    IndexPartition input_ip;
    {
      //int lo[4] = {0, 0, 0, 0};
      //int hi[4] = {extent_w-1, extent_h-1, extent_c-1, extent_n-1};
      Rect<4> extent_i(Realm::Point<4>(0, 0, 0, 0),
          Realm::Point<4>(extent_w-1, extent_h-1, extent_c-1, extent_n-1));
      input_ip = runtime->create_partition_by_restriction(ctx,
          input_is, task_is, transform, extent_i);
      assert(runtime->is_index_partition_disjoint(ctx, input_ip));
      assert(runtime->is_index_partition_complete(ctx, input_ip));
    }
    input_lps[0] = runtime->get_logical_partition(ctx, inputs[0].region, input_ip);
  }
}

hipdnnConvolutionFwdAlgo_t
selectConvolutionForwardAlgorithm(hipdnnHandle_t handle,
                                  const hipdnnTensorDescriptor_t xDesc, const void* x,
                                  const hipdnnFilterDescriptor_t wDesc, const void* w,
                                  const hipdnnConvolutionDescriptor_t convDesc,
                                  void* workSpace, size_t workSpaceSize,
                                  const hipdnnTensorDescriptor_t yDesc, void* y);
hipdnnConvolutionBwdFilterAlgo_t
selectConvolutionBackwardFilterAlgorithm(hipdnnHandle_t handle,
                                         const hipdnnTensorDescriptor_t xDesc, const void* x,
                                         const hipdnnTensorDescriptor_t dyDesc, const void* dy,
                                         const hipdnnConvolutionDescriptor_t convDesc,
                                         void* workSpace, size_t workSpaceSize,
                                         const hipdnnFilterDescriptor_t dwDesc, void* dw);
hipdnnConvolutionBwdDataAlgo_t
selectConvolutionBackwardDataAlgorithm(hipdnnHandle_t handle,
                                       const hipdnnFilterDescriptor_t wDesc, const void* w,
                                       const hipdnnTensorDescriptor_t dyDesc, const void* dy,
                                       const hipdnnConvolutionDescriptor_t convDesc,
                                       void* workSpace, size_t workSpaceSize,
                                       const hipdnnTensorDescriptor_t dxDesc, void* dx);
/*
  regions[0]: input
  regions[1]: output
  regions[2](I): filter
  regions[3](I): bias
*/
__host__
OpMeta* Conv2D::init_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  const Conv2D* conv = (Conv2D*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  const AccessorRO<float, 3> acc_input(regions[0], FID_DATA);
  const AccessorWO<float, 3> acc_output(regions[1], FID_DATA);
  const AccessorRO<float, 1> acc_filter(regions[2], FID_DATA);
  const AccessorRO<float, 1> acc_bias(regions[3], FID_DATA);
  Rect<1> rect_filter, rect_bias;
  Rect<3> rect_input, rect_output;
  rect_input = runtime->get_index_space_domain(
                   ctx, task->regions[0].region.get_index_space());
  rect_output = runtime->get_index_space_domain(
                    ctx, task->regions[1].region.get_index_space());
  rect_filter = runtime->get_index_space_domain(
                    ctx, task->regions[2].region.get_index_space());
  rect_bias = runtime->get_index_space_domain(
                  ctx, task->regions[3].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
  assert(acc_filter.accessor.is_dense_arbitrary(rect_filter));
  assert(acc_bias.accessor.is_dense_arbitrary(rect_bias));
  const float *input_ptr = acc_input.ptr(rect_input.lo);
  float *output_ptr = acc_output.ptr(rect_output.lo);
  const float *filter_ptr = acc_filter.ptr(rect_filter.lo);
  const float *bias_ptr = acc_bias.ptr(rect_bias.lo);

  Conv2DMeta* m = new Conv2DMeta(handle);
  m->relu = conv->relu;
  m->first_layer = conv->first_layer;
  checkCUDNN(hipdnnCreateTensorDescriptor(&m->inputTensor));
  checkCUDNN(hipdnnCreateTensorDescriptor(&m->biasTensor));
  checkCUDNN(hipdnnCreateTensorDescriptor(&m->outputTensor));
  checkCUDNN(hipdnnCreateFilterDescriptor(&m->filterDesc));
  checkCUDNN(hipdnnCreateConvolutionDescriptor(&m->convDesc));

  int input_w = rect_input.hi[0] - rect_input.lo[0] + 1;
  int input_h = rect_input.hi[1] - rect_input.lo[1] + 1;
  int output_w = rect_output.hi[0] - rect_output.lo[0] + 1;
  int output_h = rect_output.hi[1] - rect_output.lo[1] + 1;
  printf("init conv (input): n(%d) c(%d) h(%d) w(%d)\n", conv->inputs[0].pdim[3],
         conv->inputs[0].pdim[2], input_h, input_w);
  printf("init conv (output): n(%d) c_out(%d) h(%d) w(%d)\n", conv->output.pdim[3],
         conv->output.pdim[2], output_h, output_w);
  checkCUDNN(hipdnnSetTensor4dDescriptor(m->inputTensor,
                                        HIPDNN_TENSOR_NCHW,
                                        HIPDNN_DATA_FLOAT,
                                        conv->inputs[0].pdim[3],
                                        conv->inputs[0].pdim[2],
                                        input_h,
                                        input_w));
  
  checkCUDNN(hipdnnSetTensor4dDescriptor(m->biasTensor,
                                        HIPDNN_TENSOR_NCHW,
                                        HIPDNN_DATA_FLOAT,
                                        1,
                                        conv->output.pdim[2],
                                        1,
                                        1));

  printf("filterDim: kernel(%d %d) c_out(%d)\n", conv->kernel_h, conv->kernel_w, conv->output.pdim[2]);
  checkCUDNN(hipdnnSetFilter4dDescriptor(m->filterDesc,
                                        HIPDNN_TENSOR_NCHW,
                                        HIPDNN_DATA_FLOAT,
                                        conv->output.pdim[2],
                                        conv->inputs[0].pdim[2],
                                        conv->kernel_h,
                                        conv->kernel_w));

  //printf("convDim: padding(%d %d) stride(%d %d)\n", conv->padding_h, conv->padding_w, conv->stride_h, conv->stride_w);
  int pad_h = ((output_h - 1) * conv->stride_h + conv->kernel_h - input_h + 1) / 2;
  int pad_w = ((output_w - 1) * conv->stride_w + conv->kernel_w - input_w + 1) / 2;
  if (pad_h != conv->padding_h)
    printf("Warning: changing conv_padding_h to satisfy output_h size\n");
  if (pad_w != conv->padding_w)
    printf("Warning: changing conv_padding_w to satisfy output_w size\n");

  checkCUDNN(hipdnnSetConvolution2dDescriptor(m->convDesc,
                                             pad_h,//conv->padding_h,
                                             pad_w,//conv->padding_w,
                                             conv->stride_h,
                                             conv->stride_w,
                                             1/*upscale_x*/,
                                             1/*upscale_y*/,
                                             HIPDNN_CROSS_CORRELATION,
                                             HIPDNN_DATA_FLOAT));

  int n, c, h, w;
  checkCUDNN(hipdnnGetConvolution2dForwardOutputDim(m->convDesc,
                                                   m->inputTensor,
                                                   m->filterDesc,
                                                   &n, &c, &h, &w));
  assert(n == conv->output.pdim[3]);
  assert(c == conv->output.pdim[2]);
  assert(h == output_h);
  assert(w == output_w);

  checkCUDNN(hipdnnSetTensor4dDescriptor(m->outputTensor,
                                        HIPDNN_TENSOR_NCHW,
                                        HIPDNN_DATA_FLOAT,
                                        n, c, h, w));
  // select forward algorithm
  m->fwdAlgo = selectConvolutionForwardAlgorithm(m->handle.dnn, m->inputTensor, input_ptr,
                                                 m->filterDesc, filter_ptr, m->convDesc,
                                                 m->handle.workSpace, m->handle.workSpaceSize,
                                                 m->outputTensor, output_ptr);
  // select backward filter algorithm
  m->bwdFilterAlgo = selectConvolutionBackwardFilterAlgorithm(
                         m->handle.dnn, m->inputTensor, input_ptr,
                         m->outputTensor, output_ptr,
                         m->convDesc, m->handle.workSpace, m->handle.workSpaceSize,
                         m->filterDesc, (void*)filter_ptr);
  // select backward data algorithm
  m->bwdDataAlgo = selectConvolutionBackwardDataAlgorithm(
                       m->handle.dnn, m->filterDesc, filter_ptr,
                       m->outputTensor, output_ptr,
                       m->convDesc, m->handle.workSpace, m->handle.workSpaceSize,
                       m->inputTensor, (void*)input_ptr);
  if (m->relu) {
    checkCUDNN(hipdnnCreateActivationDescriptor(&m->actiDesc));
    checkCUDNN(hipdnnSetActivationDescriptor(m->actiDesc, HIPDNN_ACTIVATION_RELU,
                                            HIPDNN_PROPAGATE_NAN, 0.0, 0.0, 0.0));
  }
  return m;
}

/*
  regions[0](O): filter
  regions[1](O): bias
*/
__host__
void Conv2D::init_para_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const Conv2D* conv = (Conv2D*) task->args;
  const AccessorWO<float, 1> acc_filter(regions[0], FID_DATA);
  const AccessorWO<float, 1> acc_bias(regions[1], FID_DATA);
  Rect<1> rect_filter, rect_bias;
  rect_filter = runtime->get_index_space_domain(
                    ctx, task->regions[0].region.get_index_space());
  rect_bias = runtime->get_index_space_domain(
                  ctx, task->regions[1].region.get_index_space());
  assert(acc_filter.accessor.is_dense_arbitrary(rect_filter));
  assert(acc_bias.accessor.is_dense_arbitrary(rect_bias));
  float *filter_ptr = acc_filter.ptr(rect_filter.lo);
  float *bias_ptr = acc_bias.ptr(rect_bias.lo);
  // init kernel and bias
#ifdef PARAMETER_ALL_ONES
  coord_t filter_elements = conv->inputs[0].adim[2] * conv->output.adim[2] * conv->kernel_h * conv->kernel_w;
  ones_kernel<<<GET_BLOCKS(filter_elements), CUDA_NUM_THREADS>>>(
      filter_ptr, filter_elements);
  ones_kernel<<<GET_BLOCKS(filter_elements), CUDA_NUM_THREADS>>>(
      bias_ptr, conv->output.pdim[2]);
#else
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  hiprandGenerator_t genGPU;
  hiprandCreateGenerator(&genGPU, HIPRAND_RNG_PSEUDO_DEFAULT);
  hiprandSetStream(genGPU, stream);
  hiprandSetPseudoRandomGeneratorSeed(genGPU, 1234ULL);
  coord_t filter_elements = conv->inputs[0].adim[2] * conv->output.adim[2] 
                          * conv->kernel_h * conv->kernel_w;
  float factor = 1.0f / sqrt(filter_elements / conv->output.adim[2]);
  printf("factor = %.4f elements = %d\n", factor, filter_elements / conv->output.adim[2]);
  assert(filter_elements == (coord_t) rect_filter.volume());
  hiprandGenerateUniform(genGPU, filter_ptr, filter_elements);
  scale_kernel<<<GET_BLOCKS(filter_elements), CUDA_NUM_THREADS>>>(
      filter_ptr, filter_elements, -factor, factor);
  hiprandGenerateUniform(genGPU, bias_ptr, conv->output.pdim[2]);
  scale_kernel<<<GET_BLOCKS(conv->output.pdim[2]), CUDA_NUM_THREADS>>>(
      bias_ptr, conv->output.pdim[2], -factor, factor);
  hiprandDestroyGenerator(genGPU);
#endif
}

__host__
void Conv2D::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;

  // First we initialize the filter and bias parameters
  {
    TaskLauncher para_launcher(CONV2D_INIT_PARA_TASK_ID, TaskArgument(this, sizeof(Conv2D)));
    para_launcher.add_region_requirement(
        RegionRequirement(locals[0].region, WRITE_DISCARD, EXCLUSIVE, locals[0].region));
    para_launcher.add_field(0, FID_DATA);
    para_launcher.add_region_requirement(
        RegionRequirement(locals[1].region, WRITE_DISCARD, EXCLUSIVE, locals[1].region));
    para_launcher.add_field(1, FID_DATA);
    runtime->execute_task(ctx, para_launcher);
  }

  Rect<4> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<4> it(rect); it(); it++) {
    FFHandler handle = ff.handlers[idx++];
    argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));
  }
  IndexLauncher init_launcher(CONV2D_INIT_TASK_ID, task_is,
                              TaskArgument(this, sizeof(Conv2D)), argmap);
  init_launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  init_launcher.add_field(0, FID_DATA);
  init_launcher.add_region_requirement(
      RegionRequirement(output.part, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, output.region));
  init_launcher.add_field(1, FID_DATA);
  init_launcher.add_region_requirement(
      RegionRequirement(locals[0].region, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, locals[0].region));
  init_launcher.add_field(2, FID_DATA);
  init_launcher.add_region_requirement(
      RegionRequirement(locals[1].region, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, locals[1].region));
  init_launcher.add_field(3, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
  fm.wait_all_results();
  idx = 0;
  for (PointInRectIterator<4> it(rect); it(); it++) {
    meta[idx++] = fm.get_result<OpMeta*>(*it);
  }
}

/*
  regions[0](I): input
  regions[1](O): output
  regions[2](I): filter
  regions[3](I): bias
*/
__host__
void Conv2D::forward_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  float alpha = 1.0f, beta = 0.0f;
  const Conv2D* conv = (Conv2D*) task->args;
  const Conv2DMeta* m = *((Conv2DMeta**) task->local_args);
  const AccessorRO<float, 4> acc_input(regions[0], FID_DATA);
  const AccessorWO<float, 4> acc_output(regions[1], FID_DATA);
  const AccessorRO<float, 1> acc_filter(regions[2], FID_DATA);
  const AccessorRO<float, 1> acc_bias(regions[3], FID_DATA);
  Rect<4> rect_input, rect_output;
  Rect<1> rect_filter, rect_bias;
  rect_input = runtime->get_index_space_domain(
                   ctx, task->regions[0].region.get_index_space());
  rect_output = runtime->get_index_space_domain(
                    ctx, task->regions[1].region.get_index_space());
  rect_filter = runtime->get_index_space_domain(
                    ctx, task->regions[2].region.get_index_space());
  rect_bias = runtime->get_index_space_domain(
                  ctx, task->regions[3].region.get_index_space());
  //for (int i = 0; i < 3; i++) printf("rect_input.hi = %lld lo = %lld\n", rect_input.hi[i], rect_input.lo[i]);
  //for (int i = 0; i < 3; i++) printf("rect_output.hi = %lld lo = %lld\n", rect_output.hi[i], rect_output.lo[i]);
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
  assert(acc_filter.accessor.is_dense_arbitrary(rect_filter));
  assert(acc_bias.accessor.is_dense_arbitrary(rect_bias));
  const float *input_ptr = acc_input.ptr(rect_input.lo);
  float *output_ptr = acc_output.ptr(rect_output.lo);
  const float *filter_ptr = acc_filter.ptr(rect_filter.lo);
  const float *bias_ptr = acc_bias.ptr(rect_bias.lo);  

  //printf("fwdAlgo(%d), bwdFilterALgo(%d), bwdDataAlgo(%d)\n", (int)m->fwdAlgo,(int) m->bwdFilterAlgo,(int) m->bwdDataAlgo);
  cudaEvent_t t_start, t_end;
  if (conv->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(hipdnnSetStream(m->handle.dnn, stream));
  checkCUDNN(hipdnnConvolutionForward(m->handle.dnn, &alpha,
                                     m->inputTensor, input_ptr,
                                     m->filterDesc, filter_ptr,
                                     m->convDesc, m->fwdAlgo,
                                     m->handle.workSpace, m->handle.workSpaceSize,
                                     &beta, m->outputTensor, output_ptr));

  checkCUDNN(hipdnnAddTensor(m->handle.dnn, &alpha, m->biasTensor,
                            bias_ptr, &alpha, m->outputTensor, output_ptr));
  if (m->relu) {
    checkCUDNN(hipdnnActivationForward(m->handle.dnn, m->actiDesc,
                                      &alpha, m->outputTensor, output_ptr,
                                      &beta, m->outputTensor, output_ptr));
  }
  if (conv->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("Conv2D forward time (CF) = %.2fms\n", elapsed);
  }
}

__host__
void Conv2D::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<4> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<4> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(CONV2D_FWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Conv2D)), argmap);
  launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(output.part, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, output.region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(locals[0].region, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, locals[0].region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(locals[1].region, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, locals[1].region));
  launcher.add_field(3, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): input
  regions[1](O): input_grad
  regions[2](I): output
  regions[3](I/O): output_grad
  regions[4](I): filter
  regions[5](O): filter_grad
  regions[6](O): bias_grad
*/
__host__
void Conv2D::backward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  assert(regions.size() == 7);
  assert(task->regions.size() == 7);
  float alpha = 1.0f, beta = 0.0f;
  const Conv2D* conv = (Conv2D*) task->args;
  const Conv2DMeta* m = *((Conv2DMeta**) task->local_args);
  const AccessorRO<float, 4> acc_input(regions[0], FID_DATA);
  const AccessorWO<float, 4> acc_input_grad(regions[1], FID_DATA);
  const AccessorRO<float, 4> acc_output(regions[2], FID_DATA);
  const AccessorRW<float, 4> acc_output_grad(regions[3], FID_DATA);
  const AccessorRO<float, 1> acc_kernel(regions[4], FID_DATA);
  const AccessorWO<float, 1> acc_kernel_grad(regions[5], FID_DATA);
  const AccessorWO<float, 1> acc_bias_grad(regions[6], FID_DATA);
  Rect<4> rect_input, rect_input_grad, rect_output, rect_output_grad;
  Rect<1> rect_kernel, rect_kernel_grad, rect_bias_grad;
  rect_input =
    runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_input_grad =
    runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  rect_output =
    runtime->get_index_space_domain(ctx, task->regions[2].region.get_index_space());
  rect_output_grad =
    runtime->get_index_space_domain(ctx, task->regions[3].region.get_index_space());
  rect_kernel =
    runtime->get_index_space_domain(ctx, task->regions[4].region.get_index_space());
  rect_kernel_grad =
    runtime->get_index_space_domain(ctx, task->regions[5].region.get_index_space());
  rect_bias_grad =
    runtime->get_index_space_domain(ctx, task->regions[6].region.get_index_space());
  // make sure all regions are dense
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  assert(acc_input_grad.accessor.is_dense_arbitrary(rect_input_grad));
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
  assert(acc_output_grad.accessor.is_dense_arbitrary(rect_output_grad));
  assert(acc_kernel.accessor.is_dense_arbitrary(rect_kernel));
  assert(acc_kernel_grad.accessor.is_dense_arbitrary(rect_kernel_grad));
  assert(acc_bias_grad.accessor.is_dense_arbitrary(rect_bias_grad));
  const float *input_ptr = acc_input.ptr(rect_input.lo);
  float *input_grad_ptr = acc_input_grad.ptr(rect_input_grad.lo);
  const float *output_ptr = acc_output.ptr(rect_output.lo);
  float *output_grad_ptr = acc_output_grad.ptr(rect_output_grad.lo);
  const float *kernel_ptr = acc_kernel.ptr(rect_kernel.lo);
  float *kernel_grad_ptr = acc_kernel_grad.ptr(rect_kernel_grad.lo);
  float *bias_grad_ptr = acc_bias_grad.ptr(rect_bias_grad.lo);

  cudaEvent_t t_start, t_end;
  if (conv->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(hipdnnSetStream(m->handle.dnn, stream));
  if (m->relu) {
    int n = rect_output.volume();
    reluBackward<<<GET_BLOCKS(n), CUDA_NUM_THREADS>>>(output_grad_ptr, output_ptr, n);
  }
  // Compute filter gradiant
  checkCUDNN(hipdnnConvolutionBackwardFilter(m->handle.dnn, &alpha,
                                            m->inputTensor, input_ptr,
                                            m->outputTensor, output_grad_ptr,
                                            m->convDesc, m->bwdFilterAlgo,
                                            m->handle.workSpace, m->handle.workSpaceSize,
                                            &beta, m->filterDesc, kernel_grad_ptr));
  // Compute bias gradiant
  checkCUDNN(hipdnnConvolutionBackwardBias(m->handle.dnn, &alpha,
                                          m->outputTensor, output_grad_ptr,
                                          &beta, m->biasTensor, bias_grad_ptr));
  // no need to compute input_grad if we are the first layer
  if (!m->first_layer) {
    // Compute data gradiant
    checkCUDNN(hipdnnConvolutionBackwardData(m->handle.dnn, &alpha,
                                            m->filterDesc, kernel_ptr,
                                            m->outputTensor, output_grad_ptr,
                                            m->convDesc, m->bwdDataAlgo,
                                            m->handle.workSpace, m->handle.workSpaceSize,
                                            &beta, m->inputTensor, input_grad_ptr));
  }
  if (conv->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("Conv2D backward time = %.2fms\n", elapsed);
  }
}

__host__
void Conv2D::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<4> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<4> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }

  IndexLauncher launcher(CONV2D_BWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Conv2D)), argmap);
  // regions[0](I): input
  launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  // regions[1](O): input_grad (we only need grad tensors)
  launcher.add_region_requirement(
      RegionRequirement(inputs[0].part_grad, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(1, FID_DATA);
  // regions[2](I): output
  launcher.add_region_requirement(
      RegionRequirement(output.part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, output.region));
  launcher.add_field(2, FID_DATA);
  // regions[3](I/O): output_grad
  launcher.add_region_requirement(
      RegionRequirement(output.part_grad, 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, output.region_grad));
  launcher.add_field(3, FID_DATA);
  // regions[4](I): filter
  launcher.add_region_requirement(
      RegionRequirement(locals[0].region, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, locals[0].region));
  launcher.add_field(4, FID_DATA);
  // regions[5](O): filter_grad
  launcher.add_region_requirement(
      RegionRequirement(locals[0].part_grad, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, locals[0].region_grad));
  launcher.add_field(5, FID_DATA);
  // regions[6](O): bias_grad
  launcher.add_region_requirement(
      RegionRequirement(locals[1].part_grad, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, locals[1].region_grad));
  launcher.add_field(6, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  // TODO: remove this line
  //if (first_layer)
    //fm.wait_all_results();
}

/*
  regions[0](I/O): filter
  regions[1](I): filter_grad
  regions[2](I/O): bias
  regions[3](I): bias_grad
*/
__host__
void Conv2D::update_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  const Conv2D* conv = (Conv2D*) task->args;
  const AccessorRW<float, 1> acc_filter(regions[0], FID_DATA);
  const AccessorRO<float, 1> acc_filter_grad(regions[1], FID_DATA);
  const AccessorRW<float, 1> acc_bias(regions[2], FID_DATA);
  const AccessorRO<float, 1> acc_bias_grad(regions[3], FID_DATA);
  Rect<1> rect_filter, rect_filter_grad, rect_bias, rect_bias_grad;
  rect_filter =
    runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_filter_grad =
    runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  rect_bias =
    runtime->get_index_space_domain(ctx, task->regions[2].region.get_index_space());
  rect_bias_grad =
    runtime->get_index_space_domain(ctx, task->regions[3].region.get_index_space());
  size_t filter_size = rect_filter.volume();
  size_t bias_size = rect_bias.volume();
  assert(filter_size == conv->in_channels * conv->out_channels
                        * conv->kernel_w * conv->kernel_h);
  assert(bias_size == conv->out_channels);
  assert(filter_size * conv->num_replica == rect_filter_grad.volume());
  assert(bias_size * conv->num_replica == rect_bias_grad.volume());
  assert(acc_filter.accessor.is_dense_arbitrary(rect_filter));
  assert(acc_filter_grad.accessor.is_dense_arbitrary(rect_filter_grad));
  assert(acc_bias.accessor.is_dense_arbitrary(rect_bias));
  assert(acc_bias_grad.accessor.is_dense_arbitrary(rect_bias_grad));
  float *filter_ptr = acc_filter.ptr(rect_filter.lo);
  const float *filter_grad_ptr = acc_filter_grad.ptr(rect_filter_grad.lo);
  float *bias_ptr = acc_bias.ptr(rect_bias.lo);
  const float *bias_grad_ptr = acc_bias_grad.ptr(rect_bias_grad.lo);
  updateGAS(filter_ptr, filter_grad_ptr, filter_size,
            conv->num_replica, conv->learning_rate);
  updateGAS(bias_ptr, bias_grad_ptr, bias_size,
            conv->num_replica, conv->learning_rate);
}

__host__
void Conv2D::update(const FFModel& ff)
{
  // Synchronize the learning rate
  learning_rate = ff.config.learningRate;
  assert(num_replica > 0);
  // Only aggregate parameters if more than one replica
  if (num_replica > 1) {
    Context ctx = ff.config.lg_ctx;
    Runtime* runtime = ff.config.lg_hlr;
    TaskLauncher launcher(CONV2D_UPD_TASK_ID, TaskArgument(this, sizeof(Conv2D)));
    launcher.add_region_requirement(
      RegionRequirement(locals[0].region, READ_WRITE, EXCLUSIVE, locals[0].region));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
      RegionRequirement(locals[0].region_grad, READ_ONLY, EXCLUSIVE, locals[0].region_grad));
    launcher.add_field(1, FID_DATA);
    launcher.add_region_requirement(
      RegionRequirement(locals[1].region, READ_WRITE, EXCLUSIVE, locals[1].region));
    launcher.add_field(2, FID_DATA);
    launcher.add_region_requirement(
      RegionRequirement(locals[1].region_grad, READ_ONLY, EXCLUSIVE, locals[1].region_grad));
    launcher.add_field(3, FID_DATA);
    runtime->execute_task(ctx, launcher);
  }
}

hipdnnConvolutionFwdAlgo_t
selectConvolutionForwardAlgorithm(hipdnnHandle_t handle,
                                  const hipdnnTensorDescriptor_t xDesc, const void* x,
                                  const hipdnnFilterDescriptor_t wDesc, const void* w,
                                  const hipdnnConvolutionDescriptor_t convDesc,
                                  void* workSpace, size_t workSpaceSize,
                                  const hipdnnTensorDescriptor_t yDesc, void* y)
{
  const int reqAlgCnt = 8;
  int cnt = 0;
  hipdnnConvolutionFwdAlgoPerf_t perfResults[reqAlgCnt];
  checkCUDNN(hipdnnFindConvolutionForwardAlgorithmEx(
      handle, xDesc, x, wDesc, w, convDesc, yDesc, y,
      reqAlgCnt, &cnt, perfResults, workSpace, workSpaceSize));
  assert(cnt > 0);
  checkCUDNN(perfResults[0].status);
  printf("forwardAlgo(%d) time(%.2lf)\n", perfResults[0].algo, perfResults[0].time);
  return perfResults[0].algo;
}

hipdnnConvolutionBwdFilterAlgo_t
selectConvolutionBackwardFilterAlgorithm(hipdnnHandle_t handle,
                                         const hipdnnTensorDescriptor_t xDesc, const void* x,
                                         const hipdnnTensorDescriptor_t dyDesc, const void* dy,
                                         const hipdnnConvolutionDescriptor_t convDesc,
                                         void* workSpace, size_t workSpaceSize,
                                         const hipdnnFilterDescriptor_t dwDesc, void* dw)
{
  const int reqAlgCnt = 8;
  int cnt = 0;
  hipdnnConvolutionBwdFilterAlgoPerf_t perfResults[reqAlgCnt];
  checkCUDNN(hipdnnFindConvolutionBackwardFilterAlgorithmEx(
      handle, xDesc, x, dyDesc, dy, convDesc, dwDesc, dw,
      reqAlgCnt, &cnt, perfResults, workSpace, workSpaceSize));
  assert(cnt > 0);
  checkCUDNN(perfResults[0].status);
  printf("bwdFilterAlgo(%d) time(%.2lf)\n", perfResults[0].algo, perfResults[0].time);
  return perfResults[0].algo;
}

hipdnnConvolutionBwdDataAlgo_t
selectConvolutionBackwardDataAlgorithm(hipdnnHandle_t handle,
                                       const hipdnnFilterDescriptor_t wDesc, const void* w,
                                       const hipdnnTensorDescriptor_t dyDesc, const void* dy,
                                       const hipdnnConvolutionDescriptor_t convDesc,
                                       void* workSpace, size_t workSpaceSize,
                                       const hipdnnTensorDescriptor_t dxDesc, void* dx)
{
  const int reqAlgCnt = 8;
  int cnt = 0;
  hipdnnConvolutionBwdDataAlgoPerf_t perfResults[reqAlgCnt];
  checkCUDNN(hipdnnFindConvolutionBackwardDataAlgorithmEx(
      handle, wDesc, w, dyDesc, dy, convDesc, dxDesc, dx,
      reqAlgCnt, &cnt, perfResults, workSpace, workSpaceSize));
  assert(cnt > 0);
  checkCUDNN(perfResults[0].status);
  printf("bwdDataAlgo(%d) time(%.2lf)\n", perfResults[0].algo, perfResults[0].time);
  return perfResults[0].algo;
}
