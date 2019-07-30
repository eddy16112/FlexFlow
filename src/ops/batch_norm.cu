/* Copyright 2017 Stanford, NVIDIA
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

Tensor FFModel::batch_norm(std::string name, Tensor input, bool relu)
{
  assert(input.numDim == 4); //Only support 4D BN for now
  IndexSpaceT<4> task_is;
  BatchNorm *bn = new BatchNorm(name, config, input, task_is, relu);
  layers.push_back(bn);
  return bn->output;
}

/*
  locals[0] = scale
  locals[1] = bias
*/
BatchNorm::BatchNorm(std::string _name, FFConfig _config,
                     Tensor _input, IndexSpaceT<4> _task_is,
                     bool _relu)
: Op(_name, _input), relu(_relu), profiling(_config.profiling)
{
  Context ctx = _config.lg_ctx;
  HighLevelRuntime* runtime = _config.lg_hlr;
  Rect<4> part_rect = runtime->get_index_space_domain(ctx, task_is);
  num_replica = part_rect.volume();
  // Create output tensor
  int output_w = _input.adim[0];
  int output_h = _input.adim[1];
  int output_c = _input.adim[2];
  int output_n = _input.adim[3];
  int num_par_w = part_rect.hi[0] - part_rect.lo[0] + 1;
  int num_par_h = part_rect.hi[1] - part_rect.lo[1] + 1;
  int num_par_c = part_rect.hi[2] - part_rect.lo[2] + 1;
  int num_par_n = part_rect.hi[3] - part_rect.lo[3] + 1;

  FieldSpace fs = _config.field_space;
  Rect<4> output_rect(Point<4>(0, 0, 0, 0),
      Point<4>(output_w-1, output_h-1, output_c-1, output_n-1));
  IndexSpaceT<4> output_is = runtime->create_index_space(ctx, output_rect);
  LogicalRegion output_lr = runtime->create_logical_region(ctx, output_is, fs);
  LogicalRegion output_grad_lr = runtime->create_logical_region(ctx, output_is, fs);
  int extent_w = (output_w + num_par_w - 1) / num_par_w;
  int extent_h = (output_h + num_par_h - 1) / num_par_h;
  int extent_c = output_c / num_par_c;
  int extent_n = output_n / num_par_n;
  assert(output_c % num_par_c == 0);
  assert(output_n % num_par_n == 0);
  Rect<4> ext(Point<4>(0, 0, 0, 0),
      Point<4>(extent_w-1, extent_h-1, extent_c-1, extent_n-1));
  Transform<4, 4, coord_t> trans;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      trans[i][j] = 0;
  trans[0][0] = extent_w;
  trans[1][1] = extent_h;
  trans[2][2] = extent_c;
  trans[3][3] = extent_n;
  IndexPartition output_ip =
    runtime->create_partition_by_restriction(ctx, output_is, task_is, trans, ext);
  assert(runtime->is_index_partition_disjoint(ctx, output_ip));
  assert(runtime->is_index_partition_complete(ctx, output_ip));
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, output_ip);
  LogicalPartition output_grad_lp =
    runtime->get_logical_partition(ctx, output_grad_lr, output_ip);

  int bias_nc = num_replica * _input.adim[2]; /*input_channels*/
  Rect<1, coord_t> bias_grad_rect(0, bias_nc - 1);
  Rect<1, coord_t> bias_rect(0, _input.adim[2] - 1);
  IndexSpaceT<1> bias_is = runtime->create_index_space(ctx, bias_rect);
  IndexSpaceT<1> bias_grad_is = runtime->create_index_space(ctx, bias_grad_rect);
  LogicalRegion bias_lr = runtime->create_logical_region(ctx, bias_is, fs);
  LogicalRegion scale_lr = runtime->create_logical_region(ctx, bias_is, fs);
  LogicalRegion bias_grad_lr =
    runtime->create_logical_region(ctx, bias_grad_is, fs);
  LogicalRegion scale_grad_lr =
    runtime->create_logical_region(ctx, bias_grad_is, fs);
  IndexPartition bias_grad_ip =
    runtime->create_equal_partition(ctx, bias_grad_is, task_is);
  LogicalPartition bias_grad_lp =
    runtime->get_logical_partition(ctx, bias_grad_lr, bias_grad_ip);
  LogicalPartition scale_grad_lp =
    runtime->get_logical_partition(ctx, scale_grad_lr, bias_grad_ip);

  Tensor scale_tensor, bias_tensor;
  scale_tensor.region = scale_lr;
  scale_tensor.region_grad = scale_grad_lr;
  scale_tensor.part = LogicalPartition::NO_PART;
  scale_tensor.part_grad = scale_grad_lp;
  locals[0] = scale_tensor;
  bias_tensor.region = bias_lr;
  bias_tensor.region_grad = bias_grad_lr;
  bias_tensor.part = LogicalPartition::NO_PART;
  bias_tensor.part_grad = bias_grad_lp;
  locals[1] = bias_tensor;
  numLocals = 2;

  output = _input;
  output.region = output_lr;
  output.part = output_lp;
  output.region_grad = output_grad_lr;
  output.part_grad = output_grad_lp;
  printf("Create bn layer: output(%d %d %d %d)\n",
          output.adim[3], output.adim[2], output.adim[1], output.adim[0]);

  input_lps[0] = _input.part;
}

/*
  regions[0]: input
  regions[1]: output
  regions[2](I): scale
  regions[3](I): bias
*/
__host__
OpMeta* BatchNorm::init_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  const BatchNorm* bm = (BatchNorm*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  const AccessorRO<float, 4> acc_input(regions[0], FID_DATA);
  const AccessorWO<float, 4> acc_output(regions[1], FID_DATA);
  const AccessorRO<float, 1> acc_scale(regions[2], FID_DATA);
  const AccessorRO<float, 1> acc_bias(regions[3], FID_DATA);
  Rect<1> rect_scale, rect_bias;
  Rect<4> rect_input, rect_output;
  rect_input = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_output = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  rect_scale = runtime->get_index_space_domain(ctx, task->regions[2].region.get_index_space());
  rect_bias = runtime->get_index_space_domain(ctx, task->regions[3].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
  assert(acc_scale.accessor.is_dense_arbitrary(rect_scale));
  assert(acc_bias.accessor.is_dense_arbitrary(rect_bias));
  const float *input_ptr = acc_input.ptr(rect_input.lo);
  float *output_ptr = acc_output.ptr(rect_output.lo);
  const float *scale_ptr = acc_scale.ptr(rect_scale.lo);
  const float *bias_ptr = acc_bias.ptr(rect_bias.lo);

  BatchNormMeta* m = new BatchNormMeta(handle);
#ifndef DISABLE_COMPUTATION
  m->relu = bm->relu;
  m->mode = HIPDNN_BATCHNORM_SPATIAL;
#if CUDNN_VERSION >= 7000
  m->mode = HIPDNN_BATCHNORM_SPATIAL_PERSISTENT;
#endif

  checkCUDNN(hipdnnCreateTensorDescriptor(&m->inputTensor));
  checkCUDNN(hipdnnCreateTensorDescriptor(&m->outputTensor));
  checkCUDNN(hipdnnCreateTensorDescriptor(&m->biasTensor));

  assert(rect_input == rect_output);
  int input_w = rect_input.hi[0] - rect_input.lo[0] + 1;
  int input_h = rect_input.hi[1] - rect_input.lo[1] + 1;
  int channel = bm->inputs[0].pdim[2];
  checkCUDNN(hipdnnSetTensor4dDescriptor(m->inputTensor,
                                        HIPDNN_TENSOR_NCHW,
                                        HIPDNN_DATA_FLOAT,
                                        bm->inputs[0].pdim[3],
                                        channel, input_h, input_w));
  checkCUDNN(hipdnnSetTensor4dDescriptor(m->outputTensor,
                                        HIPDNN_TENSOR_NCHW,
                                        HIPDNN_DATA_FLOAT,
                                        bm->inputs[0].pdim[3],
                                        channel, input_h, input_w));
  checkCUDNN(hipdnnSetTensor4dDescriptor(m->biasTensor,
                                        HIPDNN_TENSOR_NCHW,
                                        HIPDNN_DATA_FLOAT,
                                        1, channel, 1, 1));
  //float *runningMean, *runningVar, *saveMean, *saveVar;
  checkCUDA(cudaMalloc(&m->runningMean, sizeof(float) * channel));
  checkCUDA(cudaMalloc(&m->runningVar, sizeof(float) * channel));
  checkCUDA(cudaMalloc(&m->saveMean, sizeof(float) * channel));
  checkCUDA(cudaMalloc(&m->saveVar, sizeof(float) * channel));
  if (m->relu) {
    checkCUDNN(hipdnnCreateActivationDescriptor(&m->actiDesc));
    checkCUDNN(hipdnnSetActivationDescriptor(m->actiDesc, HIPDNN_ACTIVATION_RELU,
                                            HIPDNN_PROPAGATE_NAN, 0.0, 0.0, 0.0));
    
  }
#endif
  return m;
}

/*
  regions[0](O): scale, initilized to ones
  regions[1](O): bias, initilized to zeros
*/
__host__
void BatchNorm::init_para_task(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  //const BatchNorm* bm = (BatchNorm*) task->args;
  const AccessorWO<float, 1> acc_scale(regions[0], FID_DATA);
  const AccessorWO<float, 1> acc_bias(regions[1], FID_DATA);
  Rect<1> rect_scale, rect_bias;
  rect_scale = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_bias = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  assert(acc_scale.accessor.is_dense_arbitrary(rect_scale));
  assert(acc_bias.accessor.is_dense_arbitrary(rect_bias));
  float *scale_ptr = acc_scale.ptr(rect_scale.lo);
  float *bias_ptr = acc_bias.ptr(rect_bias.lo);
  // init kernel and bias
#ifdef PARAMETER_ALL_ONES
  ones_kernel<<<GET_BLOCKS(rect_scale.volume()), CUDA_NUM_THREADS>>>(
      scale_ptr, rect_scale.volume());
  ones_kernel<<<GET_BLOCKS(rect_bias.volume()), CUDA_NUM_THREADS>>>(
      bias_ptr, rect_bias.volume());
#else
  //cudaStream_t stream;
  //checkCUDA(cudaStreamCreate(&stream));
  //curandGenerator_t genGPU;
  //curandCreateGenerator(&genGPU, CURAND_RNG_PSEUDO_DEFAULT);
  //curandSetStream(genGPU, stream);
  //curandSetPseudoRandomGeneratorSeed(genGPU, 1234ULL);
  //curandGenerateUniform(genGPU, scale_ptr, rect_scale.volume());
  assign_kernel<<<GET_BLOCKS(rect_scale.volume()), CUDA_NUM_THREADS>>>(
      scale_ptr, rect_scale.volume(), 1.0f);
  assign_kernel<<<GET_BLOCKS(rect_bias.volume()), CUDA_NUM_THREADS>>>(
      bias_ptr, rect_bias.volume(), 0.0f);
  //curandDestroyGenerator(genGPU);
#endif
}

__host__
void BatchNorm::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  // First we initialize the scale and bias parameters
  {
    TaskLauncher para_launcher(BATCHNORM_INIT_PARA_TASK_ID, TaskArgument(NULL, 0));
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
  IndexLauncher init_launcher(BATCHNORM_INIT_TASK_ID, task_is,
                              TaskArgument(this, sizeof(BatchNorm)), argmap);
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
  regions[1](O): ouptut
  regions[2](I): scale
  regions[3](I): bias
*/
__host__
void BatchNorm::forward_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime)
{
#ifndef DISABLE_COMPUTATION
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  float alpha = 1.0f, beta = 0.0f;
  const BatchNorm* bm = (BatchNorm*) task->args;
  const BatchNormMeta* m = *((BatchNormMeta**) task->local_args);
  const AccessorRO<float, 4> acc_input(regions[0], FID_DATA);
  const AccessorWO<float, 4> acc_output(regions[1], FID_DATA);
  const AccessorRO<float, 1> acc_scale(regions[2], FID_DATA);
  const AccessorRO<float, 1> acc_bias(regions[3], FID_DATA);
  Rect<4> rect_input, rect_output;
  Rect<1> rect_scale, rect_bias;
  rect_input = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_output = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  rect_scale = runtime->get_index_space_domain(ctx, task->regions[2].region.get_index_space());
  rect_bias = runtime->get_index_space_domain(ctx, task->regions[3].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
  assert(acc_scale.accessor.is_dense_arbitrary(rect_scale));
  assert(acc_bias.accessor.is_dense_arbitrary(rect_bias));
  const float *input_ptr = acc_input.ptr(rect_input.lo);
  float *output_ptr = acc_output.ptr(rect_output.lo);
  const float *scale_ptr = acc_scale.ptr(rect_scale.lo);
  const float *bias_ptr = acc_bias.ptr(rect_bias.lo);  

  cudaEvent_t t_start, t_end;
  if (bm->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start);
  }
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(hipdnnSetStream(m->handle.dnn, stream));
  coord_t numChannels = bm->inputs[0].pdim[2];
  assign_kernel<<<GET_BLOCKS(numChannels), CUDA_NUM_THREADS>>>(m->runningMean, numChannels, 0.0f);
  assign_kernel<<<GET_BLOCKS(numChannels), CUDA_NUM_THREADS>>>(m->runningVar, numChannels, 0.0f);
  checkCUDNN(hipdnnBatchNormalizationForwardTraining(
             m->handle.dnn, m->mode, &alpha, &beta, m->inputTensor, (void *)input_ptr,
             m->outputTensor, (void *)output_ptr, m->biasTensor, (void *)scale_ptr, (void *)bias_ptr,
             1.0, m->runningMean, m->runningVar, HIPDNN_BN_MIN_EPSILON,
             m->saveMean, m->saveVar));
  if (bm->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("BatchNorm forward time (BF) = %.2fms\n", elapsed);
  }
#endif
}

__host__
void BatchNorm::forward(const FFModel& ff)
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
  IndexLauncher launcher(BATCHNORM_FWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(BatchNorm)), argmap);
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
  regions[4](I): scale
  regions[5](O): scale_grad
  regions[6](O): bias_grad
*/
__host__
void BatchNorm::backward_task(const Task *task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx, Runtime *runtime)
{
#ifndef DISABLE_COMPUTATION
  assert(regions.size() == 7);
  assert(task->regions.size() == 7);
  float alpha = 1.0f, beta = 0.0f;
  const BatchNorm* bm = (BatchNorm*) task->args;
  const BatchNormMeta* m = *((BatchNormMeta**) task->local_args);
  const AccessorRO<float, 4> acc_input(regions[0], FID_DATA);
  const AccessorWO<float, 4> acc_input_grad(regions[1], FID_DATA);
  const AccessorRO<float, 4> acc_output(regions[2], FID_DATA);
  const AccessorRW<float, 4> acc_output_grad(regions[3], FID_DATA);
  const AccessorRO<float, 1> acc_scale(regions[4], FID_DATA);
  const AccessorWO<float, 1> acc_scale_grad(regions[5], FID_DATA);
  const AccessorWO<float, 1> acc_bias_grad(regions[6], FID_DATA);
  Rect<4> rect_input, rect_input_grad, rect_output, rect_output_grad;
  Rect<1> rect_scale, rect_scale_grad, rect_bias_grad;
  rect_input =
    runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_input_grad =
    runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  rect_output =
    runtime->get_index_space_domain(ctx, task->regions[2].region.get_index_space());
  rect_output_grad =
    runtime->get_index_space_domain(ctx, task->regions[3].region.get_index_space());
  rect_scale =
    runtime->get_index_space_domain(ctx, task->regions[4].region.get_index_space());
  rect_scale_grad =
    runtime->get_index_space_domain(ctx, task->regions[5].region.get_index_space());
  rect_bias_grad =
    runtime->get_index_space_domain(ctx, task->regions[6].region.get_index_space());
  // make sure all regions are dense
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  assert(acc_input_grad.accessor.is_dense_arbitrary(rect_input_grad));
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
  assert(acc_output_grad.accessor.is_dense_arbitrary(rect_output_grad));
  assert(acc_scale.accessor.is_dense_arbitrary(rect_scale));
  assert(acc_scale_grad.accessor.is_dense_arbitrary(rect_scale_grad));
  assert(acc_bias_grad.accessor.is_dense_arbitrary(rect_bias_grad));
  const float *input_ptr = acc_input.ptr(rect_input.lo);
  float *input_grad_ptr = acc_input_grad.ptr(rect_input_grad.lo);
  const float *output_ptr = acc_output.ptr(rect_output.lo);
  float *output_grad_ptr = acc_output_grad.ptr(rect_output_grad.lo);
  const float *scale_ptr = acc_scale.ptr(rect_scale.lo);
  float *scale_grad_ptr = acc_scale_grad.ptr(rect_scale_grad.lo);
  float *bias_grad_ptr = acc_bias_grad.ptr(rect_bias_grad.lo);

  cudaEvent_t t_start, t_end;
  if (bm->profiling) {
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
  checkCUDNN(hipdnnBatchNormalizationBackward(
             m->handle.dnn, m->mode, &alpha, &beta, &alpha, &beta,
             m->inputTensor, input_ptr, m->outputTensor, output_grad_ptr,
             m->inputTensor, input_grad_ptr, m->biasTensor, scale_ptr,
             scale_grad_ptr, bias_grad_ptr, HIPDNN_BN_MIN_EPSILON,
             m->saveMean, m->saveVar));
  if (bm->profiling) {
    cudaEventRecord(t_end);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("BatchNorm backward time = %.2fms\n", elapsed);
  }
#endif
}

__host__
void BatchNorm::backward(const FFModel& ff)
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

  IndexLauncher launcher(BATCHNORM_BWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(BatchNorm)), argmap);
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
}

__host__
void BatchNorm::update(const FFModel& ff)
{
  //FIXME: we didn't sync batch norm parameters for now
}
