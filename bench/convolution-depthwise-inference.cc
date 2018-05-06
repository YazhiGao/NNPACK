#include <vector>
#include <cstdio>

#include <nnpack.h>
#include <nnpack/system.h>
#include <nnpack/AlignedAllocator.h>

void run(const size_t inputChannels, const size_t outputChannels, const size_t imageSize, const size_t strideOut, const size_t ptop, const size_t pbot, const size_t pleft, const size_t pright){
  double start = read_timer();  
  pthreadpool_t pthreadpool = pthreadpool_create(0);
  // std::vector<float> input, output, bias;
  std::vector<float, AlignedAllocator<float, 32>> kernel, input, output, bias;
  //  std::vector<uint8_t, AlignedAllocator<uint8_t, 32>> transformedKernel, workspaceBuffer;
  input.resize(inputChannels * imageSize * imageSize);
  kernel.resize(outputChannels * inputChannels);
  bias.resize(outputChannels);
  output.resize(outputChannels * imageSize * imageSize);

  nnp_convolution_transform_strategy strategy = nnp_convolution_transform_strategy_compute;
  const nnp_convolution_algorithm algorithm = nnp_convolution_algorithm_implicit_gemm;
  const nnp_size imageSize2D = {imageSize, imageSize};
  const nnp_size kernelSize2D = {3, 3};
  const nnp_size outputStride2D = {strideOut, strideOut};
  const nnp_padding imagePadding = {ptop, pbot, pleft, pright};

  nnp_profile profile;
  int iters = 1000;
  for (int i = 0; i < iters; i++) {
    nnp_status status = nnp_convolution_depthwise_inference(
        algorithm, strategy, inputChannels, outputChannels, imageSize2D, imagePadding,
        kernelSize2D, outputStride2D, input.data(), kernel.data(), bias.data(), output.data(),
        NULL, NULL, nnp_activation_identity, NULL, pthreadpool, &profile);
    assert(status == nnp_status_success);
  }

  pthreadpool_destroy(pthreadpool);
  double duration = read_timer() - start;
  long items = iters * imageSize * imageSize * outputChannels;
  printf("Time used:%lf, item processed:%ld, speed:%lfG/s\n", duration, items, items/duration/(1<<20));  
  fflush(stdout);
}

int main(){
  auto status = nnp_initialize();
  assert(status == nnp_status_success);
  run(32, 32, 112, 1, 1, 1, 1, 1);	
  run(64, 64, 112, 2, 0, 1, 0, 1);
  run(128, 128, 56, 1, 1, 1, 1, 1);
  run(128, 128, 56, 2, 0, 1, 0, 1);
  run(128, 128, 28, 1, 1, 1, 1, 1);
  run(256, 256, 28, 2, 0, 1, 0, 1);
  run(512, 512, 14, 1, 1, 1, 1, 1);
  run(512, 512, 14, 2, 0, 1, 0, 1);
  run(1024, 1024, 7, 2, 0, 1, 0, 1);
  status = nnp_deinitialize();
  assert(status == nnp_status_success);
}
