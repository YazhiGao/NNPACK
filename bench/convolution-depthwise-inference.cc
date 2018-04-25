#include <vector>

#include <nnpack.h>
#include <nnpack/AlignedAllocator.h>

#include <benchmark/benchmark.h>

static void ConvolutionSetup(benchmark::internal::Benchmark *benchmark) {
  benchmark->Unit(benchmark::kMicrosecond)
      ->ArgNames({"Cin", "Cout", "ImageSize", "StrideOut", "Ptop", "Pbot",
                  "Pleft", "Pright"});
}

class NNPACK : public benchmark::Fixture {
  virtual void SetUp(const benchmark::State &) override {
    const auto status = nnp_initialize();
    assert(status == nnp_status_success);
  }

  virtual void TearDown(const benchmark::State &) override {
    const auto status = nnp_deinitialize();
    assert(status == nnp_status_success);
  }
};

BENCHMARK_DEFINE_F(NNPACK, conv3x3)(benchmark::State &state) {
  const size_t inputChannels = static_cast<size_t>(state.range(0));
  const size_t outputChannels = static_cast<size_t>(state.range(1));
  const size_t imageSize = static_cast<size_t>(state.range(2));
  const size_t strideOut = static_cast<size_t>(state.range(3));
  const size_t ptop = static_cast<size_t>(state.range(4));
  const size_t pbot = static_cast<size_t>(state.range(5));
  const size_t pleft = static_cast<size_t>(state.range(6));
  const size_t pright = static_cast<size_t>(state.range(7));

  std::vector<float> input, kernel, output, bias;
  std::vector<uint8_t, AlignedAllocator<uint8_t, 32>> transformedKernel,
      workspaceBuffer;
  input.resize(inputChannels * imageSize * imageSize);
  kernel.resize(outputChannels * inputChannels);
  bias.resize(outputChannels);
  output.resize(outputChannels * imageSize * imageSize);

  nnp_convolution_transform_strategy strategy =
      nnp_convolution_transform_strategy_compute;
  const nnp_convolution_algorithm algorithm =
      nnp_convolution_algorithm_implicit_gemm;
  const nnp_size imageSize2D = {imageSize, imageSize};
  const nnp_size kernelSize2D = {3, 3};
  const nnp_size outputStride2D = {strideOut, strideOut};
  const nnp_padding imagePadding = {ptop, pbot, pleft, pright};

  size_t workspaceSize = 0;
  nnp_status status = nnp_convolution_depthwise_inference(
      algorithm, strategy, inputChannels, outputChannels, imageSize2D,
      imagePadding, kernelSize2D, outputStride2D, NULL, NULL, NULL, NULL, NULL,
      &workspaceSize, nnp_activation_identity, NULL, NULL, NULL);
  assert(status == nnp_status_success);
  workspaceBuffer.resize(workspaceSize);

  double input_transform_share = 0.0, kernel_transform_share = 0.0,
         output_transform_share = 0.0, matmul_share = 0.0;
  for (auto _ : state) {
    nnp_profile profile;
    status = nnp_convolution_depthwise_inference(
        algorithm, strategy, inputChannels, outputChannels, imageSize2D,
        imagePadding, kernelSize2D, outputStride2D, input.data(),
        transformedKernel.empty() ? kernel.data()
                                  : static_cast<float *>(static_cast<void *>(
                                        transformedKernel.data())),
        bias.data(), output.data(), workspaceBuffer.data(), &workspaceSize,
        nnp_activation_identity, NULL, NULL, &profile);
    assert(status == nnp_status_success);

    input_transform_share += profile.input_transform;
    kernel_transform_share += profile.kernel_transform;
    output_transform_share += profile.output_transform;
    matmul_share += profile.block_multiplication;
  }
  state.counters["Ti"] =
      benchmark::Counter(input_transform_share, benchmark::Counter::kIsRate);
  state.counters["Tk"] =
      benchmark::Counter(kernel_transform_share, benchmark::Counter::kIsRate);
  state.counters["To"] =
      benchmark::Counter(output_transform_share, benchmark::Counter::kIsRate);
  state.counters["MM"] =
      benchmark::Counter(matmul_share, benchmark::Counter::kIsRate);

  state.SetItemsProcessed(state.iterations() * imageSize * imageSize *
                          inputChannels * outputChannels);
}
// conv1
BENCHMARK_REGISTER_F(NNPACK, conv3x3)
    ->Apply(ConvolutionSetup)
    ->Args({32, 32, 112, 1, 1, 1, 1, 1});
// conv2
BENCHMARK_REGISTER_F(NNPACK, conv3x3)
    ->Apply(ConvolutionSetup)
    ->Args({64, 64, 112, 2, 0, 1, 0, 1});
// conv3
BENCHMARK_REGISTER_F(NNPACK, conv3x3)
    ->Apply(ConvolutionSetup)
    ->Args({128, 128, 56, 1, 1, 1, 1, 1});
// conv4
BENCHMARK_REGISTER_F(NNPACK, conv3x3)
    ->Apply(ConvolutionSetup)
    ->Args({128, 128, 56, 2, 0, 1, 0, 1});
// conv5
BENCHMARK_REGISTER_F(NNPACK, conv3x3)
    ->Apply(ConvolutionSetup)
    ->Args({128, 128, 56, 1, 1, 1, 1, 1});
// conv6
BENCHMARK_REGISTER_F(NNPACK, conv3x3)
    ->Apply(ConvolutionSetup)
    ->Args({256, 256, 28, 2, 0, 1, 0, 1});
// conv7
BENCHMARK_REGISTER_F(NNPACK, conv3x3)
    ->Apply(ConvolutionSetup)
    ->Args({512, 512, 14, 1, 1, 1, 1, 1});
// conv8
BENCHMARK_REGISTER_F(NNPACK, conv3x3)
    ->Apply(ConvolutionSetup)
    ->Args({512, 512, 14, 2, 0, 1, 0, 1});
// conv9
BENCHMARK_REGISTER_F(NNPACK, conv3x3)
    ->Apply(ConvolutionSetup)
    ->Args({1024, 1024, 7, 2, 0, 1, 0, 1});
BENCHMARK_MAIN();
