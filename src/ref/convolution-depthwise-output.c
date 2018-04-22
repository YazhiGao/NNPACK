#include <nnpack.h>
#include <nnpack/reference.h>

struct convolution_depthwise_output_context {
  size_t input_channels;
  size_t output_channels;
  size_t depthwise_multiplier;
  struct nnp_size input_size;
  struct nnp_padding input_padding;
  struct nnp_size kernel_size;
  struct nnp_size output_size;
  struct nnp_size output_subsampling;
  const float* input_pointer;
  const float* kernel_pointer;
  const float* bias;
  float* output_pointer;
};

static void compute_convolution_depthwise_output(
    const struct convolution_depthwise_output_context
        context[restrict static 1],
    size_t sample, size_t output_channel) {
  const size_t input_channels = context->input_channels;
  const size_t output_channels = context->output_channels;
  const size_t depthwise_multiplier = context->depthwise_multiplier;
  const struct nnp_size input_size = context->input_size;
  const struct nnp_padding input_padding = context->input_padding;
  const struct nnp_size kernel_size = context->kernel_size;
  const struct nnp_size output_size = context->output_size;
  const struct nnp_size output_subsampling = context->output_subsampling;

  // input layout NHWC
  // kernel layout HWCX
  // output layout NHW(CX)
  const float(*input)[input_size.height][input_size.width][input_channels] =
      (const float(*)[input_size.height][input_size.width][input_channels])
          context->input_pointer;
  const float(
      *kernel)[kernel_size.width][input_channels][depthwise_multiplier] =
      (const float(*)[kernel_size.width][input_channels][depthwise_multiplier])
          context->kernel_pointer;
  float(*output)[output_size.height][output_size.width][output_channels] =
      (float(*)[output_size.height][output_size.width][output_channels])
          context->output_pointer;

  size_t depthwise_channel = output_channel % depthwise_multiplier;
  size_t input_channel = output_channel / depthwise_multiplier;

  for (size_t y = 0; y < output_size.height; y++) {
    for (size_t x = 0; x < output_size.width; x++) {
      double v = 0.0;
      for (size_t i = 0; i < kernel_size.height; i++) {
        const size_t s = y * output_subsampling.height + i - input_padding.top;
        if (s < input_size.height) {
          for (size_t j = 0; j < kernel_size.width; j++) {
            const size_t t =
                x * output_subsampling.width + j - input_padding.left;
            if (t < input_size.width) {
              v += input[sample][s][t][input_channel] *
                   kernel[i][j][input_channel][depthwise_channel];
            }
          }
        }
      }
      output[sample][y][x][output_channel] = v + context->bias[output_channel];
    }
  }
}

void nnp_convolution_depthwise_output__reference(
    size_t batch_size, size_t input_channels, size_t output_channels,
    size_t depthwise_multiplier, struct nnp_size input_size,
    struct nnp_padding input_padding, struct nnp_size kernel_size,
    struct nnp_size output_subsampling, const float input_pointer[],
    const float kernel_pointer[], const float bias[], float output_pointer[],
    pthreadpool_t threadpool) {
  const struct nnp_size output_size = {
      .width = (input_padding.left + input_size.width + input_padding.right -
                kernel_size.width) /
                   output_subsampling.width +
               1,
      .height = (input_padding.top + input_size.height + input_padding.bottom -
                 kernel_size.height) /
                    output_subsampling.height +
                1};
  struct convolution_depthwise_output_context
      convolution_depthwise_output_context = {
          .input_channels = input_channels,
          .output_channels = output_channels,
          .depthwise_multiplier = depthwise_multiplier,
          .input_size = input_size,
          .input_padding = input_padding,
          .kernel_size = kernel_size,
          .output_size = output_size,
          .output_subsampling = output_subsampling,
          .input_pointer = input_pointer,
          .kernel_pointer = kernel_pointer,
          .bias = bias,
          .output_pointer = output_pointer};

  pthreadpool_compute_2d(
      threadpool,
      (pthreadpool_function_2d_t)compute_convolution_depthwise_output,
      &convolution_depthwise_output_context, batch_size, output_channels);
}
