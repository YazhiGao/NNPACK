#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <fxdiv.h>

#include <nnpack.h>
#include <nnpack/macros.h>
#include <nnpack/system.h>
#include <nnpack/utils.h>

#include <nnpack/activations.h>
#include <nnpack/hwinfo.h>
#include <nnpack/validation.h>

// very naive implementation

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

  const float(*input)[input_channels][input_size.height][input_size.width] =
      (const float(*)[input_channels][input_size.height][input_size.width])
          context->input_pointer;
  const float(
      *kernel)[depthwise_multiplier][kernel_size.height][kernel_size.width] =
      (const float(*)[depthwise_multiplier][kernel_size.height]
                     [kernel_size.width])context->kernel_pointer;
  float(*output)[output_channels][output_size.height][output_size.width] =
      (float(*)[output_channels][output_size.height][output_size.width])
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
              v += input[sample][input_channel][s][t] *
                   kernel[input_channel][depthwise_channel][i][j];
            }
          }
        }
      }
      output[sample][output_channel][y][x] = v + context->bias[output_channel];
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
enum nnp_status nnp_convolution_depthwise_inference(
    enum nnp_convolution_algorithm algorithm,
    enum nnp_convolution_transform_strategy transform_strategy,
    size_t input_channels, size_t output_channels, struct nnp_size input_size,
    struct nnp_padding input_padding, struct nnp_size kernel_size,
    struct nnp_size output_subsampling, const float* input, const float* kernel,
    const float* bias, float* output, void* workspace_buffer,
    size_t* workspace_size, enum nnp_activation activation,
    const void* activation_parameters, pthreadpool_t threadpool,
    struct nnp_profile* profile) {
  NNP_TOTAL_START(profile)

  /* Basic validation of parameters. This check detects invalid, but not
   * unsupported parameters. */
  enum nnp_status status = validate_convolution_arguments(
      1, input_channels, output_channels, input_size, input_padding,
      kernel_size, output_subsampling, activation, activation_parameters);
  if (status != nnp_status_success) {
    goto cleanup;
  }

  if (activation_parameters != NULL) {
    status = nnp_status_unsupported_activation_parameters;
    goto cleanup;
  }

  const struct nnp_size output_size = {
      .width = (input_padding.left + input_size.width + input_padding.right -
                kernel_size.width) /
                   output_subsampling.width +
               1,
      .height = (input_padding.top + input_size.height + input_padding.bottom -
                 kernel_size.height) /
                    output_subsampling.height +
                1};

  nnp_convolution_depthwise_output__reference(
      1, input_channels, output_channels, output_channels / input_channels,
      input_size, input_padding, kernel_size, output_subsampling, input, kernel,
      bias, output, threadpool);

cleanup:
  NNP_TOTAL_END(profile)
  return status;
}
