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
#include <nnpack/reference.h>

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
      input_padding, kernel_size, output_subsampling, input, kernel, bias,
      output, threadpool);

cleanup:
  NNP_TOTAL_END(profile)
  return status;
}
