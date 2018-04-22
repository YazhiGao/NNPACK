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
    enum nnp_convolution_transform_strategy transform_strategy, size_t input_channels,
    size_t output_channels, struct nnp_size input_size, struct nnp_padding input_padding,
    struct nnp_size kernel_size, struct nnp_size output_subsampling, const float *input,
    const float *kernel, const float *bias, float *output, void *workspace_buffer,
    size_t *workspace_size, enum nnp_activation activation, const void *activation_parameters,
    pthreadpool_t threadpool, struct nnp_profile *profile) {
  NNP_TOTAL_START(profile)

  /* Basic validation of parameters. This check detects invalid, but not
   * unsupported parameters. */
  enum nnp_status status = validate_convolution_arguments(
      1, input_channels, output_channels, input_size, input_padding, kernel_size,
      output_subsampling, activation, activation_parameters);
  if (status != nnp_status_success) {
    goto cleanup;
  }

  if (activation_parameters != NULL) {
    status = nnp_status_unsupported_activation_parameters;
    goto cleanup;
  }

  const struct nnp_size output_size = {
      .width = (input_padding.left + input_size.width + input_padding.right - kernel_size.width) /
                   output_subsampling.width +
               1,
      .height =
          (input_padding.top + input_size.height + input_padding.bottom - kernel_size.height) /
              output_subsampling.height +
          1};

  size_t buffer_size = 1024; // Buffer for a batch of output pixels
  for (size_t out_y = 0; out_y < output_size.height; out_y++) {
    for (size_t out_x = 0; out_x < output_size.width; out_x++) {
      per_output_pixel_inference(out_x, out_y, input_channels, output_channels, input_size,
                                 input_padding, kernel_size, output_subsampling, input, kernel,
                                 output, workspace_buffer, workspace_size);
    }
  }

cleanup:
  NNP_TOTAL_END(profile)
  return status;
}

enum void per_output_pixel_inference(size_t out_x, size_t out_y, size_t input_channels,
                                     size_t output_channels, struct nnp_size input_size,
                                     struct nnp_padding input_padding, struct nnp_size kernel_size,
                                     struct nnp_size output_subsampling, const float *input,
                                     const float *kernel, float *output, void *workspace_buffer,
                                     size_t *workspace_size) {
  struct nnp_size input_offset = {.width = };
  for (size_t filter_y = 0; filter_y < kernel_size.height; filter_y++) {
    for (size_t filter_x = 0; filter_x < kernel_size.width, filter_x++) {
    }
  }
}
