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
#if NNP_BACKEND_ARM
static inline void nnp_depthwise_micro_kernel(
    const float* input, const float* kernel, float* output,
    size_t depth_multiplier, size_t input_channels, size_t simd_width) {
  float32x4_t input_simd;
  float32x4_t kernel_simd;
  float32x4_t acc_simd;
  float32x2_t h_input_simd;
  float32x2_t h_kernel_simd;
  float32x2_t h_acc_simd;
  for (size_t depth_multiplier_index = 0;
       depth_multiplier_index < depth_multiplier; depth_multiplier_index++) {
    size_t input_channel_index = 0;
    for (; input_channel_index < input_channels - simd_width;
         input_channel_index += simd_width) {
      float* out_simd = output + depth_multiplier_index * input_channels +
                        input_channel_index;
      input_simd = vld1q_f32(input + input_channel_index);
      kernel_simd = vld1q_f32(kernel + depth_multiplier_index * input_channels +
                              input_channel_index);
      acc_simd = vld1q_f32(output_simd);
      acc_simd = vmlaq_f32(acc_simd, input_simd, kernel_simd);
      vst1q_f32(output_simd, acc_simd);
    }
    size_t h_simd_width = simd_width / 2;
    for (; input_channel_index < input_channels - h_simd_width;
         input_channel_index += h_simd_width) {
      float* h_out_simd = output + depth_multiplier_index * input_channels +
                          input_channel_index;
      h_input_simd = vld1q_f32(input + input_channel_index);
      h_kernel_simd =
          vld1q_f32(kernel + depth_multiplier_index * input_channels +
                    input_channel_index);
      h_acc_simd = vld1_f32(h_output_simd);
      h_acc_simd = vmla_f32(h_acc_simd, h_input_simd, h_kernel_simd);
      vst1_f32(h_output_simd, h_acc_simd);
    }
    for (; input_channel_index < input_channels; input_channel_index++) {
      const float input_s = *(input+ input_channel_index;
      const float kernel_s = *(kernel + depth_multiplier_index * input_channels + input_channel_index);
      *(output+ depth_multiplier_index * input_channels + input_channel_index) = input_s* kernel_s;
    }
  }
}
#endif

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
