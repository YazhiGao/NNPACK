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
#include <nnpack/arm_neon.h>
#include <nnpack/hwinfo.h>
#include <nnpack/macros.h>
#include <nnpack/validation.h>
#include <pthreadpool.h>
// Assume that the output channel can be evenly divided by this block size
#define output_channel_block_size 32
typedef void (*micro_kernel_function)(size_t, size_t, struct nnp_size, struct nnp_size,
                                      struct nnp_size, struct nnp_padding, struct nnp_size,
                                      const float *, const float *, const float *, float *, size_t,
                                      size_t);
static inline void
nnp_depthwise_1_micro_kernel(size_t out_x, size_t out_y, struct nnp_size output_size,
                             struct nnp_size input_size, struct nnp_size kernel_size,
                             struct nnp_padding input_padding, struct nnp_size output_subsampling,
                             const float *bias, const float *input, const float *kernel,
                             float *output, size_t depthwise_multiplier, size_t input_channels) {
  size_t output_channels = input_channels * depthwise_multiplier;
  float *output_pos = output + (out_y * output_size.width + out_x) * output_channels;
  register float32x4_t t1, t2, t3, t4, t5, t6, t7, t8;
  for (size_t b = 0; b < output_channels / output_channel_block_size; b++) {
    size_t channel_offset = b * output_channel_block_size;
    const float *cur_bias = bias + channel_offset;
    t1 = vld1q_f32(cur_bias);
    t2 = vld1q_f32(cur_bias + 4);
    t3 = vld1q_f32(cur_bias + 8);
    t4 = vld1q_f32(cur_bias + 12);
    t5 = vld1q_f32(cur_bias + 16);
    t6 = vld1q_f32(cur_bias + 20);
    t7 = vld1q_f32(cur_bias + 24);
    t8 = vld1q_f32(cur_bias + 28);
    float32x4_t input_simd, kernel_simd;
    for (size_t filter_y = 0; filter_y < kernel_size.height; filter_y++) {
      const size_t input_y = out_y * output_subsampling.height + filter_y - input_padding.top;
      if (input_y < input_size.height) {
        for (size_t filter_x = 0; filter_x < kernel_size.width; filter_x++) {
          const size_t input_x = out_x * output_subsampling.width + filter_x - input_padding.left;
          if (input_x < input_size.width) {
            const float *input_pos =
                input + (input_y * input_size.width + input_x) * input_channels + channel_offset;
            const float *kernel_pos =
                kernel +
                (filter_y * kernel_size.width + filter_x) * input_channels * depthwise_multiplier +
                channel_offset;
            input_simd = vld1q_f32(input_pos);
            kernel_simd = vld1q_f32(kernel_pos);
            t1 = vmlaq_f32(t1, input_simd, kernel_simd);
            input_simd = vld1q_f32(input_pos + 4);
            kernel_simd = vld1q_f32(kernel_pos + 4);
            t2 = vmlaq_f32(t2, input_simd, kernel_simd);
            input_simd = vld1q_f32(input_pos + 8);
            kernel_simd = vld1q_f32(kernel_pos + 8);
            t3 = vmlaq_f32(t3, input_simd, kernel_simd);
            input_simd = vld1q_f32(input_pos + 12);
            kernel_simd = vld1q_f32(kernel_pos + 12);
            t4 = vmlaq_f32(t4, input_simd, kernel_simd);
            input_simd = vld1q_f32(input_pos + 16);
            kernel_simd = vld1q_f32(kernel_pos + 16);
            t5 = vmlaq_f32(t5, input_simd, kernel_simd);
            input_simd = vld1q_f32(input_pos + 20);
            kernel_simd = vld1q_f32(kernel_pos + 20);
            t6 = vmlaq_f32(t6, input_simd, kernel_simd);
            input_simd = vld1q_f32(input_pos + 24);
            kernel_simd = vld1q_f32(kernel_pos + 24);
            t7 = vmlaq_f32(t7, input_simd, kernel_simd);
            input_simd = vld1q_f32(input_pos + 28);
            kernel_simd = vld1q_f32(kernel_pos + 28);
            t8 = vmlaq_f32(t8, input_simd, kernel_simd);
          }
        }
      }
    }
    float *cur_output_pos = output_pos + channel_offset;
    vst1q_f32(cur_output_pos, t1);
    vst1q_f32(cur_output_pos + 4, t2);
    vst1q_f32(cur_output_pos + 8, t3);
    vst1q_f32(cur_output_pos + 12, t4);
    vst1q_f32(cur_output_pos + 16, t5);
    vst1q_f32(cur_output_pos + 20, t6);
    vst1q_f32(cur_output_pos + 24, t7);
    vst1q_f32(cur_output_pos + 28, t8);
  }
}

struct NNP_CACHE_ALIGN per_output_pixel_context {
  size_t input_channels;
  size_t output_channels;
  struct nnp_size input_size;
  struct nnp_size output_size;
  size_t depthwise_multiplier;
  struct nnp_padding input_padding;
  struct nnp_size kernel_size;
  struct nnp_size output_subsampling;
  const float *input;
  const float *kernel;
  const float *bias;
  float *output;
  enum nnp_activation activation;
  micro_kernel_function kernel_function;
};

void per_output_pixel_inference(const struct per_output_pixel_context context[restrict static 1],
                                size_t out_y_start, size_t out_y_step) {
  const size_t input_channels = context->input_channels;
  const size_t output_channels = context->output_channels;
  const struct nnp_size input_size = context->input_size;
  const struct nnp_size output_size = context->output_size;
  const size_t depthwise_multiplier = context->depthwise_multiplier;
  const struct nnp_padding input_padding = context->input_padding;
  const struct nnp_size kernel_size = context->kernel_size;
  const struct nnp_size output_subsampling = context->output_subsampling;
  const float *input = context->input;
  const float *kernel = context->kernel;
  const float *bias = context->bias;
  float *output = context->output;
  enum nnp_activation activation = context->activation;
  micro_kernel_function kernel_function = context->kernel_function;
  for (size_t out_y = out_y_start; out_y < out_y_start + out_y_step; out_y++) {
    for (size_t out_x = 0; out_x < output_size.width; out_x++) {
      float *output_pos = output + (out_y * output_size.width + out_x) * output_channels;
      kernel_function(out_x, out_y, output_size, input_size, kernel_size, input_padding,
                      output_subsampling, bias, input, kernel, output, depthwise_multiplier,
                      input_channels);
      size_t output_channel;
      float *local_output = NULL;
      switch (activation) {
      case nnp_activation_identity:
        break;
      case nnp_activation_relu:
        output_channel = 0;
        local_output = output_pos;
        for (; output_channel < output_channels; output_channel += nnp_hwinfo.simd_width) {
          float32x4_t acc = vld1q_f32(output_pos + output_channel);
          acc = vmaxq_f32(vdupq_n_f32(0.f), acc);
          vst1q_f32(local_output, acc);
          local_output += nnp_hwinfo.simd_width;
        }

        for (; output_channel < output_channels; output_channel++) {
          *local_output++ = relu(*(output_pos + output_channel), 0.0f);
        }
        break;
      default:
        NNP_UNREACHABLE;
      }
    }
  }
}

static inline void select_micro_kernel(const size_t input_channels,
                                       const size_t depthwise_multiplier,
                                       micro_kernel_function *kernel_function) {
  switch (depthwise_multiplier) {
  case 1:
    *kernel_function = nnp_depthwise_1_micro_kernel;
    break;
  }
}

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
  size_t depthwise_multiplier = output_channels / input_channels;
  size_t thread_cache_num = 1;
  if (threadpool != NULL)
    thread_cache_num = pthreadpool_get_threads_count(threadpool);
  micro_kernel_function kernel_function;
  select_micro_kernel(input_channels, depthwise_multiplier, &kernel_function);

  struct per_output_pixel_context per_output_pixel_context = {
      .input_channels = input_channels,
      .output_channels = output_channels,
      .input_size = input_size,
      .output_size = output_size,
      .depthwise_multiplier = depthwise_multiplier,
      .input_padding = input_padding,
      .kernel_size = kernel_size,
      .output_subsampling = output_subsampling,
      .input = input,
      .kernel = kernel,
      .bias = bias,
      .output = output,
      .activation = activation,
      .kernel_function = kernel_function};
  size_t step =
      output_size.height / thread_cache_num + ((output_size.height % thread_cache_num) ? 1 : 0);
  pthreadpool_compute_1d_tiled(threadpool,
                               (pthreadpool_function_1d_tiled_t)per_output_pixel_inference,
                               &per_output_pixel_context, output_size.height, step);
cleanup:
  NNP_TOTAL_END(profile)
  return status;
}
