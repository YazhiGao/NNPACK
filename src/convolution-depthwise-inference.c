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
struct NNP_CACHE_ALIGN kernel_packing_context {
  const float* kernel;
  float* packed_kernel;

  size_t reduction_size;
  size_t reduction_block_start;
  size_t reduction_block_size;
};

static void compute_kernel_packing(
    const struct kernel_packing_context context[restrict static 1],
    size_t output_channels_subblock_start, size_t reduction_block_offset,
    size_t output_channels_subblock_size, size_t reduction_block_range) {
  const size_t reduction_size = context->reduction_size;
  const size_t reduction_block_start = context->reduction_block_start;
  const size_t reduction_block_size = context->reduction_block_size;

  const float* kernel = context->kernel +
                        output_channels_subblock_start * reduction_size +
                        reduction_block_offset;
  float* packed_kernel = context->packed_kernel +
                         output_channels_subblock_start * reduction_block_size +
                         reduction_block_offset * output_channels_subblock_size;

  for (size_t output_channels_subblock_offset = 0;
       output_channels_subblock_offset < output_channels_subblock_size;
       output_channels_subblock_offset += 1) {
    packed_kernel[output_channels_subblock_offset] =
        kernel[output_channels_subblock_offset * reduction_size];
  }
}

struct NNP_CACHE_ALIGN input_packing_context {
  const float* input;
  float* packed_input;

  size_t simd_width;
  size_t reduction_block_start;
  size_t reduction_block_size;
  size_t output_image_block_start;
  struct nnp_size input_size;
  size_t input_padding_top;
  size_t input_padding_left;
  struct fxdiv_divisor_size_t kernel_elements;
  struct fxdiv_divisor_size_t kernel_width;
  struct fxdiv_divisor_size_t output_width;
  struct nnp_size output_subsampling;
};

static void compute_input_packing(
    const struct input_packing_context context[restrict static 1],
    size_t reduction_block_offset, size_t output_image_subblock_start,
    size_t reduction_block_range, size_t output_image_subblock_size) {
  const size_t simd_width = context->simd_width;
  const size_t reduction_block_start = context->reduction_block_start;
  const size_t reduction_block_size = context->reduction_block_size;
  const size_t output_image_block_start = context->output_image_block_start;
  const struct nnp_size input_size = context->input_size;
  const size_t input_padding_top = context->input_padding_top;
  const size_t input_padding_left = context->input_padding_left;
  const struct fxdiv_divisor_size_t kernel_elements = context->kernel_elements;
  const struct fxdiv_divisor_size_t kernel_width = context->kernel_width;
  const struct fxdiv_divisor_size_t output_width = context->output_width;
  const struct nnp_size output_subsampling = context->output_subsampling;

  const float(*input)[input_size.height][input_size.width] =
      (const float(*)[input_size.height][input_size.width])context->input;
  float* packed_input = context->packed_input;

  const size_t output_image_subblock_stride =
      round_up_by_power_of_2(output_image_subblock_size, simd_width);

  const size_t reduction_index = reduction_block_start + reduction_block_offset;
  const struct fxdiv_result_size_t reduction_index_divmod =
      fxdiv_divide_size_t(reduction_index, kernel_elements);
  const size_t input_channel = reduction_index_divmod.quotient;
  const struct fxdiv_result_size_t kernel_xy =
      fxdiv_divide_size_t(reduction_index_divmod.remainder, kernel_width);
  const size_t kernel_y = kernel_xy.quotient;
  const size_t kernel_x = kernel_xy.remainder;

  for (size_t output_image_subblock_offset = 0;
       output_image_subblock_offset < output_image_subblock_size;
       output_image_subblock_offset += 1) {
    const size_t output_image_index = output_image_block_start +
                                      output_image_subblock_start +
                                      output_image_subblock_offset;
    const struct fxdiv_result_size_t output_xy =
        fxdiv_divide_size_t(output_image_index, output_width);
    const size_t output_y = output_xy.quotient;
    const size_t output_x = output_xy.remainder;

    const size_t input_y =
        output_y * output_subsampling.height + kernel_y - input_padding_top;
    const size_t input_x =
        output_x * output_subsampling.width + kernel_x - input_padding_left;

    const size_t packed_index =
        output_image_subblock_start * reduction_block_size +
        reduction_block_offset * output_image_subblock_stride +
        output_image_subblock_offset;
    if ((input_x < input_size.width) && (input_y < input_size.height)) {
      packed_input[packed_index] = input[input_channel][input_y][input_x];
    } else {
      packed_input[packed_index] = 0.0f;
    }
  }
}

struct NNP_CACHE_ALIGN matrix_multiplication_context {
  const float* packed_kernel;
  const float* packed_input;
  float* output;

  size_t reduction_block_start;
  size_t reduction_block_size;
  size_t output_image_size;
  size_t output_image_block_start;
  size_t output_image_subblock_max;
  size_t output_channels_subblock_max;
};

static void compute_matrix_multiplication(
    const struct matrix_multiplication_context context[restrict static 1],
    size_t output_channels_block_start, size_t output_image_subblock_start,
    size_t output_channels_block_size, size_t output_image_subblock_size) {
  const size_t reduction_block_start = context->reduction_block_start;
  const size_t reduction_block_size = context->reduction_block_size;
  const size_t output_image_size = context->output_image_size;
  const size_t output_image_block_start = context->output_image_block_start;
  const size_t output_image_subblock_max = context->output_image_subblock_max;
  const size_t output_channels_subblock_max =
      context->output_channels_subblock_max;

  const float* packed_kernel =
      context->packed_kernel +
      output_channels_block_start * reduction_block_size;
  const float* packed_input =
      context->packed_input +
      output_image_subblock_start * reduction_block_size;
  float* output = context->output +
                  output_channels_block_start * output_image_size +
                  output_image_block_start + output_image_subblock_start;

  if (output_image_subblock_size == output_image_subblock_max) {
    const nnp_fast_sgemm_function fast_gemm = nnp_hwinfo.sgemm.only_mr_x_nr;
    while (output_channels_block_size >= output_channels_subblock_max) {
      output_channels_block_size -= output_channels_subblock_max;

      fast_gemm(reduction_block_size, reduction_block_start, packed_kernel,
                packed_input, output, output_image_size);

      packed_kernel += reduction_block_size * output_channels_subblock_max;
      output += output_image_size * output_channels_subblock_max;
    }
  }

  const nnp_full_sgemm_function full_gemm = nnp_hwinfo.sgemm.upto_mr_x_nr;
  while (output_channels_block_size != 0) {
    const size_t output_channels_subblock_size =
        min(output_channels_block_size, output_channels_subblock_max);
    output_channels_block_size -= output_channels_subblock_size;

    full_gemm(output_channels_subblock_size, output_image_subblock_size,
              reduction_block_size, reduction_block_start, packed_kernel,
              packed_input, output, output_image_size);

    packed_kernel += reduction_block_size * output_channels_subblock_max;
    output += output_image_size * output_channels_subblock_max;
  }
}
static enum nnp_status compute_gemm_convolution_inference(
    const enum nnp_convolution_transform_strategy transform_strategy,
    const size_t input_channels, const size_t output_channels,
    const struct nnp_size input_size, const struct nnp_padding input_padding,
    const struct nnp_size kernel_size, const struct nnp_size output_size,
    const struct nnp_size output_subsampling, const float* input,
    const float* kernel, const float* bias, float* output,
    void* workspace_buffer, size_t* workspace_size,
    enum nnp_activation activation, pthreadpool_t threadpool,
    struct nnp_profile* profile) {
  enum nnp_status status = nnp_status_success;
  void* memory_block = NULL;
  size_t memory_size = 0;
  const size_t simd_width = nnp_hwinfo.simd_width;

  /* Calculate cache blocking parameters */
  const size_t cache_elements_l1 = nnp_hwinfo.blocking.l1 / sizeof(float);
  const size_t cache_elements_l2 = nnp_hwinfo.blocking.l2 / sizeof(float);
  const size_t cache_elements_l3 = nnp_hwinfo.blocking.l3 / sizeof(float);

  const size_t output_channels_subblock_max = nnp_hwinfo.sgemm.mr;
  const size_t output_image_subblock_max = nnp_hwinfo.sgemm.nr;

  const size_t depth_multiplier = output_channels / input_channels;
  const size_t reduction_size = kernel_size.height * kernel_size.width;
  const size_t output_image_size = output_size.height * output_size.width;
  /* rows of joint packed input and kernel mr+nr-column elements
   */
  const size_t reduction_block_max =
      round_down(cache_elements_l1 /
                     (output_channels_subblock_max + output_image_subblock_max),
                 2);
  /* blocksize in one channel row for kernel
   */
  const size_t output_channels_block_max = round_down(
      cache_elements_l2 / reduction_block_max, output_channels_subblock_max);
  /* blocksize in one channel row for image
   */
  const size_t output_image_block_max = round_down(
      cache_elements_l3 / reduction_block_max, output_image_subblock_max);
  switch (transform_strategy) {
    case nnp_convolution_transform_strategy_compute:
    case nnp_convolution_transform_strategy_reuse: {
      /* packed kernel is now p * reduction_block_size * input_channel;
       */
      const size_t packed_kernel_size =
          depth_multiplier * min(reduction_block_max, reduction_size) *
          sizeof(float);
      const size_t packed_input_size =
          min(output_image_block_max, round_up(output_image_size, simd_width)) *
          min(reduction_block_max, reduction_size) * sizeof(float);
      memory_size = packed_kernel_size + packed_input_size;
      if (workspace_buffer == NULL) {
        if (workspace_size == NULL) {
          memory_block = allocate_memory(memory_size);
          if (memory_block == NULL) {
            return nnp_status_out_of_memory;
          }
        } else {
          *workspace_size = memory_size;
          return nnp_status_success;
        }
      } else {
        if (*workspace_size < memory_size) {
          return nnp_status_insufficient_buffer;
        }
        memory_block = workspace_buffer;
      }

      float* packed_input = memory_block;
      float* packed_kernel = memory_block + packed_input_size;
      for (size_t group = 0; group < input_channels; group++) {
        for (size_t reduction_block_start = 0;
             reduction_block_start < reduction_size;
             reduction_block_start += reduction_block_max) {
          const size_t reduction_block_size =
              min(reduction_size - reduction_block_start, reduction_block_max);

          if (transform_strategy ==
              nnp_convolution_transform_strategy_compute) {
            /* Pack kernel into memory block */
            NNP_KERNEL_TRANSFORM_START(profile)
            struct kernel_packing_context kernel_packing_context = {
                .kernel = kernel + group * depth_multiplier * (reduction_size) +
                          reduction_block_start,
                .packed_kernel = packed_kernel,
                .reduction_size = reduction_size,
                .reduction_block_start = reduction_block_start,
                .reduction_block_size = reduction_block_size,
            };
            pthreadpool_compute_2d_tiled(
                threadpool,
                (pthreadpool_function_2d_tiled_t)compute_kernel_packing,
                &kernel_packing_context, output_channels, reduction_block_size,
                output_channels_subblock_max, 1);
            NNP_KERNEL_TRANSFORM_END(profile)
          } else {
            packed_kernel =
                (void*)kernel + group * depth_multiplier * reduction_size +
                output_channels * reduction_block_start * sizeof(float);
          }

          const struct fxdiv_divisor_size_t kernel_elements_divisor =
              fxdiv_init_size_t(kernel_size.height * kernel_size.width);
          const struct fxdiv_divisor_size_t kernel_width_divisor =
              fxdiv_init_size_t(kernel_size.width);
          const struct fxdiv_divisor_size_t output_width_divisor =
              fxdiv_init_size_t(output_size.width);
          for (size_t output_image_block_start = 0;
               output_image_block_start < output_image_size;
               output_image_block_start += output_image_block_max) {
            const size_t output_image_block_size =
                min(output_image_size - output_image_block_start,
                    output_image_block_max);

            /* Pack image into L3 block */
            NNP_INPUT_TRANSFORM_START(profile)
            struct input_packing_context input_packing_context = {
                .input = input + group * (input_size.width * input_size.height),
                .packed_input = packed_input,
                .simd_width = simd_width,
                .reduction_block_start = reduction_block_start,
                .reduction_block_size = reduction_block_size,
                .output_image_block_start = output_image_block_start,
                .input_size = input_size,
                .input_padding_top = input_padding.top,
                .input_padding_left = input_padding.left,
                .kernel_elements = kernel_elements_divisor,
                .kernel_width = kernel_width_divisor,
                .output_width = output_width_divisor,
                .output_subsampling = output_subsampling,
            };
            pthreadpool_compute_2d_tiled(
                threadpool,
                (pthreadpool_function_2d_tiled_t)compute_input_packing,
                &input_packing_context, reduction_block_size,
                output_image_block_size, 1, output_image_subblock_max);
            NNP_INPUT_TRANSFORM_END(profile)

            NNP_BLOCK_MULTIPLICATION_START(profile)
            struct matrix_multiplication_context matrix_multiplication_context =
                {
                    .packed_kernel = packed_kernel,
                    .packed_input = packed_input,
                    .output =
                        output + group * (output_image_size * depth_multiplier),
                    .reduction_block_start = reduction_block_start,
                    .reduction_block_size = reduction_block_size,
                    .output_image_size = output_image_size,
                    .output_image_block_start = output_image_block_start,
                    .output_image_subblock_max = output_image_subblock_max,
                    .output_channels_subblock_max =
                        output_channels_subblock_max,
                };
            pthreadpool_compute_2d_tiled(
                threadpool,
                (pthreadpool_function_2d_tiled_t)compute_matrix_multiplication,
                &matrix_multiplication_context, output_channels,
                output_image_block_size, output_channels_block_max,
                output_image_subblock_max);
            NNP_BLOCK_MULTIPLICATION_END(profile)
          }
        }
      }
      /* Add bias */
      NNP_OUTPUT_TRANSFORM_START(profile)
      switch (activation) {
        case nnp_activation_identity:
          for (size_t output_channel = 0; output_channel < output_channels;
               output_channel += 1) {
            const float bias_value = bias[output_channel];
            for (size_t index = 0; index < output_image_size; index += 1) {
              output[output_channel * output_image_size + index] += bias_value;
            }
          }
          break;
        case nnp_activation_relu:
          for (size_t output_channel = 0; output_channel < output_channels;
               output_channel += 1) {
            const float bias_value = bias[output_channel];
            for (size_t index = 0; index < output_image_size; index += 1) {
              output[output_channel * output_image_size + index] =
                  relu(output[output_channel * output_image_size + index] +
                           bias_value,
                       0.0f);
            }
          }
          break;
        default:
          NNP_UNREACHABLE;
      }
      NNP_OUTPUT_TRANSFORM_END(profile)
      break;
    }
    case nnp_convolution_transform_strategy_precompute: {
      const size_t packed_kernel_size =
          output_channels * reduction_size * sizeof(float);
      if (workspace_buffer == NULL) {
        *workspace_size = packed_kernel_size;
        return nnp_status_success;
      } else {
        if (*workspace_size < packed_kernel_size) {
          return nnp_status_insufficient_buffer;
        }
        memory_block = workspace_buffer;
      }

      for (size_t reduction_block_start = 0;
           reduction_block_start < reduction_size;
           reduction_block_start += reduction_block_max) {
        const size_t reduction_block_size =
            min(reduction_size - reduction_block_start, reduction_block_max);

        /* Pack kernel into memory block */
        NNP_KERNEL_TRANSFORM_START(profile)
        struct kernel_packing_context kernel_packing_context = {
            .kernel = kernel + reduction_block_start,
            .packed_kernel =
                (void*)workspace_buffer +
                output_channels * reduction_block_start * sizeof(float),
            .reduction_size = reduction_size,
            .reduction_block_start = reduction_block_start,
            .reduction_block_size = reduction_block_size,
        };
        pthreadpool_compute_2d_tiled(
            threadpool, (pthreadpool_function_2d_tiled_t)compute_kernel_packing,
            &kernel_packing_context, output_channels, reduction_block_size,
            output_channels_subblock_max, 1);
        NNP_KERNEL_TRANSFORM_END(profile)
      }
      break;
    }
    default:
      return nnp_status_invalid_transform_strategy;
  }

  if (memory_block != workspace_buffer) {
    release_memory(memory_block, memory_size);
  }
  return status;
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
  status = compute_gemm_convolution_inference(
      nnp_convolution_transform_strategy_compute, input_channels,
      output_channels, input_size, input_padding, kernel_size, output_size,
      output_subsampling, input, kernel, bias, output, workspace_buffer,
      workspace_size, activation, threadpool, profile);

cleanup:
  NNP_TOTAL_END(profile)
  return status;
}
