#pragma once

#include <nnpack.h>

#include <testers/convolution.h>

namespace MobileNet {
// padding follows "SAME" scheme padding in original mobilenet implementation
/*
 * MobileNet conv_dw layer type 1:
 * input channels = 32
 * output channels = 32
 * input size = 112 x 112
 * implict padding = 1
 * kernel size = 3 x 3
 * output sampling = 1 x 1
 */
inline ConvolutionTester convdw1() {
  return std::move(ConvolutionTester()
                       .multithreading(true)
                       .inputChannels(32)
                       .outputChannels(32)
                       .inputSize(112, 112)
                       .kernelSize(3, 3)
                       .outputSubsampling(1, 1)
                       .inputPadding(1, 1, 1, 1));
}

/*
 * MobileNet conv_dw layer type 2:
 * input channels = 64
 * output channels = 64
 * input size = 112 x 112
 * implict padding = 1
 * kernel size = 3 x 3
 * output sampling = 2 x 2
 */
inline ConvolutionTester convdw2() {
  return std::move(ConvolutionTester()
                       .multithreading(true)
                       .inputChannels(64)
                       .outputChannels(64)
                       .inputSize(112, 112)
                       .kernelSize(3, 3)
                       .outputSubsampling(2, 2)
                       .inputPadding(1, 2, 1, 2));
}

/*
 * MobileNet conv_dw layer type 3:
 * input channels = 128
 * output channels = 128
 * input size = 56 x 56
 * implict padding = 1
 * kernel size = 3 x 3
 * output sampling = 1 x 1
 */
inline ConvolutionTester convdw3() {
  return std::move(ConvolutionTester()
                       .multithreading(true)
                       .inputChannels(128)
                       .outputChannels(128)
                       .inputSize(56, 56)
                       .kernelSize(3, 3)
                       .outputSubsampling(1, 1)
                       .inputPadding(1, 1, 1, 1));
}

/*
 * MobileNet conv_dw layer type 4:
 * input channels = 128
 * output channels = 128
 * input size = 56 x 56
 * implict padding = 1
 * kernel size = 3 x 3
 * output sampling = 2 x 2
 */
inline ConvolutionTester convdw4() {
  return std::move(ConvolutionTester()
                       .multithreading(true)
                       .inputChannels(128)
                       .outputChannels(128)
                       .inputSize(56, 56)
                       .kernelSize(3, 3)
                       .outputSubsampling(2, 2)
                       .inputPadding(1, 2, 1, 2));
}

/*
 * MobileNet conv_dw layer type 5:
 * input channels = 256
 * output channels = 256
 * input size = 28 x 28
 * implict padding = 1
 * kernel size = 3 x 3
 * output sampling = 1 x 1
 */
inline ConvolutionTester convdw5() {
  return std::move(ConvolutionTester()
                       .multithreading(true)
                       .inputChannels(128)
                       .outputChannels(128)
                       .inputSize(56, 56)
                       .kernelSize(3, 3)
                       .outputSubsampling(1, 1)
                       .inputPadding(1, 1, 1, 1));
}

/*
 * MobileNet conv_dw layer type 6:
 * input channels = 256
 * output channels = 256
 * input size = 28 x 28
 * implict padding = 1
 * kernel size = 3 x 3
 * output sampling = 2 x 2
 */
inline ConvolutionTester convdw6() {
  return std::move(ConvolutionTester()
                       .multithreading(true)
                       .inputChannels(128)
                       .outputChannels(128)
                       .inputSize(56, 56)
                       .kernelSize(3, 3)
                       .outputSubsampling(2, 2)
                       .inputPadding(1, 2, 1, 2));
}

/*
 * MobileNet conv_dw layer type 7:
 * input channels = 512
 * output channels = 512
 * input size = 14 x 14
 * implict padding = 1
 * kernel size = 3 x 3
 * output sampling = 1 x 1
 */
inline ConvolutionTester convdw7() {
  return std::move(ConvolutionTester()
                       .multithreading(true)
                       .inputChannels(128)
                       .outputChannels(128)
                       .inputSize(56, 56)
                       .kernelSize(3, 3)
                       .outputSubsampling(1, 1)
                       .inputPadding(1, 1, 1, 1));
}

/*
 * MobileNet conv_dw layer type 8:
 * input channels = 512
 * output channels = 512
 * input size = 14 x 14
 * implict padding = 1
 * kernel size = 3 x 3
 * output sampling = 2 x 2
 */
inline ConvolutionTester convdw8() {
  return std::move(ConvolutionTester()
                       .multithreading(true)
                       .inputChannels(128)
                       .outputChannels(128)
                       .inputSize(56, 56)
                       .kernelSize(3, 3)
                       .outputSubsampling(2, 2)
                       .inputPadding(1, 2, 1, 2));
}

/*
 * MobileNet conv_dw layer type 9:
 * input channels = 1024
 * output channels = 1024
 * input size = 7 x 7
 * implict padding = 1
 * kernel size = 3 x 3
 * output sampling = 2 x 2
 */
inline ConvolutionTester convdw9() {
  return std::move(ConvolutionTester()
                       .multithreading(true)
                       .inputChannels(128)
                       .outputChannels(128)
                       .inputSize(56, 56)
                       .kernelSize(3, 3)
                       .outputSubsampling(2, 2)
                       .inputPadding(1, 2, 1, 2));
}
}  // namespace MobileNet
