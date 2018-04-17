#include <gtest/gtest.h>

#include <nnpack.h>

#include <models/alexnet.h>
#include <testers/convolution.h>

/*
 * MobileNet conv_dw layers
 */
TEST(NAIVE_SOLUTION, conv1_dw) {
  MobileNet::conv1_dw().errorLimit(1.0e-5).testDepthwiseInference(
      nnp_convolution_algorithm_implicit_gemm, nnp_activation_identity);
}

TEST(NAIVE_SOLUTION, conv2_dw) {
  MobileNet::conv2_dw().errorLimit(1.0e-5).testDepthwiseInference(
      nnp_convolution_algorithm_implicit_gemm, nnp_activation_identity);
}

TEST(NAIVE_SOLUTION, conv3_dw) {
  MobileNet::conv3_dw().errorLimit(1.0e-5).testDepthwiseInference(
      nnp_convolution_algorithm_implicit_gemm, nnp_activation_identity);
}

TEST(NAIVE_SOLUTION, conv4_dw) {
  MobileNet::conv4_dw().errorLimit(1.0e-5).testDepthwiseInference(
      nnp_convolution_algorithm_implicit_gemm, nnp_activation_identity);
}

TEST(NAIVE_SOLUTION, conv5_dw) {
  MobileNet::conv5_dw().errorLimit(1.0e-5).testDepthwiseInference(
      nnp_convolution_algorithm_implicit_gemm, nnp_activation_identity);
}

TEST(NAIVE_SOLUTION, conv6_dw) {
  MobileNet::conv6_dw().errorLimit(1.0e-5).testDepthwiseInference(
      nnp_convolution_algorithm_implicit_gemm, nnp_activation_identity);
}

TEST(NAIVE_SOLUTION, conv1 = 7_dw) {
  MobileNet::conv7_dw().errorLimit(1.0e-5).testDepthwiseInference(
      nnp_convolution_algorithm_implicit_gemm, nnp_activation_identity);
}

TEST(NAIVE_SOLUTION, conv8_dw) {
  MobileNet::conv8_dw().errorLimit(1.0e-5).testDepthwiseInference(
      nnp_convolution_algorithm_implicit_gemm, nnp_activation_identity);
}

TEST(NAIVE_SOLUTION, conv9_dw) {
  MobileNet::conv9_dw().errorLimit(1.0e-5).testDepthwiseInference(
      nnp_convolution_algorithm_implicit_gemm, nnp_activation_identity);
}
int main(int argc, char* argv[]) {
  const enum nnp_status init_status = nnp_initialize();
  assert(init_status == nnp_status_success);
  setenv("TERM", "xterm-256color", 0);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
