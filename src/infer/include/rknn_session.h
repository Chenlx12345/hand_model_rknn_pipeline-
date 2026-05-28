// Minimal RKNN session: load .rknn, feed uint8 NHWC, get float32 outputs.
// Inputs are assumed uint8 because both rtmdet/rtmpose .rknn here have mean/std
// baked in via rknn.config(mean_values, std_values).
#pragma once

#include <rknn_api.h>

#include <cstdint>
#include <string>
#include <vector>

namespace hand_pipeline {

struct TensorInfo {
    std::string name;
    std::vector<int> dims;     // NCHW or NHWC depending on fmt
    rknn_tensor_format fmt = RKNN_TENSOR_NCHW;
    rknn_tensor_type type = RKNN_TENSOR_FLOAT32;
    std::size_t elem_count = 0;  // product of dims
};

struct OutputTensor {
    std::vector<int> dims;
    std::vector<float> data;  // float32, NCHW order as model exports
};

class RknnSession {
public:
    explicit RknnSession(const std::string& model_path);
    ~RknnSession();

    RknnSession(const RknnSession&) = delete;
    RknnSession& operator=(const RknnSession&) = delete;

    // Run with raw uint8 NHWC buffer. `buf` must be h*w*c bytes.
    std::vector<OutputTensor> Run(const std::uint8_t* nhwc_buf,
                                  std::size_t buf_size) const;

    const TensorInfo& input() const noexcept { return input_; }
    const std::vector<TensorInfo>& outputs() const noexcept { return outputs_; }

    int input_h() const noexcept { return input_h_; }
    int input_w() const noexcept { return input_w_; }
    int input_c() const noexcept { return input_c_; }

private:
    rknn_context ctx_ = 0;
    TensorInfo input_;
    std::vector<TensorInfo> outputs_;
    int input_h_ = 0, input_w_ = 0, input_c_ = 0;
};

}  // namespace hand_pipeline
