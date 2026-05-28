#include "rknn_session.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace hand_pipeline {

namespace {

TensorInfo make_info(const rknn_tensor_attr& a) {
    TensorInfo info;
    info.name = a.name[0] ? std::string(a.name)
                          : ("tensor_" + std::to_string(a.index));
    info.dims.reserve(a.n_dims);
    info.elem_count = 1;
    for (std::uint32_t i = 0; i < a.n_dims; ++i) {
        info.dims.push_back(static_cast<int>(a.dims[i]));
        info.elem_count *= a.dims[i];
    }
    info.fmt = a.fmt;
    info.type = a.type;
    return info;
}

}  // namespace

RknnSession::RknnSession(const std::string& model_path) {
    int ret = rknn_init(&ctx_, const_cast<char*>(model_path.c_str()), 0, 0, nullptr);
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("rknn_init failed (" + std::to_string(ret) +
                                 ") for " + model_path);
    }

    rknn_input_output_num io{};
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
    if (ret != RKNN_SUCC) {
        rknn_destroy(ctx_);
        ctx_ = 0;
        throw std::runtime_error("rknn_query IN_OUT_NUM failed");
    }
    if (io.n_input != 1) {
        rknn_destroy(ctx_);
        ctx_ = 0;
        throw std::runtime_error("expected single input; got " +
                                 std::to_string(io.n_input));
    }

    rknn_tensor_attr in_attr{};
    in_attr.index = 0;
    ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
    if (ret != RKNN_SUCC) {
        rknn_destroy(ctx_);
        ctx_ = 0;
        throw std::runtime_error("rknn_query INPUT_ATTR failed");
    }
    input_ = make_info(in_attr);

    // Resolve H/W/C from attr.fmt
    if (in_attr.n_dims < 4) {
        rknn_destroy(ctx_);
        ctx_ = 0;
        throw std::runtime_error("input must be 4-D");
    }
    if (in_attr.fmt == RKNN_TENSOR_NHWC) {
        input_h_ = static_cast<int>(in_attr.dims[1]);
        input_w_ = static_cast<int>(in_attr.dims[2]);
        input_c_ = static_cast<int>(in_attr.dims[3]);
    } else {
        // NCHW
        input_c_ = static_cast<int>(in_attr.dims[1]);
        input_h_ = static_cast<int>(in_attr.dims[2]);
        input_w_ = static_cast<int>(in_attr.dims[3]);
    }

    outputs_.reserve(io.n_output);
    for (std::uint32_t i = 0; i < io.n_output; ++i) {
        rknn_tensor_attr out_attr{};
        out_attr.index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));
        if (ret != RKNN_SUCC) {
            rknn_destroy(ctx_);
            ctx_ = 0;
            throw std::runtime_error("rknn_query OUTPUT_ATTR " +
                                     std::to_string(i) + " failed");
        }
        outputs_.push_back(make_info(out_attr));
    }

    // 优先 AUTO 三核分配以提升单 stream 推理吞吐；老版 librknnrt 没有该 API 时静默失败。
    rknn_core_mask mask = RKNN_NPU_CORE_AUTO;
    (void)rknn_set_core_mask(ctx_, mask);
}

RknnSession::~RknnSession() {
    if (ctx_ != 0) {
        rknn_destroy(ctx_);
        ctx_ = 0;
    }
}

std::vector<OutputTensor> RknnSession::Run(const std::uint8_t* nhwc_buf,
                                           std::size_t buf_size) const {
    const std::size_t expected = static_cast<std::size_t>(input_h_) *
                                 static_cast<std::size_t>(input_w_) *
                                 static_cast<std::size_t>(input_c_);
    if (buf_size != expected) {
        throw std::invalid_argument("input buf_size " + std::to_string(buf_size) +
                                    " != expected " + std::to_string(expected));
    }

    rknn_input in{};
    in.index = 0;
    in.type = RKNN_TENSOR_UINT8;
    in.fmt = RKNN_TENSOR_NHWC;
    in.size = static_cast<std::uint32_t>(buf_size);
    in.buf = const_cast<std::uint8_t*>(nhwc_buf);
    in.pass_through = 0;

    int ret = rknn_inputs_set(ctx_, 1, &in);
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("rknn_inputs_set failed (" + std::to_string(ret) + ")");
    }

    ret = rknn_run(ctx_, nullptr);
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("rknn_run failed (" + std::to_string(ret) + ")");
    }

    std::vector<rknn_output> raw(outputs_.size());
    for (auto& o : raw) {
        std::memset(&o, 0, sizeof(o));
        o.want_float = 1;
    }
    ret = rknn_outputs_get(ctx_, static_cast<std::uint32_t>(raw.size()),
                           raw.data(), nullptr);
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("rknn_outputs_get failed (" + std::to_string(ret) + ")");
    }

    std::vector<OutputTensor> result(outputs_.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
        result[i].dims = outputs_[i].dims;
        const std::size_t n = raw[i].size / sizeof(float);
        const float* p = static_cast<const float*>(raw[i].buf);
        result[i].data.assign(p, p + n);
    }
    rknn_outputs_release(ctx_, static_cast<std::uint32_t>(raw.size()), raw.data());
    return result;
}

}  // namespace hand_pipeline
