// vitgloves_vis_sdk — 公共结果类型(给下游 SDK 消费用)。
//
// `VvResult` 是单个 5Hz tick 的全部产物;字段一一对应 NDJSON 输出
// (见 docs/RESULT_FORMAT.md)。下游 SDK 调 `sdk.get_latest(&r)` 拿数据,
// 拿到就完全拥有这份数据(struct 自带 owning std::vector),怎么用、留多久都行。
//
// 头里只用 <cstdint> + <cstddef> + <vector>;不引 OpenCV / Eigen / nlohmann。
// 集成方编译既不会被传染依赖,也方便 C++ 17 直接 by-value 拷贝/移动。
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vitgloves {
namespace vis {

/* ── 单只手的 LED-box 6-DoF(step4 Part A 输出)─────────────────── */
struct Box6DofResult {
    int      side      = -1;          ///< 0=left, 1=right
    uint8_t  imu_id    = 0xFF;        ///< 灯盒 IMU 硬件 id:left=0x11, right=0x21
    double   T[16]     = {};          ///< 4×4 行主序,X_body = R·X_box + t
    double   reproj_px = 0.0;         ///< 7 LED 联合 cam1/cam4 平均重投影残差(像素)
};

/* ── 单个 IMU 关节的 3D 位置(step4 Part C 输出)──────────────── */
struct FingerPointResult {
    int      side       = -1;         ///< 0=left, 1=right
    uint8_t  imu_id     = 0xFF;       ///< 0x12..0x1D / 0x22..0x2D / 0x05 / 0x08
                                      ///< (含义见 docs/IMU_ID.md)
    double   xyz_body[3] = {};        ///< body 系 3D(米)
    double   reproj_l    = 0.0;       ///< 源 IR 点在 cam1 (左) 的 step2 重投影残差(px)
    double   reproj_r    = 0.0;       ///< 同上 cam4 (右)
};

/* ── step2 三角化的 IR 3D 点(每个亮光斑一条)─────────────────── */
struct StereoPoint3D {
    double   xyz_body[3] = {};        ///< body 系 3D(米)
    double   uv1[2]      = {};        ///< cam1(左 IR)像素
    double   uv4[2]      = {};        ///< cam4(右 IR)像素
    double   reproj_l    = 0.0;       ///< xyz → cam1 的重投影残差(像素)
    double   reproj_r    = 0.0;       ///< 同上 cam4
};

/* ── step3 三角化的 hand 关键点 3D(每只手 21 kpt 各一条)──────── */
struct HandKpt3DResult {
    int      side        = -1;        ///< 0=left, 1=right(RTMDet class)
    int      kpt         = -1;        ///< 0..20(RTMPose 21 joint,0=wrist,5=index_MCP, ...)
    int      cam_a       = -1;        ///< 主 RGB 相机 id(通常 2 或 3)
    int      cam_b       = -1;        ///< 副 RGB 相机 id
    double   xyz_body[3] = {};        ///< body 系 3D(米)
    double   uv_a[2]     = {};        ///< 在 cam_a 的 RTMPose 输出像素
    double   uv_b[2]     = {};        ///< 在 cam_b 的 RTMPose 输出像素
    double   reproj_a    = 0.0;       ///< xyz → cam_a 的重投影残差(像素)
    double   reproj_b    = 0.0;       ///< 同上 cam_b
};

/* ── 单个 5Hz tick 的全部结果 ─────────────────────────────────── */
struct VvResult {
    uint64_t timestamp_ns = 0;        ///< batch_mono_ns,CLOCK_MONOTONIC ns
    int      frame_id     = -1;       ///< SDK 内部 30Hz batch 计数器
    std::vector<Box6DofResult>     boxes;          ///< 0~2 个
    std::vector<FingerPointResult> fingers;        ///< 0~26 个(每手最多 13)
    std::vector<StereoPoint3D>     step2_points;   ///< 全部三角化 IR 点
    std::vector<HandKpt3DResult>   step3_points;   ///< 全部三角化 hand kpt
};

} // namespace vis
} // namespace vitgloves
