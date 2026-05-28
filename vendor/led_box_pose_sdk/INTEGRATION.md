# led_box_pose_sdk 集成说明

以二进制形式交付的手部 GT 提取库。输入来自 algo-input 服务（`services::algo`）的
6 路同步 BGR 图像与相机标定，输出为逐帧的中间结果与最终结果（当前为明文 JSON）。
本说明面向接入该库的业务工程。

## 交付内容

```
led_box_pose_sdk/
├── include/led_box_pose_sdk.hpp     公共头（仅依赖 <cstdint>/<string>/<vector>，不含 OpenCV 类型）
├── lib/libled_box_pose_sdk.a        静态库（推荐：全静态 buildroot 集成）
├── lib/libled_box_pose_sdk.so(.0)   动态库（可选：运行期有匹配的 OpenCV/OpenSSL .so 时）
├── assets/                          运行期资源
│   ├── config.json                  算法参数（阈值、门限、process_hz）
│   ├── models/rtmdet.rknn           手部检测模型
│   ├── models/rtmpose.rknn          手部姿态模型
│   ├── models/MODELS_VERSION.txt    模型版本记录
│   └── box_left.txt / box_right.txt LED box 模型
├── examples/integrate_example.cpp   接入示例（算法侧完整接线）
├── examples/algo_input_api.hpp      algo-input 接口快照（参考；实物以 das_ego_app 为准）
└── INTEGRATION.md
```

## 接口边界

库不依赖 algo-input 服务。入口 `process_batch` 接收 POD 数组 `ImageView`（裸指针 +
宽高 + cam_id）；标定经 `InitConfig.calib`（POD `CameraCalib` 数组）传入。从服务到库
入口的转换由业务工程提供。

公共 API（`led_box_pose_sdk.hpp`）关键部分：

```cpp
namespace led_box_pose {
struct ImageView   { const uint8_t* data; int width, height, channels, step_bytes, cam_id; };
struct CameraCalib { int cam_id, width, height; double K[9]; double dist[8]; int n_dist; double T_b_c[7]; };
struct InitConfig {
    std::vector<CameraCalib> calib;  // 输入：各路标定（来自 getAlgoCameraCalibration）
    std::string calib_path;          // 备选：calib 为空时回退读 camera_info.json
    std::string assets_dir;          // 输入：交付包内 assets/
    std::string output_path;         // 输出：结果文件
    int  cam_left = 1, cam_right = 4, cam_rgb_a = 2, cam_rgb_b = 3;
    bool encrypt = false;            // false=明文 JSON，true=AES-256-GCM（输出 .enc）
};
class LedBoxPoseSdk {
    explicit LedBoxPoseSdk(const InitConfig&);
    bool ok() const;
    int  process_batch(const ImageView* frames, int n, int64_t pts_us);
    int  finalize();
    int  frames_processed() const; int points_last_frame() const; int hands_last_frame() const;
    std::string last_error() const;
};
}
```

`CameraCalib` 与 `services::algo::CameraCalibration` 字段对应：`K`(3x3 内参)、
`dist`(Double Sphere 取 `D` 的 `[xi, alpha]`)、`T_b_c`(body→camera 7 维)。

## 接入流程

```cpp
#include "led_box_pose_sdk.hpp"
#include "algo_input_api.hpp"
using namespace led_box_pose;
namespace algo = services::algo;

// 1. 启动 algo-input 服务
algo::initializeAlgoInput();

// 2. 从接口取标定 -> InitConfig.calib
InitConfig cfg;
cfg.assets_dir  = "/opt/led_box_pose_sdk/assets";
cfg.output_path = "/mnt/sdcard/result.json";
for (uint8_t c = 0; c < algo::kAlgoCameraCount; ++c) {
    if (!algo::hasAlgoCameraCalibration(c)) continue;
    const algo::CameraCalibration& sc = algo::getAlgoCameraCalibration(c);
    CameraCalib cc;
    cc.cam_id = sc.cam_id; cc.width = sc.width; cc.height = sc.height;
    for (int i = 0; i < 9; ++i) cc.K[i] = sc.K[i];
    cc.n_dist = (int)std::min<size_t>(8, sc.D.size());
    for (int i = 0; i < cc.n_dist; ++i) cc.dist[i] = sc.D[i];
    for (int i = 0; i < 7; ++i) cc.T_b_c[i] = sc.T_b_c[i];
    cfg.calib.push_back(cc);
}

// 3. 构造 SDK
LedBoxPoseSdk sdk(cfg);
if (!sdk.ok()) { /* 读 sdk.last_error()，终止 */ }

// 4. 注册 batch 回调：AlgoFrame.bgr -> ImageView -> process_batch
algo::setAlgoBatchCallback([&sdk](const algo::AlgoBatch& batch) {
    ImageView views[algo::kAlgoCameraCount]; int n = 0;
    for (const auto& v : batch.cams) {
        if (v.bgr.empty()) continue;
        views[n++] = ImageView{ v.bgr.data, v.bgr.cols, v.bgr.rows,
                                v.bgr.channels(), (int)v.bgr.step, (int)v.cam_id };
    }
    sdk.process_batch(views, n, (int64_t)batch.pts_us);
});

// 5. 采集结束 / 退出前
algo::clearAlgoBatchCallback();
algo::shutdownAlgoInput();
sdk.finalize();
```

完整可编译示例见 `examples/integrate_example.cpp`。

约定：

- `process_batch` 按 `assets/config.json` 的 `process_hz` 在库内部跳帧处理，回调按
  相机原始帧率调用即可，无需外部降频。
- `ImageView` 指向的像素仅在调用期内被读取，`process_batch` 返回后即可释放。
- 回调在服务的采集线程触发，库内同步处理。

## 编译与链接（RK3588）

推荐：**静态库**。SDK 不自带 OpenCV/OpenSSL；最终可执行从目标 sysroot 解析这些依赖，
全工程使用同一套 OpenCV/OpenSSL，无版本冲突。SDK 库本身只用到 OpenCV `core` + `imgproc`
（不含 videoio）。

```sh
g++ -std=c++17 app.cpp \
    -I<sdk>/include -I<algo_input_api.hpp 所在目录> \
    <sdk>/lib/libled_box_pose_sdk.a \
    <sysroot OpenCV: libopencv_imgproc.a libopencv_core.a + 其 3rdparty .a> \
    <sysroot libcrypto.a>                 `# 仅当 SDK 以 WITH_CRYPTO=ON 构建` \
    <algo-input 实现库, das_ego_app 提供> \
    -lrknnrt -lpthread -ldl -lm -o app
```

要点：
- OpenCV/OpenSSL 由集成方 sysroot 提供。SDK 对象按 OpenCV 4.5 / OpenSSL EVP 接口编译，
  与 sysroot 的 OpenCV 4.5（静态）兼容；EVP API 在 OpenSSL 1.1 与 3.x 间一致，故链接
  sysroot 的 libcrypto（1.1 或 3）均可。
- 结果默认明文 JSON，不需要加密时可要求提供 **以 `-DLBP_WITH_CRYPTO=OFF` 构建** 的库，
  此时 SDK 完全无 OpenSSL 依赖（`encrypt=true` 退化为明文）。

动态库（可选）：`libled_box_pose_sdk.so` 运行期依赖 `libopencv_core`、`libopencv_imgproc`、
`librknnrt.so`、`libstdc++`（及 WITH_CRYPTO 时 `libcrypto`），仅适用于运行期具备**匹配版本**
这些 .so 的环境；全静态 sysroot 请用 `.a`。

## 部署到板端

- 静态链接时无需分发 SDK 库（已并入可执行）；动态链接时把 `.so*` 放入 `LD_LIBRARY_PATH` 或 `-rpath`。
- 整个 `assets/` 随包部署，`InitConfig.assets_dir` 指向它。

## 输出结果

`output_path` 指向的 JSON（当前明文）。逐帧字段：

- `ts_ns`：帧时间戳
- 顶层 `calibration`：各相机内参（Double Sphere）与外参（B2C 4x4）
- `step1`：两路 IR 的 LED 光斑检测
- `step2`：左右匹配、三角化 3D 点（body 系）、重投影误差
- `step3`：单路 RGB 上的手部检测框 + 置信度 + 21 关键点 + 关键点置信度

设 `InitConfig.encrypt = true` 时，输出为 `output_path + ".enc"`（AES-256-GCM）。

## 参数调整

修改 `assets/config.json`（无需重新编译）。常用项：`led_threshold`、`led_min_area`、
`led_max_area`（LED 光斑亮度阈值与面积范围）、`det_score_thr`（检测置信度阈值）、
`process_hz`（库内部处理频率）。JSON 中省略的键采用库内置缺省值。
