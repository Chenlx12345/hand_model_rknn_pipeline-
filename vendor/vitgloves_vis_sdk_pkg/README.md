# vitgloves_vis_sdk — 集成包

板子(RK3588 aarch64)上的静态库 + 头 + 资源,交付给集成方接入。

## 目录

```
.
├── include/vitgloves_vis_sdk.hpp        公共 API(PImpl,只暴露 InitConfig + class)
├── lib/libvitgloves_vis_sdk.a            交叉编译静态库(buildroot toolchain)
├── third_party/
│   ├── algo_input_api/algo_input_api.hpp   services::algo 接口快照(SDK 内部用)
│   └── rknn/rknn_api.h                     librknnrt 头(板上系统通常没装 SDK)
├── assets/
│   ├── config/config.json                 算法门限;改了不重编
│   ├── led/{box_left,box_right}.txt       7-LED 物理模板(mm)
│   └── model/{rtmdet,rtmpose}.rknn        RKNN 模型(如缺,联系交付方拿)
├── examples/
│   ├── demo_integrator.cpp                最小启动示例
│   └── CMakeLists.txt                     示例工程
├── build.sh                               一键编 demo
└── docs/
    └── RESULT_FORMAT.md                   NDJSON 输出每行字段说明
```

## 集成方需要做什么

1. **保留** `services::algo`(`libalgo_input`)能正常起来:
   - `initializeAlgoInput()` 成功
   - `setAlgoBatchCallback` / `setAlgoImuCallback` 能注册并被持续触发
   - `getAlgoCameraCalibration(0..5)` 都返回 `valid=true`
2. **配 assets**:把本包的 `assets/` 整目录拷到运行机,默认路径
   `/opt/vitgloves_vis_sdk/assets`(编译期 baked,运行时可用 env `VV_ASSETS_DIR` 覆盖)。
3. **链接**:`libvitgloves_vis_sdk.a` + `libalgo_input` + 系统 `librknnrt.so` + OpenCV4(集成方自己已有);
   见 `examples/CMakeLists.txt`。
4. **跑**:
   ```cpp
   vitgloves::vis::InitConfig cfg;
   cfg.output_path = "/data/recordings/vv.jsonl";
   vitgloves::vis::VitglovesVisSdk sdk(cfg);
   sdk.start();         // 内部注册回调,服务推数据进 SDK
   ...                  // 主线程做别的;5 Hz 在内部产出结果
   sdk.stop();          // 优雅注销 + 关文件
   ```
   完整代码见 [examples/demo_integrator.cpp](examples/demo_integrator.cpp)。

## 输出

每个 5Hz tick 一行 JSON 写到 `cfg.output_path`(NDJSON / JSON Lines)。
字段说明见 [docs/RESULT_FORMAT.md](docs/RESULT_FORMAT.md)。

## 依赖一览

| 组件 | 来自 | 备注 |
|---|---|---|
| C++17 | gcc 9+ | buildroot 给的 g++ ✓ |
| OpenCV4 | 集成方系统(/usr) | core/imgproc/video |
| librknnrt | 集成方系统(/usr/lib/) | 必须 |
| Eigen / nlohmann | 已静态编进 .a | 不需要集成方再装 |
| `services::algo` | 集成方 libalgo_input | 头由本包 third_party 提供 |
