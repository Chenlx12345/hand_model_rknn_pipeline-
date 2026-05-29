# 交付:打包静态库给集成方

本文档讲清两件事:

1. 怎么用集成方的 **buildroot toolchain** 把本 SDK 编成静态 `.a`;
2. 给集成方的 **tarball 里到底放了什么、放在哪儿、对方该怎么用**(参照
   `ref/guanyuan_sdk/examples/integrator_demo/` 的布局)。

板上联调(在 RK3588 上自己 cmake / 跑 demo)请看 [BUILD_ON_3588.md](./BUILD_ON_3588.md)。

---

## 1. 一键打包

```bash
cd led_hand_sdk
./scripts/pack_for_integrator.sh
```

完事拿到:
```
dist/
├── vitgloves_vis_sdk_pkg/         # 解开后就是交付目录
└── vitgloves_vis_sdk_pkg.tar.gz   # 直接发对方
```

---

## 2. 工具链来源(确定不让 ABI 漂)

|  | 路径 | 说明 |
|---|---|---|
| Toolchain | `das_ego_sdk/buildroot/output/rockchip_genrobot_rk3588/host/share/buildroot/toolchainfile.cmake` | 集成方提供的 buildroot 工具链,带 `aarch64-buildroot-linux-gnu-g++` + sysroot |
| OpenCV (aarch64 dist) | `das_ego_sdk/external/rknpu2/examples/3rdparty/opencv/opencv-linux-aarch64` | **集成方使用的同一份**;头文件 + `lib/libopencv_*.a`;ABI 与板上一致 |
| Eigen3 (header-only) | 宿主 `/usr/include/eigen3/Eigen` | 打包时拷进 `third_party/eigen/`,临时本地用 |
| nlohmann/json (多文件) | 宿主 `/usr/include/nlohmann/` | 同上,拷到 `third_party/json/nlohmann/` |
| librknnrt | buildroot sysroot 的 `usr/lib/librknnrt.so` | 编译只引用 `rknn_*` 符号,运行时由板上 `/usr/lib/librknnrt.so` 提供 |
| rknn_api.h | 本仓库 `third_party/rknn/rknn_api.h` | 已 vendor,板上系统也没装 |

**为什么用 buildroot toolchain?** `aarch64-buildroot-linux-gnu-g++` 的 libstdc++ ABI、glibc 版本、`-D_FORTIFY_SOURCE` 默认值等都和集成方 `das_ego_app` 二进制锁定;不用就会出现"链接成功但运行 segfault"的 ABI 漂移。

**为什么不能用宿主 `/usr/include/opencv4`?** buildroot 包装器会拒绝任何在工具链 sysroot 外的 `/usr` 路径(`unsafe header/library path used in cross-compilation`)。所以 OpenCV 必须指向**项目目录里的一份 aarch64 dist**。我们直接复用 das_ego_sdk 里现成的 `opencv-linux-aarch64`,集成方板上跑的就是它。

---

## 3. `pack_for_integrator.sh` 都做了什么

```text
1. 校验:toolchainfile.cmake / opencv-linux-aarch64/include / Eigen / nlohmann 都在
2. 把 Eigen + nlohmann 拷进 third_party/{eigen,json}     # 临时,gitignore 了
3. cmake 配置:
   -DCMAKE_TOOLCHAIN_FILE=<buildroot>
   -DCMAKE_BUILD_TYPE=Release
   -DVV_BUILD_HOST_DEMO=OFF              # 不编 mp4_algo_service / demo_init
   -DOpenCV_INCLUDE_DIRS=<opencv-aarch64>/include
   -DOpenCV_LIBS=""                       # .a 不真链 OpenCV,集成方链
4. cmake --build → libvitgloves_vis_sdk.a   (aarch64 ELF)
5. 把 .a + 头 + third_party + assets + 示例 + docs 拼成 dist/vitgloves_vis_sdk_pkg/
6. tar czf dist/vitgloves_vis_sdk_pkg.tar.gz
```

可覆盖的环境变量:

| 变量 | 默认 | 何时改 |
|---|---|---|
| `TOOLCHAIN_FILE` | 上表那条 | 集成方升级 buildroot |
| `OPENCV_DIR` | das_ego_sdk 那条 | OpenCV dist 换位置 |
| `EIGEN_INCLUDE` | `/usr/include/eigen3` | 容器构建 / 系统没装 |
| `JSON_INCLUDE` | `/usr/include` | 同上 |
| `PKG_NAME` | `vitgloves_vis_sdk_pkg` | 加版本号:`PKG_NAME=vitgloves_vis_sdk_v0.1.0` |
| `OUT_DIR` | `dist/` | — |

---

## 4. 交付包目录布局

```
vitgloves_vis_sdk_pkg/
├── README.md                       集成方的"读我"
├── docs/RESULT_FORMAT.md           NDJSON 每行字段
├── include/
│   └── vitgloves_vis_sdk.hpp       公共 API(PImpl,只 <cstdint>/<string>)
├── lib/
│   └── libvitgloves_vis_sdk.a      aarch64,buildroot 工具链产出
├── third_party/
│   ├── algo_input_api/
│   │   └── algo_input_api.hpp      services::algo 接口快照(集成方一般已有,这里给个对版)
│   └── rknn/
│       └── rknn_api.h              板上系统不装 SDK 也能编
├── assets/
│   ├── config/config.json          算法门限;改了不重编
│   ├── led/box_left.txt
│   ├── led/box_right.txt           7 LED 物理坐标(mm)
│   └── model/rtmdet.rknn
│       model/rtmpose.rknn          RKNN 模型(若 host 本地没有,这步会跳过 → 让集成方自己 scp)
├── examples/
│   ├── demo_integrator.cpp         最小启动示例(SIGINT 退出)
│   └── CMakeLists.txt              链 .a + 系统 OpenCV + librknnrt + libalgo_input
└── build.sh                        集成方一键编 demo
```

包大小参考(含 `assets/model/*.rknn`):**~36 MB**;去掉模型 ~5 MB。

---

## 5. 集成方拿到包之后做什么

参照 guanyuan_sdk 的 integrator_demo 流程,需要做的最少 4 步:

### ① 把 assets 拷到板上

默认编译期路径 `/opt/vitgloves_vis_sdk/assets`(SDK 启动时 baked,运行时可 env `VV_ASSETS_DIR` 覆盖):
```bash
sudo mkdir -p /opt/vitgloves_vis_sdk
sudo cp -r assets /opt/vitgloves_vis_sdk/
```

### ② 编 demo

```bash
./build.sh
# 等价于:
#   cd examples/build && cmake .. && make
```

`examples/CMakeLists.txt` 默认链:
- `lib/libvitgloves_vis_sdk.a`(本包)
- 系统 `librknnrt.so`(`/usr/lib/`)
- 集成方现成的 `libalgo_input`(也叫 das_ego_app 那个进程的 algo 服务实现)
- OpenCV4 + pthread + dl

### ③ 跑

```bash
./examples/build/demo_integrator /data/recordings/vv.jsonl
# 按 Ctrl-C 停;每秒打印 frames + imu 计数;5 Hz 写一行 JSON。
```

### ④ 读结果

输出文件是 **NDJSON / JSON Lines**;每行 4 段(boxes / finger_points / step2_points / step3_points)。
字段一览见 `docs/RESULT_FORMAT.md`。

---

## 6. ABI 与依赖契约(给集成方踩坑用)

| 项 | 我们的 .a 引用 | 谁负责满足 |
|---|---|---|
| `cv::*` | 用 opencv-linux-aarch64(buildroot 那份)头编 | 集成方:链同一份 OpenCV |
| `rknn_*` | 用 `rknn_api.h` 编 | 集成方:运行时有 `librknnrt.so` |
| `services::algo::*` | 用 vendored `third_party/algo_input_api/algo_input_api.hpp` 编 | 集成方:libalgo_input 实现这个接口 |
| `Eigen::*` | header-only,**已静态展开进 .a** | 不需要 |
| `nlohmann::json` | header-only,**已静态展开进 .a** | 不需要 |
| libc / libstdc++ | buildroot 工具链 | 集成方 sysroot 一致即可 |

确认 .a 干净的命令:
```bash
aarch64-buildroot-linux-gnu-nm libvitgloves_vis_sdk.a | awk '$1=="U"{print $2}' | c++filt | sort -u
```
应该只看到 `cv::` + `rknn_` + `services::algo::*` + 标准库函数(`__cxa_*`, `pthread_*`, `f{open,close,read,write}`, …)。**不应该出现** `nlohmann::` 或 `Eigen::` 的未解析符号 — 那两个 header-only 库已经全部进 .a。

---

## 7. 给集成方的简短"说明手册"模板

直接附 `README.md` 在包里(`pack_for_integrator.sh` 已自动生成);要点:

> **依赖**(板上确认)
> - `/usr/lib/librknnrt.so` 必有
> - 集成方自己的 OpenCV4 + libalgo_input 已就位
>
> **集成 5 步**
> ```cpp
> vitgloves::vis::InitConfig cfg;
> cfg.output_path = "/data/recordings/vv.jsonl";   // NDJSON 落盘
>
> vitgloves::vis::VitglovesVisSdk sdk(cfg);
> if (!sdk.start()) {                              // 内部自己注册回调
>     std::fprintf(stderr, "%s\n", sdk.last_error().c_str());
>     return 1;
> }
> // ... 业务循环 ...
> sdk.stop();                                       // 优雅退出,关文件
> ```
>
> **输出**:5 Hz 写一行 JSON,字段见 `docs/RESULT_FORMAT.md`。
>
> **改算法参数**:`assets/config/config.json` 直接改,不用重编(SDK start 时读)。

---

## 8. 常见交付坑

| 现象 | 原因 | 处理 |
|---|---|---|
| `unsafe header/library path used in cross-compilation: '-I/usr/include/...'` | OpenCV 用了 host /usr | 必须用 buildroot 兼容的 dist(`OPENCV_DIR`) |
| `cmake: nlohmann/adl_serializer.hpp: No such file` | 只拷了 `json.hpp`,没拷子头 | pack 脚本现在 `cp -r nlohmann/.`,确认 `third_party/json/nlohmann/` 不是只有 1 个文件 |
| 集成方链接 `undefined reference: services::algo::initializeAlgoInput()` | 没链 libalgo_input | `target_link_libraries(... algo_input)` |
| 集成方 demo 运行 `cannot open .rknn` | assets 没拷到 `/opt/...` 或没 env 覆盖 | `VV_ASSETS_DIR=/path/to/assets ./demo_integrator ...` |
| 运行 `rknn_init failed (-1)` | librknnrt.so 版本与 .rknn 不匹配 | 让集成方升级 librknnrt(板载 RKNN runtime) |

---

## 9. 重发版本时

1. 改 `include/vitgloves_vis_sdk.hpp` 里的 `VITGLOVES_VIS_SDK_VERSION_*`;
2. `PKG_NAME=vitgloves_vis_sdk_v0.2.0 ./scripts/pack_for_integrator.sh`;
3. 给集成方一份 `CHANGELOG`,标 ABI 是否变(改了 `InitConfig` 字段 / `VitglovesVisSdk` 公共方法签名 = ABI 破)。

---

## 附:验证一次真的能编

```bash
$ ./scripts/pack_for_integrator.sh
...
-- OpenCV (manual): .../opencv-linux-aarch64/include
-- Eigen3:          .../third_party/eigen
-- nlohmann/json:   .../third_party/json
-- algo_input_api:  .../third_party/algo_input_api
-- RKNN enabled:    .../sysroot/usr/lib/librknnrt.so (header .../third_party/rknn)
[100%] Linking CXX static library libvitgloves_vis_sdk.a
✓ static lib: build_pack/libvitgloves_vis_sdk.a  (3.2 MiB)
✓ 完成: dist/vitgloves_vis_sdk_pkg.tar.gz  (36 MB)
```

包里 `libvitgloves_vis_sdk.a`:
```
$ file libvitgloves_vis_sdk.a
libvitgloves_vis_sdk.a: current ar archive
$ aarch64-buildroot-linux-gnu-nm libvitgloves_vis_sdk.a | awk '$1=="U"' | wc -l
172      # 全是 cv::* / rknn_* / services::algo::* / libc / libstdc++ 符号
```
