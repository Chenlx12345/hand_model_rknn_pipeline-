# IMU 硬件 id(hex)对照

输出 JSON 里 `boxes[*].imu_id` 与 `finger_points[*].imu_id` 是 14 颗 IMU(每只手)
的硬件地址(hex 字符串,如 `"0x14"`),不是 SDK 内部用的 0..13 索引。

数据来源:`RobbyVit_v2/led_box_pose/assets/imu_id.json`(rig owner 公布的硬件出厂表)。
SDK 把它编进了 `src/config/imu_id.hpp`,改硬件 id 需要同步两边。

---

## 1. 一张表全列出来

| SDK 内部 logical | side=0 (LEFT 左手) | side=1 (RIGHT 右手) | 关节(英文) | 中文 |
|:---:|:---:|:---:|:---|:---|
| **0**  | `0x11` (raw 17) | `0x21` (raw 33) | `led_box`      | 手背的灯盒(box 6-DoF 输出这只) |
| 1  | `0x12` (18)     | `0x22` (34)     | `thumb_MCP`    | 拇指 MCP |
| 2  | `0x13` (19)     | `0x23` (35)     | `thumb_IP`     | 拇指 IP  |
| 3  | `0x14` (20)     | `0x24` (36)     | `index_MCP`    | 食指 MCP |
| 4  | `0x15` (21)     | `0x25` (37)     | `index_DIP`    | 食指 DIP |
| 5  | `0x16` (22)     | `0x26` (38)     | `middle_MCP`   | 中指 MCP |
| 6  | `0x17` (23)     | `0x27` (39)     | `middle_DIP`   | 中指 DIP |
| 7  | `0x18` (24)     | `0x28` (40)     | `ring_MCP`     | 无名指 MCP |
| 8  | `0x19` (25)     | `0x29` (41)     | `ring_DIP`     | 无名指 DIP |
| 9  | `0x1A` (26)     | `0x2A` (42)     | `pinky_MCP`    | 小指 MCP |
| 10 | `0x1B` (27)     | `0x2B` (43)     | `pinky_DIP`    | 小指 DIP |
| 11 | `0x1C` (28)     | `0x2C` (44)     | `thumb_CMC`    | 拇指 CMC |
| 12 | `0x1D` (29)     | `0x2D` (45)     | `dorsum_ulnar` | 手背尺侧 |
| 13 | `0x05` (5)      | `0x08` (8)      | `wrist`        | 手腕(腕带 IMU) |

> 规律(便于记):**左手** 0x1?,**右手** 0x2?,**手腕例外**(左 0x05 / 右 0x08;
> 跨手共用 0..A 范围,因为腕带是独立模块)。

---

## 2. 各 IMU 在哪个位置

```
                  ┌─── kpt 4 (拇指 TIP)
                  │
       kpt 3 ──── 0x13/0x23 (thumb_IP)
       │
       kpt 2 ──── 0x12/0x22 (thumb_MCP)
       │
   kpt 1 ──────── 0x1C/0x2C (thumb_CMC)
   │
   │       ┌── kpt 8 ─── 0x15/0x25 (index_DIP)  ── kpt 7
   │       ├── kpt 12 ── 0x17/0x27 (middle_DIP) ── kpt 11
   │       ├── kpt 16 ── 0x19/0x29 (ring_DIP)   ── kpt 15
   │       └── kpt 20 ── 0x1B/0x2B (pinky_DIP)  ── kpt 19
   │
   │  ┌── kpt 5 ──── 0x14/0x24 (index_MCP)
   ├──┤   kpt 9 ──── 0x16/0x26 (middle_MCP)
   │  │   kpt 13 ─── 0x18/0x28 (ring_MCP)
   │  └── kpt 17 ─── 0x1A/0x2A (pinky_MCP)
   │
   ├── 0x11/0x21 (led_box,手背正中)
   ├── 0x1D/0x2D (dorsum_ulnar,手背尺侧)
   └── kpt 0 ─── 0x05/0x08 (wrist,腕带向前臂方向 ~100mm)
```

`kpt N` 是 step3 RTMPose 输出的 21 个关键点编号(同 MANO 习惯;见
[RESULT_FORMAT.md](RESULT_FORMAT.md) 第 6 节)。

---

## 3. 用法示例

### 拿到一只手的所有 IMU 的 3D 位置

```python
import json
for line in open("vv_result.jsonl"):
    rec = json.loads(line)
    # 左手所有 IMU:0x11 (box) + 0x12..0x1D + 0x05 (wrist)
    left = {}
    for b in rec["boxes"]:
        if b["side"] == 0:
            T = b["T_body_box_row_major"]
            # box 自己的 IMU 位置就是 T 的平移分量(即 t):
            left[b["imu_id"]] = [T[3], T[7], T[11]]   # 0x11
    for f in rec["finger_points"]:
        if f["side"] == 0:
            left[f["imu_id"]] = f["xyz_body"]
    print(line[:60], "... left=", list(left.keys()))
```

输出:
```
{"timestamp_ns":1779790769954290000,"frame_id":7, ... left= ['0x11', '0x1C', '0x14', '0x05']
```

### Hex 反查 → 名字

```python
HEX_TO_NAME = {
    "0x11":"led_box_L",  "0x12":"thumb_MCP_L",  "0x13":"thumb_IP_L",
    "0x14":"index_MCP_L", "0x15":"index_DIP_L",  "0x16":"middle_MCP_L",
    "0x17":"middle_DIP_L","0x18":"ring_MCP_L",   "0x19":"ring_DIP_L",
    "0x1A":"pinky_MCP_L", "0x1B":"pinky_DIP_L",  "0x1C":"thumb_CMC_L",
    "0x1D":"dorsum_ulnar_L","0x05":"wrist_L",
    "0x21":"led_box_R",  "0x22":"thumb_MCP_R",  "0x23":"thumb_IP_R",
    "0x24":"index_MCP_R", "0x25":"index_DIP_R",  "0x26":"middle_MCP_R",
    "0x27":"middle_DIP_R","0x28":"ring_MCP_R",   "0x29":"ring_DIP_R",
    "0x2A":"pinky_MCP_R", "0x2B":"pinky_DIP_R",  "0x2C":"thumb_CMC_R",
    "0x2D":"dorsum_ulnar_R","0x08":"wrist_R",
}
```

### C++ 端(SDK 内部用)

```cpp
#include "config/imu_id.hpp"
uint8_t b = vitgloves::vis::imu_id_byte(/*side=*/0, /*logical=*/3);  // → 0x14
std::string s = vitgloves::vis::imu_id_hex (0, 3);                   // → "0x14"
```

---

## 4. 哪些 IMU 一定出/可能不出

| imu_id 段 | 一定出? | 决定因素 |
|---|---|---|
| `0x11` / `0x21` (led_box) | step4 拟合到的那只手会有 | step3 识别到这只手 → step4 拟合通过 |
| `0x12..0x1D` / `0x22..0x2D`(手指) | **不一定**,逐点输出 | 该 IR 点没被聚类、不在剩余候选里、或 Hungarian 匹配代价 > gate → 不出 |
| `0x05` / `0x08` (wrist) | 通常出 | step3 给出 wrist + MCP centroid → 算 virtual wrist → 匹配到 IR 点 |

漏检(某 IMU 这一帧没出)和漏识别(整只手不可见)在结果里都是"少一条 entry",
不会产生 NaN 或占位条目 — 上层按 `imu_id` 查表即可。

---

## 5. 校对来源

```bash
cat /home/youmu/RobbyGen/RobbyVit_v2/led_box_pose/assets/imu_id.json
```

如果以后硬件改 id,改两处:
1. `RobbyVit_v2/led_box_pose/assets/imu_id.json`(rig owner 维护的源真相)
2. `led_hand_sdk/src/config/imu_id.hpp` 里的 `kImuIdHex` 表
3. 本文档 + `docs/RESULT_FORMAT.md` 第 4 节
