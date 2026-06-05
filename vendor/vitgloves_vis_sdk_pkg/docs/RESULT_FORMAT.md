# 结果文件格式(`vitgloves_vis_sdk` 输出)

SDK 每一个 **5Hz tick** 写一行 JSON 到 `InitConfig::output_path`。
文件整体是 **NDJSON / JSON Lines**(每行一个独立 JSON 对象,可直接 `for line in f: json.loads(line)` 流式读)。

参考解析示例见 `visual_calib/scripts/visualize.py`。

---

## 1. 通用约定

| 项 | 约定 |
|---|---|
| 坐标系 | **body 系**(头部 IMU 当 origin,与相机标定的 `T_b_c` 同源) |
| 距离单位 | 米(`xyz` / `xyz_body` 都是米) |
| 像素单位 | 像素(0,0 在图像左上角;原图分辨率 1600×1300) |
| 时间 | `timestamp_ns` 纳秒,CLOCK_MONOTONIC,与 IMU 同钟 |
| 节拍 | 默认 5 Hz(由 `assets/config/config.json` 里的 `process_hz` 决定) |
| 标签 | `side`/`label`:**0 = 左手,1 = 右手**(box LED 模板对应 `box_left.txt` / `box_right.txt`) |

---

## 2. 顶层字段

```jsonc
{
  "timestamp_ns": 1779790769954290000,   // 该帧 6 路相机凑齐时刻 (CLOCK_MONOTONIC)
  "frame_id":     7,                      // SDK 内部 batch 计数器(从 0 起,每个 30Hz batch +1)
  "boxes":         [ … ],
  "finger_points": [ … ],
  "step2_points":  [ … ],
  "step3_points":  [ … ]
}
```

四个数组都可能为空,语义见下表:

| 段 | 空时含义 |
|---|---|
| `boxes` | 该 tick 没拟合出有效 6-DoF(常因 step3 未识别出手) |
| `finger_points` | 同上,没 box 就无 side 信息,不做 logical 匹配 |
| `step2_points` | IR 端没出三角化点(一般不会,除非两路检测都 ≤ 3) |
| `step3_points` | RTMDet 没框出手 / RTMPose 置信度全部 < gate |

---

## 3. `boxes[]` — step4 LED-box 6-DoF

每只识别到的手有一个 entry(0 ~ 2 个)。

```jsonc
{
  "side":   0,                            // 0=left, 1=right;对应 box_left.txt / box_right.txt
  "imu_id": "0x11",                       // 灯盒自己的 IMU 硬件 id(left=0x11, right=0x21)
  "T_body_box_row_major": [               // 4×4 齐次变换,box→body:X_body = R·X_box + t
    -0.9033, -0.3911, -0.1764, 0.1735,    //   R 是 3×3,t 是末列 3×1
     0.4238, -0.7489, -0.5095, 0.1663,    //   最后一行固定 [0,0,0,1]
     0.0671, -0.5350,  0.8422, -0.3220,
     0.0,     0.0,     0.0,     1.0
  ],
  "reproj_px": 0.418                      // 7 个 LED 模板用 (R,t) 转 body 再反投到 cam1+cam4
                                          //  的"联合重投影残差"平均(像素);典型 ≤ 2 px
}
```

复原:
```python
import numpy as np
T = np.array(box["T_body_box_row_major"]).reshape(4, 4)
R, t = T[:3, :3], T[:3, 3]
# 用 box_{left|right}.txt 的 7 个 LED 模板(mm → m):
M_body = (R @ M.T).T + t                  # M:(7,3),已转米;输出 body 系
```

---

## 4. `finger_points[]` — 各 IMU 的 3D 位置

每只手最多 13 条 entry(13 颗手指 / 腕 IMU;**灯盒自己**(left=`0x11` / right=`0x21`)
不在这一段,它在 `boxes[]` 里出来)。

```jsonc
{
  "side":     0,                          // 这只手是左/右
  "imu_id":   "0x14",                     // IMU 硬件 hex id(下表对照)
  "xyz_body": [0.1512, 0.1877, -0.3308],  // 该 IMU 关节估计的 3D 位置(米,body 系)
  "reproj_l": 0.433,                      // 源 IR 点在 cam1 的重投影残差(像素;来自 step2)
  "reproj_r": 0.476                       //                cam4
}
```

`imu_id` 来自 `RobbyVit_v2/led_box_pose/assets/imu_id.json`,完整对照见
[IMU_ID.md](IMU_ID.md)。简表如下(关节定义 + RTMPose kpt 推导公式):

| 左手 | 右手 | 关节 | 怎么定位(用 step3 21 kpt 推 3D 预测) |
|---|---|---|---|
| `0x12` | `0x22` | 拇指 MCP    | kpt 2 → 3,interior α=0.35 |
| `0x13` | `0x23` | 拇指 IP     | kpt 3 → 4,α=0.50 |
| `0x14` | `0x24` | 食指 MCP    | kpt 5 → 6,α=0.40 |
| `0x15` | `0x25` | 食指 DIP    | kpt 7 → 8,α=0.50 |
| `0x16` | `0x26` | 中指 MCP    | kpt 9 → 10,α=0.40 |
| `0x17` | `0x27` | 中指 DIP    | kpt 11 → 12,α=0.50 |
| `0x18` | `0x28` | 无名指 MCP  | kpt 13 → 14,α=0.40 |
| `0x19` | `0x29` | 无名指 DIP  | kpt 15 → 16,α=0.50 |
| `0x1A` | `0x2A` | 小指 MCP    | kpt 17 → 18,α=0.40 |
| `0x1B` | `0x2B` | 小指 DIP    | kpt 19 → 20,α=0.50 |
| `0x1C` | `0x2C` | 拇指 CMC    | kpt 1(POINT,无插值) |
| `0x1D` | `0x2D` | 手背尺侧    | kpt 0 → 17 中点 |
| `0x05` | `0x08` | 手腕        | virtual = kpt0 + α·\|kpt0 − MCP_centroid\|·proximal_dir |

> "3D 位置" = 该手剩余 IR 点(去掉被 box 用走的 7 个)里,Hungarian 距离最优配到该 IMU 的那一个;
> `reproj_l/r` 就是这个 IR 点在 step2 里的重投影残差,直接复用。

---

## 5. `step2_points[]` — step2 三角化的 IR 3D 点

每个亮点(双手 LED + 各种手指 IMU 灯)各一条。

```jsonc
{
  "xyz":      [0.2055, 0.0723, -0.4060], // body 系 3D 位置(米)
  "uv1":      [886.0,  723.5],           // 在 cam1 (左 IR) 的像素中心
  "uv4":      [264.71, 826.96],          // 在 cam4 (右 IR) 的像素中心
  "reproj_l": 0.086,                     // xyz 用 DS 反投到 cam1 后与 uv1 的像素距离
  "reproj_r": 0.098                      //                       cam4
}
```

注意:`step2_points` 是 **唯一一段所有 tick 都会有内容的段**(只要两路 IR 都检测到 > 3 个光斑就出);其他三段都依赖它有点。

---

## 6. `step3_points[]` — step3 三角化的 hand 21-kpt 3D 点

按"每只手每个 kpt 一条"输出。最多 2 手 × 21 kpt = 42 条。

```jsonc
{
  "side":     0,                          // 0=left, 1=right(RTMDet 的 class)
  "kpt":      5,                          // 0..20(RTMPose 21 关节;0=wrist, 5=index_MCP, …)
  "xyz":      [0.197, 0.221, -0.367],     // body 系 3D(米)
  "cam_a":    2,                          // 主 RGB 相机 id(IR 左密时选 cam2;反之 cam3)
  "cam_b":    3,                          // 副 RGB 相机 id
  "uv_a":     [620.4, 925.1],             // 在 cam_a 的 RTMPose 输出像素
  "uv_b":     [521.7, 901.3],             //   cam_b
  "reproj_a": 1.70,                       // xyz 反投到 cam_a 与 uv_a 的像素距离
  "reproj_b": 1.64                        //              cam_b
}
```

RTMPose 的 21 关节定义(`kpt` 字段):

```
 0: wrist
 1-4:   thumb   (CMC, MCP, IP, TIP)
 5-8:   index   (MCP, PIP, DIP, TIP)
 9-12:  middle  (MCP, PIP, DIP, TIP)
13-16:  ring    (MCP, PIP, DIP, TIP)
17-20:  pinky   (MCP, PIP, DIP, TIP)
```

骨架连接(供画图):
```
0–1–2–3–4         (thumb)
0–5–6–7–8         (index)
0–9–10–11–12      (middle)
0–13–14–15–16     (ring)
0–17–18–19–20     (pinky)
```

只有 **conf ≥ kpt_conf_thr(默认 0.30)且射线最近距离 ≤ 3 cm** 的 kpt 才进 `step3_points`,所以可能 < 42 条。

---

## 7. 完整最小示例

```json
{"timestamp_ns":1779790769954290000,"frame_id":7,
 "boxes":[
   {"side":0,"imu_id":"0x11",
    "T_body_box_row_major":[-0.903,-0.391,-0.176,0.174,
                             0.424,-0.749,-0.510,0.166,
                             0.067,-0.535, 0.842,-0.322,
                             0.0,   0.0,   0.0,   1.0],
    "reproj_px":0.42}
 ],
 "finger_points":[
   {"side":0,"imu_id":"0x1C","xyz_body":[0.151,0.188,-0.331],
    "reproj_l":0.43,"reproj_r":0.48}
 ],
 "step2_points":[
   {"xyz":[0.205,0.072,-0.406],"uv1":[886.0,723.5],"uv4":[264.7,827.0],
    "reproj_l":0.09,"reproj_r":0.10}
 ],
 "step3_points":[
   {"side":0,"kpt":0,"xyz":[0.202,0.210,-0.338],
    "cam_a":2,"cam_b":3,
    "uv_a":[605.6,940.9],"uv_b":[511.6,917.3],
    "reproj_a":1.70,"reproj_b":1.64}
 ]
}
```

---

## 8. 跨段如何关联

不同段之间没有显式外键,但可以通过"距离 / 像素位置"相互对应:

- **box ↔ step2**:用 `T_body_box` 把 7 个 LED 模板转 body,与 `step2_points[*].xyz` 距离最小者就是这只 box 用了的 7 个 IR 点;
- **finger ↔ step2**:`finger_points[*].xyz_body` **就是** 某个 `step2_points[*].xyz`(直接复用,reproj_l/r 也一致),按位置查重即可还原是哪一条 step2 点;
- **finger ↔ step3**:同手(`side` 相同)的 step3 21 kpt 用解剖学公式算出预测 3D 位置 → 匹配到剩余 IR 点 → 就是 `finger_points`。

---

## 9. 节拍与缺省

- **节流**:由 `Params::process_hz`(默认 5 Hz)决定,SDK 内部按 `batch_mono_ns - last_process_ns ≥ 1e9 / process_hz` 触发;在 5 Hz tick 之间的 30 Hz batch 不写盘。
- **空段**:任何一段在该 tick 缺少有效数据时输出 `[]`,行结构保持稳定(可以无脑 `len(d["boxes"])`)。
- **IMU**:**不写盘**。IMU 200 Hz 数据只在 SDK 内部的互补滤波器 `imu_filter` 里维护状态(`q_world_body` + `gyro_bias`),将来如要消费可加字段。
- **frame_id 不连续**:5 Hz 节流意味着典型间隔 `frame_id` 增 5~7(取决于 30 Hz batch 抖动)。
