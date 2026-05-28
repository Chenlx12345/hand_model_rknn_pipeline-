// led_box_pose_sdk — board-side (RK3588) hand-GT extraction SDK.
//
// Consumes the 6-camera live stream delivered by InferenceCameraService,
// runs the led_box_pose pipeline (IR LED triangulation → hand det/pose → box
// 6DoF), and writes both intermediate and final results into ONE encrypted
// file (AES-256-GCM). See DESIGN.md.
//
// Public surface is intentionally narrow: one class, POD structs, and NO
// third-party types (no OpenCV / Eigen / OpenSSL) so integrators link a single
// .so without pulling our dependencies. Internals are hidden via PImpl +
// -fvisibility=hidden + a linker version script.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace led_box_pose {

/* ── Version ─────────────────────────────────────────────────── */
#define LED_BOX_POSE_SDK_VERSION_MAJOR 0
#define LED_BOX_POSE_SDK_VERSION_MINOR 1
#define LED_BOX_POSE_SDK_VERSION_PATCH 0

/* ── Visibility ──────────────────────────────────────────────── */
#if defined(LED_BOX_POSE_SDK_BUILD)
  #define LED_BOX_POSE_API __attribute__((visibility("default")))
#else
  #define LED_BOX_POSE_API
#endif

/* ── Borrowed image view ─────────────────────────────────────────
 * Mirrors InferenceCameraView: the pixel buffer is owned by the caller and is
 * only valid for the duration of the process_batch() call. The SDK copies what
 * it needs; do not assume the SDK keeps the pointer. */
struct ImageView {
    const uint8_t* data        = nullptr;  ///< pixel buffer (BGR8 or 8-bit gray)
    int            width       = 0;
    int            height      = 0;
    int            channels    = 0;        ///< 3 = BGR, 1 = gray
    int            step_bytes  = 0;        ///< row stride in bytes (0 = width*channels)
    int            cam_id      = -1;       ///< 0..5, matches InferenceCameraView::cam_id
};

/* ── Camera calibration ──────────────────────────────────────────
 * One camera's intrinsics + body→camera extrinsics, mirroring the fields the
 * algo-input service exposes (services::algo::CameraCalibration). The caller
 * fills this from getAlgoCameraCalibration() and passes it via InitConfig.calib.
 * Only the Double Sphere model is supported: K holds fx,fy,cx,cy; dist holds the
 * DS params [xi, alpha]. */
struct CameraCalib {
    int    cam_id = -1;
    int    width  = 0;
    int    height = 0;
    double K[9]   = {};        ///< 3x3 intrinsics, row-major: [fx,0,cx, 0,fy,cy, 0,0,1]
    double dist[8] = {};       ///< distortion; Double Sphere: dist[0]=xi, dist[1]=alpha
    int    n_dist = 0;
    double T_b_c[7] = {};      ///< body→camera: tx,ty,tz, qx,qy,qz,qw
};

/* ── Init config ─────────────────────────────────────────────── */
struct InitConfig {
    /// INPUT (preferred): per-camera calibration from the algo-input interface.
    /// If empty, the SDK falls back to reading calib_path.
    std::vector<CameraCalib> calib;
    std::string calib_path;            ///< INPUT (fallback): camera_info.json path
    std::string assets_dir;            ///< INPUT: static assets (config, models, box model)
    std::string output_path;           ///< OUTPUT: result file (.enc when encrypt=true)
    int         cam_left  = 1;         ///< IR stereo left  (rig: cam1)
    int         cam_right = 4;         ///< IR stereo right (rig: cam4)
    int         cam_rgb_a = 2;         ///< RGB cam paired with cam_left  (rig: cam2)
    int         cam_rgb_b = 3;         ///< RGB cam paired with cam_right (rig: cam3)
    bool        encrypt   = false;     ///< false = plaintext JSON; true = AES-256-GCM
};

/* ── Error codes ─────────────────────────────────────────────── */
constexpr int LBP_OK              =  0;
constexpr int LBP_ERR_NOT_READY   = -1;   ///< construction failed / not initialized
constexpr int LBP_ERR_INVALID_ARG = -2;
constexpr int LBP_ERR_FINISHED    = -3;   ///< finalize() already called
constexpr int LBP_ERR_IO          = -4;
constexpr int LBP_ERR_INTERNAL    = -5;

/* ── SDK ─────────────────────────────────────────────────────── */
class LED_BOX_POSE_API LedBoxPoseSdk {
public:
    explicit LedBoxPoseSdk(const InitConfig& cfg);
    ~LedBoxPoseSdk();

    LedBoxPoseSdk(const LedBoxPoseSdk&) = delete;
    LedBoxPoseSdk& operator=(const LedBoxPoseSdk&) = delete;

    /** True if construction (calib/assets load) succeeded. Check before feeding. */
    bool ok() const;

    /**
     * Feed one time-synchronized 6-camera batch (as delivered by the camera
     * service callback). Frames are borrowed for the call only.
     * @param frames   array of ImageView (any subset of the 6 cams; the SDK
     *                 picks cam_left/cam_right for IR triangulation)
     * @param n_frames number of elements in frames[]
     * @param pts_us   batch hardware trigger timestamp (microseconds)
     * @return LBP_OK or a negative error code
     */
    int process_batch(const ImageView* frames, int n_frames, int64_t pts_us);

    /**
     * Flush + finalize: serialize accumulated intermediate+final results and
     * write the (encrypted) output file. Call once at end-of-stream. Idempotent
     * after the first successful call (returns LBP_ERR_FINISHED).
     * @return LBP_OK or a negative error code
     */
    int finalize();

    /* ── read-only stats ── */
    int         frames_processed() const;   ///< number of batches accepted
    int         points_last_frame() const;  ///< 3-D points from the last batch
    int         hands_last_frame() const;   ///< step3 hands from the last batch
    std::string last_error() const;         ///< human-readable last error

private:
    struct Impl;
    Impl* pimpl_;
};

} // namespace led_box_pose
