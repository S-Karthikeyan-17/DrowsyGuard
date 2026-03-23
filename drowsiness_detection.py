"""
DrowsyGuard — Driver Drowsiness Detection System
Backend: MediaPipe Face Mesh + Flask

DETECTION METHOD:
  Uses MediaPipe Face Mesh (468 landmarks) to compute:
    • EAR  — Eye Aspect Ratio  (Soukupova & Cech, 2016)
              EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
              Both eyes averaged. Falls below EAR_THRESH when eye closes.
    • MAR  — Mouth Aspect Ratio (same geometry applied to lips)
              Rises above MAR_THRESH when mouth opens wide (yawn).

  These ratios are far more robust than Haar cascades:
    Works under varied lighting, with glasses, different skin tones.
    Sub-pixel precision — no false positives from face texture.
    No separate cascade XML files needed.

Install:
    pip install mediapipe flask flask-cors numpy opencv-python

Run:
    python drowsyguard.py
    Open http://localhost:5000
"""

import cv2
import numpy as np
import base64
import traceback
import os
import math
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import mediapipe as mp

_mp_face  = mp.solutions.face_mesh
_face_mesh = _mp_face.FaceMesh(
    static_image_mode        = False,
    max_num_faces            = 1,
    refine_landmarks         = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence  = 0.5,
)

# ============================================================
#  LANDMARK INDICES  (MediaPipe 468-point Face Mesh)
# ============================================================

# Left eye  (from subject's perspective — appears on right in mirror)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
# Right eye
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Mouth landmarks for MAR
MOUTH_TOP     = 13
MOUTH_BOTTOM  = 14
MOUTH_LEFT    = 78
MOUTH_RIGHT   = 308
MOUTH_UPPER_L = 82
MOUTH_UPPER_R = 312
MOUTH_LOWER_L = 87
MOUTH_LOWER_R = 317

# ============================================================
#  TUNEABLE THRESHOLDS
# ============================================================
EAR_THRESH        = 0.22   # below this  -> eye closed
MAR_THRESH        = 0.65   # above this  -> yawn
EYE_CLOSED_FRAMES = 15     # frames -> DANGER  (~1.5 s @ 10 fps)
WARNING_FRAMES    = 8      # frames -> WARNING
YAWN_FRAME_THRESH = 10     # frames -> YAWN alert
EAR_SMOOTH_N      = 3      # rolling average window

# ============================================================
#  SESSION STATE
# ============================================================
_state = {
    "eye_closed_counter":  0,
    "yawn_counter":        0,
    "total_drowsy_events": 0,
    "total_yawn_events":   0,
    "frame_count":         0,
    "alert_level":         "NORMAL",
    "ear_history":         [],
}


def reset_state() -> None:
    _state.update({
        "eye_closed_counter":  0,
        "yawn_counter":        0,
        "total_drowsy_events": 0,
        "total_yawn_events":   0,
        "frame_count":         0,
        "alert_level":         "NORMAL",
        "ear_history":         [],
    })


# ============================================================
#  GEOMETRY HELPERS
# ============================================================

def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _lm(landmarks, idx, w, h):
    """Return pixel coords (x, y) for landmark index."""
    pt = landmarks[idx]
    return (pt.x * w, pt.y * h)


def compute_ear(landmarks, eye_indices, w, h) -> float:
    """
    Eye Aspect Ratio:
      EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    Returns 0.0 on degenerate geometry.
    """
    p = [_lm(landmarks, i, w, h) for i in eye_indices]
    vert1 = _dist(p[1], p[5])
    vert2 = _dist(p[2], p[4])
    horiz = _dist(p[0], p[3])
    if horiz < 1e-6:
        return 0.0
    return (vert1 + vert2) / (2.0 * horiz)


def compute_mar(landmarks, w, h) -> float:
    """
    Mouth Aspect Ratio using 3 vertical pairs for stability.
    Higher value = wider open mouth.
    """
    top   = _lm(landmarks, MOUTH_TOP,    w, h)
    bot   = _lm(landmarks, MOUTH_BOTTOM, w, h)
    left  = _lm(landmarks, MOUTH_LEFT,   w, h)
    right = _lm(landmarks, MOUTH_RIGHT,  w, h)
    ul    = _lm(landmarks, MOUTH_UPPER_L, w, h)
    ll    = _lm(landmarks, MOUTH_LOWER_L, w, h)
    ur    = _lm(landmarks, MOUTH_UPPER_R, w, h)
    lr    = _lm(landmarks, MOUTH_LOWER_R, w, h)

    vert1 = _dist(top, bot)
    vert2 = _dist(ul,  ll)
    vert3 = _dist(ur,  lr)
    horiz = _dist(left, right)

    if horiz < 1e-6:
        return 0.0
    return (vert1 + vert2 + vert3) / (3.0 * horiz)


# ============================================================
#  CODEC HELPERS
# ============================================================

def decode_frame(b64_image: str) -> np.ndarray:
    if "," in b64_image:
        _, data = b64_image.split(",", 1)
    else:
        data = b64_image
    rem = len(data) % 4
    if rem:
        data += "=" * (4 - rem)
    img_bytes = base64.b64decode(data)
    if not img_bytes:
        raise ValueError("Empty image data.")
    arr   = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("Failed to decode image.")
    return frame


def encode_frame(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("cv2.imencode failed.")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


# ============================================================
#  ANNOTATION HELPERS
# ============================================================

def draw_hud(frame, alert_level, ear, mar):
    colors = {"NORMAL": (0, 220, 120), "WARNING": (0, 165, 255), "DANGER": (50, 50, 255)}
    color  = colors.get(alert_level, (150, 150, 150))
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (10, 10, 20), -1)
    cv2.putText(frame,
        f"STATUS: {alert_level}   EAR: {ear:.3f} (thresh={EAR_THRESH})   MAR: {mar:.3f} (thresh={MAR_THRESH})",
        (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
    cv2.putText(frame,
        f"Eye-closed: {_state['eye_closed_counter']}   Yawn-ctr: {_state['yawn_counter']}   "
        f"Drowsy events: {_state['total_drowsy_events']}   Yawn events: {_state['total_yawn_events']}",
        (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 200, 220), 1, cv2.LINE_AA)


def draw_eye_contour(frame, lm, eye_indices, w, h, color):
    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in eye_indices]
    for i in range(len(pts)):
        cv2.line(frame, pts[i], pts[(i + 1) % len(pts)], color, 1, cv2.LINE_AA)
    xs, ys = zip(*pts)
    cv2.rectangle(frame, (min(xs)-3, min(ys)-3), (max(xs)+3, max(ys)+3), color, 1)


def draw_mouth_contour(frame, lm, w, h, color):
    idxs = [MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOTTOM,
            MOUTH_UPPER_L, MOUTH_UPPER_R, MOUTH_LOWER_L, MOUTH_LOWER_R]
    pts  = [(int(lm[i].x * w), int(lm[i].y * h)) for i in idxs]
    xs, ys = zip(*pts)
    cv2.rectangle(frame, (min(xs)-4, min(ys)-4), (max(xs)+4, max(ys)+4), color, 1)
    cv2.line(frame,
             (int(lm[MOUTH_TOP].x*w), int(lm[MOUTH_TOP].y*h)),
             (int(lm[MOUTH_BOTTOM].x*w), int(lm[MOUTH_BOTTOM].y*h)),
             color, 1, cv2.LINE_AA)


def draw_ear_bar(frame, ear, eye_closed):
    bx, by = frame.shape[1] - 28, 55
    bh     = 120
    cv2.rectangle(frame, (bx, by), (bx+18, by+bh), (30, 30, 30), -1)
    cv2.rectangle(frame, (bx, by), (bx+18, by+bh), (80, 80, 80), 1)
    ty = by + int(bh * (1.0 - EAR_THRESH / 0.40))
    cv2.line(frame, (bx-4, ty), (bx+22, ty), (0, 200, 255), 1)
    fh_bar = int(bh * min(ear / 0.40, 1.0))
    fy     = by + bh - fh_bar
    clr    = (0, 50, 255) if eye_closed else (0, 200, 120)
    if fh_bar > 0:
        cv2.rectangle(frame, (bx+1, fy), (bx+17, by+bh-1), clr, -1)
    cv2.putText(frame, "EAR", (bx-2, by-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 180, 180), 1)


def draw_alert_overlay(frame, level):
    if level not in ("DANGER", "WARNING"):
        return
    h, w = frame.shape[:2]
    cx   = w // 2
    cy   = h // 2
    if level == "DANGER":
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 200), 6)
        msg   = "!! DROWSINESS DETECTED - PULL OVER !!"
        tw, _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0], 0
        cv2.putText(frame, msg, (cx - tw//2, cy+1),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, msg, (cx - tw//2, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (50, 50, 255), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 140, 255), 4)
        msg   = "! EYES CLOSING - STAY ALERT !"
        tw, _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0], 0
        cv2.putText(frame, msg, (cx - tw//2, cy+1),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, msg, (cx - tw//2, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 140, 255), 2, cv2.LINE_AA)


def draw_eye_closure_bar(frame, ec, drowsy, warning):
    bx = 8
    by = frame.shape[0] - 20
    bw = 220
    bh = 10
    filled = int(bw * min(ec / max(EYE_CLOSED_FRAMES, 1), 1.0))
    color  = (0, 50, 255) if drowsy else (0, 140, 255) if warning else (0, 200, 120)
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (40, 40, 40), -1)
    if filled > 0:
        cv2.rectangle(frame, (bx, by), (bx+filled, by+bh), color, -1)
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (80, 80, 80), 1)
    cv2.putText(frame, f"DROWSINESS METER: {ec}/{EYE_CLOSED_FRAMES}",
                (bx+bw+8, by+9), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


# ============================================================
#  CORE DETECTION PIPELINE
# ============================================================

def process_frame(b64_image: str) -> dict:
    frame = decode_frame(b64_image)
    h, w  = frame.shape[:2]

    _state["frame_count"] += 1

    face_detected = False
    eyes_open     = False
    yawn_detected = False
    ear_avg       = 0.0
    mar           = 0.0

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = _face_mesh.process(rgb)

    if result.multi_face_landmarks:
        face_detected = True
        lm = result.multi_face_landmarks[0].landmark

        # EAR
        ear_l   = compute_ear(lm, LEFT_EYE,  w, h)
        ear_r   = compute_ear(lm, RIGHT_EYE, w, h)
        ear_avg = (ear_l + ear_r) / 2.0

        # Smooth EAR
        hist = _state["ear_history"]
        hist.append(ear_avg)
        if len(hist) > EAR_SMOOTH_N:
            hist.pop(0)
        ear_smooth = sum(hist) / len(hist)

        eyes_open = ear_smooth >= EAR_THRESH

        # MAR
        mar           = compute_mar(lm, w, h)
        yawn_detected = mar >= MAR_THRESH

        # Draw eye contours
        eye_color = (0, 220, 120) if eyes_open else (0, 60, 255)
        draw_eye_contour(frame, lm, LEFT_EYE,  w, h, eye_color)
        draw_eye_contour(frame, lm, RIGHT_EYE, w, h, eye_color)

        # EAR label near left eye corner
        lx = int(lm[LEFT_EYE[0]].x * w)
        ly = int(lm[LEFT_EYE[0]].y * h)
        cv2.putText(frame,
                    f"L:{ear_l:.2f} R:{ear_r:.2f} Avg:{ear_smooth:.2f}",
                    (max(lx - 60, 0), max(ly - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (0, 220, 120) if eyes_open else (0, 60, 255), 1, cv2.LINE_AA)

        if not eyes_open:
            ex = int(lm[RIGHT_EYE[3]].x * w)
            ey = int(lm[RIGHT_EYE[3]].y * h)
            cv2.putText(frame, "CLOSED", (ex + 4, ey),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 60, 255), 1, cv2.LINE_AA)

        # Draw mouth
        mouth_color = (0, 60, 255) if yawn_detected else (0, 200, 255)
        draw_mouth_contour(frame, lm, w, h, mouth_color)
        if yawn_detected:
            mx = int(lm[MOUTH_TOP].x * w)
            my = int(lm[MOUTH_TOP].y * h)
            cv2.putText(frame, f"YAWN MAR:{mar:.2f}",
                        (mx - 40, my - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1, cv2.LINE_AA)

        # Update eye counter
        if eyes_open:
            _state["eye_closed_counter"] = 0
        else:
            _state["eye_closed_counter"] += 1

    else:
        # No face — decay slowly
        _state["eye_closed_counter"] = max(0, _state["eye_closed_counter"] - 2)
        _state["ear_history"].clear()

    # Yawn counter
    if yawn_detected:
        _state["yawn_counter"] += 1
    else:
        _state["yawn_counter"] = max(0, _state["yawn_counter"] - 1)

    ec = _state["eye_closed_counter"]
    yc = _state["yawn_counter"]

    drowsy_alert  = ec >= EYE_CLOSED_FRAMES
    warning_alert = ec >= WARNING_FRAMES
    yawn_alert    = yc >= YAWN_FRAME_THRESH

    # Leading-edge event counting
    if drowsy_alert and _state["alert_level"] != "DANGER":
        _state["total_drowsy_events"] += 1
    if yc == YAWN_FRAME_THRESH:
        _state["total_yawn_events"] += 1

    alert_level = (
        "DANGER"  if drowsy_alert                 else
        "WARNING" if (warning_alert or yawn_alert) else
        "NORMAL"
    )
    _state["alert_level"] = alert_level

    # Draw overlays
    draw_hud(frame, alert_level, ear_avg, mar)
    draw_alert_overlay(frame, alert_level)
    draw_ear_bar(frame, ear_avg, not eyes_open and face_detected)
    if face_detected:
        draw_eye_closure_bar(frame, ec, drowsy_alert, warning_alert)

    return {
        "annotated_frame":     encode_frame(frame),
        "alert_level":         alert_level,
        "face_detected":       bool(face_detected),
        "eyes_detected":       bool(eyes_open),
        "eye_count":           2 if eyes_open else 0,
        "yawn_detected":       bool(yawn_detected),
        "drowsy_alert":        bool(drowsy_alert),
        "warning_alert":       bool(warning_alert),
        "yawn_alert":          bool(yawn_alert),
        "eye_closed_counter":  int(ec),
        "yawn_counter":        int(yc),
        "ear_avg":             round(float(ear_avg), 4),
        "mar":                 round(float(mar), 4),
        "total_drowsy_events": int(_state["total_drowsy_events"]),
        "total_yawn_events":   int(_state["total_yawn_events"]),
        "frame_count":         int(_state["frame_count"]),
        "timestamp":           datetime.now().isoformat(),
    }


# ============================================================
#  FLASK APP
# ============================================================
app = Flask(__name__, static_folder=".")

CORS(app, resources={r"/*": {
    "origins":       "*",
    "methods":       ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type"],
    "max_age":       600,
}})


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(".", "index.html")


@app.route("/process", methods=["POST", "OPTIONS"])
def process():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        data   = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "Invalid or missing JSON body"}), 400
        b64img = data.get("image", "").strip()
        if not b64img:
            return jsonify({"error": "No image provided"}), 400
        return jsonify(process_frame(b64img)), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 422
    except Exception:
        tb = traceback.format_exc()
        print("ERROR in /process:\n", tb)
        return jsonify({"error": tb}), 500


@app.route("/reset", methods=["POST", "OPTIONS"])
def reset():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    reset_state()
    return jsonify({"status": "ok", "message": "Session reset."}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":              "ok",
        "engine":              "mediapipe-face-mesh",
        "ear_threshold":       EAR_THRESH,
        "mar_threshold":       MAR_THRESH,
        "frame_count":         int(_state["frame_count"]),
        "alert_level":         _state["alert_level"],
        "eye_closed_counter":  int(_state["eye_closed_counter"]),
        "yawn_counter":        int(_state["yawn_counter"]),
        "total_drowsy_events": int(_state["total_drowsy_events"]),
        "total_yawn_events":   int(_state["total_yawn_events"]),
    }), 200


# ============================================================
#  STARTUP
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 65)
    print("  DrowsyGuard — EAR/MAR Driver Drowsiness Detection")
    print("=" * 65)
    print(f"  Engine   : MediaPipe Face Mesh (468 landmarks)")
    print(f"  Frontend : http://localhost:{port}/")
    print(f"  Health   : http://localhost:{port}/health")
    print(f"\n  Thresholds:")
    print(f"    EAR < {EAR_THRESH}   -> eyes considered CLOSED")
    print(f"    MAR > {MAR_THRESH}   -> mouth considered OPEN (yawn)")
    print(f"    WARNING  at {WARNING_FRAMES}  consecutive closed-eye frames")
    print(f"    DANGER   at {EYE_CLOSED_FRAMES} consecutive closed-eye frames")
    print(f"    YAWN     at {YAWN_FRAME_THRESH} consecutive open-mouth frames")
    print(f"\n  Install:  pip install mediapipe flask flask-cors numpy opencv-python")
    print("=" * 65 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)