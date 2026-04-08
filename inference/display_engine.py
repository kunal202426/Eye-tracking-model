"""inference/display_engine.py
--------------------------------------------------------------------------------
Fullscreen display engine for the real-time eye tracking pipeline.

Upgrade 5 -- confidence & uncertainty display:
  - Full softmax probability mini-bars for all classes per classifier.
  - Prediction text coloured by confidence level.
  - Stability indicator circle (green/yellow/red) per classifier.
  - Eye health metrics row (EAR L/R, blink rate).

Upgrade 7 -- visual quality:
  - Subtle dark grid background.
  - Fixation heatmap that decays exponentially over time.
  - Mini prediction timeline at the bottom (last 10 s, 1/3 s bins).
  - Gaze trail comet effect: radius varies 4->15 px, opacity fades.
  - Text background rectangles on all HUD text.
"""

from __future__ import annotations

import argparse
import collections
import logging
import sys
import time
from pathlib import Path
from typing import Deque, Optional

import cv2
import numpy as np

log = logging.getLogger("display_engine")

# ── Window ────────────────────────────────────────────────────────────────────
DEFAULT_WINDOW_NAME = "Eye Tracker"

# ── Canvas ────────────────────────────────────────────────────────────────────
CANVAS_BG_COLOR = (15, 15, 15)

# ── Grid ──────────────────────────────────────────────────────────────────────
GRID_SPACING = 100
GRID_COLOR   = (35, 35, 35)

# ── HUD layout ────────────────────────────────────────────────────────────────
HUD_X         = 14
HUD_Y_START   = 38
HUD_LINE_H    = 22     # small row height for class bars
HUD_SECTION_H = 88     # total height per classifier section
HUD_BAR_W     = 120    # full bar width
HUD_BAR_H     = 10
HUD_PAD       = 10
HUD_BG_ALPHA  = 0.60
HUD_BG_COLOR  = (18, 18, 18)
HUD_PANEL_W   = 320

# ── Typography ────────────────────────────────────────────────────────────────
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FS_SM      = 0.44
FS_MD      = 0.58
FS_LG      = 0.65
FT_THIN    = 1
FT_THICK   = 2

# ── Colours ───────────────────────────────────────────────────────────────────
C_WHITE   = (255, 255, 255)
C_GRAY    = (130, 130, 130)
C_DIM     = (80,  80,  80)
C_GREEN   = (0,   210,  40)
C_YELLOW  = (0,   210, 210)
C_ORANGE  = (0,   140, 255)
C_RED     = (50,   50, 220)
C_CYAN    = (200, 200,   0)
C_BLUE    = (210,  90,  30)
C_LIME    = (30,  220, 120)

ATTENTION_COLORS: dict[str, tuple[int, int, int]] = {
    "focused":    C_GREEN,
    "distracted": C_YELLOW,
    "off_task":   C_RED,
}
COGLOAD_COLORS: dict[int, tuple[int, int, int]] = {
    0: C_GREEN,
    1: C_ORANGE,
    2: C_RED,
}
EMOTION_COLORS: dict[int, tuple[int, int, int]] = {
    0: C_BLUE,
    1: C_CYAN,
    2: C_RED,
    3: C_LIME,
}

EMOTION_NAMES   = ["sad", "calm", "angry", "happy"]
ATTENTION_NAMES = ["focused", "distracted", "off_task"]
COGLOAD_NAMES   = ["low", "medium", "high"]

# ── Confidence colour gating ──────────────────────────────────────────────────
CONF_HIGH = 0.70
CONF_LOW  = 0.45

# ── Gaze dot + trail ──────────────────────────────────────────────────────────
GAZE_DOT_MAX_R   = 15
GAZE_DOT_MIN_R   = 4
GAZE_DOT_RING_R  = GAZE_DOT_MAX_R + 4
DEFAULT_TRAIL_LEN = 30

# ── Heatmap ───────────────────────────────────────────────────────────────────
HEATMAP_SCALE   = 4       # heatmap kept at 1/4 screen resolution
HEATMAP_DECAY   = 0.995
HEATMAP_SIGMA   = 10      # Gaussian sigma in downsampled coords (~40 px actual)
HEATMAP_ALPHA   = 0.30
HEATMAP_GAUSS_R = 30      # kernel radius in downsampled coords

# ── Timeline ──────────────────────────────────────────────────────────────────
TIMELINE_SECS       = 10.0
TIMELINE_BIN_FRAMES = 10   # frames per colour block (~1/3 s at 30fps)
TIMELINE_H          = 18   # height in pixels of the timeline strip
TIMELINE_Y_OFFSET   = 32   # distance from bottom edge

# ── No-face alert ─────────────────────────────────────────────────────────────
NO_FACE_COLOR = (50, 50, 220)
HEAD_TURN_COLOR = (0, 140, 255)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _letterbox(
    frame: np.ndarray, target_w: int, target_h: int,
) -> tuple[np.ndarray, int, int, float]:
    fh, fw = frame.shape[:2]
    scale  = min(target_w / fw, target_h / fh)
    new_w  = int(fw * scale)
    new_h  = int(fh * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x_off  = (target_w - new_w) // 2
    y_off  = (target_h - new_h) // 2
    return resized, x_off, y_off, scale


def _dim_color(c: tuple[int, int, int], factor: float = 0.5) -> tuple[int, int, int]:
    return (int(c[0] * factor), int(c[1] * factor), int(c[2] * factor))


def _conf_color(
    conf: Optional[float],
    base_color: tuple[int, int, int],
) -> tuple[int, int, int]:
    if conf is None:
        return C_GRAY
    if conf >= CONF_HIGH:
        return base_color
    if conf >= CONF_LOW:
        return _dim_color(base_color, 0.65)
    return C_GRAY


def _text_bg(
    canvas: np.ndarray,
    text: str,
    x: int, y: int,
    scale: float, thick: int,
) -> None:
    """Draw a dark rectangle behind text."""
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, thick)
    pad = 3
    cv2.rectangle(
        canvas,
        (x - pad, y - th - pad),
        (x + tw + pad, y + bl + pad),
        (10, 10, 10), -1,
    )


def _put_text_bg(
    canvas: np.ndarray,
    text: str,
    x: int, y: int,
    scale: float, color: tuple, thick: int = FT_THIN,
) -> None:
    """Draw text with a dark background rectangle."""
    _text_bg(canvas, text, x, y, scale, thick)
    cv2.putText(canvas, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def _build_gauss_kernel(r: int, sigma: float) -> np.ndarray:
    """Build a 2*r+1 square Gaussian kernel."""
    size = 2 * r + 1
    k    = cv2.getGaussianKernel(size, sigma)
    return (k @ k.T).astype(np.float32)


_GAUSS_KERNEL: Optional[np.ndarray] = None


def _add_gaze_to_heatmap(
    heatmap: np.ndarray,
    gx: int, gy: int,
    r: int = HEATMAP_GAUSS_R,
    sigma: float = float(HEATMAP_SIGMA),
) -> None:
    """Add a Gaussian blob to heatmap at (gx, gy)."""
    global _GAUSS_KERNEL
    if _GAUSS_KERNEL is None:
        _GAUSS_KERNEL = _build_gauss_kernel(r, sigma)

    H, W  = heatmap.shape
    x1, x2 = gx - r, gx + r + 1
    y1, y2 = gy - r, gy + r + 1

    kx1 = max(0, -x1)
    ky1 = max(0, -y1)
    kx2 = _GAUSS_KERNEL.shape[1] - max(0, x2 - W)
    ky2 = _GAUSS_KERNEL.shape[0] - max(0, y2 - H)

    x1 = max(0, x1); x2 = min(W, x2)
    y1 = max(0, y1); y2 = min(H, y2)

    if x2 > x1 and y2 > y1 and kx2 > kx1 and ky2 > ky1:
        heatmap[y1:y2, x1:x2] += _GAUSS_KERNEL[ky1:ky2, kx1:kx2] * 255.0


# ─────────────────────────────────────────────────────────────────────────────
# DisplayEngine
# ─────────────────────────────────────────────────────────────────────────────

class DisplayEngine:
    """Fullscreen rendering engine with all Upgrade 5 + 7 features."""

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        window_name: str = DEFAULT_WINDOW_NAME,
        trail_length: int = DEFAULT_TRAIL_LEN,
    ) -> None:
        self._W  = screen_w
        self._H  = screen_h
        self._wn = window_name

        # Gaze trail
        self._trail: Deque[tuple[int, int]] = collections.deque(maxlen=trail_length)

        # Heatmap (1/4 resolution)
        hw = screen_w  // HEATMAP_SCALE
        hh = screen_h  // HEATMAP_SCALE
        self._heatmap = np.zeros((hh, hw), dtype=np.float32)

        # Timeline buffer: deque of (attention_name, cogload_name)
        # Each entry = one frame; we bin them in groups of TIMELINE_BIN_FRAMES
        max_tl = int(TIMELINE_SECS * 30) + 60
        self._timeline: Deque[tuple[Optional[str], Optional[str]]] = (
            collections.deque(maxlen=max_tl)
        )

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN,
        )
        log.info(
            "DisplayEngine ready (%dx%d, trail=%d, window='%s')",
            screen_w, screen_h, trail_length, window_name,
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def render(
        self,
        annotated_frame: np.ndarray,
        preds,                          # ModelPredictions
        gaze_screen_xy: Optional[tuple[int, int]],
        fps: float,
        face_detected: bool = True,
        # Upgrade 5 new params
        left_ear: Optional[float] = None,
        right_ear: Optional[float] = None,
        blink_rate: Optional[float] = None,
        head_turned: bool = False,
        debug_mode: bool = False,
    ) -> np.ndarray:
        """Compose the full-screen display frame.

        Args:
            annotated_frame:  BGR frame with eye overlay.
            preds:            ModelPredictions from model_runner.
            gaze_screen_xy:   Calibrated screen gaze position (Kalman-smoothed).
            fps:              Current FPS estimate.
            face_detected:    Whether a face was found.
            left_ear:         Left EAR value for health metrics.
            right_ear:        Right EAR value.
            blink_rate:       Blinks/min for health metrics.
            head_turned:      Show head-turn warning if True.
            debug_mode:       Show raw logits / Kalman state panel.

        Returns:
            Composed BGR canvas at (screen_h, screen_w, 3).
        """
        # 1. Dark grid canvas
        canvas = self._make_grid_canvas()

        # 2. Letterbox webcam frame
        lb, x_off, y_off, _ = _letterbox(annotated_frame, self._W, self._H)
        lh, lw = lb.shape[:2]
        canvas[y_off:y_off + lh, x_off:x_off + lw] = lb

        # 3. Heatmap overlay (Upgrade 7)
        self._heatmap *= HEATMAP_DECAY
        if gaze_screen_xy is not None:
            hx = gaze_screen_xy[0] // HEATMAP_SCALE
            hy = gaze_screen_xy[1] // HEATMAP_SCALE
            _add_gaze_to_heatmap(self._heatmap, hx, hy)
        self._draw_heatmap(canvas)

        # 4. HUD panel (Upgrade 5)
        self._draw_hud(canvas, preds, left_ear, right_ear, blink_rate)

        # 5. FPS (top-right)
        fps_text = f"FPS: {fps:.0f}"
        (tw, _), _ = cv2.getTextSize(fps_text, FONT, FS_MD, FT_THIN)
        _put_text_bg(canvas, fps_text,
                     self._W - tw - 18, 28, FS_MD, C_GRAY)

        # 6. Status alerts
        if not face_detected:
            _put_text_bg(canvas, "No face detected",
                         self._W // 2 - 100, self._H - TIMELINE_Y_OFFSET - 30,
                         FS_MD, NO_FACE_COLOR, FT_THICK)
        if head_turned:
            _put_text_bg(canvas, "Head turned",
                         self._W // 2 - 70, self._H - TIMELINE_Y_OFFSET - 54,
                         FS_MD, HEAD_TURN_COLOR, FT_THICK)

        # 7. Gaze trail + dot (Upgrade 7 comet)
        attn_stable = getattr(preds, "attention_stable_name", None)
        if gaze_screen_xy is not None:
            self._trail.append(gaze_screen_xy)
        self._draw_gaze_comet(canvas, attn_stable)

        # 8. Timeline (Upgrade 7)
        self._timeline.append((attn_stable, getattr(preds, "cogload_stable_name", None)))
        self._draw_timeline(canvas)

        # 9. Debug overlay
        if debug_mode:
            self._draw_debug(canvas, preds)

        return canvas

    def show(self, canvas: np.ndarray) -> int:
        cv2.imshow(self._wn, canvas)
        return int(cv2.waitKey(1) & 0xFF)

    def is_open(self) -> bool:
        try:
            return cv2.getWindowProperty(self._wn, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def close(self) -> None:
        self._trail.clear()
        try:
            cv2.destroyWindow(self._wn)
        except cv2.error:
            pass
        log.info("DisplayEngine closed.")

    # ── Private: canvas helpers ───────────────────────────────────────────────

    def _make_grid_canvas(self) -> np.ndarray:
        """Create dark BGR canvas with subtle grid lines (Upgrade 7)."""
        canvas = np.full((self._H, self._W, 3), CANVAS_BG_COLOR, dtype=np.uint8)
        for x in range(0, self._W, GRID_SPACING):
            cv2.line(canvas, (x, 0), (x, self._H), GRID_COLOR, 1)
        for y in range(0, self._H, GRID_SPACING):
            cv2.line(canvas, (0, y), (self._W, y), GRID_COLOR, 1)
        return canvas

    # ── Private: heatmap (Upgrade 7) ─────────────────────────────────────────

    def _draw_heatmap(self, canvas: np.ndarray) -> None:
        """Render the decaying fixation heatmap as a semi-transparent overlay."""
        hm = self._heatmap
        max_val = hm.max()
        if max_val < 1.0:
            return

        norm   = np.clip(hm / max_val * 255.0, 0, 255).astype(np.uint8)
        colour = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        # Zero regions -> fully transparent (avoid blue tint on empty areas)
        mask    = (norm > 10)
        big     = cv2.resize(colour, (self._W, self._H), interpolation=cv2.INTER_LINEAR)
        big_msk = cv2.resize(mask.astype(np.uint8) * 255, (self._W, self._H),
                             interpolation=cv2.INTER_NEAREST)
        alpha_mask = (big_msk > 0)

        blended = cv2.addWeighted(big, HEATMAP_ALPHA, canvas, 1.0, 0)
        canvas[alpha_mask] = blended[alpha_mask]

    # ── Private: HUD (Upgrade 5) ──────────────────────────────────────────────

    def _draw_hud(
        self,
        canvas: np.ndarray,
        preds,
        left_ear: Optional[float],
        right_ear: Optional[float],
        blink_rate: Optional[float],
    ) -> None:
        """Draw semi-transparent HUD with full probability bars (Upgrade 5)."""
        # Build section list
        sections = [
            ("EMOTION",   EMOTION_NAMES,   EMOTION_COLORS,
             getattr(preds, "emotion_probs",        None),
             getattr(preds, "emotion_stable_name",  None),
             getattr(preds, "emotion_stable_conf",  None)),
            ("ATTENTION", ATTENTION_NAMES,  {i: ATTENTION_COLORS.get(n, C_WHITE) for i, n in enumerate(ATTENTION_NAMES)},
             getattr(preds, "attention_probs",       None),
             getattr(preds, "attention_stable_name", None),
             getattr(preds, "attention_stable_conf", None)),
            ("LOAD",      COGLOAD_NAMES,   COGLOAD_COLORS,
             getattr(preds, "cogload_probs",         None),
             getattr(preds, "cogload_stable_name",   None),
             getattr(preds, "cogload_stable_conf",   None)),
        ]

        # Metrics row height
        metrics_h = 28

        # Total panel height
        n_sections = len(sections)
        panel_h    = n_sections * HUD_SECTION_H + metrics_h + 2 * HUD_PAD
        x0         = HUD_X - HUD_PAD
        y0         = HUD_Y_START - 20

        # Semi-transparent bg
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0),
                      (x0 + HUD_PANEL_W, y0 + panel_h), HUD_BG_COLOR, -1)
        cv2.addWeighted(overlay, HUD_BG_ALPHA, canvas, 1 - HUD_BG_ALPHA, 0, canvas)

        y_cursor = HUD_Y_START

        for (section_label, class_names, class_colors,
             probs, stable_name, stable_conf) in sections:

            # Section header
            cv2.putText(canvas, section_label, (HUD_X, y_cursor),
                        FONT, FS_SM, C_GRAY, FT_THIN, cv2.LINE_AA)

            # Stability indicator circle (Upgrade 5)
            indicator_col = C_RED
            if stable_conf is not None:
                if stable_conf >= CONF_HIGH:
                    indicator_col = C_GREEN
                elif stable_conf >= CONF_LOW:
                    indicator_col = C_YELLOW
            cx_ind = HUD_X + 130
            cy_ind = y_cursor - 6
            cv2.circle(canvas, (cx_ind, cy_ind), 5, indicator_col, -1)

            # Stable prediction label
            if stable_name in (None, "uncertain") or (stable_conf is not None and stable_conf < CONF_LOW):
                disp_text = "LOW CONF"
                text_col  = C_GRAY
            else:
                disp_text = stable_name
                text_col  = _conf_color(stable_conf,
                                        class_colors.get(
                                            class_names.index(stable_name)
                                            if stable_name in class_names else 0,
                                            C_WHITE,
                                        ))
            conf_str = f"{stable_conf * 100:.0f}%" if stable_conf is not None else " ?%"
            cv2.putText(canvas,
                        f"{disp_text:<12}{conf_str}",
                        (HUD_X + 150, y_cursor),
                        FONT, FS_MD, text_col, FT_THIN, cv2.LINE_AA)

            y_cursor += HUD_LINE_H

            # Per-class mini bars (Upgrade 5)
            for ci, cname in enumerate(class_names):
                p     = float(probs[ci]) if probs is not None else 0.0
                col   = class_colors.get(ci, C_WHITE)
                bar_w = max(1, int(p * HUD_BAR_W))

                # Dim bar background
                cv2.rectangle(canvas,
                              (HUD_X, y_cursor - HUD_BAR_H + 2),
                              (HUD_X + HUD_BAR_W, y_cursor + 2),
                              C_DIM, 1)
                # Filled bar
                if bar_w > 1:
                    cv2.rectangle(canvas,
                                  (HUD_X, y_cursor - HUD_BAR_H + 2),
                                  (HUD_X + bar_w, y_cursor + 2),
                                  col, -1)
                # Class label
                pct_text = f"{cname[:8]:<8} {p * 100:4.0f}%"
                cv2.putText(canvas, pct_text,
                            (HUD_X + HUD_BAR_W + 6, y_cursor),
                            FONT, FS_SM, _dim_color(col, 0.9), FT_THIN, cv2.LINE_AA)
                y_cursor += HUD_LINE_H

            y_cursor += 6  # section gap

        # Eye health metrics row (Upgrade 5)
        ear_l = f"EAR L:{left_ear:.2f}" if left_ear is not None else "EAR L:---"
        ear_r = f"R:{right_ear:.2f}"     if right_ear is not None else "R:---"
        blk   = f"Blink:{blink_rate:.0f}/min" if blink_rate is not None else "Blink:---"
        health_text = f"{ear_l}  {ear_r}  {blk}"
        cv2.putText(canvas, health_text,
                    (HUD_X, y_cursor + 16),
                    FONT, FS_SM, C_CYAN, FT_THIN, cv2.LINE_AA)

    # ── Private: gaze comet trail (Upgrade 7) ────────────────────────────────

    def _draw_gaze_comet(
        self,
        canvas: np.ndarray,
        attention_name: Optional[str],
    ) -> None:
        """Draw gaze trail as a comet (size + opacity both fade)."""
        trail_list = list(self._trail)
        n = len(trail_list)
        if n == 0:
            return

        dot_color = ATTENTION_COLORS.get(attention_name, C_CYAN) if attention_name else C_CYAN

        for j, (tx, ty) in enumerate(trail_list):
            frac   = (j + 1) / n
            # Radius: linear from min to max
            radius = max(GAZE_DOT_MIN_R,
                         int(GAZE_DOT_MIN_R + (GAZE_DOT_MAX_R - GAZE_DOT_MIN_R) * frac))
            # Opacity: linear from 0.08 (oldest) to 1.0 (newest)
            alpha  = 0.08 + 0.92 * frac

            if j < n - 1:
                overlay = canvas.copy()
                cv2.circle(overlay, (tx, ty), radius, dot_color, -1)
                cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)
            else:
                # Current dot: solid + white ring
                cv2.circle(canvas, (tx, ty), GAZE_DOT_RING_R, C_WHITE, 2)
                cv2.circle(canvas, (tx, ty), GAZE_DOT_MAX_R, dot_color, -1)

    # ── Private: timeline (Upgrade 7) ────────────────────────────────────────

    def _draw_timeline(self, canvas: np.ndarray) -> None:
        """Draw mini prediction timeline strip at the bottom of the canvas."""
        tl = list(self._timeline)
        if not tl:
            return

        # Bin frames into groups of TIMELINE_BIN_FRAMES
        bins: list[tuple[Optional[str], Optional[str]]] = []
        for i in range(0, len(tl), TIMELINE_BIN_FRAMES):
            chunk = tl[i:i + TIMELINE_BIN_FRAMES]
            # Most frequent attention and cogload in chunk
            attn_counts: dict[Optional[str], int] = {}
            cog_counts:  dict[Optional[str], int] = {}
            for a, c in chunk:
                attn_counts[a] = attn_counts.get(a, 0) + 1
                cog_counts[c]  = cog_counts.get(c, 0) + 1
            best_a = max(attn_counts, key=lambda k: attn_counts[k])
            best_c = max(cog_counts,  key=lambda k: cog_counts[k])
            bins.append((best_a, best_c))

        n_bins  = len(bins)
        bin_w   = max(1, self._W // max(n_bins, 1))
        y_base  = self._H - TIMELINE_Y_OFFSET
        half_h  = TIMELINE_H // 2

        for i, (attn, cog) in enumerate(bins):
            x1 = i * bin_w
            x2 = x1 + bin_w - 1

            # Top half = attention colour
            a_col = ATTENTION_COLORS.get(attn, C_DIM) if attn else C_DIM
            cv2.rectangle(canvas, (x1, y_base - TIMELINE_H),
                          (x2, y_base - half_h), a_col, -1)

            # Bottom half = cogload colour
            cog_idx = COGLOAD_NAMES.index(cog) if cog in COGLOAD_NAMES else -1
            c_col = COGLOAD_COLORS.get(cog_idx, C_DIM)
            cv2.rectangle(canvas, (x1, y_base - half_h),
                          (x2, y_base), c_col, -1)

        # Border
        cv2.rectangle(canvas, (0, y_base - TIMELINE_H),
                      (self._W - 1, y_base), C_GRAY, 1)

    # ── Private: debug overlay ─────────────────────────────────────────────────

    def _draw_debug(self, canvas: np.ndarray, preds) -> None:
        """Small debug panel showing raw logits and Kalman state."""
        lines = []
        if getattr(preds, "emotion_probs", None) is not None:
            p = preds.emotion_probs
            lines.append(f"emo_probs: {' '.join(f'{x:.2f}' for x in p)}")
        if getattr(preds, "attention_probs", None) is not None:
            p = preds.attention_probs
            lines.append(f"att_probs: {' '.join(f'{x:.2f}' for x in p)}")
        if getattr(preds, "cogload_probs", None) is not None:
            p = preds.cogload_probs
            lines.append(f"cog_probs: {' '.join(f'{x:.2f}' for x in p)}")
        if getattr(preds, "kalman_state", None) is not None:
            k = preds.kalman_state
            lines.append(f"kalman: cx={k[0]:.3f} cy={k[1]:.3f} vx={k[2]:.4f} vy={k[3]:.4f}")
        if getattr(preds, "gaze_cx", None) is not None:
            lines.append(f"gaze_raw: ({preds.gaze_cx:.3f},{preds.gaze_cy:.3f})  "
                         f"smooth: ({preds.gaze_cx_smooth:.3f},{preds.gaze_cy_smooth:.3f})")

        x_dbg = self._W - 420
        y_dbg = 50
        panel_h = len(lines) * 18 + 10
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x_dbg - 4, y_dbg - 14),
                      (x_dbg + 416, y_dbg + panel_h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        for i, ln in enumerate(lines):
            cv2.putText(canvas, ln, (x_dbg, y_dbg + i * 18),
                        FONT, FS_SM, C_LIME, FT_THIN, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# CLI (synthetic rendering test)
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic rendering smoke test. Displays random-noise frame "
            "with dummy predictions for 5 seconds then exits."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path", type=Path, required=True,
                        help="Project root directory.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run synthetic rendering smoke test."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    import dataclasses as _dc

    _root = Path(__file__).parent.parent
    sys.path.insert(0, str(_root))
    from calibration import get_screen_resolution

    parser    = _build_arg_parser()
    args      = parser.parse_args(argv)
    data_path = args.data_path.resolve()

    if not data_path.exists():
        log.error("data_path does not exist: %s", data_path)
        sys.exit(1)

    screen_w, screen_h = get_screen_resolution()
    log.info("Screen: %dx%d", screen_w, screen_h)

    @_dc.dataclass
    class _FakePreds:
        emotion_stable_name:    str   = "calm"
        emotion_stable_conf:    float = 0.72
        emotion_probs:          object = _dc.field(default_factory=lambda: np.array([0.05,0.72,0.10,0.13]))
        attention_stable_name:  str   = "focused"
        attention_stable_conf:  float = 0.85
        attention_probs:        object = _dc.field(default_factory=lambda: np.array([0.85,0.10,0.05]))
        cogload_stable_name:    str   = "medium"
        cogload_stable_conf:    float = 0.60
        cogload_probs:          object = _dc.field(default_factory=lambda: np.array([0.20,0.60,0.20]))
        gaze_cx:                float = 0.5
        gaze_cy:                float = 0.5
        gaze_cx_smooth:         float = 0.5
        gaze_cy_smooth:         float = 0.5
        kalman_state:           object = _dc.field(default_factory=lambda: np.array([0.5,0.5,0.0,0.0]))

    engine = DisplayEngine(screen_w, screen_h)
    preds  = _FakePreds()
    dummy  = np.random.randint(50, 180, (480, 640, 3), dtype=np.uint8)
    n = 0
    t_start = time.monotonic()

    log.info("Synthetic render test (5 s). Press Q to exit early.")
    try:
        while time.monotonic() - t_start < 5.0:
            gaze = (
                int(screen_w * 0.5 + 120 * np.sin(n * 0.08)),
                int(screen_h * 0.5 + 80  * np.cos(n * 0.06)),
            )
            canvas = engine.render(
                dummy, preds,
                gaze_screen_xy = gaze,
                fps            = 30.0,
                face_detected  = True,
                left_ear       = 0.30 + 0.05 * np.sin(n * 0.2),
                right_ear      = 0.31 + 0.04 * np.cos(n * 0.2),
                blink_rate     = 14.0,
                debug_mode     = True,
            )
            key = engine.show(canvas)
            n  += 1
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        engine.close()

    log.info("Rendered %d synthetic frames. SMOKE TEST PASSED.", n)


if __name__ == "__main__":
    main()
