# High Jump Analyzer

Upload a video of one high jump and get:

- **An annotated video**: a skeleton with anchor points at the shoulders, hips, knees and ankles, with live knee and hip angles and your lean into the curve.
- **Key moments**: curve, plant, takeoff, peak and landing frames.
- **Your numbers against your targets**: approach speed, curve lean, takeoff knee and hip angles, takeoff angle, hip rise and back angle at the peak. Targets adjust for your height and gender.
- **Vertical jump**: how far your hips (center of mass) rose after takeoff, cross-checked against time in the air.
- **Bar clearance** (optional): enter the bar height to check whether your head, shoulders, hips, knees and feet each cleared the bar, with a verdict (cleared, very close, or likely hit and with what). Tap the bar ends and the floor on the video for an exact bar line and scale.
- **A 0-100 jump rating** with a score for each phase (approach, curve lean, takeoff, bar clearance, landing) and tips on what to work on.

**Try it:** https://zeyung.github.io/high-jump-analyzer/

Everything runs in your browser using [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker). Your video is never uploaded.

## How tracking works

1. **Find everyone.** The pose model looks at every frame on its own (so a wrong lock-on cannot carry over) and reports every person it sees.
2. **Pick the jumper.** Detections are linked into tracks, and the track that actually runs and jumps wins. People the model imagines in the landing mat, crowd or equipment barely move, so they lose. If it still picks wrong, tap **Wrong person? Pick the jumper**.
3. **Follow up close.** The model runs again on a zoomed crop that follows the jumper frame by frame, giving it several times more pixels of the athlete.
4. **Clean up.** Left/right mix-ups are fixed, short gaps filled, and jitter smoothed.
5. **Key moments.** Plant and takeoff come from when the takeoff foot lands and leaves the ground; peak from the highest hip point; landing from when the fall stops. Any of them can be set by hand in **Fix points**.

## The bar

Enter the bar height and the app finds the bar in the video on its own (a thin straight line that stays put while the jumper moves), checking it against the athlete's size so it does not pick a fence or the mat edge. That bar is taken to be exactly the height you entered, which sets the scale for the vertical jump, hip height and clearance. Use **Adjust the bar** to drag its ends if the line is off.

## On the video

Each joint has its own color: a solid dot for the left side and a ring for the right, so overlapping joints stay visible. A legend next to the video names every point and lets you show or hide each one, plus the skeleton, lean line, bar, clearance check and phase label. Joint angles, lean and hip height for the current frame are shown beside the video instead of on it.

## Training data

**Fix points** lets you drag any wrong joint to the right spot. **Export labeled frames** saves frames with their joint positions as JSON. Collected over many jumps, that is the training data a high-jump-specific pose model would need.

## Filming tips

- Film from the takeoff side with the bar running across the picture.
- Keep both standards and the floor under them in frame so you can mark the bar.
- Keep the whole body in frame from the last 4-5 steps through the landing.
- Hold the camera still and do not zoom.
- One jump per clip; up to 15 seconds is analyzed from the start time you pick.

## Accuracy

These are estimates from a single 2D video, not lab measurements. Your height sets the pixel-to-centimetre scale. Target ranges are rounded from published elite high jump research and widened for non-elite jumpers:

- Dapena, J. (1980). Mechanics of translation in the Fosbury-flop. *Medicine and Science in Sports and Exercise*, 12(1), 37-44.
- Dapena, J. (1980). Mechanics of rotation in the Fosbury-flop. *Medicine and Science in Sports and Exercise*, 12(1), 45-53.
- Isolehto, J., Virmavirta, M., Kyröläinen, H., & Komi, P. V. (2007). Biomechanical analysis of the high jump at the 2005 IAAF World Championships. *New Studies in Athletics*, 22(2), 17-27.

## Development

- `analysis.js`: the jump math (no browser needed).
- `tracking.js`: picking the jumper and cleaning up the pose output.
- `bar.js`: finding the bar in the video.
- `app.js`: video, pose model, drawing and report.
- `npm test`: runs the tests on synthetic jumps (`test/`).
- Serve the folder with any static server, for example `python3 -m http.server`.
