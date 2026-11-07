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

## Better tracking

- The cleanup follows the jumper (not coaches or spectators), fixes left/right mix-ups, fills short gaps and smooths jitter.
- **High accuracy** uses a larger pose model at 60 frames a second.
- **Clip speed** handles phone slow-motion clips, so times and speeds stay correct.
- **Fix points**: drag any wrong joint to the right spot and re-analyze.
- **Export labeled frames** saves frames and their (fixed) joint positions as JSON. Collected across many jumps, this is the training data a high-jump-specific pose model would need.

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
- `tracking.js`: cleanup of the raw pose output.
- `app.js`: video, pose model, drawing and report.
- `npm test`: runs the tests on synthetic jumps (`test/`).
- Serve the folder with any static server, for example `python3 -m http.server`.
