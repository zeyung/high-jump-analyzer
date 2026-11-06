# High Jump Analyzer

Upload a video of one high jump and get:

- **An annotated video**: a skeleton with anchor points at the shoulders, hips, knees and ankles, with live knee and hip angles and your lean into the curve.
- **Key moments**: curve, plant, takeoff, peak and landing frames.
- **Your numbers against your targets**: approach speed, curve lean, takeoff knee and hip angles, takeoff angle, hip rise and back angle at the peak. Targets adjust for your height and gender.
- **A 0-100 jump rating** with a score for each phase (approach, curve lean, takeoff, bar clearance, landing) and tips on what to work on.

**Try it:** https://zeyung.github.io/high-jump-analyzer/

Everything runs in your browser using [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker). Your video is never uploaded.

## Filming tips

- Film from the side, roughly in line with the bar.
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
- `app.js`: video, pose model, drawing and report.
- `npm test`: runs the tests on synthetic jumps (`test/`).
- Serve the folder with any static server, for example `python3 -m http.server`.
