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

## How it works

1. **Fill in the form**: video, athlete height, bar height, gender, takeoff foot.
2. **Pick the jumper and the bar** (the pose model loads meanwhile): tap the jumper; optionally tap them again on a frame over the bar; tap the two ends of the bar, then the floor under it.
3. **Tracking** follows only the person you tapped, with a zoomed crop on every frame. Candidates are scored on distance, body size and confidence, so a coach or athlete standing nearby cannot take over, and the crop is also tried sideways and upside down for the flight over the bar. With two taps it also tracks back from the second tap and joins the runs where they agree.
4. **Camera pans** are measured from the background and removed, so a handheld, panning phone still works.
5. **Key moments**: plant and takeoff from the takeoff foot landing and leaving the ground; the peak from the hips' height above the jumper's own feet; landing from when the fall stops. Any of them can be set by hand.
6. **Heights**: the bar is taken to be exactly the height entered, so it measures the peak hip height and clearance at the depth where the jumper crosses. Without a marked bar, heights are approximate.

## Coaching

The report lists what went well, what to fix first (ranked by score and phase weight), drills for the weakest phase, approach and technique fundamentals (J-curve, last-step rhythm, plant, drive, bar clearance) and a practice checklist (consistent filming, measured marks, logging jumps, one change at a time). It also measures takeoff contact time and the rhythm and length of the last steps.

## On the video

Each joint has its own color: a solid dot for the left side and a ring for the right, so overlapping joints stay visible. A legend next to the video names every point and lets you show or hide each one, plus the skeleton, lean line, bar, clearance check and phase label. Joint angles, lean and hip height for the current frame are shown beside the video instead of on it.

## Training data

**Fix points** lets you drag any wrong joint to the right spot. **Export labeled frames** saves frames with their joint positions as JSON. Collected over many jumps, that is the training data a high-jump-specific pose model would need.

## Filming tips

- Film the takeoff side-on, with the bar running across the picture. Takeoff angle, approach speed and lean are only accurate side-on.
- Keep a bar end and the floor under it in frame so you can mark the bar.
- A tripod is best; a smooth pan to follow the jumper also works.
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
- `bar.js`: finding line candidates for the bar (kept for experiments).
- `motion.js`: measuring and removing camera pans.
- `coaching.js`: feedback, drills, fundamentals and the practice checklist.
- `app.js`: video, pose model, drawing and report.
- `npm test`: runs the tests on synthetic jumps (`test/`).
- Serve the folder with any static server, for example `python3 -m http.server`.
