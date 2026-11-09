# Task Progress

Progress log for the Violin UI tasks in `violin_ui_code_review_and_next_tasks_updated.md`. Newest entries go at the bottom.

---

## Task 3F — Preprocessing and inverse coordinate transform for any orientation

**Branch:** `task-3f-preprocessing` (created from `Violin_UI` at `451746e`, the Task 3C commit; fast-forwarded to `d3650d8` on 2026-11-09)
**Date:** 2026-10-24
**Status:** ✅ Implemented and merged into local `Violin_UI` (see *Update: Integration into Violin_UI* below). ⏳ On-device verification is still pending.
**Committed:** Yes (2026-11-09). Not pushed.

### Summary of completed work

- Added a standalone `FramePreprocessor` that works out where a camera frame goes inside the model input tensor. It supports two modes:
  - `LEGACY_STRETCH` (the default) reproduces the current `try.tflite` placement.
  - `LETTERBOX` scales the frame evenly, centers it and pads the short side.
- It provides the forward (source pixel → tensor pixel) and inverse (tensor pixel → source pixel) coordinate transforms.
- It takes the model input size as a parameter, so it is not tied to 640×640.
- Added 16 JVM unit tests, covering portrait, landscape and square frames and the 1 px round-trip acceptance criterion.
- The live inference path is unchanged. `Detector.kt` still uses its own `PREPROCESS_*` constants.

### Files created or modified

| File | Status | Purpose |
|---|---|---|
| `modules/camerax/android/src/main/java/expo/modules/camerax/FramePreprocessor.kt` | **new** | `FramePreprocessor`, `ResizeMode`, `PreprocessPlan` |
| `modules/camerax/android/src/test/java/expo/modules/camerax/FramePreprocessorTest.kt` | **new** | 16 geometry and round-trip tests |
| `task_progress.md` | **new** | This log |

These are **not** modified: `Detector.kt`, `ModelOutputParser.kt` (`LetterboxInfo`), `LegacyObbParser.kt`, `CameraxView.kt`, `OverlayView.kt`, and the detection thresholds, angle semantics and classification logic.

### Implementation details and design decisions

**API**

```kotlin
val plan = FramePreprocessor(ResizeMode.LETTERBOX).plan(sourceW, sourceH, tensorW, tensorH)
plan.contentWidth / contentHeight / padX / padY   // integer placement used to draw the bitmap
plan.letterbox                                    // existing LetterboxInfo, handed to parsers
plan.sourceToTensorX/Y(...)                       // forward
plan.tensorToSourceX/Y(...)                       // inverse
```

**Modes**
- **`LEGACY_STRETCH` (default).** Content height equals the tensor height, and content width is `round(tensorHeight · 9/16)`, placed at (0, 0) with no padding. For a 640×640 input this gives exactly the current 360×640 region, for every frame shape, so `try.tflite` behavior is preserved. Landscape frames are still distorted in this mode.
- **`LETTERBOX`.** `scale = min(tensorW/srcW, tensorH/srcH)`. Content size is `round(src · scale)` and padding is `(tensor − content) / 2` (integer division, so any odd pixel goes to the right or bottom). For example, a 960×540 frame in a 640×640 input gives 640×360 content with `padY = 140`.

**Design decisions**
- **Pure Kotlin, no Android types.** Bitmap drawing stays with the caller (Detector), so all the geometry is unit-testable on the JVM, like the 3C parsers.
- **Reuses `LetterboxInfo` unchanged.** The task text proposed `LetterboxInfo(scale, padX, padY, srcW, srcH)`. I kept the existing contract so `ModelOutputParser`, `LegacyObbParser` and their tests don't change. The source size lives in `PreprocessPlan`. `tensorToSource*` is defined as `LetterboxInfo.toNormalized* × sourceSize`, so the parser path and the inverse transform can't drift apart.
- **Whole-pixel content and padding.** The plan describes the bitmap that will actually be drawn, and the inverse uses those same rounded numbers.
- **Tensor size is passed per call**, not stored, so the same preprocessor works for any model input shape (Detector already reads `tensorWidth`/`tensorHeight` from the interpreter).
- **Legacy stays the default.** Per the task list, letterbox vs. stretch must eventually match training (Task 4B).

### Unit tests and results

Command (JDK 17; the machine's default JDK 26 fails Gradle 8.10.2):

```
cd android && JAVA_HOME=$(/usr/libexec/java_home -v 17) ./gradlew :camerax:testDebugUnitTest --rerun-tasks
```

| Suite | Result |
|---|---|
| `FramePreprocessorTest` (new) | ✅ 16 / 16 |
| `LegacyObbParserTest` (existing) | ✅ 12 / 12 |
| `ViolinBowClassifierTest` (existing) | ✅ 14 / 14 |
| **Total** | ✅ 42 tests, 0 failures |

`FramePreprocessorTest` covers:
- **Legacy behavior:**
  - default mode is `LEGACY_STRETCH`
  - on a 640×640 input the plan equals the current Detector placement (360×640 at (0,0)) for portrait, landscape and square frames
  - the legacy region scales with non-square tensors (640×384 → 216×384)
  - a landscape frame's aspect ratio is distorted by more than 3× (documents the known limitation)
- **Letterbox placement:**
  - exact placement for landscape 960×540, portrait 1080×1920 and square 720×720 frames
  - for 6 frame sizes × 5 tensor sizes (640², 320², 640×384, 384×640, 1024²), content fits inside the tensor, one side fills it, it is centered within 1 px, and the aspect ratio is kept up to rounding
- **Round trip (acceptance criterion, within 1 px):**
  - source → tensor → source in both modes for all frame/tensor combinations, including corners, the center and off-grid points
  - the parser's normalized path (`toNormalized* × size`) agrees with the inverse transform within 1e-9
  - rounding the content size to whole pixels moves a point less than 1 px from an ideal unrounded letterbox
  - padding maps outside the source frame, and the content edges map exactly to 0 and to the frame size
- **End to end with `LegacyObbParser`:**
  - letterbox mode: a rotated box (0.4 rad) at known source coordinates, mapped into the tensor and parsed, returns to the source within 1 px for portrait, landscape and square frames and all 5 tensor sizes
  - legacy mode: the same check with an axis-aligned box on a 640×640 input
- **Validation:** zero source or tensor sizes throw `IllegalArgumentException`.

The accuracy claims above are geometric, checked with synthetic tensors. **They say nothing about real detection accuracy on device.**

### Issues, limitations, and open questions

- **No effect on the app yet.** Until Detector integration lands, the live pipeline still stretches 960×540 landscape frames into 360×640.
- **Training convention for `try.tflite` is unknown.** Letterbox only helps if it matches how the model was trained, including padding color (currently black, where Ultralytics uses gray 114) and centered vs. top-left placement. Switching modes without that information could make results worse. Open question for the model team (ties into Task 4B).
- **Angles survive only under uniform scaling.** In `LEGACY_STRETCH`, x and y are scaled differently, so a rotated box in the tensor is not the same rotated box in the source. `LegacyObbParser` applies the angle in tensor space and then maps the corners back, which is unchanged from before. Letterbox mode preserves angles.
- **Downstream code still assumes 640.** `Detector.toLegacyPoints()` (×640), `OverlayView`'s `/640` mapping and the string-lock thresholds are in 640-unit space. A model with a different input size would need those addressed (Tasks 3G and 3D).
- **Device box accuracy is still unvalidated** (see `task_3c_implementation_report.md` §7). Preprocessing is a known contributor, not a confirmed sole cause.

### Remaining work for Task 3F

*As of the foundation step. Steps 1 and part of 3 are now done; see the update below for current status.*

1. **Detector integration**
   - Add a constructor parameter `preprocessor: FramePreprocessor = FramePreprocessor()` (defaults to legacy).
   - In `detect()`, call `preprocessor.plan(frame.width, frame.height, tensorWidth, tensorHeight)`. Scale the frame to `contentWidth × contentHeight`, draw it at `(padX, padY)` on a `tensorWidth × tensorHeight` bitmap, and pass `plan.letterbox` to `parser.parse()`.
   - Remove `PREPROCESS_CONTENT_WIDTH/HEIGHT`, `PREPROCESS_TENSOR_SIZE` and `PREPROCESS_LETTERBOX`.
   - Keep `CameraxView` on the default so behavior doesn't change.
2. **Optional pad-color setting**, if the training convention requires it.
3. **Verification after integration**
   - Re-run unit tests and `:app:assembleDebug`.
   - On a phone, check that legacy mode looks the same as before in portrait and landscape.
   - Try `LETTERBOX` mode in landscape and compare box placement. If boxes are still off, investigate the other candidates from the 3C report: the model itself, the angle convention, overlay mapping (3G) and mirroring.
4. **Update this log** with the integration results, then commit (not yet requested).

Out of scope for 3F: `OverlayView` screen mapping (3G), removing `toLegacyPoints()`, and classifier changes (3D).

---

### Update: Detector integration (2026-10-24)

**Status:** ✅ Integrated and builds. ⏳ Not yet verified on a device. **Committed:** No.

#### What changed

| File | Status | Change |
|---|---|---|
| `modules/camerax/android/src/main/java/expo/modules/camerax/Detector.kt` | modified | Uses `FramePreprocessor` for frame placement and the parser `LetterboxInfo`; obsolete constants removed (+9 / −18 lines) |
| `task_progress.md` | modified | This update |

`FramePreprocessor.kt` and `FramePreprocessorTest.kt` are unchanged since the foundation step. These are still **not** modified: `CameraxView.kt`, `OverlayView.kt`, `ModelOutputParser.kt` (`LetterboxInfo`), `LegacyObbParser.kt`, thresholds, angle semantics and classification.

#### Implementation details

- **New constructor parameter.** `preprocessor: FramePreprocessor = FramePreprocessor()`, which defaults to `LEGACY_STRETCH`. `CameraxView` still calls `Detector(context, this)`, so it gets the legacy default without any change.
- **`detect()` now does the following:**
  1. `plan = preprocessor.plan(frame.width, frame.height, tensorWidth, tensorHeight)`, using the input size `Detector` already reads from the interpreter in `init`.
  2. Scales the frame to `plan.contentWidth × plan.contentHeight` (same `createScaledBitmap(..., filter = false)` as before).
  3. Draws it at `(plan.padX, plan.padY)` on a `tensorWidth × tensorHeight` ARGB bitmap. Padding is transparent black, same as before.
  4. Passes `plan.letterbox` to `parser.parse()`.
- **Removed:** `PREPROCESS_CONTENT_WIDTH`, `PREPROCESS_CONTENT_HEIGHT`, `PREPROCESS_TENSOR_SIZE`, `PREPROCESS_LETTERBOX`, and a dead commented-out resize line. No fixed 640 remains in preprocessing.
- **Kept:** `LEGACY_POINT_SCALE` / `toLegacyPoints()` (×640). These are the downstream point space for `classify()`, `OverlayView` and `Profile`, not preprocessing (Tasks 3G/3D).

#### Why `try.tflite` behavior is unchanged

- I read the input tensor from the model file itself: `images_nhwc [1, 640, 640, 3]`, output `output_0 [1, 7, 8400]`. So `Detector` gets `tensorWidth = tensorHeight = 640`.
- With a 640×640 tensor, `LEGACY_STRETCH` gives content 360×640 at (0, 0) and `LetterboxInfo(640, 640, 360f, 640f, 0f, 0f)` for every frame shape. That's exactly the old constants. The unit test `legacy_640Tensor_matchesCurrentDetectorPlacement_forEveryFrameShape` covers this.
- So for `try.tflite` the input pixels, the parser's `LetterboxInfo`, the output points and the classification inputs are identical to before.
- **One intentional difference:** the padded bitmap is now sized from the interpreter's input shape, not a fixed 640×640. For `try.tflite` these are the same. For a model with a different input size, the old code would have fed a wrongly sized buffer.

#### Test and build results

JDK 17: `cd android && JAVA_HOME=$(/usr/libexec/java_home -v 17) ./gradlew ...`

| Command | Result |
|---|---|
| `:camerax:testDebugUnitTest --rerun-tasks` | ✅ 42 tests, 0 failures (`FramePreprocessorTest` 16/16, `LegacyObbParserTest` 12/12, `ViolinBowClassifierTest` 14/14) |
| `:app:assembleDebug` | ✅ BUILD SUCCESSFUL (APK rebuilt) |

`Detector` uses Android `Bitmap`/`Canvas` and the TFLite interpreter, so the integration itself is covered only by compilation plus the `FramePreprocessor` tests. There is no JVM test that runs `Detector.detect()`.

#### Limitations

- **The app's behavior is intentionally unchanged.** Landscape 960×540 frames are still stretched into 360×640, because `LEGACY_STRETCH` is the default and `CameraxView` isn't touched. Letterbox is available only by constructing `Detector(..., preprocessor = FramePreprocessor(ResizeMode.LETTERBOX))`.
- **Letterbox is untested on device and may not suit `try.tflite`.** Its training preprocessing (centered vs. top-left, pad color) is unknown. The current pad is black (0 after /255). Ultralytics trains with gray 114. A pad-color option was **not** added, since there's no spec requiring it yet.
- **In letterbox mode, downstream points mean something different.** `toLegacyPoints()` scales normalized *source-frame* coordinates by 640. In legacy mode that equals the old numbers. In letterbox mode the points are still in 0–640 source-relative space, but `OverlayView` stretches them over the whole view and ignores FIT_CENTER offsets (3G). So letterbox boxes may be correctly located in the frame but still drawn misaligned on screen.
- **Per-frame cost is unchanged** (one scaled bitmap + one padded bitmap per frame, as before). Reusing bitmaps is a possible later optimization, not done here.
- **Device box accuracy is still unvalidated.** Preprocessing is one known contributor, not a confirmed sole cause (see `task_3c_implementation_report.md` §7).

#### Remaining work for Task 3F

- [x] Detector integration with `LEGACY_STRETCH` as the default
- [x] Remove obsolete `PREPROCESS_*` constants
- [x] Unit tests and `:app:assembleDebug`
- [ ] **On-device check, legacy mode.** Portrait and landscape should look the same as the Task 3C build.
- [ ] **On-device experiment, letterbox mode.** Temporarily pass `FramePreprocessor(ResizeMode.LETTERBOX)` and compare box placement in landscape. Interpret the result with the overlay limitation above in mind.
- [ ] **Decide how the mode is chosen.** Keep it in code, or expose it as config/Expo prop. That would need a `CameraxView` change, which is outside this step.
- [ ] Pad-color option, only if the model spec (Task 4B) requires it.
- [ ] Commit once reviewed. Not requested yet.

Out of scope for 3F: `OverlayView` screen mapping (3G), removing `toLegacyPoints()`, and classifier changes (3D).

---

## Task 3B — Remove mock and scratch code from production sources

**Branch:** `xinru/task-3b-cleanup`, rebased onto `Violin_UI` at `287dc50`
**Date:** 2026-10-30
**Status:** ✅ Merged into local `Violin_UI` as `c716327`. ⏳ Not pushed, and not yet verified on a device.

### What was integrated

| Commit | Author | Change |
|---|---|---|
| `5388cef` | jiangxinru | Mock and debug cleanup, with `Detector.kt` conflicts resolved in favor of Task 3C |
| `a3ade69` | jiangxinru | Restored real detection and removed the mock bow test |
| `b2379d2` | Junyong Lee | Removed unused `f1`/`f2`/`f3` helpers and the commented-out calibration prop |

- Kept the `Time Comparing` bow inference timing for Task 4D. It still logs on every frame, including release builds (see Task 3H).
- Deleted `ViolinDetector.kt` and the Task 1A mock block.
- `ModelOutputParser.kt`, `LegacyObbParser.kt`, `ViolinDetection.kt`, `ViolinBowClassifier.kt`, `Profile.kt` and their tests are unchanged.
- Thresholds and classification logic are unchanged. Live detection still uses the legacy geometry in `Detector.classify()`.

### Test and build results (merge commit, JDK 17)

| Command | Result |
|---|---|
| `:camerax:testDebugUnitTest --rerun-tasks` | ✅ 26 tests, 0 failures (`ViolinBowClassifierTest` 14/14, `LegacyObbParserTest` 12/12) |
| `:app:assembleDebug --rerun-tasks` | ✅ BUILD SUCCESSFUL |

### Follow-up

- Task 3F is still uncommitted and based on `287dc50`, so it needs moving onto the updated `Violin_UI`. Its `Detector.kt` patch applied cleanly on `b2379d2`, which has the same tree as the merge commit, and passed 42/42 tests.
- `origin/xinru/task-3b-cleanup` (`25a51f1`) and `origin/Violin_UI` (`287dc50`) don't have this work yet.

---

### Update: Integration into Violin_UI (2026-11-09)

**Status:** ✅ Committed on `task-3f-preprocessing` and merged into local `Violin_UI`. ⏳ Not pushed, and not yet verified on a device.

#### How it was integrated

- `task-3f-preprocessing` had no commits of its own (`287dc50` is already in `Violin_UI`), so it was fast-forwarded to `Violin_UI` at `d3650d8`. That commit includes Task 3B (`c716327`) and Task 3E (`d3650d8`).
- The uncommitted 3F files were stashed by path, the branch was fast-forwarded, and the stash was restored. `Detector.kt` merged automatically with no conflicts. The restored files are byte-identical to the pre-integration versions.
- Task 3K (`xinru/task-3k-lifecycle`) is **not** included.

#### Compatibility with Tasks 3B, 3C and 3E

- 3F's `Detector.kt` edits touch only the constructor, the companion `PREPROCESS_*` constants and the preprocessing in `detect()`. None of them touch `classify()`, 3E's `BowHeight`/`BowAngle` code, thresholds or `maxAngle`.
- 3C's parser contract (`ModelOutputParser`, `LegacyObbParser`, `LetterboxInfo`) is unchanged.
- `LEGACY_STRETCH` is still the default. For `try.tflite` (`[1, 640, 640, 3]`) the model input and the parser's `LetterboxInfo` are identical to before, so detection and classification behavior is preserved.
- Task 3K was checked separately (in a scratch worktree): 3F applies cleanly on top of it, and 3F's preprocessing runs only after 3K's null-interpreter early return.

#### Test and build results (JDK 17, `Violin_UI` + 3F)

| Command | Result |
|---|---|
| `:camerax:testDebugUnitTest` | ✅ 50 tests, 0 failures (`BowClassificationTest` 8/8, `FramePreprocessorTest` 16/16, `LegacyObbParserTest` 12/12, `ViolinBowClassifierTest` 14/14) |
| `:app:assembleDebug` | ✅ BUILD SUCCESSFUL |
| `tsc --noEmit` | 8 errors, all pre-existing on `Violin_UI` (none in files touched by 3E or 3F) |

#### Remaining work

- [ ] **On-device check, legacy mode.** Portrait and landscape should look the same as the Task 3E build.
- [ ] **On-device experiment, letterbox mode.** Only after the model team confirms the training preprocessing (Task 4B).
- [ ] Decide how the mode is chosen (code vs. config/Expo prop).
- [ ] Pad-color option, only if the model spec requires it.

Out of scope for 3F: `OverlayView` screen mapping and the remaining `/640f` debug-drawing literals (3G), removing `toLegacyPoints()`, and classifier changes (3D).
