# Task 3C Implementation Report — Model-Agnostic Detection Contract

**Branch:** `Violin_UI`
**Task:** 3C from `violin_ui_code_review_and_next_tasks_updated.md`
**Goal:** When the trained violin model is ready, adding a new parser should be enough, with no rewrite of `Detector.kt`.

---

## 1. Old vs. new architecture

### Before

`Detector.kt` did everything in one class:

```
Bitmap ─► preprocess (360x640 → 640x640) ─► interpreter.run()
       ─► reshape [1,8400,7] / [1,7,N]          ┐
       ─► newBestBox (scores, +0.3 boost, 0.6)  │ all hard-coded for try.tflite
       ─► rotatedRectToPoints (·640/360)        ┘
       ─► YoloResults ─► classify() ─► CameraxView / OverlayView / Profile
```

`init` read `numChannel`/`numElements` from output 0 and assumed a rank-3 tensor. The asset name `try.tflite` was a private constant.

### After

```
Bitmap ─► preprocess (unchanged) ─► interpreter.runForMultipleInputsOutputs()
       ─► Map<Int, OutputTensor>  (one entry per output tensor, shape + floats)
       ─► ModelOutputParser.parse(outputs, LetterboxInfo)      ◄── swappable
       ─► ViolinDetection (bow/strings as Quad in normalized [0,1] coords)
       ─► Quad.toLegacyPoints() (×640) ─► YoloResults ─► classify() ─► … (unchanged)
```

`Detector` now handles only interpreter/delegate setup, preprocessing and running inference. Parsing the output layout is the parser's job.

---

## 2. Files changed

| File | Status | Purpose |
|---|---|---|
| `modules/camerax/android/src/main/java/expo/modules/camerax/ViolinDetection.kt` | **new** | `ViolinDetection` and `Quad` (the model-independent result) |
| `modules/camerax/android/src/main/java/expo/modules/camerax/ModelOutputParser.kt` | **new** | `ModelOutputParser` interface, `OutputTensor`, `LetterboxInfo` |
| `modules/camerax/android/src/main/java/expo/modules/camerax/LegacyObbParser.kt` | **new** | The 7-channel OBB logic moved out of `Detector` |
| `modules/camerax/android/src/main/java/expo/modules/camerax/Detector.kt` | modified | Uses the parser; configurable asset and parser (+44 / −171 lines) |
| `modules/camerax/android/src/test/java/expo/modules/camerax/LegacyObbParserTest.kt` | **new** | 12 synthetic-tensor unit tests |

`CameraxView.kt`, `OverlayView.kt`, `Profile.kt` and `ViolinBowClassifier.kt` are **not** changed. `CameraxView` still calls `Detector(context, this)`, and the new constructor parameters have defaults.

---

## 3. What moved out of `Detector.kt`

| Removed from `Detector` | New home |
|---|---|
| `numChannel`, `numElements` fields and the output-shape reading in `init` | `LegacyObbParser.parse` (reads `OutputTensor.shape`) |
| `[1,8400,7]` → `[1,7,N]` layout check | `LegacyObbParser.parse` |
| `transposeN7To7N()` | `LegacyObbParser.transposeN7To7N()` (unchanged) |
| `newBestBox()` (class pick, +0.3 bow boost, 0.6 threshold) | `LegacyObbParser.newBestBox()` (unchanged) |
| `CONFIDENCE_THRESHOLD` | `LegacyObbParser.CONFIDENCE_THRESHOLD` (+ `BOW_CONFIDENCE_BOOST`) |
| `OrientedBoundingBox` data class | `LegacyObbParser.OrientedBoundingBox` (internal) |
| Best-box-per-class loop in `detect()` | `LegacyObbParser.parse` |
| `rotatedRectToPoints()` incl. the `·640/360` x-fix | `LegacyObbParser.rotatedRectToQuad()` + `LetterboxInfo.toNormalized*()` |

What `Detector` gained:
- Constructor parameters `modelAsset: String = DEFAULT_MODEL_ASSET` (`"try.tflite"`) and `parser: ModelOutputParser = LegacyObbParser()`.
- Output buffers are allocated generically for every output tensor, using the interpreter's own shapes.
- `PREPROCESS_LETTERBOX`, which describes the current 360×640-into-640×640 placement. The literals now live in named constants.
- `Quad.toLegacyPoints()`, which scales normalized corners by 640 so downstream code gets the same numbers as before.

---

## 4. How the parser abstraction works

```kotlin
interface ModelOutputParser {
    fun parse(outputs: Map<Int, OutputTensor>, letterbox: LetterboxInfo): ViolinDetection
}
```

- **`OutputTensor(shape: IntArray, data: FloatArray)`**: one raw output tensor, keyed by its output index. A future model with several outputs (for example separate boxes/scores/classes tensors) just shows up as several map entries. `Detector` doesn't need to change.
- **`LetterboxInfo(tensorWidth, tensorHeight, contentWidth, contentHeight, padX, padY)`**: where the source bitmap sits inside the input tensor. Parsers call `toNormalizedX/Y()` to map tensor pixels back to normalized source coordinates. Width and height are separate, so the format covers both today's non-uniform stretch and a real aspect-preserving letterbox (Task 3F).
- **`ViolinDetection(bow: Quad?, strings: Quad?, bowConfidence, stringConfidence)`**: the contract. `Quad` holds exactly 4 corners in **normalized source-bitmap coordinates**: (0,0) is the top-left of the bitmap passed to `detect()` and (1,1) is the bottom-right.
- **Errors:** if the outputs aren't in a layout the parser understands, it throws `IllegalArgumentException`. `Detector` catches it, logs `MODEL ERROR`, and returns empty results *without* calling the listener. That's the same observable behavior as the old "invalid tensor shapes" check.

**Adding the trained model later** should look like this:
1. Write `class TrainedModelParser : ModelOutputParser` once the real output format is known.
2. Construct `Detector(context, this, modelAsset = "violin.tflite", parser = TrainedModelParser())` in `CameraxView`.
3. Add synthetic-tensor tests like `LegacyObbParserTest`.

### Design decisions that differ from the task text (and why)

- **`Map<Int, OutputTensor>` instead of `Map<Int, Any>`.** A typed, Android-free tensor type keeps parsers pure Kotlin, so they run in plain JVM unit tests. With `Any` every parser would need unchecked casts.
- **`LetterboxInfo` fields are not `(scale, padX, padY, srcW, srcH)`.** The current preprocessing stretches the frame unevenly (x and y scale differently), so a single `scale` can't describe it. Source width/height aren't needed to produce normalized coordinates, so I left them out to keep the class small. 3F can add them if needed.
- **No `FramePreprocessor.kt` yet.** The task list puts it under 3C, but its content (an aspect-preserving letterbox) is the scope of 3F. I left preprocessing in `Detector` unchanged and only named its constants.
- **Parsers output normalized coordinates, but `Detector` still emits legacy 0..640 points.** `OverlayView` divides by 640, and the string-lock thresholds (`MAX_Y_DELTA_THRESHOLD = 3`) are in those units. Changing them would change behavior, so the conversion sits in one helper (`toLegacyPoints`) that 3F/3G can remove later.

---

## 5. Behavior preservation

For `try.tflite` the pipeline produces the same points as before:

- Legacy: `x = xRot · 640 / 360`, `y = yRot`
- New: `x = (xRot / 360) · 640`, `y = (yRot / 640) · 640`

The two agree up to double-precision rounding (~1e-13). The test `rotatedBox_matchesLegacyCornerFormula` checks the parser against a copy of the old formula. These are unchanged:
- Score logic, +0.3 bow boost, 0.6 threshold, strict `>` best-per-class (first anchor wins ties)
- String points are still sorted with `sortStringPoints` before going out; bow points are not
- `classify()`, temporal smoothing, string-box locking, `maxAngle`, labels, and listener calls (`detected` / `noDetect`)
- Delegate fallback order (QNN → GPU → CPU) and the default asset `try.tflite`

Small, intentional differences:
- `interpreter.run(...)` became `interpreter.runForMultipleInputsOutputs(...)`. `run` is a thin wrapper around it, so the inference is the same.
- The shape check now runs after inference instead of before, so a bad model wastes one inference per frame before logging the error. Valid models aren't affected.
- I added `C >= 7` as a hardening check. A malformed `[1,C<7,N]` output used to crash with an index error; now it logs `MODEL ERROR`.
- The per-box `println("ANGLE …")` and the per-box `BOX INITAL BOW/STRING` logs are replaced by one `CheckDelBox` summary log per frame.

---

## 6. Tests and build results

Environment: JDK 17 (Temurin 17.0.20.1). The machine's default JDK 26 fails Gradle 8.10.2 at settings evaluation, so commands were run with `JAVA_HOME=$(/usr/libexec/java_home -v 17)`.

| Command | Result |
|---|---|
| `./gradlew :camerax:testDebugUnitTest` (baseline, before changes) | ✅ BUILD SUCCESSFUL |
| `./gradlew :camerax:testDebugUnitTest` (after changes) | ✅ BUILD SUCCESSFUL. 26 tests, 0 failures |
| ↳ `LegacyObbParserTest` | ✅ 12 / 12 |
| ↳ `ViolinBowClassifierTest` (existing) | ✅ 14 / 14 |
| `./gradlew :app:assembleDebug` | ✅ BUILD SUCCESSFUL (full debug APK builds) |

No new compiler warnings in the new or changed code. All remaining warnings in `Detector.kt` were already there.

`LegacyObbParserTest` covers:
- no anchor above threshold → empty detection
- bow +0.3 boost lets a 0.35 bow through; strings at 0.55 are rejected
- best box per class is kept; on equal scores the first anchor wins
- rotated boxes match the legacy corner formula exactly (within 1e-6 px)
- a full-content box maps to exactly (0,0)…(1,1); letterbox padding is removed
- `[1,8400,7]` and `[1,7,N]` layouts give identical results
- missing output 0, non-rank-3 output, and fewer than 7 channels all throw `IllegalArgumentException`

Re-run on 2026-10-24 after removing the temporary device-validation logging (see §7):

| Command | Result |
|---|---|
| `./gradlew :camerax:testDebugUnitTest --rerun-tasks` | ✅ BUILD SUCCESSFUL. 26 tests, 0 failures (`LegacyObbParserTest` 12/12, `ViolinBowClassifierTest` 14/14) |
| `./gradlew :app:assembleDebug` | ✅ BUILD SUCCESSFUL |

---

## 7. Physical-device findings

A manual session ran on a physical Android phone with `try.tflite`, using temporary `Task3CConf` / `Task3CCoord` log tags to trace raw scores and coordinates through each stage (camera frame → detector input → normalized quad → legacy points → screen).

**What was confirmed**
- The live pipeline runs end to end: `CameraxView.performDetection(rotatedBitmap)` → `Detector.detect()` → `LegacyObbParser.parse()` → `YoloResults` → `classify()` → overlay. Inference runs, the parser accepts the model's output layout, and no `MODEL ERROR` is logged.

**What was not validated**
- **Bounding-box accuracy.** The boxes drawn on screen do not reliably line up with the real bow and strings. Task 3C is a structural refactor that is unit-tested to reproduce the old coordinates exactly (§5), so it does not claim to fix accuracy, and it does not yet show that accuracy is correct.

**Observed preprocessing mismatch**
- In camera-only landscape mode, the detector receives a **960×540** (landscape) frame. `Detector.detect()` still stretches every frame into a fixed **360×640 (portrait)** region at the top-left of the 640×640 tensor (`PREPROCESS_CONTENT_WIDTH/HEIGHT`). A 16:9 landscape image is squeezed to 9:16. That is a non-uniform distortion of roughly 3.2× on the aspect ratio, so the model sees an image very different from what it was trained on.
- This is a **known contributor**, but it is **not established as the only cause** of the inaccurate boxes. Other things that have not been ruled out:
  - how well `try.tflite` itself performs on these scenes (it is a placeholder, not the trained violin model)
  - what `try.tflite` expects as input (the original export resolution and letterbox convention haven't been confirmed)
  - the angle convention used in the OBB corner reconstruction
  - the overlay mapping: legacy points are drawn as `x/640 · viewWidth` across the whole view, while hands use the FIT_CENTER preview content rect
  - front-camera mirroring

**Temporary instrumentation removed.** All `Task3CConf` / `Task3CCoord` logging was deleted after the session, including `LegacyObbParser.debugRawMaxScores()` and the stage traces in `CameraxView`, `Detector` and `OverlayView`. `CameraxView.kt` and `OverlayView.kt` are back to identical with `HEAD`. Production logs are kept: the `MODEL ERROR` log for parser failures, the inference timing logs, and the one-line `CheckDelBox` detection summary.

---

## 8. Task 3F follow-up (not implemented here)

Task 3C deliberately leaves preprocessing, overlay geometry, thresholds and classification unchanged. For 3F:

1. **Orientation-aware, aspect-preserving letterbox.** Replace the fixed 360×640 stretch with a uniform scale plus padding computed from the actual frame size (so 960×540 → 640×360 content with vertical padding, not 360×640). The `LetterboxInfo` passed to the parser should come from that computation instead of the constant `PREPROCESS_LETTERBOX`. The parser API already supports this, and the parsers themselves won't need changes.
2. **Revalidate on device afterward.** Repeat the coordinate trace in portrait and landscape to see how much of the box error goes away with correct preprocessing. If error remains, investigate the other candidates in §7 (overlay mapping is 3G, model choice is the trained-model work).
3. **Add tests** for the letterbox computation in both orientations (synthetic frame sizes → expected `LetterboxInfo` and round-trip coordinates).

---

## 9. Overlap with `xinru/task-3b-cleanup`

`origin/xinru/task-3b-cleanup` is one commit (`fe11636`, "Clean up temporary mock and debug code") on top of the same base as this work (`7222230`). A simulated merge (`git merge-tree`, working tree untouched) shows:

- **`Detector.kt`: 2 content conflicts**, both inside code that 3C moved to `LegacyObbParser` (the old `detect()` box loop and `rotatedRectToPoints` / `newBestBox` / `OrientedBoundingBox`). Resolve both in favor of the 3C side. 3B also removes the inference-timing and `CheckDelBox` logs, which 3C keeps. That's a judgment call for whoever does the merge.
- **⚠️ `CameraxView.kt`: auto-merges with no conflict, but brings in a mock.** Despite its commit message, `fe11636` comments out `performDetection(rotatedBitmap)` and adds an active call to a new `testMockBowResult()` that injects a fixed fake bow/string result every frame. If 3B is merged as-is, the real detector is never called. Fix this on the 3B branch (restore `performDetection(rotatedBitmap)` and delete `testMockBowResult()`) before merging it, or after merging check that `CameraxView` calls `performDetection(rotatedBitmap)`.
- No overlap on the new 3C files. 3B deletes `ViolinDetector.kt`, which is unrelated to 3C's `ViolinDetection.kt`.

On `Violin_UI` with the 3C changes, `performDetection(rotatedBitmap)` is active and there is no mock.

---

## 10. Remaining limitations

- **Preprocessing is still hard-coded** (360×640 stretch into 640×640, no real letterbox; landscape 960×540 frames are distorted, as seen on device). That's Task 3F (§8). The parser API already accepts the `LetterboxInfo` that 3F will produce.
- **Box accuracy is not validated on device** (§7). The cause is not fully isolated.
- **Downstream code is still in legacy 0..640 point space.** `toLegacyPoints()` is a compatibility shim until `OverlayView` (3G) and `classify` (3D) take normalized `Quad`s directly.
- **Float outputs only.** `Detector` allocates `FLOAT32` output buffers, as before. A quantized (UINT8/INT8) model would need dequantization in `Detector` or a dtype field on `OutputTensor`.
- **Parser and asset are chosen in code, not runtime config.** `CameraxView` uses the defaults. Exposing them as Expo props or config is a follow-up (it ties in with 3K, configurable asset plus graceful failure if the asset is missing).
- **Confidence semantics are parser-specific.** `LegacyObbParser` reports the boosted bow score. A future parser should document what its confidences mean.
- **`[1,N,7]` with N ≠ 8400** is still treated as `[1,7,N]`, as before. I preserved this on purpose; a new model should get its own parser rather than stretching this one.
- `ViolinBowClassifier` is still not on the live path (Task 3D).
