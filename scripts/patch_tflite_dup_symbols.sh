#!/bin/bash
# MediaPipeTasksCommon bundles its own full copy of the TensorFlow Lite runtime,
# so the TensorFlowLiteC pod's binaries collide with it at app link time
# (51 duplicate TfLite*/TFLGpu* symbols). This localizes those symbols in the
# TensorFlowLiteC pod's device binaries: internal references inside each binary
# are unaffected, but the symbols stop being exported, so MediaPipe's copies win
# at final link and MediaPipe's runtime stays fully self-consistent. Our own
# detector path is unaffected: it talks to TFLite through the high-level C API
# (none of which is localized) and uses the CoreML delegate, which is not in
# the duplicate set. Only the Metal-delegate fallback binds to MediaPipe's
# TFLGpuDelegate* implementations instead of 2.14's.
#
# Idempotent: skips binaries that already export none of the duplicate symbols.
set -euo pipefail

# Resolve ios/Pods relative to this script (repo-root/scripts/ -> repo-root/ios/Pods).
PODS_ROOT="$(cd "$(dirname "$0")/../ios/Pods" && pwd)"

DUP_SYMBOLS=(
  TFLGpuDelegateBindMetalBufferToTensor TFLGpuDelegateCreate TFLGpuDelegateDelete
  TfLiteBackendBufferCreate TfLiteBackendBufferDelete TfLiteBackendBufferGetPtr
  TfLiteBackendBufferSetPtr TfLiteDelegateCreate TfLiteFloatArrayCopy
  TfLiteFloatArrayCreate TfLiteFloatArrayFree TfLiteFloatArrayGetSizeInBytes
  TfLiteIntArrayCopy TfLiteIntArrayCreate TfLiteIntArrayEqual
  TfLiteIntArrayEqualsArray TfLiteIntArrayFree TfLiteIntArrayGetSizeInBytes
  TfLiteQuantizationFree TfLiteSparsityFree TfLiteSynchronizationCreate
  TfLiteSynchronizationDelete TfLiteSynchronizationGetPtr TfLiteSynchronizationSetPtr
  TfLiteTelemetryConversionMetadataGetModelOptimizationModes
  TfLiteTelemetryConversionMetadataGetNumModelOptimizationModes
  TfLiteTelemetryGpuDelegateSettingsGetBackend
  TfLiteTelemetryGpuDelegateSettingsGetNumNodesDelegated
  TfLiteTelemetryInterpreterSettingsGetConversionMetadata
  TfLiteTelemetryInterpreterSettingsGetNumSubgraphInfo
  TfLiteTelemetryInterpreterSettingsGetSubgraphInfo
  TfLiteTelemetrySubgraphInfoGetNumQuantizations
  TfLiteTelemetrySubgraphInfoGetQuantizations
  TfLiteTensorCopy TfLiteTensorDataFree TfLiteTensorFree TfLiteTensorRealloc
  TfLiteTensorReset TfLiteTensorResizeMaybeCopy TfLiteTypeGetName
  TfLiteXNNPackDelegateCreate TfLiteXNNPackDelegateCreateWithThreadpool
  TfLiteXNNPackDelegateDelete TfLiteXNNPackDelegateGetFlags
  TfLiteXNNPackDelegateGetThreadPool TfLiteXNNPackDelegateOptionsDefault
  TfLiteXNNPackDelegateWeightsCacheCreate TfLiteXNNPackDelegateWeightsCacheCreateWithSize
  TfLiteXNNPackDelegateWeightsCacheDelete TfLiteXNNPackDelegateWeightsCacheFinalizeHard
  TfLiteXNNPackDelegateWeightsCacheFinalizeSoft
)

DUP_FILE="$(mktemp)"
printf "_%s\n" "${DUP_SYMBOLS[@]}" > "$DUP_FILE"
trap 'rm -f "$DUP_FILE"' EXIT

patch_binary() {
  local bin="$1"
  if [ ! -f "$bin" ]; then
    echo "skip (missing): $bin"
    return
  fi

  local exported_dups
  exported_dups=$( (nm -g "$bin" 2>/dev/null | grep -E ' (T|S|D) ' | awk '{print $NF}' | grep -F -x -f "$DUP_FILE" || true) | wc -l | tr -d ' ')
  if [ "$exported_dups" -eq 0 ]; then
    echo "already patched: $bin"
    return
  fi

  local workdir keep thin
  workdir="$(mktemp -d)"
  keep="$workdir/keep.txt"
  thin="$workdir/thin.o"

  # The pod ships single-arch universal wrappers; extract the arm64 slice.
  lipo -thin arm64 "$bin" -output "$thin" 2>/dev/null || cp "$bin" "$thin"

  # May legitimately be empty (the Metal binary's only globals are duplicates).
  (nm -g "$thin" | grep -E ' [A-TV-Z] ' | awk '{print $NF}' | grep -F -x -v -f "$DUP_FILE" || true) | sort -u > "$keep"

  ld -r "$thin" -exported_symbols_list "$keep" -o "$workdir/patched.o"
  cp "$bin" "$bin.orig"
  lipo -create "$workdir/patched.o" -output "$bin"
  rm -rf "$workdir"
  echo "patched ($exported_dups dup symbols localized): $bin"
}

patch_binary "$PODS_ROOT/TensorFlowLiteC/Frameworks/TensorFlowLiteC.xcframework/ios-arm64/TensorFlowLiteC.framework/TensorFlowLiteC"
patch_binary "$PODS_ROOT/TensorFlowLiteC/Frameworks/TensorFlowLiteCMetal.xcframework/ios-arm64/TensorFlowLiteCMetal.framework/TensorFlowLiteCMetal"
