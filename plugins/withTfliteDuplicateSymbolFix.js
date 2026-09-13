const { withDangerousMod } = require('expo/config-plugins');
const fs = require('fs');
const path = require('path');

/**
 * MediaPipeTasksCommon ships its own complete TensorFlow Lite runtime baked
 * into a vendored static archive. Its exported TfLite and TFLGpu symbols
 * collide with the TensorFlowLiteC pod's copy of the same symbols when both
 * link into the app, failing the build with 51 duplicate symbols.
 *
 * scripts/patch_tflite_dup_symbols.sh resolves this by un-exporting the
 * duplicates from the TensorFlowLiteC binaries, so MediaPipe keeps using its
 * own self-consistent runtime. That has to re-run after every `pod install`
 * (CocoaPods restores the pristine binaries), so this plugin wires it into the
 * generated Podfile's post_install hook.
 *
 * It lives as a config plugin rather than a hand edit because `expo prebuild`
 * regenerates ios/Podfile from scratch, which would silently drop the hook and
 * break the build for anyone who re-runs prebuild or clones the repo.
 */

// Presence of the script name marks the hook as already injected.
const HOOK_MARKER = 'patch_tflite_dup_symbols.sh';

const HOOK_SNIPPET = `
    # Injected by plugins/withTfliteDuplicateSymbolFix.js — see that file for why.
    # Un-exports the TensorFlowLite symbols that collide with MediaPipe's bundled
    # copy. Idempotent, so re-running pod install is safe.
    system("bash", File.join(__dir__, "..", "scripts", "patch_tflite_dup_symbols.sh")) or
      raise "TFLite duplicate-symbol patch failed (scripts/patch_tflite_dup_symbols.sh)"
`;

const POST_INSTALL_ANCHOR = 'post_install do |installer|';

/**
 * Injects the hook call at the top of the Podfile's existing post_install
 * block. It must go inside the existing block rather than adding a second
 * `post_install` — CocoaPods keeps only the last definition, so a second one
 * would silently discard React Native's own post-install work.
 */
function addHookToPodfile(contents) {
  if (contents.includes(HOOK_MARKER)) {
    return contents;
  }

  const anchorIndex = contents.indexOf(POST_INSTALL_ANCHOR);
  if (anchorIndex === -1) {
    throw new Error(
      '[withTfliteDuplicateSymbolFix] No `post_install do |installer|` block found in ' +
        'ios/Podfile, so the TFLite duplicate-symbol patch could not be installed. ' +
        'The build will fail to link MediaPipe until this is resolved.'
    );
  }

  const insertAt = anchorIndex + POST_INSTALL_ANCHOR.length;
  return contents.slice(0, insertAt) + HOOK_SNIPPET + contents.slice(insertAt);
}

const withTfliteDuplicateSymbolFix = (config) =>
  withDangerousMod(config, [
    'ios',
    async (config) => {
      const podfilePath = path.join(config.modRequest.platformProjectRoot, 'Podfile');
      const contents = fs.readFileSync(podfilePath, 'utf8');
      const patched = addHookToPodfile(contents);

      if (patched !== contents) {
        fs.writeFileSync(podfilePath, patched);
      }

      return config;
    },
  ]);

module.exports = withTfliteDuplicateSymbolFix;
// Exported for testing the string manipulation without running a full prebuild.
module.exports.addHookToPodfile = addHookToPodfile;
