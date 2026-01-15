package expo.modules.camerax

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import com.qualcomm.qti.QnnDelegate
import com.qualcomm.qti.QnnDelegate.Options.BackendType
import java.io.File
import java.nio.ByteBuffer

/**
 * Unified delegation manager for all TFLite models (Detector, Hands, Pose).
 *
 * Advantages:
 * - Probe ONCE at startup, cache result
 * - All models use same delegate (guaranteed)
 * - One place to force override for testing
 * - Logging shows global state, not per-model
 *
 * Log filter:
 *   adb logcat -s CheckDel:* CheckDelBow:* CheckDelHands:* CheckDelHandsLM:* CheckDelPose:* CheckDelPoseLM:*
 */
object DelegateManager {
    private const val TAG = "CheckDel"

    enum class DelegateType { HTP, GPU, CPU }

    private var activeDelegate: DelegateType? = null
    private var forcedDelegate: DelegateType? = null
    private var qnnDelegate: QnnDelegate? = null
    private var gpuDelegate: GpuDelegate? = null
    private var skelDir: String? = null
    private var initialized = false

    /**
     * Force a specific delegate for testing.
     * Call BEFORE any model initialization.
     * Pass null to return to automatic fallback.
     */
    @JvmStatic
    fun forceDelegate(type: DelegateType?) {
        forcedDelegate = type
        if (type != null) {
            activeDelegate = type
            Log.i(TAG, "FORCED delegate: $type (automatic fallback disabled)")
        } else {
            Log.i(TAG, "Force cleared, returning to automatic fallback")
            activeDelegate = null
            initialized = false
        }
    }

    @JvmStatic
    fun getActiveDelegate(): DelegateType {
        return activeDelegate ?: DelegateType.CPU
    }

    @JvmStatic
    fun isForced(): Boolean = forcedDelegate != null

    @JvmStatic
    fun isHtp(): Boolean = getActiveDelegate() == DelegateType.HTP

    /**
     * Create interpreter with unified delegation.
     * Uses forced delegate if set, otherwise automatic fallback.
     *
     * @param context Android context
     * @param modelPath Asset path to the model file
     * @param logTag Tag suffix for logging (e.g., "Hands" -> "CheckDelHands")
     */
    @JvmStatic
    fun createInterpreter(
        context: Context,
        modelPath: String,
        logTag: String
    ): Interpreter {
        val model = FileUtil.loadMappedFile(context, modelPath)
        return createInterpreterFromBuffer(context, model, logTag)
    }

    /**
     * Create interpreter from a ByteBuffer with unified delegation.
     */
    @JvmStatic
    fun createInterpreterFromBuffer(
        context: Context,
        model: ByteBuffer,
        logTag: String
    ): Interpreter {
        // If forced, use that delegate directly
        if (forcedDelegate != null) {
            val result = createWithDelegate(context, model, forcedDelegate!!, logTag)
            if (result != null) return result
            Log.w(TAG, "Forced delegate $forcedDelegate failed, falling back to CPU")
        }

        // Automatic fallback (Shri's logic)
        return createWithFallbacks(context, model, logTag)
    }

    private fun createWithFallbacks(
        context: Context,
        model: ByteBuffer,
        logTag: String
    ): Interpreter {
        val fullTag = "CheckDel$logTag"

        // 1) Try HTP (NPU)
        val skelDir = tryLoadQnnAndPickSkelDir(context)
        if (skelDir != null) {
            try {
                val opts = Interpreter.Options()
                val qOpts = QnnDelegate.Options().apply {
                    setBackendType(BackendType.HTP_BACKEND)
                    setSkelLibraryDir(skelDir)
                }
                val delegate = QnnDelegate(qOpts)
                opts.addDelegate(delegate)
                val interp = Interpreter(model as ByteBuffer, opts)

                // Cache for future calls
                qnnDelegate = delegate
                activeDelegate = DelegateType.HTP
                initialized = true

                Log.i(fullTag, "delegate=HTP")
                return interp
            } catch (t: Throwable) {
                Log.w(TAG, "$logTag: HTP failed: ${t.message}")
            }
        }

        // 2) Try GPU
        try {
            val cl = CompatibilityList()
            if (cl.isDelegateSupportedOnThisDevice) {
                val opts = Interpreter.Options()
                val delegate = GpuDelegate(cl.bestOptionsForThisDevice)
                opts.addDelegate(delegate)
                val interp = Interpreter(model as ByteBuffer, opts)

                // Cache for future calls
                gpuDelegate = delegate
                activeDelegate = DelegateType.GPU
                initialized = true

                Log.i(fullTag, "delegate=GPU")
                return interp
            }
        } catch (t: Throwable) {
            Log.w(TAG, "$logTag: GPU failed: ${t.message}")
        }

        // 3) CPU Fallback
        val opts = Interpreter.Options()
        try { opts.setUseXNNPACK(true) } catch (_: Throwable) {}
        opts.setNumThreads(4)
        activeDelegate = DelegateType.CPU
        initialized = true

        Log.i(fullTag, "delegate=CPU")
        return Interpreter(model as ByteBuffer, opts)
    }

    private fun createWithDelegate(
        context: Context,
        model: ByteBuffer,
        delegate: DelegateType,
        logTag: String
    ): Interpreter? {
        val fullTag = "CheckDel$logTag"

        return when (delegate) {
            DelegateType.HTP -> {
                val skelDir = tryLoadQnnAndPickSkelDir(context) ?: return null
                try {
                    val opts = Interpreter.Options()
                    val qOpts = QnnDelegate.Options().apply {
                        setBackendType(BackendType.HTP_BACKEND)
                        setSkelLibraryDir(skelDir)
                    }
                    val qnnDel = QnnDelegate(qOpts)
                    opts.addDelegate(qnnDel)
                    qnnDelegate = qnnDel
                    activeDelegate = DelegateType.HTP
                    Log.i(fullTag, "delegate=HTP (forced)")
                    Interpreter(model as ByteBuffer, opts)
                } catch (t: Throwable) {
                    Log.e(TAG, "$logTag: Forced HTP failed: ${t.message}")
                    null
                }
            }
            DelegateType.GPU -> {
                try {
                    val cl = CompatibilityList()
                    if (!cl.isDelegateSupportedOnThisDevice) return null
                    val opts = Interpreter.Options()
                    val gpuDel = GpuDelegate(cl.bestOptionsForThisDevice)
                    opts.addDelegate(gpuDel)
                    gpuDelegate = gpuDel
                    activeDelegate = DelegateType.GPU
                    Log.i(fullTag, "delegate=GPU (forced)")
                    Interpreter(model as ByteBuffer, opts)
                } catch (t: Throwable) {
                    Log.e(TAG, "$logTag: Forced GPU failed: ${t.message}")
                    null
                }
            }
            DelegateType.CPU -> {
                val opts = Interpreter.Options()
                try { opts.setUseXNNPACK(true) } catch (_: Throwable) {}
                opts.setNumThreads(4)
                activeDelegate = DelegateType.CPU
                Log.i(fullTag, "delegate=CPU (forced)")
                Interpreter(model as ByteBuffer, opts)
            }
        }
    }

    private fun tryLoadQnnAndPickSkelDir(context: Context): String? {
        // Return cached result if already loaded
        skelDir?.let { return it }

        val mustLoad = listOf("QnnSystem", "QnnHtp", "QnnHtpPrepare")
        for (name in mustLoad) {
            try { System.loadLibrary(name) }
            catch (e: UnsatisfiedLinkError) {
                Log.w(TAG, "QNN: failed to load $name: ${e.message}")
                return null
            }
        }
        val base = context.applicationInfo.nativeLibraryDir
        val skels = listOf(
            "libQnnHtpV79Skel.so",
            "libQnnHtpV75Skel.so",
            "libQnnHtpV73Skel.so",
            "libQnnHtpV69Skel.so"
        )
        val chosen = skels.firstOrNull { File("$base/$it").exists() }
        if (chosen == null) {
            Log.w(TAG, "QNN: no HTP skel found under $base")
            return null
        }
        Log.d(TAG, "QNN: using skel=$chosen in $base")
        skelDir = base
        return base
    }

    /**
     * Close delegates and reset state.
     * Call when shutting down the camera.
     */
    @JvmStatic
    fun close() {
        try { qnnDelegate?.close() } catch (_: Throwable) {}
        try { gpuDelegate?.close() } catch (_: Throwable) {}
        qnnDelegate = null
        gpuDelegate = null
        activeDelegate = null
        forcedDelegate = null
        skelDir = null
        initialized = false
    }

    /**
     * Reset only the active delegate tracking (keep force setting).
     * Useful when re-initializing models.
     */
    @JvmStatic
    fun resetDelegateState() {
        try { qnnDelegate?.close() } catch (_: Throwable) {}
        try { gpuDelegate?.close() } catch (_: Throwable) {}
        qnnDelegate = null
        gpuDelegate = null
        if (forcedDelegate == null) {
            activeDelegate = null
        }
        initialized = false
    }
}
