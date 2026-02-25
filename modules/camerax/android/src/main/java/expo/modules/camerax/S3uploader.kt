package expo.modules.camerax

import android.content.Context
import android.util.Log
import com.amazonaws.auth.CognitoCachingCredentialsProvider
import com.amazonaws.mobileconnectors.s3.transferutility.TransferListener
import com.amazonaws.mobileconnectors.s3.transferutility.TransferState
import com.amazonaws.mobileconnectors.s3.transferutility.TransferUtility
import com.amazonaws.regions.Region
import com.amazonaws.regions.Regions
import com.amazonaws.services.s3.AmazonS3Client
import java.io.File
import java.io.FileWriter

class S3Uploader(private val context: Context) {

    private val credentialsProvider: CognitoCachingCredentialsProvider
    private val s3Client: AmazonS3Client
    private val transferUtility: TransferUtility

    init {
        // need this for automating credentials
        credentialsProvider = CognitoCachingCredentialsProvider(
            context,
            AwsConfig.COGNITO_IDENTITY_POOL_ID,
            Regions.fromName(AwsConfig.AWS_REGION)
        )


        s3Client = AmazonS3Client(credentialsProvider, Region.getRegion(Regions.fromName(AwsConfig.AWS_REGION)))

        transferUtility = TransferUtility.builder()
            .context(context)
            .s3Client(s3Client)
            .defaultBucket(AwsConfig.S3_BUCKET_NAME)
            .build()
    }

    /**
     * Upload session JSON files to S3
     * @param userId The user ID
     * @param timestamp The session timestamp (formatted for filename)
     * @param detailJson The detail session JSON content
     * @param summaryJson The summary JSON content
     */
    fun uploadSessionData(
        userId: String,
        timestamp: String,
        detailJson: String?,
        summaryJson: String?,
        onComplete: (success: Boolean, message: String) -> Unit
    ) {
        // Get the Cognito identity ID (unique per device)
        val identityId = credentialsProvider.identityId
        Log.d(TAG, "Cognito Identity ID: $identityId")

        // Track upload status
        var detailUploaded = detailJson == null  // If no detail JSON, mark as done
        var summaryUploaded = summaryJson == null  // If no summary JSON, mark as done
        var hasError = false
        val errorMessages = mutableListOf<String>()

        fun checkCompletion() {
            if (detailUploaded && summaryUploaded) {
                if (hasError) {
                    onComplete(false, "Upload failed: ${errorMessages.joinToString(", ")}")
                } else {
                    onComplete(true, "All files uploaded successfully")
                }
            }
        }

        // Upload detail JSON
        if (detailJson != null) {
            uploadJsonToS3(
                identityId = identityId,
                userId = userId,
                timestamp = timestamp,
                jsonContent = detailJson,
                fileType = "detail"
            ) { success, message ->
                detailUploaded = true
                if (!success) {
                    hasError = true
                    errorMessages.add("Detail: $message")
                }
                checkCompletion()
            }
        }

        // Upload summary JSON
        if (summaryJson != null) {
            uploadJsonToS3(
                identityId = identityId,
                userId = userId,
                timestamp = timestamp,
                jsonContent = summaryJson,
                fileType = "summary"
            ) { success, message ->
                summaryUploaded = true
                if (!success) {
                    hasError = true
                    errorMessages.add("Summary: $message")
                }
                checkCompletion()
            }
        }

        // If both are null, complete immediately
        if (detailJson == null && summaryJson == null) {
            onComplete(false, "No JSON data to upload")
        }
    }

    private fun uploadJsonToS3(
        identityId: String,
        userId: String,
        timestamp: String,
        jsonContent: String,
        fileType: String,
        onComplete: (success: Boolean, message: String) -> Unit
    ) {
        try {
            // Create temporary file
            val fileName = "session_${userId}_${timestamp}${if (fileType == "summary") "_summary" else ""}.json"
            val tempFile = File(context.cacheDir, fileName)

            // NEW
            val prettyJson = try {
                org.json.JSONObject(jsonContent).toString(4)
            } catch (e: Exception) {
                Log.w(TAG, "Could not pretty-print JSON, using original format: ${e.message}")
                jsonContent // Fallback to original if parsing fails
            }
            // Write JSON content to file
            FileWriter(tempFile).use { writer ->
                writer.write(prettyJson)
            }


            val date = timestamp.substring(0, 8)
            val time = timestamp.substring(9)

            val s3Key = "sessions/$userId/$date/$time/$fileName"

            Log.d(TAG, "Uploading $fileType JSON to S3: $s3Key")

            // Upload using TransferUtility
            val uploadObserver = transferUtility.upload(
                AwsConfig.S3_BUCKET_NAME,
                s3Key,
                tempFile
            )

            uploadObserver.setTransferListener(object : TransferListener {
                override fun onStateChanged(id: Int, state: TransferState) {
                    when (state) {
                        TransferState.COMPLETED -> {
                            Log.d(TAG, "$fileType JSON uploaded successfully: $s3Key")
                            tempFile.delete()  // Clean up temp file
                            onComplete(true, "Uploaded successfully")
                        }
                        TransferState.FAILED -> {
                            Log.e(TAG, "$fileType JSON upload failed: $s3Key")
                            tempFile.delete()  // Clean up temp file
                            onComplete(false, "Upload failed")
                        }
                        TransferState.CANCELED -> {
                            Log.w(TAG, "$fileType JSON upload canceled: $s3Key")
                            tempFile.delete()  // Clean up temp file
                            onComplete(false, "Upload canceled")
                        }
                        else -> {
                            // IN_PROGRESS, WAITING, etc.
                            Log.d(TAG, "$fileType JSON upload state: $state")
                        }
                    }
                }

                override fun onProgressChanged(id: Int, bytesCurrent: Long, bytesTotal: Long) {
                    val percentage = ((bytesCurrent.toFloat() / bytesTotal.toFloat()) * 100).toInt()
                    Log.d(TAG, "$fileType JSON upload progress: $percentage%")
                }

                override fun onError(id: Int, ex: Exception) {
                    Log.e(TAG, "$fileType JSON upload error: ${ex.message}", ex)
                    tempFile.delete()  // Clean up temp file
                    onComplete(false, "Error: ${ex.message}")
                }
            })

        } catch (e: Exception) {
            Log.e(TAG, "Error preparing $fileType JSON upload: ${e.message}", e)
            onComplete(false, "Error: ${e.message}")
        }
    }

    companion object {
        private const val TAG = "S3Uploader"
    }
}