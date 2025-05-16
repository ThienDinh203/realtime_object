package com.programminghut.realtime_object

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import android.view.TextureView
import android.widget.Button
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.programminghut.realtime_object.ml.ModelStableV1
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp


class MainActivity : AppCompatActivity() {
    companion object {
        private var index = 0
    }

    lateinit var labels:List<String>
    var colors = listOf<Int>(
        Color.BLUE, Color.GREEN, Color.RED, Color.CYAN, Color.GRAY, Color.BLACK,
        Color.DKGRAY, Color.MAGENTA, Color.YELLOW, Color.RED)
    val paint = Paint()
    lateinit var imageProcessor:ImageProcessor
    lateinit var bitmap:Bitmap
    lateinit var imageView:ImageView
    lateinit var cameraDevice:CameraDevice
    lateinit var handler:Handler
    lateinit var cameraManager:CameraManager
    lateinit var textureView:TextureView
    lateinit var model:ModelStableV1
    private var isNotifying = false
    private val notifyThreshold = 0.5f
    private val probabilityWindow = mutableListOf<Float>()
    private val smoothingWindowSize = 10 // average over last 10 frames
    private val highThreshold = 0.4f
    private val frameStates = mutableListOf<String>()
    private var totalDrowsyCount = 0
    private var totalAwakeCount = 0
    private var useFrontCamera = true
// lateinit var mediaPlayer: MediaPlayer
    private fun loadDetectionCounts() {
        val prefs = getSharedPreferences("DetectionStats", MODE_PRIVATE)
        totalDrowsyCount = prefs.getInt("TotalDrowsy", 0)
        totalAwakeCount = prefs.getInt("TotalAwake", 0)
        Log.d("FrameStats", "Loaded totals – Drowsy: $totalDrowsyCount, Awake: $totalAwakeCount")
    }

//    private fun getFrontCameraId(): String? {
//        for (id in cameraManager.cameraIdList) {
//            val characteristics = cameraManager.getCameraCharacteristics(id)
//            val lensFacing = characteristics.get(android.hardware.camera2.CameraCharacteristics.LENS_FACING)
//            if (lensFacing == android.hardware.camera2.CameraCharacteristics.LENS_FACING_FRONT) {
//                return id
//            }
//        }
//        return null // If no front camera found
//    }

    private fun getCameraId(isFront: Boolean): String? {
        for (id in cameraManager.cameraIdList) {
            val characteristics = cameraManager.getCameraCharacteristics(id)
            val lensFacing = characteristics.get(android.hardware.camera2.CameraCharacteristics.LENS_FACING)
            if ((isFront && lensFacing == android.hardware.camera2.CameraCharacteristics.LENS_FACING_FRONT) ||
                (!isFront && lensFacing == android.hardware.camera2.CameraCharacteristics.LENS_FACING_BACK)) {
                return id
            }
        }
        return null
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        loadDetectionCounts()
        setContentView(R.layout.activity_main)
        val switchCameraButton = findViewById<Button>(R.id.switchCameraButton)
        switchCameraButton.setOnClickListener {
            cameraDevice.close() // Close current camera
            useFrontCamera = !useFrontCamera // Toggle camera
            open_camera() // Re-open with new selection
        }
        get_permission()

        labels = FileUtil.loadLabels(this, "labels.txt")
        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
            .build()

// imageProcessor = ImageProcessor.Builder().add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR)).build()
// model = SsdMobilenetV11Metadata1.newInstance(this)
        model = ModelStableV1.newInstance(this)
        val handlerThread = HandlerThread("videoThread")
        handlerThread.start()
        handler = Handler(handlerThread.looper)

        imageView = findViewById(R.id.imageView)

        textureView = findViewById(R.id.textureView)
        textureView.surfaceTextureListener = object:TextureView.SurfaceTextureListener{

            override fun onSurfaceTextureAvailable(p0: SurfaceTexture, p1: Int, p2: Int) {
                open_camera()
            }
            override fun onSurfaceTextureSizeChanged(p0: SurfaceTexture, p1: Int, p2: Int) {
            }

            override fun onSurfaceTextureDestroyed(p0: SurfaceTexture): Boolean {
                return false
            }
            override fun onSurfaceTextureUpdated(p0: SurfaceTexture) {
                // bitmap = textureView.bitmap!!
                // var image = TensorImage.fromBitmap(bitmap)
                // image = imageProcessor.process(image)

                val startTime = System.currentTimeMillis()
                val startTimeFormatted = java.text.SimpleDateFormat("HH:mm:ss.SSS", java.util.Locale.getDefault()).format(java.util.Date())
                Log.d("Timing", "Start frame at $startTimeFormatted")

                bitmap = textureView.bitmap!!
                var image = TensorImage(DataType.FLOAT32)
                image.load(bitmap)
                image = imageProcessor.process(image)

                val inferenceStartTime = System.currentTimeMillis()
                Log.d("Timing", "Start model inference: ${inferenceStartTime - startTime} ms after frame start")

                val tensorBuffer = image.tensorBuffer
                val outputs = model.process(tensorBuffer)
                val probability = outputs.outputFeature0AsTensorBuffer.floatArray[0]

                val endTime = System.currentTimeMillis() // End timing
                Log.d("Timing", "Model finished inference: ${endTime - inferenceStartTime} ms (inference time)")
                Log.d("Timing", "Total processing time: ${endTime - startTime} ms")
                val delay = endTime - startTime

                var mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                val canvas = Canvas(mutable)

                val h = mutable.height
                val w = mutable.width
                paint.textSize = h / 15f
                paint.strokeWidth = h / 85f

                val status = if (probability > highThreshold) "Drowsy" else "Awake"

                if (status == "Drowsy") {
                    totalDrowsyCount++
                } else {
                    totalAwakeCount++
                }
                Log.d("DinhThien", "Running Total – Drowsy: $totalDrowsyCount, Awake: $totalAwakeCount")


                frameStates.add(status)

                if (frameStates.size > 4) {
                    val drowsyCount = frameStates.count { it == "Drowsy" }
                    val awakeCount = frameStates.count { it == "Awake" }

                    val finalState = if (drowsyCount >= 3) "Drowsy" else "Awake"
                    Log.d("FinalState", "After 5 frames => Drowsy: $drowsyCount, Awake: $awakeCount → Final: $finalState")

                    frameStates.removeAt(0);
                }

                paint.color = if (probability > highThreshold) Color.RED else Color.GREEN
                paint.style = Paint.Style.FILL

                canvas.drawText("$status : ", 50f, 100f, paint)
//                       + " ${(probability * 100).toInt()}%", 50f, 100f, paint)

                // 1. Add probability to sliding window and compute average
                probabilityWindow.add(probability)
                if (probabilityWindow.size > smoothingWindowSize) {
                    probabilityWindow.removeAt(0)
                }
                val averageProbability = probabilityWindow.average().toFloat()

                val currentTime = java.text.SimpleDateFormat("HH:mm:ss.SSS", java.util.Locale.getDefault())
                    .format(java.util.Date())
                Log.d("FrameTime", "Current time: $currentTime")


//                val status = if (averageProbability > highThreshold) "Drowsy" else "Awake"
//                paint.color = if (averageProbability > highThreshold) Color.RED else Color.GREEN
//                paint.style = Paint.Style.FILL
//
//                canvas.drawText("$status = ${(averageProbability * 100).toInt()}%", 50f, 100f, paint)

                // TODO: Add media
//                if (sustainedDrowsy && !isNotifying) {
//                    isNotifying = true
//
//                    Handler(Looper.getMainLooper()).postDelayed({
//                        if (probability > notifyThreshold) {
//                        // if (!mediaPlayer.isPlaying) {
//                        // mediaPlayer.start()
//                        // }
//                            Log.d("DrowsyWarning", "Warning Sleep! $status: ${(probability * 100).toInt()}%")
//                        }
//                        isNotifying = false
//                    }, 1500)
//                }
                imageView.setImageBitmap(mutable)
            }
        }

        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager

    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()
        saveDetectionCounts()
// if (::mediaPlayer.isInitialized) {
// mediaPlayer.release()
// }
    }

    private fun saveDetectionCounts() {
        val prefs = getSharedPreferences("DetectionStats", MODE_PRIVATE)
        val editor = prefs.edit()
        editor.putInt("TotalDrowsy", totalDrowsyCount)
        editor.putInt("TotalAwake", totalAwakeCount)
        editor.apply()
    }

    @SuppressLint("MissingPermission")
    fun open_camera(){
//        val cameraId = getFrontCameraId()
        val cameraId = getCameraId(useFrontCamera)
        if (cameraId == null) {
            Log.e("CameraError", "Selected camera not found.")
            return
        }

        cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(p0: CameraDevice) {
                cameraDevice = p0

                val surfaceTexture = textureView.surfaceTexture
                val surface = Surface(surfaceTexture)

                val captureRequest = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                captureRequest.addTarget(surface)

                cameraDevice.createCaptureSession(listOf(surface), object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(p0: CameraCaptureSession) {
                        p0.setRepeatingRequest(captureRequest.build(), null, null)
                    }

                    override fun onConfigureFailed(p0: CameraCaptureSession) {
                        Log.e("CameraError", "Capture session configuration failed.")
                    }
                }, handler)
            }

            override fun onDisconnected(p0: CameraDevice) {}

            override fun onError(p0: CameraDevice, p1: Int) {
                Log.e("CameraError", "Camera device error: $p1")
            }
        }, handler)
    }

    fun get_permission(){
        if(ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED){
            requestPermissions(arrayOf(Manifest.permission.CAMERA), 101)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if(grantResults[0] != PackageManager.PERMISSION_GRANTED){
            get_permission()
        }
    }
}