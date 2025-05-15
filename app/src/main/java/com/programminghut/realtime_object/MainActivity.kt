package com.programminghut.realtime_object

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.renderscript.Element
import android.util.Log
import android.view.Surface
import android.view.TextureView
import android.widget.ImageView
import androidx.core.content.ContextCompat
import com.programminghut.realtime_object.ml.SsdMobilenetV11Metadata1
import com.programminghut.realtime_object.ml.Model
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import android.graphics.Rect


class MainActivity : AppCompatActivity() {

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
    lateinit var model:Model
    private var isNotifying = false
    private val notifyThreshold = 0.5f
    private val probabilityWindow = mutableListOf<Float>()
    private val smoothingWindowSize = 10 // average over last 10 frames
    private val highThreshold = 0.7f
    private var drowsyStartTime: Long = 0
    private val drowsyDurationThreshold = 2000L // 2 seconds

// lateinit var mediaPlayer: MediaPlayer


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        get_permission()

        labels = FileUtil.loadLabels(this, "labels.txt")
        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
            .build()

// imageProcessor = ImageProcessor.Builder().add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR)).build()
// model = SsdMobilenetV11Metadata1.newInstance(this)
        model = Model.newInstance(this)
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

                bitmap = textureView.bitmap!!
                var image = TensorImage(DataType.FLOAT32) // Sử dụng FLOAT32 thay vì mặc định
                image.load(bitmap)
                image = imageProcessor.process(image)

                val tensorBuffer = image.tensorBuffer
                val outputs = model.process(tensorBuffer)
                val probability = outputs.outputFeature0AsTensorBuffer.floatArray[0]

                val endTime = System.currentTimeMillis() // End timing
                val delay = endTime - startTime

                var mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                val canvas = Canvas(mutable)

                val h = mutable.height
                val w = mutable.width
                paint.textSize = h / 15f
                paint.strokeWidth = h / 85f

                val status = if (probability > 0.5) "Drowsy" else "Awake"
                paint.color = if (probability > 0.5) Color.RED else Color.GREEN

                // 1. Add probability to sliding window and compute average
                probabilityWindow.add(probability)
                if (probabilityWindow.size > smoothingWindowSize) {
                    probabilityWindow.removeAt(0)
                }
                val averageProbability = probabilityWindow.average().toFloat()

// 2. Determine state using smoothed average and higher threshold
                val isDrowsy = averageProbability > highThreshold
                val currentTime = System.currentTimeMillis()

// 3. Track duration of drowsy state
                if (isDrowsy) {
                    if (drowsyStartTime == 0L) {
                        drowsyStartTime = currentTime
                    }
                } else {
                    drowsyStartTime = 0L
                }

                val sustainedDrowsy = drowsyStartTime != 0L && (currentTime - drowsyStartTime >= drowsyDurationThreshold)

//                val status = if (sustainedDrowsy) "Drowsy" else "Awake"
//                paint.color = if (sustainedDrowsy) Color.RED else Color.GREEN

                paint.style = Paint.Style.FILL

                canvas.drawText("$status: ${(averageProbability * 100).toInt()}%", 50f, 100f, paint)

                paint.color = Color.WHITE
                paint.textSize = h / 50f
                canvas.drawText("Delay: ${delay}ms", 50f, 150f, paint)
                Log.d("InferenceTime", "Model inference delay: ${delay}ms")

                if (sustainedDrowsy && !isNotifying) {
                    isNotifying = true

                    Handler(Looper.getMainLooper()).postDelayed({
                        if (probability > notifyThreshold) {
                        // if (!mediaPlayer.isPlaying) {
                        // mediaPlayer.start()
                        // }
                            Log.d("DrowsyWarning", "Warning Sleep! $status: ${(probability * 100).toInt()}%")
                        }
                        isNotifying = false
                    }, 1500)
                }
                imageView.setImageBitmap(mutable)
            }
        }

        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager

    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()
// if (::mediaPlayer.isInitialized) {
// mediaPlayer.release()
// }
    }

    @SuppressLint("MissingPermission")
    fun open_camera(){
        cameraManager.openCamera(cameraManager.cameraIdList[0], object:CameraDevice.StateCallback(){
            override fun onOpened(p0: CameraDevice) {
                cameraDevice = p0

                var surfaceTexture = textureView.surfaceTexture
                var surface = Surface(surfaceTexture)

                var captureRequest = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                captureRequest.addTarget(surface)

                cameraDevice.createCaptureSession(listOf(surface), object: CameraCaptureSession.StateCallback(){
                    override fun onConfigured(p0: CameraCaptureSession) {
                        p0.setRepeatingRequest(captureRequest.build(), null, null)
                    }
                    override fun onConfigureFailed(p0: CameraCaptureSession) {
                    }
                }, handler)
            }

            override fun onDisconnected(p0: CameraDevice) {

            }

            override fun onError(p0: CameraDevice, p1: Int) {

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