
import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
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
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.programminghut.realtime_object.R
import com.programminghut.realtime_object.ml.ModelStableV1
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.text.SimpleDateFormat


class MainActivity : AppCompatActivity() {

    // ---------- Thuộc tính ----------
    private lateinit var textureView: TextureView
    private lateinit var imageView  : ImageView

    private lateinit var cameraDevice: CameraDevice
    private lateinit var cameraManager: CameraManager
    private lateinit var handler     : Handler

    private lateinit var faceDetector: FaceDetectorHelper
    private lateinit var classifier  : ModelStableV1         // model drowsy/awake

    private lateinit var imageProcessor: ImageProcessor

    private val paintBox  = Paint().apply {
        style = Paint.Style.STROKE; strokeWidth = 5f; color = Color.BLUE
    }
    private val paintText = Paint().apply {
        style = Paint.Style.FILL; textSize = 60f; color = Color.WHITE
    }

    private val highThreshold = 0.7f

    // ---------- Lifecycle ----------
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        /* permission */
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), 1001)
            return
        }

        /* View */
        textureView = findViewById(R.id.textureView)
        imageView   = findViewById(R.id.imageView)

        /* Model load */
        faceDetector = FaceDetectorHelper(this)
        classifier   = ModelStableV1.newInstance(this)

        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        /* Camera thread */
        val handlerThread = HandlerThread("cameraThread").apply { start() }
        handler = Handler(handlerThread.looper)

        textureView.surfaceTextureListener = surfaceCallback
        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager
    }

    override fun onDestroy() {
        super.onDestroy()
        faceDetector.close()
        classifier.close()
    }

    // ---------- Camera ----------
    private val surfaceCallback = object : TextureView.SurfaceTextureListener {
        override fun onSurfaceTextureAvailable(st: SurfaceTexture, w: Int, h: Int) =
            openCamera()
        override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, w: Int, h: Int) = Unit
        override fun onSurfaceTextureDestroyed(st: SurfaceTexture) = false
        override fun onSurfaceTextureUpdated(st: SurfaceTexture) = processFrame()
    }

    @SuppressLint("MissingPermission")
    private fun openCamera() {
        val id = cameraManager.cameraIdList.first()      // lấy camera sau; đổi theo nhu cầu
        cameraManager.openCamera(id, object : CameraDevice.StateCallback() {
            override fun onOpened(device: CameraDevice) {
                cameraDevice = device
                val surface  = Surface(textureView.surfaceTexture)
                val req      = cameraDevice.createCaptureRequest(
                    CameraDevice.TEMPLATE_PREVIEW
                ).apply { addTarget(surface) }

                cameraDevice.createCaptureSession(
                    listOf(surface),
                    object : CameraCaptureSession.StateCallback() {
                        override fun onConfigured(sess: CameraCaptureSession) =
                            sess.setRepeatingRequest(req.build(), null, handler)
                        override fun onConfigureFailed(sess: CameraCaptureSession) = Unit
                    },
                    handler
                )
            }
            override fun onDisconnected(device: CameraDevice) = Unit
            override fun onError(device: CameraDevice, err: Int) = Unit
        }, handler)
    }

    // ---------- Xử lý mỗi frame ----------
    private fun processFrame() {
        val original = textureView.bitmap ?: return

        /* 1. Detect face */
        val detection = faceDetector.detect(original)
        if (detection == null) {
            imageView.setImageBitmap(original)           // không thấy mặt
            return
        }
        val faceRect = faceDetector.detectionToRect(detection, original)

        /* 2. Crop mặt & tiền xử lý */
        val cropped = Bitmap.createBitmap(
            original,
            faceRect.left, faceRect.top,
            faceRect.width(), faceRect.height()
        )
        var tensor = TensorImage(DataType.FLOAT32).apply { load(cropped) }
        tensor = imageProcessor.process(tensor)

        /* 3. Phân loại */
        val prob = classifier
            .process(tensor.tensorBuffer)
            .outputFeature0AsTensorBuffer.floatArray[0]

        val status = if (prob > highThreshold) "Drowsy" else "Awake"

        /* 4. Vẽ lên preview */
        val mutable = original.copy(Bitmap.Config.ARGB_8888, true)
        val canvas  = Canvas(mutable)

        canvas.drawRect(faceRect, paintBox)
        paintText.color = if (prob > highThreshold) Color.RED else Color.GREEN
        canvas.drawText(
            "$status : ${(prob * 100).toInt()}%",
            faceRect.left.toFloat(),
            (faceRect.top - 20).coerceAtLeast(60f),
            paintText
        )

        imageView.setImageBitmap(mutable)
    }
}
