package com.programminghut.realtime_object

import android.content.Context
import android.graphics.*
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.Detection
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.core.RotatedRect
import com.google.mediapipe.tasks.vision.facedetection.*
import kotlin.math.max
import kotlin.math.min

class FaceDetectorHelper(
    context: Context,
    modelAsset: String = "face_detection_front.tflite",
    private val scoreThreshold: Float = 0.5f
) {

    private val faceDetector: FaceDetector

    init {
        val options = FaceDetectorOptions.builder()
            .setScoreThreshold(scoreThreshold)
            .setRunningMode(RunningMode.IMAGE)     // chạy 1-shot (không stream)
            .setModelAssetPath(modelAsset)
            .build()

        faceDetector = FaceDetector.createFromOptions(context, options)
    }

    /** Trả về detection (khung) có score cao nhất, null nếu không có mặt */
    fun detect(bitmap: Bitmap): Detection? {
        val mpImage = BitmapImageBuilder(bitmap).build()
        val result = faceDetector.detect(mpImage)
        return result.detections().maxByOrNull { it.categories()[0].score() }
            ?.takeIf { it.categories()[0].score() >= scoreThreshold }
    }

    /** Chuyển bounding box chuẩn Mediapipe -> Rect pixel của bitmap gốc */
    fun detectionToRect(d: Detection, src: Bitmap): Rect {
        val r: RotatedRect = d.boundingBox()      // toạ độ gốc + kích thước tính theo px
        val left   = max(0f, r.centerX() - r.width() / 2f)
        val top    = max(0f, r.centerY() - r.height() / 2f)
        val right  = min(src.width.toFloat(),  r.centerX() + r.width() / 2f)
        val bottom = min(src.height.toFloat(), r.centerY() + r.height() / 2f)
        return Rect(left.toInt(), top.toInt(), right.toInt(), bottom.toInt())
    }

    fun close() = faceDetector.close()
}