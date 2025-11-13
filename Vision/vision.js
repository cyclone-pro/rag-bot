// vision.js
// Lightweight on-device vision using MediaPipe Tasks.
// No frames/landmarks leave the browser; only coarse events are emitted.

let faceLandmarker, faceDetector;
let visInterval = null;
let lastMouthOpen = null;
let visionUnavailableReason = null;
let multiFaceDisabledReason = null;
let mediapipeLoader = null;

const visState = {
  lookAwayFrames: 0,
  multiFaceFrames: 0
};

export function getMouthOpenApprox() {
  return lastMouthOpen;
}

export function getVisionState() {
  return {
    ...visState,
    visionUnavailableReason,
    multiFaceDisabledReason
  };
}

export async function startVision(videoEl, cfg, onEvent) {
  visionUnavailableReason = null;
  try {
    const { FaceLandmarker, FaceDetector, FilesetResolver } = await _loadVisionTasks();
    const resolver = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.10/wasm"
    );

    faceLandmarker = await FaceLandmarker.createFromOptions(resolver, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFacialTransformationMatrixes: true
    });
    faceDetector = await _maybeCreateFaceDetector(FaceDetector, resolver, cfg, onEvent);

    const periodMs = Math.max(50, Math.floor(1000 / (cfg.fps || 5)));

    const tick = async () => {
      const ts = performance.now();
      if (faceDetector) {
        const det = await faceDetector.detect(videoEl);
        const faces = det.detections ?? [];
        if (faces.length > 1) visState.multiFaceFrames++; else visState.multiFaceFrames = 0;
        if (visState.multiFaceFrames >= (cfg.multiFaceMinFrames || 5)) {
          onEvent?.({ multiFaceEvent: true });
          visState.multiFaceFrames = 0;
        }
      } else {
        visState.multiFaceFrames = 0;
      }

      const out = await faceLandmarker.detectForVideo(videoEl, ts);
      const hasFace = out.faceLandmarks && out.faceLandmarks.length > 0;
      let mouthOpen = null;
      if (hasFace) {
        const lm = out.faceLandmarks[0];
        mouthOpen = _mouthOpenAmount(lm);
        const yawDeg = _estimateYawDeg(lm);
        const frontFacing = Math.abs(yawDeg) <= (cfg.lookAwayMaxYawDeg || 25);
        if (!frontFacing) visState.lookAwayFrames++; else visState.lookAwayFrames = 0;
      } else {
        visState.lookAwayFrames++;
      }
      lastMouthOpen = mouthOpen;

      if (visState.lookAwayFrames >= (cfg.lookAwayMinFrames || 10)) {
        const bump = (cfg.lookAwayMinFrames || 10) / (cfg.fps || 5);
        onEvent?.({ lookAwayBumpSec: bump });
        visState.lookAwayFrames = 0;
      }
    };

    const runner = () => {
      tick().catch(err => console.error("vision tick failed", err));
    };

    runner();
    visInterval = setInterval(runner, periodMs);
    return;
  } catch (err) {
    console.warn("Vision init failed; falling back", err);
    await stopVision();
    visionUnavailableReason = err?.message || "MediaPipe assets blocked";
    lastMouthOpen = null;
    visState.lookAwayFrames = 0;
    visState.multiFaceFrames = 0;
    onEvent?.({ visionStatus: "unavailable", reason: visionUnavailableReason });
  }
}

export async function stopVision() {
  if (visInterval) clearInterval(visInterval);
  visInterval = null;
  try { await faceLandmarker?.close(); } catch {}
  try { await faceDetector?.close(); } catch {}
  faceLandmarker = null;
  faceDetector = null;
}

async function _loadVisionTasks() {
  if (!mediapipeLoader) {
    mediapipeLoader = import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.10")
      .catch(err => {
        mediapipeLoader = null;
        throw err;
      });
  }
  return mediapipeLoader;
}

async function _maybeCreateFaceDetector(FaceDetector, resolver, cfg, onEvent) {
  multiFaceDisabledReason = null;
  try {
    return await FaceDetector.createFromOptions(resolver, {
      baseOptions: {
        modelAssetPath: cfg?.modelPaths?.faceDetector
          || "https://storage.googleapis.com/mediapipe-models/face_detector/face_detector/float16/1/face_detector.task"
      }
    });
  } catch (err) {
    multiFaceDisabledReason = err?.message || "face detector model unavailable";
    console.warn("FaceDetector unavailable; multi-person monitoring disabled", err);
    onEvent?.({
      visionStatus: "degraded",
      reason: multiFaceDisabledReason,
      disabledFeatures: ["multi_face_detection"]
    });
    return null;
  }
}

// --- small helpers ---
function _mouthOpenAmount(landmarks) {
  const up = landmarks[13], lo = landmarks[14];
  if (!up || !lo) return 0;
  return Math.abs(up.y - lo.y);
}

function _estimateYawDeg(landmarks) {
  const L = landmarks[33], R = landmarks[263];
  const nose = landmarks[1] || landmarks[4];
  if (!L || !R || !nose) return 0;
  const eyeCx = (L.x + R.x)/2;
  const dx = nose.x - eyeCx;
  return (dx / 0.08) * 25;
}
