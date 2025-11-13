// audio.js
// Web Audio API energy + simple VAD heuristic. No audio leaves the device.

let audioCtx, analyser, timeBuf, timer = null;

// ring buffer of small buckets
const buckets = [];

export function startAudio(stream, cfg, getMouthOpen, onEvent) {
  stopAudio();

  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const src = audioCtx.createMediaStreamSource(stream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 2048;
  src.connect(analyser);
  timeBuf = new Float32Array(analyser.fftSize);

  const bucketMs = cfg.rmsFrameMs || 100;

  const tick = () => {
    analyser.getFloatTimeDomainData(timeBuf);
    let sum = 0; for (let i=0;i<timeBuf.length;i++) sum += timeBuf[i]*timeBuf[i];
    const rms = Math.sqrt(sum / timeBuf.length);
    const vad = rms > (cfg.vadEnergyThreshold || 0.02);

    const mouthSample = typeof getMouthOpen === "function" ? getMouthOpen() : undefined;
    const mouthOpenLikely = typeof mouthSample === "number"
      ? mouthSample > (cfg.mouthOpenThreshold || 0.018)
      : undefined;

    buckets.push({ rms, vad, mouthOpen: mouthOpenLikely });
    if (buckets.length > 64) buckets.shift();

    // background speech: consecutive vad true (sustained)
    const consecVad = _countConsecFromEnd(buckets, b => b.vad);
    if (consecVad >= (cfg.bgSpeechMinBuckets || 4)) {
      onEvent?.({ bgSpeechEvent: true });
      buckets.length = 0; // cooldown
    }

    // another voice while candidate silent (vad true + mouth closed sustained)
    const consecBgWhileMouthClosed = _countConsecFromEnd(
      buckets,
      b => b.vad && b.mouthOpen === false
    );
    if (consecBgWhileMouthClosed >= (cfg.silentWhileSpeechMinBuckets || 8)) {
      onEvent?.({ anotherVoiceWhileSilentEvent: true });
      buckets.length = 0;
    }

    timer = setTimeout(tick, bucketMs);
  };
  tick();
}

export function stopAudio() {
  if (timer) clearTimeout(timer);
  timer = null;
  try { audioCtx && audioCtx.close(); } catch {}
  audioCtx = null;
  analyser = null;
  timeBuf = null;
  buckets.length = 0;
}

// helpers
function _countConsecFromEnd(arr, pred) {
  let c=0; for (let i=arr.length-1;i>=0;i--) { if (pred(arr[i])) c++; else break; } return c;
}
