const video = document.getElementById("video");
const snapshotCanvas = document.getElementById("snapshotCanvas");
const startCameraBtn = document.getElementById("startCameraBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");
const recordGestureBtn = document.getElementById("recordGestureBtn");
const recordVoiceBtn = document.getElementById("recordVoiceBtn");
const speakTextBtn = document.getElementById("speakTextBtn");
const resultBox = document.getElementById("resultBox");
const statusText = document.getElementById("statusText");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const imageInput = document.getElementById("imageInput");
const uploadImageBtn = document.getElementById("uploadImageBtn");

let stream = null;
let recognition = null;
let isRecordingVoice = false;

const SEQ_LEN = 30;
const CAPTURE_INTERVAL_MS = 80;

function setResultText(text) {
  resultBox.textContent = text || "尚未產生文字";
}

function setStatus(text) {
  statusText.textContent = text || "";
}

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setResultText("此瀏覽器不支援攝影機功能");
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false
    });

    video.srcObject = stream;

    cameraPlaceholder.style.display = "none";
    startCameraBtn.disabled = true;
    stopCameraBtn.disabled = false;
    recordGestureBtn.disabled = false;

    setStatus("鏡頭已開啟");
  } catch (error) {
    setResultText("無法開啟鏡頭：" + (error.message || error.name || "未知錯誤"));
    setStatus("請確認瀏覽器已允許相機權限");
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }

  video.srcObject = null;
  cameraPlaceholder.style.display = "flex";

  startCameraBtn.disabled = false;
  stopCameraBtn.disabled = true;
  recordGestureBtn.disabled = true;

  setStatus("鏡頭已關閉");
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function captureFrame() {
  const width = video.videoWidth;
  const height = video.videoHeight;

  if (!width || !height) {
    return null;
  }

  snapshotCanvas.width = width;
  snapshotCanvas.height = height;

  const ctx = snapshotCanvas.getContext("2d");
  ctx.drawImage(video, 0, 0, width, height);

  return snapshotCanvas.toDataURL("image/jpeg", 0.85);
}

async function recordGestureSequence() {
  if (!stream) {
    setStatus("請先開啟鏡頭");
    return;
  }

  recordGestureBtn.disabled = true;
  setResultText("正在收集 30 幀，請完成手語動作");
  setStatus("錄製中...");

  const frames = [];

  for (let i = 0; i < SEQ_LEN; i++) {
    const frame = captureFrame();

    if (frame) {
      frames.push(frame);
    }

    setStatus(`錄製中：${i + 1} / ${SEQ_LEN}`);
    await sleep(CAPTURE_INTERVAL_MS);
  }

  setResultText("辨識中請稍後");
  setStatus("正在送出序列資料...");

  try {
    const response = await fetch("/predict_sequence", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ frames })
    });

    const data = await response.json();

    setResultText(data.result || "沒有結果");
    setStatus(data.ok ? "辨識完成" : "辨識失敗");
  } catch (error) {
    setResultText("送出辨識失敗：" + (error.message || "未知錯誤"));
    setStatus("請稍後再試");
  }

  recordGestureBtn.disabled = false;
}

async function uploadImagePredict() {
  const file = imageInput.files[0];

  if (!file) {
    setStatus("請先選擇圖片");
    return;
  }

  setResultText("圖片辨識中請稍後");
  setStatus("正在讀取圖片...");

  const reader = new FileReader();

  reader.onload = async function(event) {
    try {
      const imageData = event.target.result;

      const response = await fetch("/predict_image", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ image: imageData })
      });

      const data = await response.json();

      setResultText(data.result || "沒有結果");
      setStatus(data.ok ? "辨識完成" : "辨識失敗");
    } catch (error) {
      setResultText("圖片辨識失敗：" + (error.message || "未知錯誤"));
      setStatus("請稍後再試");
    }
  };

  reader.readAsDataURL(file);
}

function getRecognitionCtor() {
  return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

function setupRecognition() {
  const RecognitionCtor = getRecognitionCtor();

  if (!RecognitionCtor) {
    return null;
  }

  const rec = new RecognitionCtor();
  rec.lang = "zh-TW";
  rec.interimResults = true;
  rec.continuous = false;

  rec.onstart = () => {
    isRecordingVoice = true;
    recordVoiceBtn.textContent = "錄音中...";
    setStatus("請開始說話");
  };

  rec.onresult = (event) => {
    let transcript = "";

    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }

    transcript = transcript.trim();

    if (transcript) {
      setResultText(transcript);
    }
  };

  rec.onerror = (event) => {
    setStatus("錄音功能發生問題：" + (event.error || "未知錯誤"));
  };

  rec.onend = () => {
    isRecordingVoice = false;
    recordVoiceBtn.textContent = "語音轉文字";
  };

  return rec;
}

function toggleVoiceRecording() {
  if (!recognition) {
    setStatus("目前瀏覽器不支援語音辨識，建議使用 Chrome");
    return;
  }

  if (isRecordingVoice) {
    recognition.stop();
    return;
  }

  try {
    recognition.start();
  } catch (error) {
    setStatus("無法開始錄音：" + (error.message || "未知錯誤"));
  }
}

function speakCurrentText() {
  const text = resultBox.textContent.trim();

  if (!text || text === "尚未產生文字" || text.includes("請稍後")) {
    setStatus("目前沒有可播放的文字");
    return;
  }

  if (!("speechSynthesis" in window)) {
    setStatus("目前瀏覽器不支援語音播放功能");
    return;
  }

  window.speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "zh-TW";
  utterance.rate = 1;
  utterance.pitch = 1;

  utterance.onstart = () => setStatus("正在播放文字語音...");
  utterance.onend = () => setStatus("語音播放完成");
  utterance.onerror = () => setStatus("語音播放失敗");

  window.speechSynthesis.speak(utterance);
}

recognition = setupRecognition();

startCameraBtn.addEventListener("click", startCamera);
stopCameraBtn.addEventListener("click", stopCamera);
recordGestureBtn.addEventListener("click", recordGestureSequence);
uploadImageBtn.addEventListener("click", uploadImagePredict);
recordVoiceBtn.addEventListener("click", toggleVoiceRecording);
speakTextBtn.addEventListener("click", speakCurrentText);
