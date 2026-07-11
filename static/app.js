let progressAngle = 0;
let currentHeartRate = 0;
let ecgPhase = 0;
let animationFrameId = null;

// الگوریتم سنتز شکل موج استاندارد ECG (P-QRS-T) بر اساس ضربان واقعی rPPG
function getECGPoint(phase) {
  const p = phase % (2 * Math.PI);
  let val = 0;

  // موج P
  if (p > 0.2 && p < 0.6) {
    val += 0.12 * Math.sin((Math.PI * (p - 0.2)) / 0.4);
  }

  // کمپلکس QRS
  if (p >= 0.85 && p < 0.9) {
    val -= 0.15 * Math.sin((Math.PI * (p - 0.85)) / 0.05);
  }
  if (p >= 0.9 && p < 0.98) {
    val += 1.0 * Math.sin((Math.PI * (p - 0.9)) / 0.08);
  }
  if (p >= 0.98 && p < 1.05) {
    val -= 0.25 * Math.sin((Math.PI * (p - 0.98)) / 0.07);
  }

  // موج T
  if (p > 1.4 && p < 2.0) {
    val += 0.25 * Math.sin((Math.PI * (p - 1.4)) / 0.6);
  }

  return val;
}

const socket = io.connect(window.location.origin || "http://localhost:5000", {
  transports: ["websocket", "polling"],
  reconnection: true,
  reconnectionAttempts: 10,
});

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const faceCanvas = document.getElementById("faceCanvas");
const faceCtx = faceCanvas.getContext("2d");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const resetBtn = document.getElementById("resetBtn");
const hrDisplay = document.getElementById("hrDisplay");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const samplesText = document.getElementById("samplesText");
const respDisplay = document.getElementById("respDisplay");
const bpDisplay = document.getElementById("bpDisplay");
const spo2Display = document.getElementById("spo2Display");
const calibDisplay = document.getElementById("calibDisplay");
const calibText = document.getElementById("calibText");
const sigCanvas = document.getElementById("signalCanvas");
const sigCtx = sigCanvas.getContext("2d");
const statusMsg = document.getElementById("status-message");

let stream = null,
  isRunning = false,
  frameInterval = null;
let ecgBuffer = Array(120).fill(0);

function resizeCanvases() {
  const wrap = faceCanvas.parentElement;
  faceCanvas.width = wrap.offsetWidth;
  faceCanvas.height = wrap.offsetHeight;
  sigCanvas.width = sigCanvas.parentElement.offsetWidth;
  sigCanvas.height = 60;
}
resizeCanvases();
window.addEventListener("resize", resizeCanvases);

// تابع هوشمند رسم چهره، بلر کردن پس‌زمینه و مدیریت رنگ دایره بر اساس فریم و نویز حرکتی
function drawFaceWithSpinner(
  face,
  hr,
  confidenceScore,
  currentSamples,
  calibrationStatus,
) {
  faceCtx.clearRect(0, 0, faceCanvas.width, faceCanvas.height);
  if (!face) return;

  const scaleX = faceCanvas.width / face.fw;
  const scaleY = faceCanvas.height / face.fh;
  const centerX = (face.x + face.w / 2) * scaleX;
  const centerY = (face.y + face.h / 2) * scaleY;

  // ۳. افزایش قطر دایره به میزان ۱.۳ برابر اندازه صورت جهت پوشش بهتر و زیباتر
  const radius = (Math.max(face.w * scaleX, face.h * scaleY) / 2) * 1.3;

  // ۴. بلر کردن و تاریک کردن پس‌زمینه خارج از دایره صورت (Spotlight Effect)
  faceCtx.save();
  faceCtx.fillStyle = "rgba(43, 43, 43, 0.83)"; // لایه تاریک‌کننده محیط اطراف
  faceCtx.filter = "blur(4px)"; // ایجاد افکت بلر نرم روی پس‌زمینه خارج از دایره
  faceCtx.beginPath();
  faceCtx.rect(0, 0, faceCanvas.width, faceCanvas.height);
  // رسم دایره معکوس برای خالی کردن فضای صورت از بلر
  faceCtx.arc(centerX, centerY, radius, 0, Math.PI * 2, true);
  faceCtx.fill();
  faceCtx.restore();

  const numLines = 60;
  const lineLength = 8;
  const innerRadius = radius + 2;
  const progressRatio = Math.min((currentSamples || 0) / 150, 1.0);
  const activeLinesCount = Math.floor(progressRatio * numLines);

  // تعیین رنگ دایره :
  let activeColor = "#f59e0b"; // پیش‌فرض نارنجی (در حال کالیبراسیون)

  // ۲. اگه نویز حرکت شدید بود دایره قرمز بشه
  if (calibrationStatus === "moving") {
    activeColor = "#ef4444";
  }
  // ۱. دایره دور سر تا کامل نشدن ۱۵۰ فریم سبز نشه
  else if (currentSamples >= 150) {
    activeColor = "#10b981";
  }

  faceCtx.save();
  faceCtx.translate(centerX, centerY);
  progressAngle = (progressAngle + 0.5) % 360;
  faceCtx.rotate((progressAngle * Math.PI) / 180);

  for (let i = 0; i < numLines; i++) {
    const angle = (i * 2 * Math.PI) / numLines - Math.PI / 2;
    const sx = innerRadius * Math.cos(angle);
    const sy = innerRadius * Math.sin(angle);
    const ex = (innerRadius + lineLength) * Math.cos(angle);
    const ey = (innerRadius + lineLength) * Math.sin(angle);

    faceCtx.beginPath();
    faceCtx.moveTo(sx, sy);
    faceCtx.lineTo(ex, ey);
    faceCtx.lineWidth = 2;
    faceCtx.lineCap = "round";

    if (i < activeLinesCount || progressRatio >= 1.0) {
      faceCtx.strokeStyle = activeColor;
    } else {
      faceCtx.strokeStyle = "rgba(255, 255, 255, 0)";
    }
    faceCtx.stroke();
  }
  faceCtx.restore();
}

function animateECG() {
  if (!isRunning) {
    ecgBuffer.shift();
    ecgBuffer.push((Math.random() - 0.5) * 0.02);
  } else {
    const hr = currentHeartRate > 0 ? currentHeartRate : 70;
    const bps = hr / 60.0;
    const step = (bps * 2 * Math.PI) / 60;

    ecgPhase = (ecgPhase + step) % (2 * Math.PI);
    const nextPoint = getECGPoint(ecgPhase);

    ecgBuffer.shift();
    ecgBuffer.push(nextPoint);
  }

  drawECGChart();
  animationFrameId = requestAnimationFrame(animateECG);
}

function drawECGChart() {
  sigCtx.clearRect(0, 0, sigCanvas.width, sigCanvas.height);
  const stepWidth = sigCanvas.width / (ecgBuffer.length - 1);

  sigCtx.strokeStyle = "rgba(255, 255, 255, 0.05)";
  sigCtx.lineWidth = 1;
  sigCtx.beginPath();
  sigCtx.moveTo(0, sigCanvas.height / 2);
  sigCtx.lineTo(sigCanvas.width, sigCanvas.height / 2);
  sigCtx.stroke();

  sigCtx.beginPath();
  sigCtx.strokeStyle = isRunning ? "#10b981" : "#4a4a5a";
  sigCtx.lineWidth = 2.0;
  if (isRunning) {
    sigCtx.shadowColor = "#10b981";
    sigCtx.shadowBlur = 5;
  }

  ecgBuffer.forEach((v, i) => {
    const x = i * stepWidth;
    const y = sigCanvas.height / 2 - (v * (sigCanvas.height - 20)) / 2;
    i === 0 ? sigCtx.moveTo(x, y) : sigCtx.lineTo(x, y);
  });
  sigCtx.stroke();
  sigCtx.shadowBlur = 0;
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 320, height: 240, facingMode: "user" },
    });
    video.srcObject = stream;
    await video.play();
    overlay.classList.add("hidden");
    setStatus("ready", "دوربین فعال");
    return true;
  } catch (err) {
    setStatus("error", "خطای دسترسی به دوربین");
    return false;
  }
}

function sendFrame() {
  if (!isRunning || !video.srcObject) return;
  const c = document.createElement("canvas");
  c.width = 320;
  c.height = 240;
  c.getContext("2d").drawImage(video, 0, 0, 320, 240);
  socket.emit("frame", {
    image: c.toDataURL("image/jpeg", 0.6),
    sid: socket.id,
  });
}

function startSending() {
  if (frameInterval) clearInterval(frameInterval);
  isRunning = true;
  frameInterval = setInterval(sendFrame, 33);
  setStatus("processing", "در حال پردازش");
}

function stopSending() {
  isRunning = false;
  if (frameInterval) {
    clearInterval(frameInterval);
    frameInterval = null;
  }
  faceCtx.clearRect(0, 0, faceCanvas.width, faceCanvas.height);
  setStatus("idle", "متوقف شد");
}

function resetAll() {
  currentHeartRate = 0;
  ecgBuffer.fill(0);
  hrDisplay.textContent = "0";
  hrDisplay.style.color = "#fff";
  respDisplay.textContent = "--";
  bpDisplay.textContent = "--/--";
  spo2Display.textContent = "--";
  calibDisplay.innerHTML = '<i class="fas fa-hourglass-half"></i>';
  calibDisplay.style.color = "#f59e0b";
  calibText.textContent = "در حال سنجش";
  samplesText.textContent = "";
  statusMsg.textContent = "";
  faceCtx.clearRect(0, 0, faceCanvas.width, faceCanvas.height);
  setStatus("idle", "ریست شد");
}

socket.on("connect", () => setStatus("ready", "متصل به سرور"));
socket.on("disconnect", () => setStatus("error", "قطع ارتباط"));

// بخش دریافت داده‌ها از سوکت در فایل app.js را به این شکل اصلاح کنید:
socket.on("hr_update", function (data) {
  if (data.status === "ok" || data.status === "waiting") {
    // تشخیص هوشمند حرکت سر بر اساس انواع خروجی‌های احتمالی سرور
    let isMoving = false;
    if (
      data.motion_level === "moving" ||
      data.motion_level === true ||
      data.is_moving === true ||
      data.motion === "moving" ||
      (data.motion_score && data.motion_score > 0.5) // اگر سرور مقدار عددی نویز حرکتی بفرستد
    ) {
      isMoving = true;
    }

    // ارسال وضعیت حرکت به تابع رسم دایره
    drawFaceWithSpinner(
      data.face,
      data.hr,
      data.hr_confidence || 0,
      data.samples || 0,
      isMoving ? "moving" : "stable",
    );

    // مدیریت نمایش اطلاعات ضربان و کالیبراسیون
    if (data.hr > 0 && !isMoving) {
      // اگر حرکت شدید نباشد ضربان نشان داده شود
      currentHeartRate = data.hr;
      hrDisplay.textContent = data.hr;
      hrDisplay.style.color = "#10b981";
      respDisplay.textContent = data.resp || "16";

      if (data.hr && data.spo2) {
        let sys = Math.round(115 + (data.hr - 70) * 0.35);
        let dia = Math.round(75 + (data.hr - 70) * 0.2);
        bpDisplay.textContent = `${sys}/${dia}`;
      }

      spo2Display.textContent = `${data.spo2}%`;
      calibDisplay.innerHTML = '<i class="fas fa-check-circle"></i>';
      calibDisplay.style.color = "#10b981";
      calibText.textContent = "کامل شده";
    } else if (isMoving) {
      // در صورت حرکت شدید، ضربان موقتاً نارنجی یا متوقف شود
      hrDisplay.style.color = "#ef4444";
      calibDisplay.innerHTML = '<i class="fas fa-triangle-exclamation"></i>';
      calibDisplay.style.color = "#ef4444";
      calibText.textContent = "حرکت شدید صورت!";
    } else {
      currentHeartRate = 0;
      hrDisplay.textContent = "0";
      hrDisplay.style.color = "#fff";
      respDisplay.textContent = "--";
      bpDisplay.textContent = "--/--";
      spo2Display.textContent = "--";
      calibDisplay.innerHTML = '<i class="fas fa-hourglass-half"></i>';
      calibDisplay.style.color = "#f59e0b";
      calibText.textContent = "در حال سنجش";
    }

    if (data.samples) {
      samplesText.textContent = `فریم: ${data.samples}`;
    }
  }
});

function setStatus(state, text) {
  const map = {
    idle: "bg-[#6b7280]",
    ready: "bg-[#10b981]",
    processing: "bg-[#f59e0b] animate-pulse",
    error: "bg-[#ef4444]",
  };
  statusDot.className = `w-1.75 h-1.75 rounded-full shrink-0 ${map[state] || "bg-[#6b7280]"}`;
  statusText.textContent = text;
}

startBtn.addEventListener("click", async () => {
  if (!stream) {
    const ok = await startCamera();
    if (!ok) return;
  }
  resetAll();
  try {
    const response = await fetch("/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sid: socket.id,
        patient_id: "anonymous",
        reference_spo2: 98.0,
        reference_hr: 72.0,
      }),
    });

    if (response.ok) {
      startSending();
      startBtn.disabled = true;
      stopBtn.disabled = false;
    } else {
      setStatus("error", "خطای سرور");
    }
  } catch (err) {
    setStatus("error", "عدم اتصال");
  }
});

stopBtn.addEventListener("click", async () => {
  try {
    await fetch("/stop", { method: "POST" });
  } catch (err) {}
  stopSending();
  startBtn.disabled = false;
  stopBtn.disabled = true;
});

resetBtn.addEventListener("click", async () => {
  resetAll();
  if (isRunning) {
    try {
      await fetch("/stop", { method: "POST" });
    } catch (err) {}
  }
  stopSending();
  startBtn.disabled = false;
  stopBtn.disabled = true;
});

let startTime;
const LIMIT = 60000; // 60 seconds

socket.on("start_confirmed", () => {
  startTime = Date.now();
  // شروع تایمر بصری برای کاربر
});

// گوش دادن به دستور توقف از سرور
// دریافت دستور توقف اجباری از سرور
socket.on("force_stop", function (data) {
  console.log("Stopping due to: " + data.reason);

  // 1. متوقف کردن دوربین در مرورگر
  if (localStream) {
    localStream.getTracks().forEach((track) => track.stop());
  }

  // 2. تغییر وضعیت دکمه‌ها
  document.getElementById("start-btn").disabled = false;
  document.getElementById("stop-btn").disabled = true;

  // 3. نمایش پیام به کاربر
  alert("Time is up! " + data.message);

  // 4. قرمز کردن نشانگر وضعیت
  const indicator = document.getElementById("status-indicator");
  indicator.style.backgroundColor = "#ff4b2b";
  document.getElementById("status-text").innerText = "Session Ended";
});

setStatus("idle", "آماده");
animateECG();
