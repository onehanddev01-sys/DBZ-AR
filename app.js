// ============================================================================
//  DBZ AR  -  ตรวจจับท่าทางมือด้วยกล้อง แล้วเล่น GIF + เสียง
//  ใช้โมเดล MediaPipe Gesture Recognizer (Google) ตัวเต็ม + เร่งด้วย GPU
// ============================================================================

import {
  GestureRecognizer,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ----------------------------------------------------------------------------
//  ⚙️  ตั้งค่าไฟล์ GIF + เสียง  (แก้ชื่อไฟล์ตรงนี้ได้เลย)
//      เอาไฟล์ไปวางในโฟลเดอร์ media/
// ----------------------------------------------------------------------------
const CONFIG = {
  punch: { gif: "media/punch.gif", sound: "media/punch.mp3" },
  fingers: { gif: "media/fingers.gif", sound: "media/fingers.mp3" },
  hands: { gif: "media/hands.gif", sound: "media/hands.mp3" },
  middle: { gif: "media/middle.gif", sound: "media/middle.mp3" },
  love: { gif: "media/love.gif", sound: "media/love.mp3" },
  good: { gif: "media/good.gif", sound: "media/good.mp3" },
  frieza: { gif: "media/frieza.gif", sound: "media/frieza.mp3" },
};

// แหล่งโมเดล / WASM (เปลี่ยนเวอร์ชันได้ถ้าต้องการ)
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";

// พารามิเตอร์การตรวจจับ
const STABLE_FRAMES = 4;   // ต้องเห็นท่าเดิมติดกันกี่เฟรมถึงจะนับ (กันตรวจพลาด)
const COOLDOWN_MS = 500; // หน่วงหลังเล่นจบก่อนเล่นซ้ำได้

// ขนาด GIF: สัดส่วนพื้นที่จอที่ให้ GIF กินได้ (1 = เต็มจอพอดี, 0.95 = เว้นขอบนิด)
const FILL_RATIO = 1.0;

// ----------------------------------------------------------------------------
//  อ้างอิง element
// ----------------------------------------------------------------------------
const video = document.getElementById("camera");
const backdrop = document.getElementById("backdrop");
const overlay = document.getElementById("overlay");
const statusEl = document.getElementById("status");
const startScr = document.getElementById("startScreen");
const startBtn = document.getElementById("startBtn");
const muteBtn = document.getElementById("muteBtn");
const volDown = document.getElementById("volDown");
const volUp = document.getElementById("volUp");
const volSlider = document.getElementById("volSlider");
const flipBtn = document.getElementById("flipBtn");
const debugBtn = document.getElementById("debugBtn");

// ----------------------------------------------------------------------------
//  สถานะ
// ----------------------------------------------------------------------------
let recognizer = null;
let stream = null;
let running = false;
let facingMode = "user"; // "user" = กล้องหน้า, "environment" = กล้องหลัง

let volume = 0.8;
let muted = false;

let playing = false;     // กำลังเล่นเอฟเฟกต์อยู่ไหม (เล่นทีละอัน)
let lastCandidate = "none";
let stableCount = 0;
let cooldownUntil = 0;
let armed = true;      // ต้องปล่อยมือ (ไม่มีท่า) ก่อนถึงจะยิงท่าเดิมซ้ำได้

// โหลดเสียงล่วงหน้า
const audios = {};
for (const key in CONFIG) {
  const a = new Audio(CONFIG[key].sound);
  a.preload = "auto";
  audios[key] = a;
}

// ----------------------------------------------------------------------------
//  โหลดโมเดล
// ----------------------------------------------------------------------------
async function initModel() {
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  const make = (delegate) =>
    GestureRecognizer.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate },
      runningMode: "VIDEO",
      numHands: 2,
    });
  try {
    recognizer = await make("GPU"); // ใช้ GPU ก่อน (แม่น + เร็ว)
  } catch (e) {
    console.warn("GPU ไม่ได้ ใช้ CPU แทน", e);
    recognizer = await make("CPU");
  }
}

// ----------------------------------------------------------------------------
//  กล้อง
// ----------------------------------------------------------------------------
async function startCamera() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  // กล้องหน้า = พลิกกระจก, กล้องหลัง = ปกติ
  video.classList.toggle("mirror", facingMode === "user");
}

// ----------------------------------------------------------------------------
//  ตัดสินใจว่าท่าตอนนี้คือท่าอะไร
//  MediaPipe ให้ชื่อท่า: Closed_Fist, Open_Palm, Pointing_Up,
//                        Thumb_Up, Thumb_Down, Victory, ILoveYou, None
// ----------------------------------------------------------------------------
function detect(results) {
  const names = (results.gestures || [])
    .map((g) => g[0]?.categoryName)
    .filter(Boolean);

  const hands = results.landmarks || [];

  const openPalms = names.filter((n) => n === "Open_Palm").length;
  const middleOnly = hands.some(isMiddleOnly);

  // Gesture เฉพาะก่อน
  if (names.includes("ILoveYou")) return "love";
  if (names.includes("Thumb_Up")) return "good";
  if (names.includes("Thumb_Down")) return "frieza";
  if (names.includes("Victory")) return "fingers";

  // Gesture ที่กว้างกว่าไว้ทีหลัง
  if (middleOnly) return "middle";
  if (names.includes("Closed_Fist")) return "punch";
  if (openPalms >= 2) return "hands";

  return "none";
}

// เช็คว่าเป็นท่า "นิ้วกลางนิ้วเดียว" จากจุดข้อต่อมือ (lm = 21 จุดของมือหนึ่งข้าง)
// นิ้วเหยียด = ปลายนิ้ว (tip) อยู่สูงกว่าข้อกลางนิ้ว (pip)  → ค่า y น้อยกว่า
function isMiddleOnly(lm) {
  if (!lm || lm.length < 21) return false;
  const up = (tip, pip) => lm[tip].y < lm[pip].y;
  const index = up(8, 6);
  const middle = up(12, 10);
  const ring = up(16, 14);
  const pinky = up(20, 18);
  return middle && !index && !ring && !pinky;
}

// ----------------------------------------------------------------------------
//  เล่นเอฟเฟกต์ (GIF วนลูป + เสียง, ความยาวยึดตามเสียง)
// ----------------------------------------------------------------------------
function trigger(key) {
  if (!CONFIG[key]) return;
  playing = true;

  // แสดงพื้นหลังดำ + GIF (ใส่ ?t= เพื่อบังคับให้ GIF เริ่มเล่นใหม่ตั้งแต่ต้น)
  backdrop.classList.add("show");
  overlay.src = CONFIG[key].gif + "?t=" + Date.now();
  overlay.classList.add("show");

  const a = audios[key];
  a.currentTime = 0;
  a.volume = muted ? 0 : volume;
  a.loop = false;
  a.onended = endPlayback;

  a.play().catch(() => {
    // ถ้าไม่มีไฟล์เสียง/เล่นไม่ได้ → โชว์ GIF 3 วิ แล้วซ่อน
    setTimeout(endPlayback, 3000);
  });
}

function endPlayback() {
  overlay.classList.remove("show");
  backdrop.classList.remove("show");
  overlay.removeAttribute("src");
  playing = false;
  cooldownUntil = performance.now() + COOLDOWN_MS;
}

// เช็คขนาดจริงของ GIF แล้วคำนวณให้พอดีจอ (เห็นทั้งภาพ ไม่โดนตัดขอบ)
function fitOverlay() {
  const w = overlay.naturalWidth;
  const h = overlay.naturalHeight;
  if (!w || !h) return;
  const scale = Math.min(window.innerWidth / w, window.innerHeight / h) * FILL_RATIO;
  overlay.style.width = w * scale + "px";
  overlay.style.height = h * scale + "px";
}
overlay.addEventListener("load", fitOverlay); // คำนวณใหม่ทุกครั้งที่เปลี่ยน GIF
window.addEventListener("resize", fitOverlay); // หมุนจอ/ปรับขนาดหน้าต่าง

// ----------------------------------------------------------------------------
//  ลูปตรวจจับ
// ----------------------------------------------------------------------------
function loop() {
  if (!running) return;

  if (recognizer && video.readyState >= 2) {
    const now = performance.now();
    const results = recognizer.recognizeForVideo(video, now);
    const cand = detect(results);

    statusEl.textContent = cand === "none" ? "—" : cand;

    // นับความนิ่งของท่า
    if (cand !== "none" && cand === lastCandidate) stableCount++;
    else stableCount = cand === "none" ? 0 : 1;
    lastCandidate = cand;

    if (cand === "none") armed = true; // ปล่อยมือแล้ว ยิงซ้ำได้

    if (
      cand !== "none" &&
      stableCount >= STABLE_FRAMES &&
      !playing &&
      armed &&
      now >= cooldownUntil
    ) {
      trigger(cand);
      armed = false;
      stableCount = 0;
    }
  }

  requestAnimationFrame(loop);
}

// ----------------------------------------------------------------------------
//  ปุ่มเริ่ม
// ----------------------------------------------------------------------------
startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  startBtn.textContent = "กำลังโหลดโมเดล...";
  try {
    await initModel();
    await startCamera();
    running = true;
    startScr.classList.add("hidden");
    loop();
  } catch (e) {
    alert("เริ่มไม่สำเร็จ: " + e.message);
    startBtn.disabled = false;
    startBtn.textContent = "แตะเพื่อเริ่ม";
  }
});

// ----------------------------------------------------------------------------
//  ปุ่มควบคุมเสียง
// ----------------------------------------------------------------------------
function updateMuteIcon() {
  muteBtn.textContent = muted || volume === 0 ? "🔇" : "🔊";
}
function setVolume(v) {
  volume = Math.min(1, Math.max(0, v));
  volSlider.value = Math.round(volume * 100);
  if (volume > 0) muted = false;
  updateMuteIcon();
}

volSlider.addEventListener("input", () => setVolume(volSlider.value / 100));
volDown.addEventListener("click", () => setVolume(volume - 0.1));
volUp.addEventListener("click", () => setVolume(volume + 0.1));
muteBtn.addEventListener("click", () => {
  muted = !muted;
  updateMuteIcon();
});

// ----------------------------------------------------------------------------
//  สลับกล้อง / ดีบัก
// ----------------------------------------------------------------------------
flipBtn.addEventListener("click", async () => {
  facingMode = facingMode === "user" ? "environment" : "user";
  try {
    await startCamera();
  } catch (e) {
    alert("สลับกล้องไม่ได้: " + e.message);
  }
});

debugBtn.addEventListener("click", () => statusEl.classList.toggle("show"));

// ----------------------------------------------------------------------------
//  ทดสอบด้วยคีย์บอร์ด 1 / 2 / 3 (ไว้ลองโดยไม่ต้องทำท่า)
// ----------------------------------------------------------------------------
window.addEventListener("keydown", (e) => {
  if (playing) return;
  if (e.key === "1") trigger("punch");
  if (e.key === "2") trigger("fingers");
  if (e.key === "3") trigger("hands");
  if (e.key === "4") trigger("middle");
  if (e.key === "5") trigger("love");
  if (e.key === "6") trigger("good");
  if (e.key === "7") trigger("frieza");
});

updateMuteIcon();
