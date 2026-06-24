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

// ----------------------------------------------------------------------------
//  🔥 Fusion: เล่นวิดีโอ .mp4 ต่อคิว (อยู่ในโฟลเดอร์ "media duo/")
//     เล่น "fusion dance" ก่อนเสมอ → แล้วสุ่ม 1 สาขา
//     fat มี 2 คลิป (fail → form), ที่เหลือมีคลิปเดียว
// ----------------------------------------------------------------------------
const FUSION = {
  dance: "media duo/fusion dance.mp4",
  branches: {
    gogeta: ["media duo/gogeta/gogeta.mp4"],
    fat: ["media duo/fat/fat fail.mp4", "media duo/fat/fat form.mp4"],
    thin: ["media duo/thin/thin form.mp4"],
  },
};

// ----------------------------------------------------------------------------
//  👊 Fist Bump: กำหมัดสองกำ (สองคน) ชนกันกลางจอ → เล่นวิดีโอ (ฟังก์ชันสองคน)
//     ⬅️  วางไฟล์วิดีโอที่ path ด้านล่าง (ยังไม่มีไฟล์ก็รันได้ แค่จะไม่เล่นอะไร)
//     ใส่ได้หลายคลิป จะเล่นต่อคิวให้ (เหมือน fat fail → fat form)
// ----------------------------------------------------------------------------
const FISTBUMP = {
  videos: ["media duo/fist bump/fist bump.mp4"],
};

// ----------------------------------------------------------------------------
//  🌀 Kamehameha: ฝ่ามือเปิด "สามมือเรียงแนวนอน" → เล่นวิดีโอ (ฟังก์ชันหลายคน)
//     ⬅️  วางไฟล์วิดีโอที่ path ด้านล่าง (ยังไม่มีไฟล์ก็รันได้ แค่จะไม่เล่นอะไร)
//     ใส่ได้หลายคลิป จะเล่นต่อคิวให้
// ----------------------------------------------------------------------------
const KAMEHAMEHA = {
  videos: ["media duo/double kamehameha/double kamehameha.mp4"],
};

// ----------------------------------------------------------------------------
//  🐉 Father-Son Kamehameha: ฝ่ามือเปิด "สามมือเรียงแนวตั้ง" → เล่นวิดีโอ
//     ⬅️  วางไฟล์วิดีโอที่ path ด้านล่าง (ยังไม่มีไฟล์ก็รันได้ แค่จะไม่เล่นอะไร)
//     ใส่ได้หลายคลิป จะเล่นต่อคิวให้
// ----------------------------------------------------------------------------
const FATHERSON = {
  videos: ["media duo/father son kamehameha/father son kamehameha.mp4"],
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

// Fusion: ระยะห่างปลายนิ้วชี้สองนิ้วที่ถือว่า "จ่อกัน" (0-1 ของขนาดเฟรม, ค่ามาก = ง่ายขึ้น)
const FUSION_DIST = 0.2;
// Fusion: ต้องทำกลางจอถึงจะติด — จุดกึ่งกลางนิ้วต้องห่างจากกลางจอไม่เกินค่านี้ (ค่ามาก = ง่ายขึ้น)
const CENTER_RADIUS = 0.22;

// Fist Bump: ระยะห่างจุดกลางฝ่ามือสองกำที่ถือว่า "ชนกัน" (0-1 ของขนาดเฟรม, ค่ามาก = ง่ายขึ้น)
const FISTBUMP_DIST = 0.22;

// Kamehameha: ต้องเจอฝ่ามือเปิดอย่างน้อยกี่มือถึงจะนับว่า "เรียงกัน"
const KAME_MIN_HANDS = 3;
// Kamehameha: แกนหลักของแนวเรียงต้องกว้างอย่างน้อยเท่านี้ (0-1) กันมือกระจุกกันแล้วติดมั่ว
const LINE_SPREAD_MIN = 0.25;

// ----------------------------------------------------------------------------
//  อ้างอิง element
// ----------------------------------------------------------------------------
const video = document.getElementById("camera");
const backdrop = document.getElementById("backdrop");
const overlay = document.getElementById("overlay");
const voverlay = document.getElementById("voverlay");
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
      numHands: 6, // รองรับ 3 คน (สำหรับ kamehameha สามมือเรียงกัน)
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

  // 🔥 Fusion: นิ้วชี้ของสองมือ (สองคน) จ่อใกล้กัน → ตรวจก่อนเพื่อน
  const tips = hands.filter(isIndexOnly).map((lm) => lm[8]);
  for (let i = 0; i < tips.length; i++)
    for (let j = i + 1; j < tips.length; j++)
      if (dist(tips[i], tips[j]) < FUSION_DIST) {
        // จุดกึ่งกลางระหว่างปลายนิ้วทั้งสอง ต้องอยู่ "กลางจอ" ถึงจะติด
        const mid = {
          x: (tips[i].x + tips[j].x) / 2,
          y: (tips[i].y + tips[j].y) / 2,
        };
        if (dist(mid, { x: 0.5, y: 0.5 }) < CENTER_RADIUS) return "fusion";
      }

  // 👊 Fist Bump: กำหมัดสองกำ (สองคน) ชนกันกลางจอ → ตรวจก่อนท่า punch มือเดียว
  const fists = [];
  (results.gestures || []).forEach((g, i) => {
    if (g[0]?.categoryName === "Closed_Fist" && hands[i]) fists.push(hands[i][9]);
  });
  for (let i = 0; i < fists.length; i++)
    for (let j = i + 1; j < fists.length; j++)
      if (dist(fists[i], fists[j]) < FISTBUMP_DIST) {
        const mid = {
          x: (fists[i].x + fists[j].x) / 2,
          y: (fists[i].y + fists[j].y) / 2,
        };
        if (dist(mid, { x: 0.5, y: 0.5 }) < CENTER_RADIUS) return "fistbump";
      }

  // 🌀 Kamehameha: ฝ่ามือเปิด "สามมือเรียงกัน" → ตรวจก่อนท่า hands (คนเดียวชูสองมือ)
  //    เรียงแนวนอน = kamehameha, เรียงแนวตั้ง = father son kamehameha
  const palms = [];
  (results.gestures || []).forEach((g, i) => {
    if (g[0]?.categoryName === "Open_Palm" && hands[i]) palms.push(hands[i][9]);
  });
  if (palms.length >= KAME_MIN_HANDS) {
    const xs = palms.map((p) => p.x);
    const ys = palms.map((p) => p.y);
    const xRange = Math.max(...xs) - Math.min(...xs); // ความกว้างแนวนอนของกลุ่มมือ
    const yRange = Math.max(...ys) - Math.min(...ys); // ความสูงแนวตั้งของกลุ่มมือ
    // แกนไหนกว้างกว่า = ทิศที่มือเรียงตัว (และต้องกว้างพอถึงจะนับว่า "เรียงกัน")
    if (xRange > yRange && xRange >= LINE_SPREAD_MIN) return "kamehameha"; // แนวนอน
    if (yRange > xRange && yRange >= LINE_SPREAD_MIN) return "fatherson"; // แนวตั้ง
  }

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

// ระยะห่างระหว่างสองจุด (พิกัด normalize 0-1)
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

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

// เช็คว่าเป็นท่า "นิ้วชี้นิ้วเดียว" (ใช้ระยะจากข้อมือ จึงชี้ได้ทุกทิศ รวมถึงชี้เข้าหากัน)
// นิ้วเหยียด = ปลายนิ้วไกลจากข้อมือ (จุด 0) มากกว่าข้อกลางนิ้ว
function isIndexOnly(lm) {
  if (!lm || lm.length < 21) return false;
  const out = (tip, pip) => dist(lm[0], lm[tip]) > dist(lm[0], lm[pip]);
  const index = out(8, 6);
  const middle = out(12, 10);
  const ring = out(16, 14);
  const pinky = out(20, 18);
  return index && !middle && !ring && !pinky;
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
  overlay.removeAttribute("src");
  voverlay.classList.remove("show");
  voverlay.onended = null;
  voverlay.pause();
  voverlay.removeAttribute("src");
  backdrop.classList.remove("show");
  playing = false;
  cooldownUntil = performance.now() + COOLDOWN_MS;
}

// ----------------------------------------------------------------------------
//  🔥 Fusion: เล่นวิดีโอต่อคิว (fusion dance → สุ่มผลลัพธ์)
// ----------------------------------------------------------------------------
let videoQueue = [];
let videoIndex = 0;

// เล่นวิดีโอเป็นคิว (ใช้ร่วมกันทั้ง Fusion และ Fist Bump)
function playVideoQueue(list, label) {
  playing = true;
  backdrop.classList.add("show");

  videoQueue = list;
  videoIndex = 0;

  statusEl.textContent = label;
  voverlay.classList.add("show");
  voverlay.onended = playNextVideo; // คลิปจบ → เล่นคลิปถัดไปในคิว
  playNextVideo();
}

function triggerFusion() {
  // เล่น fusion dance ก่อนเสมอ แล้วสุ่มสาขา (gogeta / fat / thin)
  const keys = Object.keys(FUSION.branches);
  const pick = keys[Math.floor(Math.random() * keys.length)];
  playVideoQueue([FUSION.dance, ...FUSION.branches[pick]], "fusion → " + pick);
}

function triggerFistBump() {
  playVideoQueue([...FISTBUMP.videos], "fist bump");
}

function triggerKamehameha() {
  playVideoQueue([...KAMEHAMEHA.videos], "kamehameha");
}

function triggerFatherSon() {
  playVideoQueue([...FATHERSON.videos], "father son kamehameha");
}

function playNextVideo() {
  if (videoIndex >= videoQueue.length) {
    endPlayback(); // หมดคิว → จบ
    return;
  }
  const src = videoQueue[videoIndex++];
  voverlay.src = encodeURI(src); // encode เผื่อชื่อไฟล์/โฟลเดอร์มีช่องว่าง
  voverlay.muted = muted;
  voverlay.volume = muted ? 0 : volume;
  voverlay.play().catch(() => {
    // คลิปไหนโหลด/เล่นไม่ได้ → ข้ามไปคลิปถัดไป
    setTimeout(playNextVideo, 50);
  });
}

// คำนวณขนาดวิดีโอให้พอดีจอ (เห็นทั้งภาพ ไม่โดนตัดขอบ)
function fitVideo() {
  const w = voverlay.videoWidth;
  const h = voverlay.videoHeight;
  if (!w || !h) return;
  const scale = Math.min(window.innerWidth / w, window.innerHeight / h) * FILL_RATIO;
  voverlay.style.width = w * scale + "px";
  voverlay.style.height = h * scale + "px";
}
voverlay.addEventListener("loadedmetadata", fitVideo);
window.addEventListener("resize", fitVideo);

// ตัวกระจาย: fusion เล่นแบบวิดีโอ, ท่าอื่นเล่นแบบ GIF+เสียง
function fire(key) {
  if (key === "fusion") triggerFusion();
  else if (key === "fistbump") triggerFistBump();
  else if (key === "kamehameha") triggerKamehameha();
  else if (key === "fatherson") triggerFatherSon();
  else trigger(key);
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
      fire(cand);
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
  voverlay.volume = muted ? 0 : volume; // ปรับเสียงวิดีโอ Fusion ระหว่างเล่นด้วย
  updateMuteIcon();
}

volSlider.addEventListener("input", () => setVolume(volSlider.value / 100));
volDown.addEventListener("click", () => setVolume(volume - 0.1));
volUp.addEventListener("click", () => setVolume(volume + 0.1));
muteBtn.addEventListener("click", () => {
  muted = !muted;
  voverlay.muted = muted; // ปิด/เปิดเสียงวิดีโอ Fusion ที่กำลังเล่นด้วย
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
  if (e.key === "1") fire("punch");
  if (e.key === "2") fire("fingers");
  if (e.key === "3") fire("hands");
  if (e.key === "4") fire("middle");
  if (e.key === "5") fire("love");
  if (e.key === "6") fire("good");
  if (e.key === "7") fire("frieza");
  if (e.key === "8") fire("fusion");
  if (e.key === "9") fire("fistbump");
  if (e.key === "0") fire("kamehameha");
  if (e.key === "-") fire("fatherson");
});

updateMuteIcon();
