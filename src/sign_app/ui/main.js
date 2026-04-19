import * as THREE from "three"
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js"
import { playSentence, setCallbacks } from "./signEngine.js"
import { FINGERSPELL, FINGER_BONE_MAP } from "./fingerspellDictionary.js"

/* ═══════════════════════════════════════════════════════════
   GLOBALS
   ═══════════════════════════════════════════════════════════ */

window.targetRotations = {}
window.signing = false

let idleTime = 0
let baseRotations = {}
let bones = {}
let avatar

/* ─── UI Elements ──────────────────────────────────────── */
const statusDot   = document.getElementById("status-dot")
const statusText  = document.getElementById("status-text")
const micBtn      = document.getElementById("mic-btn")
const micSvg      = document.getElementById("mic-svg")
const waveformEl  = document.getElementById("waveform")
const signLabel   = document.getElementById("sign-label")
const transcriptLabel = document.getElementById("transcript-label")
const transcriptText  = document.getElementById("transcript-text")
const glossTokensEl   = document.getElementById("gloss-tokens")

const replayBtn    = document.getElementById("replay-btn")
const textInput    = document.getElementById("text-input")
const sendBtn     = document.getElementById("send-btn")

let lastSignSequence = []

/* ═══════════════════════════════════════════════════════════
   THREE.JS SCENE
   ═══════════════════════════════════════════════════════════ */

const scene = new THREE.Scene()

const camera = new THREE.PerspectiveCamera(
  45,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
)
camera.position.set(0, 1.25, 2.4)
camera.lookAt(0, 1.25, 0)

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
renderer.setSize(window.innerWidth, window.innerHeight)
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
renderer.shadowMap.enabled = true
renderer.setClearColor(0x000000, 0) // transparent for gradient bg

document.body.prepend(renderer.domElement)
renderer.domElement.id = "avatar-canvas"

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight
  camera.updateProjectionMatrix()
  renderer.setSize(window.innerWidth, window.innerHeight)
})

/* ─── Lighting ─────────────────────────────────────────── */
const keyLight = new THREE.DirectionalLight(0xffffff, 1.2)
keyLight.position.set(2, 4, 3)
scene.add(keyLight)

const fillLight = new THREE.DirectionalLight(0x93b5ff, 0.5)
fillLight.position.set(-2, 2, 2)
scene.add(fillLight)

const rimLight = new THREE.DirectionalLight(0xc084fc, 0.4)
rimLight.position.set(0, 2, -3)
scene.add(rimLight)

const ambient = new THREE.AmbientLight(0xffffff, 0.35)
scene.add(ambient)

const signSpot = new THREE.SpotLight(0xffffff, 1.5)
signSpot.position.set(0, 3, 2)
signSpot.target.position.set(0, 1.4, 0)
scene.add(signSpot)
scene.add(signSpot.target)

/* ─── Avatar ───────────────────────────────────────────── */
const loader = new GLTFLoader()

loader.load("avatar.glb", (gltf) => {
  avatar = gltf.scene
  scene.add(avatar)

  avatar.scale.set(1.0, 1.0, 1.0)
  avatar.position.set(0, -0.3, 0)

  avatar.traverse((obj) => {
    if (obj.isBone) {
      bones[obj.name] = obj
      // Save the default rest pose for applying relative offsets
      baseRotations[obj.name] = { 
        x: obj.rotation.x, 
        y: obj.rotation.y, 
        z: obj.rotation.z 
      }
    }
    if (obj.isMesh) {
      obj.frustumCulled = false // Fixes meshes (like the face) randomly getting cutoff/disappearing
      if (obj.morphTargetDictionary) {
        if (!window.faceMesh || Object.keys(obj.morphTargetDictionary).length > Object.keys(window.faceMesh.morphTargetDictionary).length) {
          window.faceMesh = obj
        }
      }
    }
  })

  // Start visible
  avatar.visible = true
})

/* ═══════════════════════════════════════════════════════════
   ANIMATION LOOP
   ═══════════════════════════════════════════════════════════ */

function animate() {
  requestAnimationFrame(animate)
  idleTime += 0.03

  // Lerp all bones toward target rotations + base resting pose
  for (const boneName in window.targetRotations) {
    const bone = bones[boneName]
    const base = baseRotations[boneName]
    if (!bone || !base) continue

    const targetOffset = window.targetRotations[boneName]
    const speed = 0.15

    bone.rotation.x = THREE.MathUtils.lerp(bone.rotation.x, base.x + targetOffset.x, speed)
    bone.rotation.y = THREE.MathUtils.lerp(bone.rotation.y, base.y + targetOffset.y, speed)
    bone.rotation.z = THREE.MathUtils.lerp(bone.rotation.z, base.z + targetOffset.z, speed)
  }

  // Idle breathing
  if (!window.signing && bones.Spine) {
    bones.Spine.rotation.x = Math.sin(idleTime) * 0.015
  }

  renderer.render(scene, camera)
}

animate()

/* ═══════════════════════════════════════════════════════════
   HANDSHAPE SYSTEM (expanded)
   ═══════════════════════════════════════════════════════════ */

const HANDSHAPES = {

  OPEN: {
    RightHandThumb1: { x: 0, y: 0, z: -0.3 },
    RightHandThumb2: { x: 0, y: 0, z: 0 },
    RightHandThumb3: { x: 0, y: 0, z: 0 },
    RightHandIndex1: { x: 0, y: 0, z: 0 }, RightHandIndex2: { x: 0, y: 0, z: 0 }, RightHandIndex3: { x: 0, y: 0, z: 0 },
    RightHandMiddle1: { x: 0, y: 0, z: 0 }, RightHandMiddle2: { x: 0, y: 0, z: 0 }, RightHandMiddle3: { x: 0, y: 0, z: 0 },
    RightHandRing1: { x: 0, y: 0, z: 0 }, RightHandRing2: { x: 0, y: 0, z: 0 }, RightHandRing3: { x: 0, y: 0, z: 0 },
    RightHandPinky1: { x: 0, y: 0, z: 0 }, RightHandPinky2: { x: 0, y: 0, z: 0 }, RightHandPinky3: { x: 0, y: 0, z: 0 },
    RightHand: { x: 0, y: 0, z: 0 },
  },

  FIST: {
    RightHandThumb1: { x: 0.3, y: 0, z: 0.3 },
    RightHandThumb2: { x: 0.3, y: 0, z: 0 },
    RightHandThumb3: { x: 0.2, y: 0, z: 0 },
    RightHandIndex1: { x: 1.2, y: 0, z: 0 }, RightHandIndex2: { x: 1.2, y: 0, z: 0 }, RightHandIndex3: { x: 1.0, y: 0, z: 0 },
    RightHandMiddle1: { x: 1.2, y: 0, z: 0 }, RightHandMiddle2: { x: 1.2, y: 0, z: 0 }, RightHandMiddle3: { x: 1.0, y: 0, z: 0 },
    RightHandRing1: { x: 1.2, y: 0, z: 0 }, RightHandRing2: { x: 1.2, y: 0, z: 0 }, RightHandRing3: { x: 1.0, y: 0, z: 0 },
    RightHandPinky1: { x: 1.2, y: 0, z: 0 }, RightHandPinky2: { x: 1.2, y: 0, z: 0 }, RightHandPinky3: { x: 1.0, y: 0, z: 0 },
    RightHand: { x: 0, y: 0, z: 0 },
  },

  POINT: {
    RightHandThumb1: { x: 0.5, y: 0, z: 0.3 },
    RightHandThumb2: { x: 0.4, y: 0, z: 0 },
    RightHandThumb3: { x: 0.2, y: 0, z: 0 },
    RightHandIndex1: { x: 0, y: 0, z: 0 }, RightHandIndex2: { x: 0, y: 0, z: 0 }, RightHandIndex3: { x: 0, y: 0, z: 0 },
    RightHandMiddle1: { x: 1.2, y: 0, z: 0 }, RightHandMiddle2: { x: 1.0, y: 0, z: 0 }, RightHandMiddle3: { x: 0.8, y: 0, z: 0 },
    RightHandRing1: { x: 1.2, y: 0, z: 0 }, RightHandRing2: { x: 1.0, y: 0, z: 0 }, RightHandRing3: { x: 0.8, y: 0, z: 0 },
    RightHandPinky1: { x: 1.2, y: 0, z: 0 }, RightHandPinky2: { x: 1.0, y: 0, z: 0 }, RightHandPinky3: { x: 0.8, y: 0, z: 0 },
    RightHand: { x: 0, y: 0, z: 0 },
  },

  FLAT: {
    RightHandThumb1: { x: 0.2, y: 0, z: 0.3 },
    RightHandThumb2: { x: 0.1, y: 0, z: 0 },
    RightHandThumb3: { x: 0, y: 0, z: 0 },
    RightHandIndex1: { x: 0, y: 0, z: 0 }, RightHandIndex2: { x: 0, y: 0, z: 0 }, RightHandIndex3: { x: 0, y: 0, z: 0 },
    RightHandMiddle1: { x: 0, y: 0, z: 0 }, RightHandMiddle2: { x: 0, y: 0, z: 0 }, RightHandMiddle3: { x: 0, y: 0, z: 0 },
    RightHandRing1: { x: 0, y: 0, z: 0 }, RightHandRing2: { x: 0, y: 0, z: 0 }, RightHandRing3: { x: 0, y: 0, z: 0 },
    RightHandPinky1: { x: 0, y: 0, z: 0 }, RightHandPinky2: { x: 0, y: 0, z: 0 }, RightHandPinky3: { x: 0, y: 0, z: 0 },
    RightHand: { x: 0, y: 0, z: 0 },
  },

  CLAW: {
    RightHandThumb1: { x: 0.4, y: 0, z: -0.2 },
    RightHandThumb2: { x: 0.3, y: 0, z: 0 },
    RightHandThumb3: { x: 0.2, y: 0, z: 0 },
    RightHandIndex1: { x: 0.6, y: 0, z: 0 }, RightHandIndex2: { x: 0.5, y: 0, z: 0 }, RightHandIndex3: { x: 0.4, y: 0, z: 0 },
    RightHandMiddle1: { x: 0.6, y: 0, z: 0 }, RightHandMiddle2: { x: 0.5, y: 0, z: 0 }, RightHandMiddle3: { x: 0.4, y: 0, z: 0 },
    RightHandRing1: { x: 0.6, y: 0, z: 0 }, RightHandRing2: { x: 0.5, y: 0, z: 0 }, RightHandRing3: { x: 0.4, y: 0, z: 0 },
    RightHandPinky1: { x: 0.6, y: 0, z: 0 }, RightHandPinky2: { x: 0.5, y: 0, z: 0 }, RightHandPinky3: { x: 0.4, y: 0, z: 0 },
    RightHand: { x: 0, y: 0, z: 0 },
  },
}

function applyHandshape(shape) {
  if (!bones.RightHandIndex1) return

  const shapeUpper = (shape || "OPEN").toUpperCase()

  let pose = HANDSHAPES.OPEN // Default

  // Check built-in shapes first
  if (HANDSHAPES[shapeUpper]) {
    pose = HANDSHAPES[shapeUpper]
  } else if (FINGERSPELL[shapeUpper]) {
    // Check the fingerspell dictionary for letter-based handshapes
    pose = {}
    for (const key in FINGERSPELL[shapeUpper]) {
      const boneName = FINGER_BONE_MAP[key]
      if (boneName) pose[boneName] = FINGERSPELL[shapeUpper][key]
    }
  }

  for (const bone in pose) {
    window.targetRotations[bone] = { ...pose[bone] }
  }
}

/* ═══════════════════════════════════════════════════════════
   LOCATION SYSTEM (expanded)
   ═══════════════════════════════════════════════════════════ */

const LOCATIONS = {
  neutral_space: {
    RightShoulder: { x: 0, y: 0, z: -0.15 },
    RightArm: { x: -0.3, y: 0, z: 0 },
    RightForeArm: { x: -0.4, y: 0.2, z: 0 },
  },
  chest: {
    RightShoulder: { x: 0, y: 0, z: -0.2 },
    RightArm: { x: -0.5, y: 0, z: 0 },
    RightForeArm: { x: -0.6, y: 0.3, z: 0 },
  },
  chin: {
    RightShoulder: { x: 0, y: 0, z: -0.3 },
    RightArm: { x: -0.7, y: 0.1, z: 0 },
    RightForeArm: { x: -0.85, y: 0.3, z: 0 },
  },
  mouth: {
    RightShoulder: { x: 0, y: 0, z: -0.3 },
    RightArm: { x: -0.75, y: 0.1, z: 0 },
    RightForeArm: { x: -0.9, y: 0.3, z: 0 },
  },
  nose: {
    RightShoulder: { x: 0, y: 0, z: -0.32 },
    RightArm: { x: -0.85, y: 0.1, z: 0 },
    RightForeArm: { x: -1.0, y: 0.3, z: 0 },
  },
  forehead: {
    RightShoulder: { x: 0, y: 0, z: -0.35 },
    RightArm: { x: -0.9, y: 0.1, z: 0 },
    RightForeArm: { x: -1.05, y: 0.3, z: 0 },
  },
  temple: {
    RightShoulder: { x: 0, y: 0, z: -0.35 },
    RightArm: { x: -0.85, y: 0.15, z: 0.1 },
    RightForeArm: { x: -1.0, y: 0.35, z: 0 },
  },
  side: {
    RightShoulder: { x: 0, y: 0, z: -0.1 },
    RightArm: { x: -0.3, y: 0, z: 0.3 },
    RightForeArm: { x: -0.4, y: 0.2, z: 0 },
  },
  shoulder: {
    RightShoulder: { x: 0, y: 0, z: -0.15 },
    RightArm: { x: -0.4, y: 0, z: 0.15 },
    RightForeArm: { x: -0.7, y: 0.2, z: 0 },
  },
  ear: {
    RightShoulder: { x: 0, y: 0, z: -0.35 },
    RightArm: { x: -0.85, y: 0.2, z: 0.2 },
    RightForeArm: { x: -1.0, y: 0.35, z: 0 },
  },
  waist: {
    RightShoulder: { x: 0, y: 0, z: -0.1 },
    RightArm: { x: -0.2, y: 0, z: 0 },
    RightForeArm: { x: -0.3, y: 0.15, z: 0 },
  },
}

function applyLocation(loc) {
  if (!bones.RightArm) return

  const key = (loc || "neutral_space").toLowerCase()
  const pose = LOCATIONS[key] || LOCATIONS.neutral_space

  for (const bone in pose) {
    if (window.targetRotations) {
      window.targetRotations[bone] = { ...pose[bone] }
    }
  }
}

/* ═══════════════════════════════════════════════════════════
   FINGERSPELLING (from dictionary)
   ═══════════════════════════════════════════════════════════ */

function applyFingerSpell(letter) {
  const l = (letter || "").toUpperCase()
  const pose = FINGERSPELL[l]
  if (!pose) return

  for (const key in pose) {
    const boneName = FINGER_BONE_MAP[key]
    if (boneName) {
      window.targetRotations[boneName] = { ...pose[key] }
    }
  }

  applyLocation("neutral_space")
}

/* ─── Expose to signEngine ─────────────────────────────── */
window.applyHandshape = applyHandshape
window.applyLocation = applyLocation
window.applyFingerSpell = applyFingerSpell
window.playSentence = playSentence

window.clearPose = function() {
  for (const boneName in window.targetRotations) {
    if (window.targetRotations[boneName]) {
      window.targetRotations[boneName] = { x: 0, y: 0, z: 0 }
    }
  }
}

/* ═══════════════════════════════════════════════════════════
   FACIAL EXPRESSIONS (Morph Targets)
   ═══════════════════════════════════════════════════════════ */
const EXPRESSIONS = {
  neutral: {},
  happy: { mouthSmile: 1.0 },
  sad: { mouthOpen: 0.1 },
  surprise: { mouthOpen: 0.8, mouthSmile: 0.2 },
  angry: { mouthOpen: 0.3 },
  question: { mouthOpen: 0.2 }
}

window.applyExpression = function(exprName) {
  if (!window.faceMesh || !window.faceMesh.morphTargetDictionary) return
  
  const dict = window.faceMesh.morphTargetDictionary
  const influences = window.faceMesh.morphTargetInfluences
  
  // Reset all
  for (const key in dict) {
    influences[dict[key]] = 0
  }
  
  const expr = EXPRESSIONS[(exprName || "neutral").toLowerCase()]
  if (!expr) return
  
  for (const key in expr) {
    let targetIdx = dict[key]
    if (targetIdx === undefined) {
      // Find case-insensitive partial match
      const matchingKey = Object.keys(dict).find(k => k.toLowerCase().includes(key.toLowerCase()))
      if (matchingKey) targetIdx = dict[matchingKey]
    }
    
    if (targetIdx !== undefined) {
      influences[targetIdx] = expr[key]
    }
  }
}

/* ═══════════════════════════════════════════════════════════
   SIGN ENGINE CALLBACKS (UI updates)
   ═══════════════════════════════════════════════════════════ */

setCallbacks({
  onSignStart: (index, sign) => {
    // Highlight active gloss token
    const tokens = glossTokensEl.querySelectorAll(".gloss-token")
    tokens.forEach((el, i) => {
      el.classList.remove("active")
      if (i < index) el.classList.add("done")
      if (i === index) el.classList.add("active")
    })

    const label = sign.type === "fingerspell"
      ? `Spelling: ${sign.letter || sign.handshape}`
      : `Signing: ${sign.gloss || sign.handshape}`

    signLabel.textContent = label
    signLabel.classList.add("visible")
  },

  onSignEnd: () => {
    signLabel.classList.remove("visible")
    const tokens = glossTokensEl.querySelectorAll(".gloss-token")
    tokens.forEach(el => el.classList.add("done"))

    setUIState("ready")
  },

  onLetterStart: (letter) => {
    signLabel.textContent = `Spelling: ${letter}`
  }
})

/* ═══════════════════════════════════════════════════════════
   VOICE RECORDING — VAD (Voice Activity Detection)
   ═══════════════════════════════════════════════════════════ */

let recorder = null
let audioChunks = []
let audioStream = null
let audioContext = null
let analyser = null
let silenceTimer = null
let isRecording = false

const SILENCE_THRESHOLD = 0.015    // RMS threshold for "silence"
const SILENCE_DURATION  = 10000    // ms of silence before auto-send (updated to 10 seconds)
const MIN_RECORD_TIME   = 500      // ms minimum recording

// Create waveform bars
for (let i = 0; i < 24; i++) {
  const bar = document.createElement("div")
  bar.className = "wave-bar"
  waveformEl.appendChild(bar)
}
const waveBars = waveformEl.querySelectorAll(".wave-bar")

async function startRecording() {
  if (isRecording) {
    // Toggle off — stop recording and send
    stopAndSend()
    return
  }

  try {
    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true })
  } catch (err) {
    console.error("Mic access denied:", err)
    setUIState("ready")
    return
  }

  isRecording = true
  audioChunks = []

  // Set up MediaRecorder
  recorder = new MediaRecorder(audioStream)
  recorder.ondataavailable = e => audioChunks.push(e.data)
  recorder.onstop = sendAudioToBackend
  recorder.start()

  // Set up audio analysis for VAD + waveform
  audioContext = new AudioContext()
  const source = audioContext.createMediaStreamSource(audioStream)
  analyser = audioContext.createAnalyser()
  analyser.fftSize = 256
  source.connect(analyser)

  setUIState("recording")
  monitorAudio()
}

function monitorAudio() {
  if (!isRecording || !analyser) return

  const data = new Uint8Array(analyser.frequencyBinCount)
  analyser.getByteTimeDomainData(data)

  // Calculate RMS
  let sum = 0
  for (let i = 0; i < data.length; i++) {
    const normalized = (data[i] - 128) / 128
    sum += normalized * normalized
  }
  const rms = Math.sqrt(sum / data.length)

  // Update waveform visual
  updateWaveform(data)

  // Voice activity detection
  if (rms > SILENCE_THRESHOLD) {
    // Voice detected — reset silence timer
    if (silenceTimer) {
      clearTimeout(silenceTimer)
      silenceTimer = null
    }
  } else {
    // Silence — start countdown if not already
    if (!silenceTimer && audioChunks.length > 0) {
      silenceTimer = setTimeout(() => {
        if (isRecording) stopAndSend()
      }, SILENCE_DURATION)
    }
  }

  requestAnimationFrame(monitorAudio)
}

function updateWaveform(data) {
  const step = Math.floor(data.length / waveBars.length)
  waveBars.forEach((bar, i) => {
    const value = Math.abs(data[i * step] - 128) / 128
    const height = Math.max(4, value * 36)
    bar.style.height = `${height}px`
  })
}

function stopAndSend() {
  isRecording = false
  if (silenceTimer) { clearTimeout(silenceTimer); silenceTimer = null }

  if (recorder && recorder.state === "recording") {
    recorder.stop()
  }

  if (audioStream) {
    audioStream.getTracks().forEach(t => t.stop())
    audioStream = null
  }

  if (audioContext) {
    audioContext.close()
    audioContext = null
  }

  setUIState("processing")
}

/* ═══════════════════════════════════════════════════════════
   SEND TO BACKEND
   ═══════════════════════════════════════════════════════════ */

async function sendAudioToBackend() {
  const blob = new Blob(audioChunks, { type: "audio/webm" })
  const formData = new FormData()
  formData.append("file", blob, "recording.webm")

  try {
    const response = await fetch("/voice-to-text/", {
      method: "POST",
      body: formData
    })

    if (!response.ok) throw new Error(`HTTP ${response.status}`)

    const data = await response.json()
    console.log("Backend response:", data)

    displayTranscript(data)

    if (data.sign_sequence && data.sign_sequence.length > 0) {
      lastSignSequence = data.sign_sequence
      replayBtn.disabled = false
      setUIState("signing")
      if (avatar) avatar.visible = true
      playSentence(data.sign_sequence)
    } else {
      setUIState("ready")
    }

  } catch (err) {
    console.error("Backend error:", err)
    transcriptLabel.textContent = ""
    transcriptText.textContent = "Connection error — is the backend running?"
    transcriptText.classList.add("visible")
    setUIState("ready")
  }
}

/* ═══════════════════════════════════════════════════════════
   UI STATE MANAGEMENT
   ═══════════════════════════════════════════════════════════ */

function setUIState(state) {
  micBtn.className = ""
  statusDot.className = "status-dot"
  waveformEl.classList.remove("active")

  switch (state) {
    case "ready":
      statusText.textContent = "Ready"
      statusDot.className = "status-dot"
      micSvg.innerHTML = '<path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm-1-9c0-.55.45-1 1-1s1 .45 1 1v6c0 .55-.45 1-1 1s-1-.45-1-1V5z"/><path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>'
      break

    case "recording":
      statusText.textContent = "Listening..."
      statusDot.className = "status-dot recording"
      micBtn.className = "recording"
      waveformEl.classList.add("active")
      // Stop icon
      micSvg.innerHTML = '<rect x="7" y="7" width="10" height="10" rx="1" fill="white"/>'
      break

    case "processing":
      statusText.textContent = "Processing..."
      statusDot.className = "status-dot processing"
      micBtn.className = "processing"
      // Spinner icon
      micSvg.innerHTML = '<circle cx="12" cy="12" r="8" stroke="white" stroke-width="2" fill="none" stroke-dasharray="20 30"/>'
      break

    case "signing":
      statusText.textContent = "Signing..."
      statusDot.className = "status-dot signing"
      break
  }
}

function displayTranscript(data) {
  // Show original transcription
  transcriptLabel.textContent = "TRANSCRIPTION"
  transcriptText.textContent = data.cleaned_transcription || data.raw_transcription || ""
  transcriptText.classList.add("visible")

  // Show gloss tokens
  glossTokensEl.innerHTML = ""
  if (data.sign_friendly_text && data.sign_friendly_text.length > 0) {
    data.sign_friendly_text.forEach(word => {
      const el = document.createElement("span")
      el.className = "gloss-token"
      el.textContent = word
      glossTokensEl.appendChild(el)
    })
  }
}

/* ─── Mic button handler ───────────────────────────────── */
micBtn.onclick = startRecording

/* ─── Text Input handlers ──────────────────────────────── */
async function sendTextToBackend() {
  const text = textInput.value.trim()
  if (!text) return

  textInput.value = ""
  setUIState("processing")

  try {
    const response = await fetch("/text-to-sign/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    })

    if (!response.ok) throw new Error(`HTTP ${response.status}`)

    const data = await response.json()
    console.log("Text backend response:", data)

    displayTranscript(data)

    if (data.sign_sequence && data.sign_sequence.length > 0) {
      lastSignSequence = data.sign_sequence
      replayBtn.disabled = false
      setUIState("signing")
      if (avatar) avatar.visible = true
      playSentence(data.sign_sequence)
    } else {
      setUIState("ready")
    }

  } catch (err) {
    console.error("Text backend error:", err)
    transcriptLabel.textContent = ""
    transcriptText.textContent = "Connection error — is the backend running?"
    transcriptText.classList.add("visible")
    setUIState("ready")
  }
}

sendBtn.onclick = sendTextToBackend
textInput.onkeydown = (e) => {
  if (e.key === "Enter") sendTextToBackend()
}

/* ─── Replay handler ──────────────────────────────────── */
replayBtn.onclick = () => {
  if (lastSignSequence.length > 0) {
    // Re-trigger the tokens visualization
    displayTranscript({
      cleaned_transcription: transcriptText.textContent,
      sign_friendly_text: Array.from(glossTokensEl.querySelectorAll(".gloss-token")).map(el => el.textContent),
      sign_sequence: lastSignSequence
    })
    
    setUIState("signing")
    playSentence(lastSignSequence)
  }
}

/* ─── Initial state ────────────────────────────────────── */
setUIState("ready")
/* ─── Theme Toggle ─────────────────────────────────────── */
const themeBtn = document.getElementById("theme-btn")
if (themeBtn) {
  themeBtn.onclick = () => {
    document.body.classList.toggle("light-mode")
    const isLight = document.body.classList.contains("light-mode")
    localStorage.setItem("signapp-theme", isLight ? "light" : "dark")
    updateThemeIcon(isLight)

    // Also update scene lights to match a bit better in light mode
    if (isLight) {
        scene.background = new THREE.Color(0xf1f5f9)
    } else {
        scene.background = null // transparent
    }
  }
}
function updateThemeIcon(isLight) {
  const icon = document.getElementById("theme-icon")
  if (!icon) return
  if (isLight) {
    icon.innerHTML = '<path d="M12 3c-4.97 0-9 4.03-9 9s4.03 9 9 9 9-4.03 9-9c0-.46-.04-.92-.1-1.36-.98 1.37-2.58 2.26-4.4 2.26-3.03 0-5.5-2.47-5.5-5.5 0-1.82.89-3.42 2.26-4.4C12.92 3.04 12.46 3 12 3zm0 16c-3.86 0-7-3.14-7-7s3.14-7 7-7c.22 0 .44.03.65.08C11.13 6.36 10 7.97 10 9.85c0 2.85 2.3 5.15 5.15 5.15 1.88 0 3.49-1.13 4.77-2.65.05.21.08.43.08.65 0 3.86-3.14 7-7 7z"/>'
  } else {
    icon.innerHTML = '<path d="M12 7c-2.76 0-5 2.24-5 5s2.24 5 5 5 5-2.24 5-5-2.24-5-5-5zM2 13h2c.55 0 1-.45 1-1s-.45-1-1-1H2c-.55 0-1 .45-1 1s.45 1 1 1zm18 0h2c.55 0 1-.45 1-1s-.45-1-1-1h-2c-.55 0-1 .45-1 1s.45 1 1 1zM11 2v2c0 .55.45 1 1 1s1-.45 1-1V2c0-.55-.45-1-1-1s-1 .45-1 1zm0 18v2c0 .55.45 1 1 1s1-.45 1-1v-2c0-.55-.45-1-1-1s-1 .45-1 1zM5.99 4.58c-.39-.39-1.03-.39-1.41 0-.39.39-.39 1.03 0 1.41l1.06 1.06c.39.39 1.03.39 1.41 0 .39-.39.39-1.03 0-1.41L5.99 4.58zm12.37 12.37c-.39-.39-1.03-.39-1.41 0-.39.39-.39 1.03 0 1.41l1.06 1.06c.39.39 1.03.39 1.41 0 .39-.39.39-1.03 0-1.41l-1.06-1.06zm1.06-10.96c.39-.39.39-1.03 0-1.41-.39-.39-1.03-.39-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41.39.39 1.03.39 1.41 0l1.06-1.06zM7.05 18.36c.39-.39.39-1.03 0-1.41-.39-.39-1.03-.39-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41.39.39 1.03.39 1.41 0l1.06-1.06z"/>'
  }
}
if (localStorage.getItem("signapp-theme") === "light") {
  document.body.classList.add("light-mode")
  updateThemeIcon(true)
  scene.background = new THREE.Color(0xf1f5f9)
}
