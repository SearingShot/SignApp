import { FINGERSPELL, FINGER_BONE_MAP } from "./fingerspellDictionary.js"

/**
 * Sign Engine — orchestrates sign playback, movements, and fingerspelling
 */

/* ── State ─────────────────────────────────────────────── */
let _currentSignIndex = -1
let _playing = false

/* ── Exported callbacks (set by main.js) ───────────────── */
export let onSignStart = null    // (index, sign) => void
export let onSignEnd = null      // () => void
export let onLetterStart = null  // (letter) => void

export function setCallbacks({ onSignStart: s, onSignEnd: e, onLetterStart: l }) {
  onSignStart = s
  onSignEnd = e
  onLetterStart = l
}

/* ── Play a full sentence of signs ─────────────────────── */
export function playSentence(signs) {
  if (_playing) return
  _playing = true
  _currentSignIndex = 0

  window.signing = true

  playNext(signs)
}

function playNext(signs) {
  if (_currentSignIndex >= signs.length) {
    // Done — return to neutral after a brief hold
    setTimeout(() => {
      returnToNeutral()
      _playing = false
      _currentSignIndex = -1
      window.signing = false
      if (onSignEnd) onSignEnd()
    }, 800)
    return
  }

  const sign = signs[_currentSignIndex]

  if (onSignStart) onSignStart(_currentSignIndex, sign)

  if (sign.type === "fingerspell") {
    playFingerspellLetter(sign, () => {
      _currentSignIndex++
      setTimeout(() => playNext(signs), 250) // short gap between letters
    })
  } else {
    playSign(sign, () => {
      _currentSignIndex++
      setTimeout(() => playNext(signs), 700) // longer gap between signs
    })
  }
}

/* ── Play a single sign (handshape + location + movement + expression) ── */
function playSign(sign, done) {
  // 1. Apply handshape
  applyHandshapePose(sign.handshape)

  // 2. Apply location (arm positioning)
  window.applyLocation(sign.location)

  // 3. Apply expression
  if (window.applyExpression) {
    window.applyExpression(sign.expression || "neutral")
  }

  // 4. Apply movement animation
  const moveDuration = applyMovement(sign.movement)

  // Wait for the pose to settle + movement to complete
  setTimeout(() => {
    if (window.applyExpression) window.applyExpression("neutral")
    done()
  }, Math.max(600, moveDuration + 200))
}

/* ── Fingerspell a single letter ───────────────────────── */
function playFingerspellLetter(sign, done) {
  const letter = (sign.letter || sign.handshape || "").toUpperCase()

  if (onLetterStart) onLetterStart(letter)

  const pose = FINGERSPELL[letter]
  if (!pose) {
    done()
    return
  }

  // Apply full multi-joint fingerspell pose
  for (const key in pose) {
    const boneName = FINGER_BONE_MAP[key]
    if (boneName && window.targetRotations !== undefined) {
      window.targetRotations[boneName] = { ...pose[key] }
    }
  }

  // Ensure neutral hand position for fingerspelling
  window.applyLocation("neutral_space")

  setTimeout(done, 450) // hold each letter
}

/* ── Handshape pose application ────────────────────────── */
function applyHandshapePose(shape) {
  if (!shape) return

  // Check if this shape exists in FINGERSPELL dictionary first (reuse finger poses)
  const shapeUpper = shape.toUpperCase()
  if (FINGERSPELL[shapeUpper]) {
    const pose = FINGERSPELL[shapeUpper]
    for (const key in pose) {
      const boneName = FINGER_BONE_MAP[key]
      if (boneName && window.targetRotations !== undefined) {
        window.targetRotations[boneName] = { ...pose[key] }
      }
    }
    return
  }

  // Built-in handshape aliases
  if (window.applyHandshape) {
    window.applyHandshape(shapeUpper)
  }
}

/* ── Movement animations ───────────────────────────────── */

/**
 * Apply a movement animation. Returns duration in ms.
 */
function applyMovement(movement) {
  if (!movement || movement === "none") return 0

  switch (movement.toLowerCase()) {

    case "tap": return animateTap()

    case "double_tap": return animateDoubleTap()

    case "circle_clockwise": return animateCircle(1)

    case "circle_counterclockwise": return animateCircle(-1)

    case "forward": return animateForward()

    case "pull_in": return animatePullIn()

    case "down": return animateDown()

    case "up": return animateUp()

    case "side_to_side": return animateSideToSide()

    case "nod": return animateNod()

    case "twist": return animateTwist()

    case "wave": return animateWave()

    case "touch": return animateTouch()

    default:
      console.log("Unknown movement:", movement)
      return 0
  }
}

/* ── Movement implementations ──────────────────────────── */

function animateTap() {
  const boneR = "RightForeArm"
  const boneL = "LeftForeArm"
  const currentR = getCurrentRotation(boneR)
  const currentL = getCurrentRotation(boneL)
  
  if (window.targetRotations[boneR]) window.targetRotations[boneR] = { ...currentR, x: currentR.x - 0.15 }
  if (window.targetRotations[boneL]) window.targetRotations[boneL] = { ...currentL, x: currentL.x - 0.15 }

  setTimeout(() => {
    if (window.targetRotations[boneR]) window.targetRotations[boneR] = currentR
    if (window.targetRotations[boneL]) window.targetRotations[boneL] = currentL
  }, 200)

  return 400
}

function animateDoubleTap() {
  const boneR = "RightForeArm"
  const boneL = "LeftForeArm"
  const currentR = getCurrentRotation(boneR)
  const currentL = getCurrentRotation(boneL)
  
  const applyTap = () => {
    if (window.targetRotations[boneR]) window.targetRotations[boneR] = { ...currentR, x: currentR.x - 0.15 }
    if (window.targetRotations[boneL]) window.targetRotations[boneL] = { ...currentL, x: currentL.x - 0.15 }
  }
  const applyReset = () => {
    if (window.targetRotations[boneR]) window.targetRotations[boneR] = currentR
    if (window.targetRotations[boneL]) window.targetRotations[boneL] = currentL
  }

  applyTap()
  setTimeout(() => {
    applyReset()
    setTimeout(() => {
      applyTap()
      setTimeout(() => applyReset(), 150)
    }, 200)
  }, 150)

  return 700
}

function animateCircle(direction) {
  const boneR = "RightHand"
  const boneL = "LeftHand"
  const currentR = getCurrentRotation(boneR)
  const currentL = getCurrentRotation(boneL)
  const steps = 8
  const radius = 0.2
  let step = 0

  const interval = setInterval(() => {
    const angle = (step / steps) * Math.PI * 2 * direction
    if (window.targetRotations[boneR]) {
      window.targetRotations[boneR] = {
        x: currentR.x + Math.sin(angle) * radius,
        y: currentR.y + Math.cos(angle) * radius,
        z: currentR.z
      }
    }
    if (window.targetRotations[boneL]) {
      // Mirror circle on left hand (invert Y)
      window.targetRotations[boneL] = {
        x: currentL.x + Math.sin(angle) * radius,
        y: currentL.y - Math.cos(angle) * radius,
        z: currentL.z
      }
    }
    step++
    if (step >= steps) {
      clearInterval(interval)
      if (window.targetRotations[boneR]) window.targetRotations[boneR] = currentR
      if (window.targetRotations[boneL]) window.targetRotations[boneL] = currentL
    }
  }, 80)

  return steps * 80 + 100
}

function animateForward() {
  const boneR = "RightForeArm"
  const boneL = "LeftForeArm"
  const currentR = getCurrentRotation(boneR)
  const currentL = getCurrentRotation(boneL)

  if (window.targetRotations[boneR]) window.targetRotations[boneR] = { ...currentR, x: currentR.x - 0.3 }
  if (window.targetRotations[boneL]) window.targetRotations[boneL] = { ...currentL, x: currentL.x - 0.3 }

  setTimeout(() => {
    if (window.targetRotations[boneR]) window.targetRotations[boneR] = currentR
    if (window.targetRotations[boneL]) window.targetRotations[boneL] = currentL
  }, 350)

  return 500
}

function animatePullIn() {
  const boneR = "RightForeArm"
  const boneL = "LeftForeArm"
  const currentR = getCurrentRotation(boneR)
  const currentL = getCurrentRotation(boneL)

  if (window.targetRotations[boneR]) window.targetRotations[boneR] = { ...currentR, x: currentR.x + 0.3 }
  if (window.targetRotations[boneL]) window.targetRotations[boneL] = { ...currentL, x: currentL.x + 0.3 }

  setTimeout(() => {
    if (window.targetRotations[boneR]) window.targetRotations[boneR] = currentR
    if (window.targetRotations[boneL]) window.targetRotations[boneL] = currentL
  }, 350)

  return 500
}

function animateDown() {
  const boneR = "RightArm"
  const boneL = "LeftArm"
  const currentR = getCurrentRotation(boneR)
  const currentL = getCurrentRotation(boneL)

  if (window.targetRotations[boneR]) window.targetRotations[boneR] = { ...currentR, x: currentR.x + 0.3 }
  if (window.targetRotations[boneL]) window.targetRotations[boneL] = { ...currentL, x: currentL.x + 0.3 }

  setTimeout(() => {
    if (window.targetRotations[boneR]) window.targetRotations[boneR] = currentR
    if (window.targetRotations[boneL]) window.targetRotations[boneL] = currentL
  }, 350)

  return 500
}

function animateUp() {
  const boneR = "RightArm"
  const boneL = "LeftArm"
  const currentR = getCurrentRotation(boneR)
  const currentL = getCurrentRotation(boneL)

  if (window.targetRotations[boneR]) window.targetRotations[boneR] = { ...currentR, x: currentR.x - 0.3 }
  if (window.targetRotations[boneL]) window.targetRotations[boneL] = { ...currentL, x: currentL.x - 0.3 }

  setTimeout(() => {
    if (window.targetRotations[boneR]) window.targetRotations[boneR] = currentR
    if (window.targetRotations[boneL]) window.targetRotations[boneL] = currentL
  }, 350)

  return 500
}

function animateSideToSide() {
  const boneR = "RightHand"
  const boneL = "LeftHand"
  const currentR = getCurrentRotation(boneR)
  const currentL = getCurrentRotation(boneL)

  if (window.targetRotations[boneR]) window.targetRotations[boneR] = { ...currentR, z: currentR.z - 0.2 }
  if (window.targetRotations[boneL]) window.targetRotations[boneL] = { ...currentL, z: currentL.z + 0.2 }
  
  setTimeout(() => {
    if (window.targetRotations[boneR]) window.targetRotations[boneR] = { ...currentR, z: currentR.z + 0.2 }
    if (window.targetRotations[boneL]) window.targetRotations[boneL] = { ...currentL, z: currentL.z - 0.2 }
    setTimeout(() => {
      if (window.targetRotations[boneR]) window.targetRotations[boneR] = currentR
      if (window.targetRotations[boneL]) window.targetRotations[boneL] = currentL
    }, 200)
  }, 200)

  return 600
}

function animateNod() {
  const boneR = "RightHand"
  const boneL = "LeftHand"
  const currentR = getCurrentRotation(boneR)
  const currentL = getCurrentRotation(boneL)

  if (window.targetRotations[boneR]) window.targetRotations[boneR] = { ...currentR, x: currentR.x + 0.25 }
  if (window.targetRotations[boneL]) window.targetRotations[boneL] = { ...currentL, x: currentL.x + 0.25 }

  setTimeout(() => {
    if (window.targetRotations[boneR]) window.targetRotations[boneR] = currentR
    if (window.targetRotations[boneL]) window.targetRotations[boneL] = currentL
  }, 250)

  return 450
}

function animateTwist() {
  const boneR = "RightHand"
  const boneL = "LeftHand"
  const currentR = getCurrentRotation(boneR)
  const currentL = getCurrentRotation(boneL)

  if (window.targetRotations[boneR]) window.targetRotations[boneR] = { ...currentR, y: currentR.y + 0.4 }
  if (window.targetRotations[boneL]) window.targetRotations[boneL] = { ...currentL, y: currentL.y - 0.4 }
  
  setTimeout(() => {
    if (window.targetRotations[boneR]) window.targetRotations[boneR] = { ...currentR, y: currentR.y - 0.4 }
    if (window.targetRotations[boneL]) window.targetRotations[boneL] = { ...currentL, y: currentL.y + 0.4 }
    setTimeout(() => {
      if (window.targetRotations[boneR]) window.targetRotations[boneR] = currentR
      if (window.targetRotations[boneL]) window.targetRotations[boneL] = currentL
    }, 200)
  }, 250)

  return 650
}

function animateWave() {
  const boneR = "RightHand"
  const boneL = "LeftHand"
  const currentR = getCurrentRotation(boneR)
  const currentL = getCurrentRotation(boneL)
  let step = 0
  const steps = 6

  const interval = setInterval(() => {
    const val = Math.sin(step * 1.2) * 0.2
    if (window.targetRotations[boneR]) window.targetRotations[boneR] = { ...currentR, z: currentR.z + val }
    if (window.targetRotations[boneL]) window.targetRotations[boneL] = { ...currentL, z: currentL.z - val }
    step++
    if (step >= steps) {
      clearInterval(interval)
      if (window.targetRotations[boneR]) window.targetRotations[boneR] = currentR
      if (window.targetRotations[boneL]) window.targetRotations[boneL] = currentL
    }
  }, 100)

  return steps * 100 + 100
}

function animateTouch() {
  const boneR = "RightForeArm"
  const boneL = "LeftForeArm"
  const currentR = getCurrentRotation(boneR)
  const currentL = getCurrentRotation(boneL)

  if (window.targetRotations[boneR]) window.targetRotations[boneR] = { ...currentR, x: currentR.x + 0.15 }
  if (window.targetRotations[boneL]) window.targetRotations[boneL] = { ...currentL, x: currentL.x + 0.15 }

  setTimeout(() => {
    if (window.targetRotations[boneR]) window.targetRotations[boneR] = currentR
    if (window.targetRotations[boneL]) window.targetRotations[boneL] = currentL
  }, 200)

  return 400
}

/* ── Helpers ───────────────────────────────────────────── */

function getCurrentRotation(boneName) {
  if (window.targetRotations && window.targetRotations[boneName]) {
    return { ...window.targetRotations[boneName] }
  }
  return { x: 0, y: 0, z: 0 }
}

function returnToNeutral() {
  if (window.clearPose) window.clearPose()
}