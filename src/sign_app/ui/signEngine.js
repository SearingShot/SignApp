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

/* ── Play a single sign (handshape + location + movement) ── */
function playSign(sign, done) {
  // 1. Apply handshape
  applyHandshapePose(sign.handshape)

  // 2. Apply location (arm positioning)
  window.applyLocation(sign.location)

  // 3. Apply movement animation
  const moveDuration = applyMovement(sign.movement)

  // Wait for the pose to settle + movement to complete
  setTimeout(done, Math.max(600, moveDuration + 200))
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

    case "down": return animateDown()

    case "up": return animateUp()

    case "side_to_side": return animateSideToSide()

    case "nod": return animateNod()

    case "twist": return animateTwist()

    case "wave": return animateWave()

    default:
      console.log("Unknown movement:", movement)
      return 0
  }
}

/* ── Movement implementations ──────────────────────────── */

function animateTap() {
  // Small forward-backward pulse of the forearm
  const bone = "RightForeArm"
  const current = getCurrentRotation(bone)
  const tap = { ...current, x: current.x - 0.15 }

  window.targetRotations[bone] = tap
  setTimeout(() => {
    window.targetRotations[bone] = current
  }, 200)

  return 400
}

function animateDoubleTap() {
  const bone = "RightForeArm"
  const current = getCurrentRotation(bone)
  const tap = { ...current, x: current.x - 0.15 }

  window.targetRotations[bone] = tap
  setTimeout(() => {
    window.targetRotations[bone] = current
    setTimeout(() => {
      window.targetRotations[bone] = tap
      setTimeout(() => {
        window.targetRotations[bone] = current
      }, 150)
    }, 200)
  }, 150)

  return 700
}

function animateCircle(direction) {
  // Circular wrist/forearm motion
  const bone = "RightHand"
  const current = getCurrentRotation(bone)
  const steps = 8
  const radius = 0.2
  let step = 0

  const interval = setInterval(() => {
    const angle = (step / steps) * Math.PI * 2 * direction
    window.targetRotations[bone] = {
      x: current.x + Math.sin(angle) * radius,
      y: current.y + Math.cos(angle) * radius,
      z: current.z
    }
    step++
    if (step >= steps) {
      clearInterval(interval)
      window.targetRotations[bone] = current
    }
  }, 80)

  return steps * 80 + 100
}

function animateForward() {
  // Arm moves forward (extend forearm)
  const bone = "RightForeArm"
  const current = getCurrentRotation(bone)
  const forward = { ...current, x: current.x - 0.3 }

  window.targetRotations[bone] = forward
  setTimeout(() => {
    window.targetRotations[bone] = current
  }, 350)

  return 500
}

function animateDown() {
  // Arm moves downward
  const bone = "RightArm"
  const current = getCurrentRotation(bone)
  const down = { ...current, x: current.x + 0.3 }

  window.targetRotations[bone] = down
  setTimeout(() => {
    window.targetRotations[bone] = current
  }, 350)

  return 500
}

function animateUp() {
  // Arm moves upward
  const bone = "RightArm"
  const current = getCurrentRotation(bone)
  const up = { ...current, x: current.x - 0.3 }

  window.targetRotations[bone] = up
  setTimeout(() => {
    window.targetRotations[bone] = current
  }, 350)

  return 500
}

function animateSideToSide() {
  const bone = "RightHand"
  const current = getCurrentRotation(bone)

  window.targetRotations[bone] = { ...current, z: current.z - 0.2 }
  setTimeout(() => {
    window.targetRotations[bone] = { ...current, z: current.z + 0.2 }
    setTimeout(() => {
      window.targetRotations[bone] = current
    }, 200)
  }, 200)

  return 600
}

function animateNod() {
  const bone = "RightHand"
  const current = getCurrentRotation(bone)
  const nod = { ...current, x: current.x + 0.25 }

  window.targetRotations[bone] = nod
  setTimeout(() => {
    window.targetRotations[bone] = current
  }, 250)

  return 450
}

function animateTwist() {
  const bone = "RightHand"
  const current = getCurrentRotation(bone)

  window.targetRotations[bone] = { ...current, y: current.y + 0.4 }
  setTimeout(() => {
    window.targetRotations[bone] = { ...current, y: current.y - 0.4 }
    setTimeout(() => {
      window.targetRotations[bone] = current
    }, 200)
  }, 250)

  return 650
}

function animateWave() {
  // Small wave-like motion for greeting signs
  const bone = "RightHand"
  const current = getCurrentRotation(bone)
  let step = 0
  const steps = 6

  const interval = setInterval(() => {
    const val = Math.sin(step * 1.2) * 0.2
    window.targetRotations[bone] = { ...current, z: current.z + val }
    step++
    if (step >= steps) {
      clearInterval(interval)
      window.targetRotations[bone] = current
    }
  }, 100)

  return steps * 100 + 100
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