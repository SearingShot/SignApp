/**
 * ASL Fingerspelling Dictionary
 *
 * Each letter maps to rotations for ALL finger bones (3 knuckles each)
 * plus thumb (3 joints) and optional wrist rotation.
 *
 * Values are in radians:
 *   0    = straight / open
 *   1.2  = fully curled
 *   0.6  = half curled
 *
 * Thumb uses both X (curl) and Z (spread/opposition) axes.
 */
export const FINGERSPELL = {

  A: {
    thumb1: { x: 0.2, y: 0, z: 0.3 },
    thumb2: { x: 0.1, y: 0, z: 0 },
    thumb3: { x: 0, y: 0, z: 0 },
    index1: { x: 1.2, y: 0, z: 0 }, index2: { x: 1.2, y: 0, z: 0 }, index3: { x: 1.0, y: 0, z: 0 },
    middle1: { x: 1.2, y: 0, z: 0 }, middle2: { x: 1.2, y: 0, z: 0 }, middle3: { x: 1.0, y: 0, z: 0 },
    ring1: { x: 1.2, y: 0, z: 0 }, ring2: { x: 1.2, y: 0, z: 0 }, ring3: { x: 1.0, y: 0, z: 0 },
    pinky1: { x: 1.2, y: 0, z: 0 }, pinky2: { x: 1.2, y: 0, z: 0 }, pinky3: { x: 1.0, y: 0, z: 0 },
    wrist: { x: 0, y: 0, z: 0 }
  },

  B: {
    thumb1: { x: 0.8, y: 0, z: 0.4 },
    thumb2: { x: 0.6, y: 0, z: 0 },
    thumb3: { x: 0.3, y: 0, z: 0 },
    index1: { x: 0, y: 0, z: 0 }, index2: { x: 0, y: 0, z: 0 }, index3: { x: 0, y: 0, z: 0 },
    middle1: { x: 0, y: 0, z: 0 }, middle2: { x: 0, y: 0, z: 0 }, middle3: { x: 0, y: 0, z: 0 },
    ring1: { x: 0, y: 0, z: 0 }, ring2: { x: 0, y: 0, z: 0 }, ring3: { x: 0, y: 0, z: 0 },
    pinky1: { x: 0, y: 0, z: 0 }, pinky2: { x: 0, y: 0, z: 0 }, pinky3: { x: 0, y: 0, z: 0 },
    wrist: { x: 0, y: 0, z: 0 }
  },

  C: {
    thumb1: { x: 0.3, y: 0, z: 0.2 },
    thumb2: { x: 0.3, y: 0, z: 0 },
    thumb3: { x: 0.2, y: 0, z: 0 },
    index1: { x: 0.5, y: 0, z: 0 }, index2: { x: 0.4, y: 0, z: 0 }, index3: { x: 0.3, y: 0, z: 0 },
    middle1: { x: 0.5, y: 0, z: 0 }, middle2: { x: 0.4, y: 0, z: 0 }, middle3: { x: 0.3, y: 0, z: 0 },
    ring1: { x: 0.5, y: 0, z: 0 }, ring2: { x: 0.4, y: 0, z: 0 }, ring3: { x: 0.3, y: 0, z: 0 },
    pinky1: { x: 0.5, y: 0, z: 0 }, pinky2: { x: 0.4, y: 0, z: 0 }, pinky3: { x: 0.3, y: 0, z: 0 },
    wrist: { x: 0, y: 0, z: 0 }
  },

  D: {
    thumb1: { x: 0.6, y: 0, z: 0.4 },
    thumb2: { x: 0.5, y: 0, z: 0 },
    thumb3: { x: 0.3, y: 0, z: 0 },
    index1: { x: 0, y: 0, z: 0 }, index2: { x: 0, y: 0, z: 0 }, index3: { x: 0, y: 0, z: 0 },
    middle1: { x: 1.2, y: 0, z: 0 }, middle2: { x: 1.0, y: 0, z: 0 }, middle3: { x: 0.8, y: 0, z: 0 },
    ring1: { x: 1.2, y: 0, z: 0 }, ring2: { x: 1.0, y: 0, z: 0 }, ring3: { x: 0.8, y: 0, z: 0 },
    pinky1: { x: 1.2, y: 0, z: 0 }, pinky2: { x: 1.0, y: 0, z: 0 }, pinky3: { x: 0.8, y: 0, z: 0 },
    wrist: { x: 0, y: 0, z: 0 }
  },

  E: {
    thumb1: { x: 0.3, y: 0, z: 0.3 },
    thumb2: { x: 0.2, y: 0, z: 0 },
    thumb3: { x: 0.1, y: 0, z: 0 },
    index1: { x: 1.0, y: 0, z: 0 }, index2: { x: 0.8, y: 0, z: 0 }, index3: { x: 0.6, y: 0, z: 0 },
    middle1: { x: 1.0, y: 0, z: 0 }, middle2: { x: 0.8, y: 0, z: 0 }, middle3: { x: 0.6, y: 0, z: 0 },
    ring1: { x: 1.0, y: 0, z: 0 }, ring2: { x: 0.8, y: 0, z: 0 }, ring3: { x: 0.6, y: 0, z: 0 },
    pinky1: { x: 1.0, y: 0, z: 0 }, pinky2: { x: 0.8, y: 0, z: 0 }, pinky3: { x: 0.6, y: 0, z: 0 },
    wrist: { x: 0, y: 0, z: 0 }
  },

  F: {
    thumb1: { x: 0.6, y: 0, z: 0.3 },
    thumb2: { x: 0.5, y: 0, z: 0 },
    thumb3: { x: 0.3, y: 0, z: 0 },
    index1: { x: 0.8, y: 0, z: 0 }, index2: { x: 0.6, y: 0, z: 0 }, index3: { x: 0.4, y: 0, z: 0 },
    middle1: { x: 0, y: 0, z: 0 }, middle2: { x: 0, y: 0, z: 0 }, middle3: { x: 0, y: 0, z: 0 },
    ring1: { x: 0, y: 0, z: 0 }, ring2: { x: 0, y: 0, z: 0 }, ring3: { x: 0, y: 0, z: 0 },
    pinky1: { x: 0, y: 0, z: 0 }, pinky2: { x: 0, y: 0, z: 0 }, pinky3: { x: 0, y: 0, z: 0 },
    wrist: { x: 0, y: 0, z: 0 }
  },

  G: {
    thumb1: { x: 0.2, y: 0, z: -0.4 },
    thumb2: { x: 0.1, y: 0, z: 0 },
    thumb3: { x: 0, y: 0, z: 0 },
    index1: { x: 0, y: 0, z: 0 }, index2: { x: 0, y: 0, z: 0 }, index3: { x: 0, y: 0, z: 0 },
    middle1: { x: 1.2, y: 0, z: 0 }, middle2: { x: 1.0, y: 0, z: 0 }, middle3: { x: 0.8, y: 0, z: 0 },
    ring1: { x: 1.2, y: 0, z: 0 }, ring2: { x: 1.0, y: 0, z: 0 }, ring3: { x: 0.8, y: 0, z: 0 },
    pinky1: { x: 1.2, y: 0, z: 0 }, pinky2: { x: 1.0, y: 0, z: 0 }, pinky3: { x: 0.8, y: 0, z: 0 },
    wrist: { x: 0, y: -0.5, z: 0 }
  },

  H: {
    thumb1: { x: 0.3, y: 0, z: 0.3 },
    thumb2: { x: 0.2, y: 0, z: 0 },
    thumb3: { x: 0.1, y: 0, z: 0 },
    index1: { x: 0, y: 0, z: 0 }, index2: { x: 0, y: 0, z: 0 }, index3: { x: 0, y: 0, z: 0 },
    middle1: { x: 0, y: 0, z: 0 }, middle2: { x: 0, y: 0, z: 0 }, middle3: { x: 0, y: 0, z: 0 },
    ring1: { x: 1.2, y: 0, z: 0 }, ring2: { x: 1.0, y: 0, z: 0 }, ring3: { x: 0.8, y: 0, z: 0 },
    pinky1: { x: 1.2, y: 0, z: 0 }, pinky2: { x: 1.0, y: 0, z: 0 }, pinky3: { x: 0.8, y: 0, z: 0 },
    wrist: { x: 0, y: -0.5, z: 0 }
  },

  I: {
    thumb1: { x: 0.6, y: 0, z: 0.3 },
    thumb2: { x: 0.5, y: 0, z: 0 },
    thumb3: { x: 0.3, y: 0, z: 0 },
    index1: { x: 1.2, y: 0, z: 0 }, index2: { x: 1.0, y: 0, z: 0 }, index3: { x: 0.8, y: 0, z: 0 },
    middle1: { x: 1.2, y: 0, z: 0 }, middle2: { x: 1.0, y: 0, z: 0 }, middle3: { x: 0.8, y: 0, z: 0 },
    ring1: { x: 1.2, y: 0, z: 0 }, ring2: { x: 1.0, y: 0, z: 0 }, ring3: { x: 0.8, y: 0, z: 0 },
    pinky1: { x: 0, y: 0, z: 0 }, pinky2: { x: 0, y: 0, z: 0 }, pinky3: { x: 0, y: 0, z: 0 },
    wrist: { x: 0, y: 0, z: 0 }
  },

  J: {
    thumb1: { x: 0.6, y: 0, z: 0.3 },
    thumb2: { x: 0.5, y: 0, z: 0 },
    thumb3: { x: 0.3, y: 0, z: 0 },
    index1: { x: 1.2, y: 0, z: 0 }, index2: { x: 1.0, y: 0, z: 0 }, index3: { x: 0.8, y: 0, z: 0 },
    middle1: { x: 1.2, y: 0, z: 0 }, middle2: { x: 1.0, y: 0, z: 0 }, middle3: { x: 0.8, y: 0, z: 0 },
    ring1: { x: 1.2, y: 0, z: 0 }, ring2: { x: 1.0, y: 0, z: 0 }, ring3: { x: 0.8, y: 0, z: 0 },
    pinky1: { x: 0, y: 0, z: 0 }, pinky2: { x: 0, y: 0, z: 0 }, pinky3: { x: 0, y: 0, z: 0 },
    wrist: { x: 0, y: 0.6, z: 0 }  // J = I + wrist hook
  },

  K: {
    thumb1: { x: 0.3, y: 0, z: 0.2 },
    thumb2: { x: 0.2, y: 0, z: 0 },
    thumb3: { x: 0.1, y: 0, z: 0 },
    index1: { x: 0, y: 0, z: 0 }, index2: { x: 0, y: 0, z: 0 }, index3: { x: 0, y: 0, z: 0 },
    middle1: { x: 0, y: 0, z: 0 }, middle2: { x: 0, y: 0, z: 0 }, middle3: { x: 0, y: 0, z: 0 },
    ring1: { x: 1.2, y: 0, z: 0 }, ring2: { x: 1.0, y: 0, z: 0 }, ring3: { x: 0.8, y: 0, z: 0 },
    pinky1: { x: 1.2, y: 0, z: 0 }, pinky2: { x: 1.0, y: 0, z: 0 }, pinky3: { x: 0.8, y: 0, z: 0 },
    wrist: { x: 0, y: 0, z: 0 }
  },

  L: {
    thumb1: { x: 0, y: 0, z: -0.6 },
    thumb2: { x: 0, y: 0, z: 0 },
    thumb3: { x: 0, y: 0, z: 0 },
    index1: { x: 0, y: 0, z: 0 }, index2: { x: 0, y: 0, z: 0 }, index3: { x: 0, y: 0, z: 0 },
    middle1: { x: 1.2, y: 0, z: 0 }, middle2: { x: 1.0, y: 0, z: 0 }, middle3: { x: 0.8, y: 0, z: 0 },
    ring1: { x: 1.2, y: 0, z: 0 }, ring2: { x: 1.0, y: 0, z: 0 }, ring3: { x: 0.8, y: 0, z: 0 },
    pinky1: { x: 1.2, y: 0, z: 0 }, pinky2: { x: 1.0, y: 0, z: 0 }, pinky3: { x: 0.8, y: 0, z: 0 },
    wrist: { x: 0, y: 0, z: 0 }
  },

  M: {
    thumb1: { x: 0.6, y: 0, z: 0.4 },
    thumb2: { x: 0.5, y: 0, z: 0 },
    thumb3: { x: 0.3, y: 0, z: 0 },
    index1: { x: 1.2, y: 0, z: 0 }, index2: { x: 1.0, y: 0, z: 0 }, index3: { x: 0.8, y: 0, z: 0 },
    middle1: { x: 1.2, y: 0, z: 0 }, middle2: { x: 1.0, y: 0, z: 0 }, middle3: { x: 0.8, y: 0, z: 0 },
    ring1: { x: 1.2, y: 0, z: 0 }, ring2: { x: 1.0, y: 0, z: 0 }, ring3: { x: 0.8, y: 0, z: 0 },
    pinky1: { x: 1.2, y: 0, z: 0 }, pinky2: { x: 1.0, y: 0, z: 0 }, pinky3: { x: 0.8, y: 0, z: 0 },
    wrist: { x: 0, y: 0, z: 0 }
  },

  N: {
    thumb1: { x: 0.6, y: 0, z: 0.4 },
    thumb2: { x: 0.5, y: 0, z: 0 },
    thumb3: { x: 0.3, y: 0, z: 0 },
    index1: { x: 1.2, y: 0, z: 0 }, index2: { x: 1.0, y: 0, z: 0 }, index3: { x: 0.8, y: 0, z: 0 },
    middle1: { x: 1.2, y: 0, z: 0 }, middle2: { x: 1.0, y: 0, z: 0 }, middle3: { x: 0.8, y: 0, z: 0 },
    ring1: { x: 1.2, y: 0, z: 0 }, ring2: { x: 1.0, y: 0, z: 0 }, ring3: { x: 0.8, y: 0, z: 0 },
    pinky1: { x: 1.2, y: 0, z: 0 }, pinky2: { x: 1.0, y: 0, z: 0 }, pinky3: { x: 0.8, y: 0, z: 0 },
    wrist: { x: 0, y: 0, z: 0 }
  },

  O: {
    thumb1: { x: 0.4, y: 0, z: 0.3 },
    thumb2: { x: 0.3, y: 0, z: 0 },
    thumb3: { x: 0.2, y: 0, z: 0 },
    index1: { x: 0.6, y: 0, z: 0 }, index2: { x: 0.5, y: 0, z: 0 }, index3: { x: 0.4, y: 0, z: 0 },
    middle1: { x: 0.6, y: 0, z: 0 }, middle2: { x: 0.5, y: 0, z: 0 }, middle3: { x: 0.4, y: 0, z: 0 },
    ring1: { x: 0.6, y: 0, z: 0 }, ring2: { x: 0.5, y: 0, z: 0 }, ring3: { x: 0.4, y: 0, z: 0 },
    pinky1: { x: 0.6, y: 0, z: 0 }, pinky2: { x: 0.5, y: 0, z: 0 }, pinky3: { x: 0.4, y: 0, z: 0 },
    wrist: { x: 0, y: 0, z: 0 }
  },

  P: {
    thumb1: { x: 0.2, y: 0, z: -0.3 },
    thumb2: { x: 0.1, y: 0, z: 0 },
    thumb3: { x: 0, y: 0, z: 0 },
    index1: { x: 0, y: 0, z: 0 }, index2: { x: 0, y: 0, z: 0 }, index3: { x: 0, y: 0, z: 0 },
    middle1: { x: 0.4, y: 0, z: 0 }, middle2: { x: 0.3, y: 0, z: 0 }, middle3: { x: 0.2, y: 0, z: 0 },
    ring1: { x: 1.2, y: 0, z: 0 }, ring2: { x: 1.0, y: 0, z: 0 }, ring3: { x: 0.8, y: 0, z: 0 },
    pinky1: { x: 1.2, y: 0, z: 0 }, pinky2: { x: 1.0, y: 0, z: 0 }, pinky3: { x: 0.8, y: 0, z: 0 },
    wrist: { x: 0.5, y: 0, z: 0 }  // hand tilted down
  },

  Q: {
    thumb1: { x: 0.3, y: 0, z: -0.3 },
    thumb2: { x: 0.2, y: 0, z: 0 },
    thumb3: { x: 0.1, y: 0, z: 0 },
    index1: { x: 0.4, y: 0, z: 0 }, index2: { x: 0.3, y: 0, z: 0 }, index3: { x: 0.2, y: 0, z: 0 },
    middle1: { x: 1.2, y: 0, z: 0 }, middle2: { x: 1.0, y: 0, z: 0 }, middle3: { x: 0.8, y: 0, z: 0 },
    ring1: { x: 1.2, y: 0, z: 0 }, ring2: { x: 1.0, y: 0, z: 0 }, ring3: { x: 0.8, y: 0, z: 0 },
    pinky1: { x: 1.2, y: 0, z: 0 }, pinky2: { x: 1.0, y: 0, z: 0 }, pinky3: { x: 0.8, y: 0, z: 0 },
    wrist: { x: 0.5, y: 0, z: 0 }
  },

  R: {
    thumb1: { x: 0.6, y: 0, z: 0.3 },
    thumb2: { x: 0.5, y: 0, z: 0 },
    thumb3: { x: 0.3, y: 0, z: 0 },
    index1: { x: 0, y: 0, z: -0.1 }, index2: { x: 0, y: 0, z: 0 }, index3: { x: 0, y: 0, z: 0 },
    middle1: { x: 0.1, y: 0, z: 0.2 }, middle2: { x: 0.2, y: 0, z: 0 }, middle3: { x: 0.1, y: 0, z: 0 },
    ring1: { x: 1.5, y: 0, z: 0 }, ring2: { x: 1.2, y: 0, z: 0 }, ring3: { x: 1.0, y: 0, z: 0 },
    pinky1: { x: 1.5, y: 0, z: 0 }, pinky2: { x: 1.2, y: 0, z: 0 }, pinky3: { x: 1.0, y: 0, z: 0 },
    wrist: { x: 0, y: 0, z: 0 }
  },

  S: {
    thumb1: { x: 0.3, y: 0, z: 0.3 },
    thumb2: { x: 0.2, y: 0, z: 0 },
    thumb3: { x: 0.1, y: 0, z: 0 },
    index1: { x: 1.2, y: 0, z: 0 }, index2: { x: 1.2, y: 0, z: 0 }, index3: { x: 1.0, y: 0, z: 0 },
    middle1: { x: 1.2, y: 0, z: 0 }, middle2: { x: 1.2, y: 0, z: 0 }, middle3: { x: 1.0, y: 0, z: 0 },
    ring1: { x: 1.2, y: 0, z: 0 }, ring2: { x: 1.2, y: 0, z: 0 }, ring3: { x: 1.0, y: 0, z: 0 },
    pinky1: { x: 1.2, y: 0, z: 0 }, pinky2: { x: 1.2, y: 0, z: 0 }, pinky3: { x: 1.0, y: 0, z: 0 },
    wrist: { x: 0, y: 0, z: 0 }
  },

  T: {
    thumb1: { x: 0.2, y: 0, z: 0.2 },
    thumb2: { x: 0.2, y: 0, z: 0 },
    thumb3: { x: 0.1, y: 0, z: 0 },
    index1: { x: 1.2, y: 0, z: 0 }, index2: { x: 1.0, y: 0, z: 0 }, index3: { x: 0.8, y: 0, z: 0 },
    middle1: { x: 1.2, y: 0, z: 0 }, middle2: { x: 1.0, y: 0, z: 0 }, middle3: { x: 0.8, y: 0, z: 0 },
    ring1: { x: 1.2, y: 0, z: 0 }, ring2: { x: 1.0, y: 0, z: 0 }, ring3: { x: 0.8, y: 0, z: 0 },
    pinky1: { x: 1.2, y: 0, z: 0 }, pinky2: { x: 1.0, y: 0, z: 0 }, pinky3: { x: 0.8, y: 0, z: 0 },
    wrist: { x: 0, y: 0, z: 0 }
  },

  U: {
    thumb1: { x: 0.6, y: 0, z: 0.3 },
    thumb2: { x: 0.5, y: 0, z: 0 },
    thumb3: { x: 0.3, y: 0, z: 0 },
    index1: { x: 0, y: 0, z: 0 }, index2: { x: 0, y: 0, z: 0 }, index3: { x: 0, y: 0, z: 0 },
    middle1: { x: 0, y: 0, z: 0 }, middle2: { x: 0, y: 0, z: 0 }, middle3: { x: 0, y: 0, z: 0 },
    ring1: { x: 1.2, y: 0, z: 0 }, ring2: { x: 1.0, y: 0, z: 0 }, ring3: { x: 0.8, y: 0, z: 0 },
    pinky1: { x: 1.2, y: 0, z: 0 }, pinky2: { x: 1.0, y: 0, z: 0 }, pinky3: { x: 0.8, y: 0, z: 0 },
    wrist: { x: 0, y: 0, z: 0 }
  },

  V: {
    thumb1: { x: 0.6, y: 0, z: 0.3 },
    thumb2: { x: 0.5, y: 0, z: 0 },
    thumb3: { x: 0.3, y: 0, z: 0 },
    index1: { x: 0, y: 0, z: -0.15 }, index2: { x: 0, y: 0, z: 0 }, index3: { x: 0, y: 0, z: 0 },
    middle1: { x: 0, y: 0, z: 0.15 }, middle2: { x: 0, y: 0, z: 0 }, middle3: { x: 0, y: 0, z: 0 },
    ring1: { x: 1.2, y: 0, z: 0 }, ring2: { x: 1.0, y: 0, z: 0 }, ring3: { x: 0.8, y: 0, z: 0 },
    pinky1: { x: 1.2, y: 0, z: 0 }, pinky2: { x: 1.0, y: 0, z: 0 }, pinky3: { x: 0.8, y: 0, z: 0 },
    wrist: { x: 0, y: 0, z: 0 }
  },

  W: {
    thumb1: { x: 0.6, y: 0, z: 0.4 },
    thumb2: { x: 0.5, y: 0, z: 0 },
    thumb3: { x: 0.3, y: 0, z: 0 },
    index1: { x: 0, y: 0, z: -0.15 }, index2: { x: 0, y: 0, z: 0 }, index3: { x: 0, y: 0, z: 0 },
    middle1: { x: 0, y: 0, z: 0 }, middle2: { x: 0, y: 0, z: 0 }, middle3: { x: 0, y: 0, z: 0 },
    ring1: { x: 0, y: 0, z: 0.15 }, ring2: { x: 0, y: 0, z: 0 }, ring3: { x: 0, y: 0, z: 0 },
    pinky1: { x: 1.2, y: 0, z: 0 }, pinky2: { x: 1.0, y: 0, z: 0 }, pinky3: { x: 0.8, y: 0, z: 0 },
    wrist: { x: 0, y: 0, z: 0 }
  },

  X: {
    thumb1: { x: 0.6, y: 0, z: 0.3 },
    thumb2: { x: 0.5, y: 0, z: 0 },
    thumb3: { x: 0.3, y: 0, z: 0 },
    index1: { x: 0.6, y: 0, z: 0 }, index2: { x: 0.8, y: 0, z: 0 }, index3: { x: 0.6, y: 0, z: 0 },
    middle1: { x: 1.2, y: 0, z: 0 }, middle2: { x: 1.0, y: 0, z: 0 }, middle3: { x: 0.8, y: 0, z: 0 },
    ring1: { x: 1.2, y: 0, z: 0 }, ring2: { x: 1.0, y: 0, z: 0 }, ring3: { x: 0.8, y: 0, z: 0 },
    pinky1: { x: 1.2, y: 0, z: 0 }, pinky2: { x: 1.0, y: 0, z: 0 }, pinky3: { x: 0.8, y: 0, z: 0 },
    wrist: { x: 0, y: 0, z: 0 }
  },

  Y: {
    thumb1: { x: 0, y: 0, z: -0.5 },
    thumb2: { x: 0, y: 0, z: 0 },
    thumb3: { x: 0, y: 0, z: 0 },
    index1: { x: 1.2, y: 0, z: 0 }, index2: { x: 1.0, y: 0, z: 0 }, index3: { x: 0.8, y: 0, z: 0 },
    middle1: { x: 1.2, y: 0, z: 0 }, middle2: { x: 1.0, y: 0, z: 0 }, middle3: { x: 0.8, y: 0, z: 0 },
    ring1: { x: 1.2, y: 0, z: 0 }, ring2: { x: 1.0, y: 0, z: 0 }, ring3: { x: 0.8, y: 0, z: 0 },
    pinky1: { x: 0, y: 0, z: 0 }, pinky2: { x: 0, y: 0, z: 0 }, pinky3: { x: 0, y: 0, z: 0 },
    wrist: { x: 0, y: 0, z: 0 }
  },

  Z: {
    thumb1: { x: 0.6, y: 0, z: 0.3 },
    thumb2: { x: 0.5, y: 0, z: 0 },
    thumb3: { x: 0.3, y: 0, z: 0 },
    index1: { x: 0, y: 0, z: 0 }, index2: { x: 0, y: 0, z: 0 }, index3: { x: 0, y: 0, z: 0 },
    middle1: { x: 1.2, y: 0, z: 0 }, middle2: { x: 1.0, y: 0, z: 0 }, middle3: { x: 0.8, y: 0, z: 0 },
    ring1: { x: 1.2, y: 0, z: 0 }, ring2: { x: 1.0, y: 0, z: 0 }, ring3: { x: 0.8, y: 0, z: 0 },
    pinky1: { x: 1.2, y: 0, z: 0 }, pinky2: { x: 1.0, y: 0, z: 0 }, pinky3: { x: 0.8, y: 0, z: 0 },
    wrist: { x: 0, y: 0.4, z: 0 }  // Z = index point + wrist trace
  }
}

/**
 * Maps short key names to actual avatar bone names
 */
export const FINGER_BONE_MAP = {
  thumb1: "RightHandThumb1",
  thumb2: "RightHandThumb2",
  thumb3: "RightHandThumb3",
  index1: "RightHandIndex1",
  index2: "RightHandIndex2",
  index3: "RightHandIndex3",
  middle1: "RightHandMiddle1",
  middle2: "RightHandMiddle2",
  middle3: "RightHandMiddle3",
  ring1: "RightHandRing1",
  ring2: "RightHandRing2",
  ring3: "RightHandRing3",
  pinky1: "RightHandPinky1",
  pinky2: "RightHandPinky2",
  pinky3: "RightHandPinky3",
  wrist: "RightHand"
}