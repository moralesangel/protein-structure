// Cartoon (ribbon) geometry for a Ca-only backbone, in the style molecular
// viewers use: flat ribbons through helices, arrows along strands, thin tubes
// for coil. Built from Ca positions plus a per-residue secondary-structure
// string, with no dependency beyond three.js.

const SUBDIV = 10; // spline samples per residue

/** Catmull-Rom through the control points, clamped at the ends. */
function spline(points, subdiv) {
  const out = [];
  const n = points.length;
  const at = (i) => points[Math.max(0, Math.min(n - 1, i))];
  for (let i = 0; i < n - 1; i++) {
    const p0 = at(i - 1), p1 = at(i), p2 = at(i + 1), p3 = at(i + 2);
    for (let s = 0; s < subdiv; s++) {
      const t = s / subdiv;
      const t2 = t * t, t3 = t2 * t;
      out.push([0, 1, 2].map((k) =>
        0.5 * ((2 * p1[k]) +
               (-p0[k] + p2[k]) * t +
               (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2 +
               (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3)));
    }
  }
  out.push(at(n - 1).slice());
  return out;
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const norm = (v) => {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
};

/**
 * Per-residue ribbon frames. The "side" vector is what gives a helix its flat
 * face; taking it from the local curvature (Ca(i-1), Ca(i), Ca(i+1)) is the
 * standard trick, with sign flips removed so the ribbon does not twist 180
 * degrees between neighbours.
 */
function frames(ca, breaks) {
  const n = ca.length;
  const out = [];
  let prevSide = null;
  for (let i = 0; i < n; i++) {
    const a = ca[Math.max(0, i - 1)];
    const b = ca[i];
    const c = ca[Math.min(n - 1, i + 1)];
    const tangent = norm(sub(c, a));
    let side = norm(cross(sub(c, b), sub(a, b)));
    if (!isFinite(side[0]) || Math.hypot(...side) < 1e-6) {
      side = prevSide ? prevSide.slice() : [0, 0, 1];
    }
    // Keep the ribbon face continuous along a chain.
    if (prevSide && !breaks.has(i)) {
      const dot = side[0] * prevSide[0] + side[1] * prevSide[1] + side[2] * prevSide[2];
      if (dot < 0) side = [-side[0], -side[1], -side[2]];
    }
    prevSide = side;
    const up = norm(cross(tangent, side));
    out.push({ tangent, side, up });
  }
  return out;
}

/** Linearly interpolate the per-residue frames onto the dense spline samples. */
function densify(values, subdiv, lerp) {
  const out = [];
  for (let i = 0; i < values.length - 1; i++) {
    for (let s = 0; s < subdiv; s++) out.push(lerp(values[i], values[i + 1], s / subdiv));
  }
  out.push(values[values.length - 1]);
  return out;
}

/**
 * Build one cartoon mesh for a chain segment.
 * `colours` is one THREE.Color per residue; vertices carry colour so the
 * prediction gradient flows along the ribbon.
 */
function buildSegment(THREE, ca, ss, colours, positions, normals, colorArr, indices) {
  const n = ca.length;
  if (n < 2) return;

  const breaks = new Set();
  const fr = frames(ca, breaks);

  const path = spline(ca, SUBDIV);
  const sides = densify(fr.map((f) => f.side), SUBDIV, (a, b, t) =>
    norm([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]));
  const ups = densify(fr.map((f) => f.up), SUBDIV, (a, b, t) =>
    norm([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]));
  const cols = densify(colours, SUBDIV, (a, b, t) => a.clone().lerp(b, t));

  // Cross-section size per sample, from the secondary structure of the residue
  // it came from. Strands taper to an arrow head at their final residue.
  const half = [];
  for (let k = 0; k < path.length; k++) {
    const resF = Math.min(n - 1, k / SUBDIV);
    const i = Math.floor(resF);
    const frac = resF - i;
    const t = ss[i] ?? 'C';
    if (t === 'H') {
      half.push([1.55, 0.32]);
    } else if (t === 'E') {
      const isLast = ss[i + 1] !== 'E';
      // Arrow head: wide at the base, narrowing to a point.
      half.push(isLast ? [1.9 * (1 - frac) + 0.15, 0.26] : [1.25, 0.26]);
    } else {
      half.push([0.32, 0.32]);
    }
  }

  // Extrude a rounded rectangle along the path.
  const RING = 8;
  const base = positions.length / 3;
  for (let k = 0; k < path.length; k++) {
    const [w, h] = half[k];
    const p = path[k], s = sides[k], u = ups[k];
    const c = cols[k];
    for (let j = 0; j < RING; j++) {
      const a = (j / RING) * Math.PI * 2;
      const cx = Math.cos(a) * w;
      const cy = Math.sin(a) * h;
      positions.push(p[0] + s[0] * cx + u[0] * cy,
                     p[1] + s[1] * cx + u[1] * cy,
                     p[2] + s[2] * cx + u[2] * cy);
      const nv = norm([s[0] * cx / (w * w) + u[0] * cy / (h * h),
                       s[1] * cx / (w * w) + u[1] * cy / (h * h),
                       s[2] * cx / (w * w) + u[2] * cy / (h * h)]);
      normals.push(nv[0], nv[1], nv[2]);
      colorArr.push(c.r, c.g, c.b);
    }
  }
  for (let k = 0; k < path.length - 1; k++) {
    for (let j = 0; j < RING; j++) {
      const a = base + k * RING + j;
      const b = base + k * RING + ((j + 1) % RING);
      const c = a + RING;
      const d = b + RING;
      indices.push(a, b, d, a, d, c);
    }
  }
}

/**
 * One merged cartoon mesh for the whole structure.
 * `chainBreaks` holds residue indices where a new chain starts.
 */
export function buildCartoon(THREE, coords, ss, colours, chainBreaks) {
  const positions = [], normals = [], colorArr = [], indices = [];

  let start = 0;
  const cut = (i) =>
    chainBreaks.has(i) ||
    Math.hypot(coords[i][0] - coords[i - 1][0],
               coords[i][1] - coords[i - 1][1],
               coords[i][2] - coords[i - 1][2]) > 4.6;

  for (let i = 1; i <= coords.length; i++) {
    if (i === coords.length || cut(i)) {
      if (i - start >= 3) {
        buildSegment(THREE, coords.slice(start, i), ss.slice(start, i),
                     colours.slice(start, i), positions, normals, colorArr, indices);
      }
      start = i;
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colorArr, 3));
  geo.setIndex(indices);
  return geo;
}
