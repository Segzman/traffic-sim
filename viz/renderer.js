/**
 * viz/renderer.js  — Traffic Simulation Platform canvas renderer
 *
 * Coordinate system
 * -----------------
 *   World  : Web-Mercator metres  (+X = east, +Y = north)
 *   Canvas : CSS px               (+X = right, +Y = down)
 *
 *   _tx(wx) =  wx * S + Bx + Px
 *   _ty(wy) = -wy * S + By + Py
 *   where S = base.scale * zoom,  Bx/By = fit-to-screen base offsets,
 *   Px/Py = user pan offsets (all in CSS px).
 *
 * Zoom centering invariant:
 *   newPx = (cx − Bx)·(1 − R) + Px·R
 *   newPy = (cy − By)·(1 − R) + Py·R
 */

'use strict';

// Web-Mercator constants (metres)
const MERC_HALF = 20037508.342789244;
const MERC_FULL = 40075016.685578488;

// ─────────────────────────────────────────────────────────────────────────────
// Colour palette
// ─────────────────────────────────────────────────────────────────────────────
const COLOUR = {
  VEHICLE_FREE:      '#00e676',
  VEHICLE_SLOWING:   '#ffab00',
  VEHICLE_CONGESTED: '#ff1744',
  VEHICLE_STOPPED:   '#78909c',
  PEDESTRIAN:        '#40c4ff',
  SIGNAL_GREEN:      '#00e676',
  SIGNAL_YELLOW:     '#ffd600',
  SIGNAL_RED:        '#ff1744',
  BACKGROUND:        '#17212b',
};

// ─────────────────────────────────────────────────────────────────────────────
// Road tier styles  (speed_limit m/s thresholds)
//   0 ≥ 20   motorway   amber/gold
//   1 ≥ 13   primary    steel-blue
//   2 ≥ 9.5  secondary  slate
//   3 ≥ 5.5  residential  dim
//   4 < 5.5  service    barely visible
// ─────────────────────────────────────────────────────────────────────────────
const TIER_STYLE = [
  { fill: '#e8b84b', casing: '#5c3800', fw: 8.0, cw: 2.0 },  // motorway
  { fill: '#8ab3d0', casing: '#162333', fw: 5.0, cw: 1.5 },  // primary
  { fill: '#4a7a9b', casing: '#0d1e2c', fw: 3.0, cw: 1.0 },  // secondary
  { fill: '#253f55', casing: null,      fw: 1.8, cw: 0   },  // residential
  { fill: '#182530', casing: null,      fw: 0.8, cw: 0   },  // service
];

const LANE_WIDTH_M = 3.5;   // physical lane width in metres

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
function vehicleColour(speed, limit) {
  if (speed <= 0.3) return COLOUR.VEHICLE_STOPPED;
  const r = limit > 0 ? speed / limit : 0;
  if (r >= 0.8) return COLOUR.VEHICLE_FREE;
  if (r >= 0.4) return COLOUR.VEHICLE_SLOWING;
  return COLOUR.VEHICLE_CONGESTED;
}

// Blend two hex colours at fraction t (0=all a, 1=all b)
function _blendColour(a, b, t) {
  const parse = h => [
    parseInt(h.slice(1,3),16),
    parseInt(h.slice(3,5),16),
    parseInt(h.slice(5,7),16),
  ];
  const [ar,ag,ab_] = parse(a);
  const [br,bg,bb_] = parse(b);
  const r = Math.round(ar + (br-ar)*t);
  const g = Math.round(ag + (bg-ag)*t);
  const bl= Math.round(ab_ + (bb_-ab_)*t);
  return `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${bl.toString(16).padStart(2,'0')}`;
}

// Apply a subtle type-specific colour tint over the speed colour
function _typeColourOverlay(type, base) {
  if (type === 'truck') return _blendColour(base, '#ff8f00', 0.25);  // warm amber
  if (type === 'bus')   return _blendColour(base, '#29b6f6', 0.28);  // light blue
  if (type === 'van')   return _blendColour(base, '#ce93d8', 0.18);  // soft purple
  return base;  // cars: pure speed colour
}

function signalColour(state) {
  if (state === 'green')  return COLOUR.SIGNAL_GREEN;
  if (state === 'yellow') return COLOUR.SIGNAL_YELLOW;
  return COLOUR.SIGNAL_RED;
}


// ─────────────────────────────────────────────────────────────────────────────
// _TileLayer  —  slippy-map tile layer (CartoDB Dark Matter by default)
//
// Tile URL variables:  {s} subdomain · {z} zoom · {x} col · {y} row · {r} retina
// ─────────────────────────────────────────────────────────────────────────────
class _TileLayer {
  constructor(url) {
    // CartoDB Dark Matter (no labels) — free, dark, CORS-enabled, no API key
    this._url  = url ||
      'https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png';
    this._subs = ['a', 'b', 'c', 'd'];
    this._cache = new Map();   // `z/x/y` → HTMLImageElement
  }

  _urlFor(z, x, y) {
    const s   = this._subs[(x + y + z) % 4];
    const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
    const r   = dpr >= 2 ? '@2x' : '';
    return this._url
      .replace('{s}', s).replace('{z}', z)
      .replace('{x}', x).replace('{y}', y)
      .replace('{r}', r);
  }

  /** Return (or start loading) the tile image for (z, x, y). */
  _get(z, x, y, onLoad) {
    const key = `${z}/${x}/${y}`;
    if (this._cache.has(key)) return this._cache.get(key);
    const img = new Image();
    img.crossOrigin = 'anonymous';
    if (onLoad) img.addEventListener('load', onLoad, { once: true });
    img.src = this._urlFor(z, x, y);
    this._cache.set(key, img);
    // Evict oldest if cache grows large
    if (this._cache.size > 1024)
      this._cache.delete(this._cache.keys().next().value);
    return img;
  }

  /**
   * Draw map tiles into ctx.
   *
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} S   - current px/metre scale (base.scale × zoom)
   * @param {number} Bx  - base x offset (CSS px)
   * @param {number} By  - base y offset (CSS px)
   * @param {number} Px  - pan x offset (CSS px)
   * @param {number} Py  - pan y offset (CSS px)
   * @param {number} W   - viewport width  (CSS px)
   * @param {number} H   - viewport height (CSS px)
   * @param {Function} [onLoad] - called when a tile finishes loading
   */
  draw(ctx, S, Bx, By, Px, Py, W, H, onLoad) {
    if (S <= 0) return;

    // Zoom level so tile ≈ 256 CSS px wide, clamped to OSM range [1, 19]
    let z = Math.round(Math.log2(MERC_FULL * S / 256));
    z = Math.max(1, Math.min(19, z));
    const n      = 1 << z;                    // tile grid dimension (2^z)
    const tilePx = MERC_FULL / n * S + 0.5;   // tile width in CSS px (+0.5 fills seams)

    // ---- Visible tile range ----
    // Convert screen edges → world metres → tile indices
    const wx0 = (-Bx - Px) / S;               // world-x at screen left
    const wx1 = ( W - Bx - Px) / S;           // world-x at screen right
    const wy0 = ( By + Py) / S;               // world-y at screen top    (north)
    const wy1 = ( By + Py - H) / S;           // world-y at screen bottom (south)

    // Tile y increases southward (OSM convention), so bigger wy → smaller ty
    const tx0 = Math.max(0,     Math.floor((wx0 + MERC_HALF) / MERC_FULL * n));
    const tx1 = Math.min(n - 1, Math.ceil( (wx1 + MERC_HALF) / MERC_FULL * n));
    const ty0 = Math.max(0,     Math.floor((MERC_HALF - wy0) / MERC_FULL * n));
    const ty1 = Math.min(n - 1, Math.ceil( (MERC_HALF - wy1) / MERC_FULL * n));

    for (let tx = tx0; tx <= tx1; tx++) {
      for (let ty = ty0; ty <= ty1; ty++) {
        const img = this._get(z, tx, ty, onLoad);
        if (!img.complete || !img.naturalWidth) continue;

        // Top-left world-metre corner of this tile
        const tileWX =  tx / n * MERC_FULL - MERC_HALF;
        const tileWY = MERC_HALF - ty / n * MERC_FULL;

        // CSS px position of tile top-left
        const sx =  tileWX * S + Bx + Px;
        const sy = -tileWY * S + By + Py;

        ctx.drawImage(img, sx, sy, tilePx, tilePx);
      }
    }
  }
}


// ─────────────────────────────────────────────────────────────────────────────
// TrafficRenderer
// ─────────────────────────────────────────────────────────────────────────────
class TrafficRenderer {
  constructor(canvas) {
    this._canvas = canvas;
    this._ctx    = canvas.getContext('2d');
    this._net    = null;
    this._tiles  = new _TileLayer();

    // Viewport state
    this._base = { x: 0, y: 0, scale: 1.0 };
    this._zoom  = 1.0;
    this._panX  = 0.0;
    this._panY  = 0.0;
    this._drag  = null;

    this._fps      = 0;
    this._lastTime = null;
    this._lastVehicleCount = 0;

    this._tierEdges = [[], [], [], [], []];
    this._setupInteraction();
  }

  // ── public API ────────────────────────────────────────────────────────────

  get pixelsPerMeter() { return this._base.scale * this._zoom; }
  get currentZoom()    { return this._zoom; }

  zoomBy(factor, cx, cy) {
    const newZoom = Math.min(150, Math.max(0.06, this._zoom * factor));
    this._applyZoom(newZoom, cx, cy);
  }

  resetView() { this._zoom = 1.0; this._panX = 0.0; this._panY = 0.0; }

  get stats() {
    return {
      fps:          Math.round(this._fps),
      vehicleCount: this._lastVehicleCount,
      zoom:         this._zoom,
    };
  }

  // ── coordinate transforms ─────────────────────────────────────────────────

  _S()    { return this._base.scale * this._zoom; }
  _tx(wx) { return  wx * this._S() + this._base.x + this._panX; }
  _ty(wy) { return -wy * this._S() + this._base.y + this._panY; }
  _logW() { return this._canvas._logicalWidth  || this._canvas.width;  }
  _logH() { return this._canvas._logicalHeight || this._canvas.height; }

  _applyZoom(newZoom, cx, cy) {
    const R  = newZoom / this._zoom;
    const bx = this._base.x, by = this._base.y;
    this._panX = (cx - bx) * (1 - R) + this._panX * R;
    this._panY = (cy - by) * (1 - R) + this._panY * R;
    this._zoom = newZoom;
  }

  // ── network setup ──────────────────────────────────────────────────────────

  setNetwork(net) {
    this._net = net;
    this._fitViewport();
    this._buildTierGroups();
  }

  _fitViewport() {
    if (!this._net) return;
    const nodes = Object.values(this._net.nodes);
    if (!nodes.length) return;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const n of nodes) {
      if (n.x < x0) x0 = n.x;  if (n.x > x1) x1 = n.x;
      if (n.y < y0) y0 = n.y;  if (n.y > y1) y1 = n.y;
    }
    const W = this._logW(), H = this._logH(), pad = 48;
    const sx = (W - 2 * pad) / Math.max(1, x1 - x0);
    const sy = (H - 2 * pad) / Math.max(1, y1 - y0);
    this._base.scale = Math.min(sx, sy);
    this._base.x     = -x0 * this._base.scale + pad;
    this._base.y     =  y1 * this._base.scale + pad;
    this._zoom = 1.0; this._panX = 0.0; this._panY = 0.0;
  }

  refitBase() { this._fitViewport(); }

  _roadTier(e) {
    const s = e.speed_limit || 8.3;
    if (s >= 20)  return 0;
    if (s >= 13)  return 1;
    if (s >= 9.5) return 2;
    if (s >= 5.5) return 3;
    return 4;
  }

  _buildTierGroups() {
    this._tierEdges = [[], [], [], [], []];
    for (const e of Object.values(this._net.edges))
      this._tierEdges[this._roadTier(e)].push(e);
  }

  // ── interaction ───────────────────────────────────────────────────────────

  _setupInteraction() {
    const c = this._canvas;

    c.addEventListener('wheel', e => {
      e.preventDefault();
      this.zoomBy(e.deltaY < 0 ? 1.18 : 0.847, e.offsetX, e.offsetY);
    }, { passive: false });

    c.addEventListener('mousedown', e => {
      if (e.button) return;
      this._drag = { x: e.offsetX - this._panX, y: e.offsetY - this._panY };
      c.style.cursor = 'grabbing';
    });
    c.addEventListener('mousemove', e => {
      if (!this._drag) return;
      this._panX = e.offsetX - this._drag.x;
      this._panY = e.offsetY - this._drag.y;
    });
    const end = () => { this._drag = null; c.style.cursor = 'grab'; };
    c.addEventListener('mouseup',    end);
    c.addEventListener('mouseleave', end);
    c.style.cursor = 'grab';

    // Touch
    let _td = null;
    c.addEventListener('touchstart', e => {
      if (e.touches.length === 1) {
        const t = e.touches[0];
        this._drag = { x: t.clientX - this._panX, y: t.clientY - this._panY };
        _td = null;
      } else if (e.touches.length === 2) {
        this._drag = null;
        const [a, b] = e.touches;
        _td = {
          d:  Math.hypot(b.clientX - a.clientX, b.clientY - a.clientY),
          mx: (a.clientX + b.clientX) / 2,
          my: (a.clientY + b.clientY) / 2,
        };
      }
    }, { passive: true });
    c.addEventListener('touchmove', e => {
      e.preventDefault();
      if (e.touches.length === 1 && this._drag) {
        const t = e.touches[0];
        this._panX = t.clientX - this._drag.x;
        this._panY = t.clientY - this._drag.y;
      } else if (e.touches.length === 2 && _td) {
        const [a, b] = e.touches;
        const d  = Math.hypot(b.clientX - a.clientX, b.clientY - a.clientY);
        const mx = (a.clientX + b.clientX) / 2, my = (a.clientY + b.clientY) / 2;
        const nz = Math.min(150, Math.max(0.06, this._zoom * d / _td.d));
        this._applyZoom(nz, mx, my);
        _td = { d, mx, my };
      }
    }, { passive: false });
    c.addEventListener('touchend', () => { this._drag = null; _td = null; });
  }

  // ── road widths ───────────────────────────────────────────────────────────

  _roadWidths(tier, numLanes) {
    const st   = TIER_STYLE[tier];
    const phys = numLanes * LANE_WIDTH_M * this._S();
    const fw   = Math.max(st.fw, phys);
    return { fw, cw: st.cw > 0 ? fw + st.cw * 2 : 0 };
  }

  // ── level of detail ───────────────────────────────────────────────────────

  _visibleTiers() {
    const z = this._zoom;
    if (z < 0.20) return [0];
    if (z < 0.55) return [0, 1];
    if (z < 1.20) return [0, 1, 2, 3];
    return [0, 1, 2, 3, 4];
  }

  // ── main render ───────────────────────────────────────────────────────────

  render(simState) {
    const ctx = this._ctx;
    const W   = this._logW(), H = this._logH();

    const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
    if (this._lastTime !== null)
      this._fps = 0.9 * this._fps + 0.1 * (1000 / Math.max(1, now - this._lastTime));
    this._lastTime = now;
    this._lastVehicleCount = simState.vehicles?.length ?? 0;

    // Apply DPR so all drawing is in CSS px
    const dpr = (this._canvas._logicalWidth && this._canvas.width !== W)
      ? this._canvas.width / W : 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // 1. Background (shows through where tiles haven't loaded yet)
    ctx.fillStyle = COLOUR.BACKGROUND;
    ctx.fillRect(0, 0, W, H);

    if (!this._net) return;

    // 2. Map tile layer
    const S  = this._S();
    const Bx = this._base.x, By = this._base.y;
    const Px = this._panX,   Py = this._panY;
    this._tiles.draw(ctx, S, Bx, By, Px, Py, W, H);

    // 3. Road network
    this._drawEdges();
    if (this._zoom >= 3.5) this._drawLaneMarkings();

    // 4. Simulation overlays
    if (simState.signals?.length)     this._drawSignals(simState.signals);
    if (simState.vehicles?.length)    this._drawVehicles(simState.vehicles);
    if (simState.pedestrians?.length) this._drawPedestrians(simState.pedestrians);
  }

  // ── road edges ────────────────────────────────────────────────────────────

  _drawEdges() {
    const ctx     = this._ctx;
    const visible = this._visibleTiers();
    ctx.lineJoin = 'round';
    ctx.lineCap  = 'round';

    // Draw low-tier roads first (behind) → high-tier last (on top)
    for (let tier = 4; tier >= 0; tier--) {
      if (!visible.includes(tier)) continue;
      const st    = TIER_STYLE[tier];
      const edges = this._tierEdges[tier];
      if (!edges.length) continue;

      // Sub-group by lane count for batching
      const byLanes = new Map();
      for (const e of edges) {
        const k = e.num_lanes || 1;
        if (!byLanes.has(k)) byLanes.set(k, []);
        byLanes.get(k).push(e);
      }

      for (const [lanes, grp] of byLanes) {
        const { fw, cw } = this._roadWidths(tier, lanes);
        // Casing pass
        if (cw > 0 && st.casing) {
          ctx.strokeStyle = st.casing;
          ctx.lineWidth   = cw;
          ctx.beginPath();
          for (const e of grp) this._traceSmoothPath(e);
          ctx.stroke();
        }
        // Fill pass
        ctx.strokeStyle = st.fill;
        ctx.lineWidth   = fw;
        ctx.beginPath();
        for (const e of grp) this._traceSmoothPath(e);
        ctx.stroke();
      }
    }
  }

  // ── smooth bezier path ────────────────────────────────────────────────────

  /**
   * Trace a smooth bezier curve through all geometry points of an edge.
   *
   * Uses midpoint-smoothing: each interior via-node becomes a quadratic
   * bezier control point whose endpoint is the midpoint to the next node.
   * This produces C¹-continuous curves that naturally follow road bends.
   */
  _traceSmoothPath(e) {
    const pts = this._edgePts(e);
    const n   = pts.length;
    if (n < 2) return;

    const ctx = this._ctx;
    ctx.moveTo(this._tx(pts[0][0]), this._ty(pts[0][1]));

    if (n === 2) {
      // Straight segment — no intermediate points
      ctx.lineTo(this._tx(pts[1][0]), this._ty(pts[1][1]));
      return;
    }

    // Midpoint smoothing: for each interior point P_i use it as a bezier
    // control point, landing at midpoint(P_i, P_{i+1}).
    for (let i = 1; i < n - 1; i++) {
      const cx  = this._tx(pts[i][0]),     cy  = this._ty(pts[i][1]);
      const nx  = this._tx(pts[i + 1][0]), ny  = this._ty(pts[i + 1][1]);
      const mx  = (cx + nx) * 0.5,         my  = (cy + ny) * 0.5;
      ctx.quadraticCurveTo(cx, cy, mx, my);
    }
    // Final straight segment to the last point
    ctx.lineTo(this._tx(pts[n - 1][0]), this._ty(pts[n - 1][1]));
  }

  _edgePts(e) {
    if (e.geometry?.length >= 2) return e.geometry;
    const f = this._net.nodes[e.from_node], t = this._net.nodes[e.to_node];
    return (f && t) ? [[f.x, f.y], [t.x, t.y]] : [];
  }

  // ── lane markings (zoom ≥ 3.5×) ──────────────────────────────────────────

  _drawLaneMarkings() {
    const ctx = this._ctx;
    ctx.save();
    ctx.setLineDash([5, 7]);
    ctx.lineWidth = 0.7;
    ctx.lineJoin  = 'round';
    ctx.lineCap   = 'round';

    for (const tier of [0, 1, 2]) {
      ctx.strokeStyle = tier === 0
        ? 'rgba(255,220,100,0.22)'
        : 'rgba(180,210,240,0.18)';
      for (const e of this._tierEdges[tier]) {
        const lanes = e.num_lanes || 1;
        if (lanes < 2) continue;
        const pts = this._edgePts(e);
        if (pts.length < 2) continue;
        for (let li = 1; li < lanes; li++) {
          const off = (li - lanes / 2) * LANE_WIDTH_M;
          ctx.beginPath();
          let started = false;
          for (let i = 0; i < pts.length - 1; i++) {
            const dx = pts[i + 1][0] - pts[i][0], dy = pts[i + 1][1] - pts[i][1];
            const len = Math.hypot(dx, dy);
            if (len < 0.001) continue;
            const nx = dy / len, ny = -dx / len;
            const sx = this._tx(pts[i][0]     + nx * off), sy = this._ty(pts[i][1]     + ny * off);
            const ex = this._tx(pts[i + 1][0] + nx * off), ey = this._ty(pts[i + 1][1] + ny * off);
            if (!started) { ctx.moveTo(sx, sy); started = true; } else ctx.lineTo(sx, sy);
            ctx.lineTo(ex, ey);
          }
          ctx.stroke();
        }
      }
    }
    ctx.restore();
  }

  // ── signals ───────────────────────────────────────────────────────────────

  _drawSignals(signals) {
    const ctx = this._ctx;
    const S   = this._S();

    // Traffic light geometry (CSS px, scaled with zoom)
    const ICON_MIN_S = 1.0;   // start showing icon above this px/m scale

    for (const sig of signals) {
      const n = this._net.nodes[sig.node_id];
      if (!n) continue;
      const cx = this._tx(n.x), cy = this._ty(n.y);
      const state = sig.state;

      if (S >= ICON_MIN_S) {
        // ── Detailed traffic-light icon ──────────────────────────────────
        //  Scale housing with zoom but clamp to a readable range
        const scale = Math.min(3.0, Math.max(1.0, S * 0.8));
        const HW = 5  * scale;   // housing half-width
        const HH = 11 * scale;   // housing half-height
        const LR = 3  * scale;   // light circle radius
        const GAP = HH * 0.6;    // vertical gap between lights

        // Housing (dark rounded rectangle)
        ctx.fillStyle   = '#101418';
        ctx.strokeStyle = '#000';
        ctx.lineWidth   = 0.5 * scale;
        ctx.beginPath();
        ctx.rect(cx - HW, cy - HH, HW * 2, HH * 2);
        ctx.fill();
        ctx.stroke();

        // Three lights: red (top), yellow (mid), green (bottom)
        const LIGHTS = [
          { dy: -GAP, col: COLOUR.SIGNAL_RED,    name: 'red'    },
          { dy:    0, col: COLOUR.SIGNAL_YELLOW,  name: 'yellow' },
          { dy: +GAP, col: COLOUR.SIGNAL_GREEN,   name: 'green'  },
        ];
        for (const lgt of LIGHTS) {
          const active = state === lgt.name;
          ctx.beginPath();
          ctx.arc(cx, cy + lgt.dy, LR, 0, Math.PI * 2);
          if (active) {
            ctx.fillStyle   = lgt.col;
            ctx.shadowColor = lgt.col;
            ctx.shadowBlur  = 6 * scale;
            ctx.fill();
            ctx.shadowBlur  = 0;
          } else {
            // Dim inactive lights to 15 % brightness
            ctx.fillStyle = lgt.col + '26';
            ctx.fill();
          }
        }
      } else {
        // ── Simple dot at low zoom ───────────────────────────────────────
        const r = Math.max(3, Math.min(6, 4 * S));
        ctx.beginPath();
        ctx.arc(cx, cy, r + 2, 0, Math.PI * 2);
        ctx.fillStyle = '#111827';
        ctx.fill();
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        const col = signalColour(state);
        ctx.fillStyle   = col;
        ctx.shadowColor = col;
        ctx.shadowBlur  = 6;
        ctx.fill();
        ctx.shadowBlur  = 0;
      }
    }
  }

  // ── vehicles ──────────────────────────────────────────────────────────────

  _drawVehicles(vehicles) {
    const ctx = this._ctx;
    const S   = this._S();

    for (const v of vehicles) {
      const e = this._net.edges[v.edge_id];
      if (!e) continue;

      // Physical dimensions → pixels  (fallback to generic car if missing)
      const VL = Math.max(4, (v.length || 4.5) * S);
      const VW = Math.max(2, (v.width  || 1.8) * S);

      const pts = this._edgePts(e);
      if (pts.length < 2) continue;

      const numLanes = e.num_lanes || 1;
      const laneId   = v.lane_id ?? 0;
      const lateralM = ((numLanes - 1) / 2 - laneId) * LANE_WIDTH_M;

      const { x, y, ang, rnx, rny } = this._posOnPolyline(pts, v.position_s || 0);
      const cx  = this._tx(x + rnx * lateralM);
      const cy  = this._ty(y + rny * lateralM);
      const baseCol = vehicleColour(v.speed || 0, v.speed_limit || e.speed_limit || 13.9);
      const col     = _typeColourOverlay(v.type, baseCol);

      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(ang);
      ctx.shadowColor = col;
      ctx.shadowBlur  = this._zoom >= 3 ? 12 : 5;
      ctx.fillStyle   = col;
      ctx.fillRect(-VL / 2, -VW / 2, VL, VW);
      ctx.shadowBlur  = 0;
      if (VW >= 3.5) {
        ctx.strokeStyle = 'rgba(255,255,255,0.5)';
        ctx.lineWidth   = 0.6;
        ctx.strokeRect(-VL / 2, -VW / 2, VL, VW);
      }
      ctx.restore();
    }
  }

  _posOnPolyline(pts, s) {
    let acc = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      const dx = pts[i + 1][0] - pts[i][0], dy = pts[i + 1][1] - pts[i][1];
      const len = Math.sqrt(dx * dx + dy * dy);
      if (acc + len >= s || i === pts.length - 2) {
        const t   = len > 0 ? Math.min(1, (s - acc) / len) : 0;
        const inv = len > 0 ? 1 / len : 0;
        return {
          x:   pts[i][0] + t * dx,
          y:   pts[i][1] + t * dy,
          ang: Math.atan2(-dy * this._S(), dx * this._S()),
          rnx:  dy * inv,    // right-hand normal x
          rny: -dx * inv,    // right-hand normal y
        };
      }
      acc += len;
    }
    const last = pts[pts.length - 1];
    return { x: last[0], y: last[1], ang: 0, rnx: 0, rny: 0 };
  }

  // ── pedestrians ───────────────────────────────────────────────────────────

  _drawPedestrians(pedestrians) {
    const ctx = this._ctx;
    const r   = Math.max(2, 1.2 * this._S());
    for (const p of pedestrians) {
      ctx.beginPath();
      ctx.arc(this._tx(p.x), this._ty(p.y), r, 0, Math.PI * 2);
      ctx.fillStyle = COLOUR.PEDESTRIAN;
      ctx.fill();
    }
  }
}


// ─────────────────────────────────────────────────────────────────────────────
// Module exports
// ─────────────────────────────────────────────────────────────────────────────
if (typeof module !== 'undefined' && module.exports)
  module.exports = { TrafficRenderer, vehicleColour, signalColour, COLOUR };
