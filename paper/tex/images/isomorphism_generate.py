"""
isomorphism_generate.py
------------------------
Generates isomorphism.png — a figure illustrating the correspondence:

    Attention Block (r, c)  <-->  Physical KV Page phi(c)

Three-panel layout (inspired by vLLM Figure 6):
  Left   : Attention block grid  (query rows x KV columns)
  Centre : Block table phi       (logical c -> physical page)
  Bottom : Physical KV page pool (GPU DRAM, non-contiguous)

Usage:
    pip install matplotlib
    python isomorphism_generate.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Wider figure to give pages room ──
fig, ax = plt.subplots(figsize=(20, 13.5))
ax.set_xlim(0, 20)
ax.set_ylim(-1.5, 13)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Palette ────────────────────────────────────────────────────────────────
C_PREFIX_L = '#C8DCF0'; C_PREFIX_D = '#2E7DBE'
C_NEW_L    = '#FCE8C0'; C_NEW_D    = '#D4900A'
C_SEL_L    = '#FADADA'; C_SEL_D    = '#C0392B'
C_SKIP_L   = '#F0F0F0'; C_SKIP_D   = '#999999'
C_HDR      = '#2C3E50'; C_TXT      = '#1A1A1A'; C_EDGE = '#555555'


def box(ax, x, y, w, h, fc, ec=C_EDGE, lw=1.2, text='',
        fs=9, bold=False, tc='#1A1A1A', rad=0.07):
    """Draw a rounded rectangle with an optional centred label."""
    p = mpatches.FancyBboxPatch(
        (x + 0.03, y + 0.03), w - 0.06, h - 0.06,
        boxstyle=f"round,pad=0,rounding_size={rad}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
    ax.add_patch(p)
    if text:
        ax.text(x + w / 2, y + h / 2, text,
                ha='center', va='center', fontsize=fs,
                fontweight='bold' if bold else 'normal',
                color=tc, zorder=3)


def arr(ax, x0, y0, x1, y1, color, lw=1.6, rad=0.0):
    """Draw a curved arrow from (x0,y0) to (x1,y1)."""
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'))


# ══════════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════════
NR, NP, NN = 3, 4, 2        # query rows, prefix cols, new-token cols
NC = NP + NN                 # total KV columns = 6
CW, CH = 0.88, 0.88          # cell width/height for attention grid

GX, GY = 0.55, 6.00          # grid origin — raised to make room below

TX, TY = 7.00, 5.80          # block table origin — raised too
TCW, TH = 1.30, 0.70         # table cell width / row height

# Block table mapping: logical column c -> physical page index
PHI = {0: 5, 1: 2, 2: 7, 3: 1, 4: 6, 5: 3}
N_PHYS = 8                   # total physical pages in pool

# ── Physical page pool — FULL WIDTH, generous spacing ──
PW, PH   = 1.05, 1.35        # page cell: slightly larger
PGAP     = 0.95              # much wider gap between pages
pool_total_w = N_PHYS * PW + (N_PHYS - 1) * PGAP
PX0      = (20 - pool_total_w) / 2   # centre the pool
PY0      = 2.20                        # raised — room for labels below

# Selected (sparse) blocks — (row, col) 0-indexed
SELECTED = {(0, 0), (0, 3), (1, 1), (1, 4), (2, 2), (2, 5)}

# Derived sets
accessed_phys = {PHI[c] for c in range(NC)
                 if any((r, c) in SELECTED for r in range(NR))}
phys_to_c = {v: k for k, v in PHI.items()}


# ══════════════════════════════════════════════════════════════════════════
# 1. Attention Block Grid (left panel)
# ══════════════════════════════════════════════════════════════════════════
ax.text(GX + NC * CW / 2, GY + NR * CH + 1.60,
        'Attention Block Grid',
        ha='center', va='bottom', fontsize=13, fontweight='bold', color=C_TXT)

# Double-headed brackets above grid
ax.annotate('', xy=(GX + NP * CW, GY + NR * CH + 1.10),
            xytext=(GX, GY + NR * CH + 1.10),
            arrowprops=dict(arrowstyle='<->', color=C_PREFIX_D, lw=2.0))
ax.text(GX + NP * CW / 2, GY + NR * CH + 1.25,
        r'$n_p/b$  prefix columns',
        ha='center', va='bottom', fontsize=9, color=C_PREFIX_D, fontweight='bold')

ax.annotate('', xy=(GX + NC * CW, GY + NR * CH + 1.10),
            xytext=(GX + NP * CW, GY + NR * CH + 1.10),
            arrowprops=dict(arrowstyle='<->', color=C_NEW_D, lw=2.0))
ax.text(GX + NP * CW + NN * CW / 2, GY + NR * CH + 1.25,
        r'$n_q/b$  new cols',
        ha='center', va='bottom', fontsize=9, color=C_NEW_D, fontweight='bold')

# Column headers
for c in range(NC):
    ax.text(GX + c * CW + CW / 2, GY + NR * CH + 0.65, f'c={c}',
            ha='center', va='center', fontsize=9,
            color=C_PREFIX_D if c < NP else C_NEW_D, fontweight='bold')

# Y-axis label and row labels
ax.text(GX - 0.55, GY + NR * CH / 2, 'Query rows (r)',
        ha='center', va='center', fontsize=9, color=C_TXT, rotation=90)
for r in range(NR):
    ax.text(GX - 0.18, GY + (NR - 1 - r) * CH + CH / 2, f'r={r}',
            ha='right', va='center', fontsize=9, color=C_TXT)

# Grid cells
for r in range(NR):
    for c in range(NC):
        cx, cy = GX + c * CW, GY + (NR - 1 - r) * CH
        sel = (r, c) in SELECTED
        pfx = c < NP
        if sel:
            fc, ec, lw, lbl, tc, bold = C_SEL_L, C_SEL_D, 2.2, f'({r},{c})', C_SEL_D, True
        elif pfx:
            fc, ec, lw, lbl, tc, bold = C_PREFIX_L, C_PREFIX_D, 0.9, '', C_TXT, False
        else:
            fc, ec, lw, lbl, tc, bold = C_NEW_L, C_NEW_D, 0.9, '', C_TXT, False
        box(ax, cx, cy, CW - 0.04, CH - 0.04, fc, ec, lw,
            text=lbl, fs=8, bold=bold, tc=tc)


# ══════════════════════════════════════════════════════════════════════════
# 2. Block Table phi (centre panel)
# ══════════════════════════════════════════════════════════════════════════
ax.text(TX + TCW, TY + NC * TH + 0.82, 'Block Table  φ',
        ha='center', va='bottom', fontsize=13, fontweight='bold', color=C_TXT)

# Header row
box(ax, TX,       TY + NC * TH, TCW, TH * 0.72, C_HDR, C_HDR, 1,
    text='Logical\nc', fs=8, bold=True, tc='white')
box(ax, TX + TCW, TY + NC * TH, TCW, TH * 0.72, C_HDR, C_HDR, 1,
    text='Physical\nφ(c)', fs=8, bold=True, tc='white')

# Data rows (one per logical column c)
for i, c in enumerate(range(NC)):
    ry  = TY + (NC - 1 - i) * TH
    sc  = any((r, c) in SELECTED for r in range(NR))
    pfx = c < NP
    box(ax, TX,       ry, TCW, TH - 0.06,
        C_PREFIX_L if pfx else C_NEW_L, C_EDGE, 0.8,
        text=str(c), fs=10, bold=True, tc=C_TXT)
    box(ax, TX + TCW, ry, TCW, TH - 0.06,
        C_SEL_L if sc else C_SKIP_L,
        C_SEL_D if sc else C_SKIP_D,
        2.0 if sc else 0.7,
        text=str(PHI[c]), fs=10, bold=sc,
        tc=C_SEL_D if sc else C_SKIP_D)


# ══════════════════════════════════════════════════════════════════════════
# 3. Physical KV Page Pool (full-width bottom panel)
# ══════════════════════════════════════════════════════════════════════════
ax.text(10, PY0 + PH + 1.15,
        'Physical KV Page Pool  (GPU DRAM)',
        ha='center', va='bottom', fontsize=13, fontweight='bold', color=C_TXT)
ax.text(10, PY0 + PH + 0.70,
        '(non-contiguous — managed by PagedAttention block table)',
        ha='center', va='bottom', fontsize=8.5, color='#666666', style='italic')

for p in range(N_PHYS):
    px  = PX0 + p * (PW + PGAP)
    c_l = phys_to_c.get(p, None)
    pfx = (c_l is not None and c_l < NP)
    acc = p in accessed_phys

    if   acc and pfx:         fc, ec, lw, tc = C_PREFIX_L, C_PREFIX_D, 2.5, C_PREFIX_D
    elif acc:                 fc, ec, lw, tc = C_NEW_L,    C_NEW_D,    2.5, C_NEW_D
    elif pfx:                 fc, ec, lw, tc = C_PREFIX_L, C_SKIP_D,   0.8, C_SKIP_D
    elif c_l is not None:     fc, ec, lw, tc = C_NEW_L,    C_SKIP_D,   0.8, C_SKIP_D
    else:                     fc, ec, lw, tc = C_SKIP_L,   C_SKIP_D,   0.6, C_SKIP_D

    box(ax, px, PY0, PW, PH, fc, ec, lw,
        text=f'Page\n{p}', fs=9, bold=acc, tc=tc, rad=0.10)

    # checkmark / dash badge
    ax.text(px + PW / 2, PY0 + PH - 0.13, '✓' if acc else '–',
            ha='center', va='top', fontsize=12,
            color=C_SEL_D if acc else C_SKIP_D, fontweight='bold', zorder=4)

    # logical mapping label below page
    if c_l is not None:
        ax.text(px + PW / 2, PY0 - 0.22, f'φ({c_l})={p}',
                ha='center', va='top', fontsize=8,
                color=C_PREFIX_D if pfx else C_NEW_D,
                fontweight='bold' if acc else 'normal')
    ax.text(px + PW / 2, PY0 - 0.58, f'phys {p}',
            ha='center', va='top', fontsize=8, color='#999999')


# ══════════════════════════════════════════════════════════════════════════
# Arrows: selected grid column c -> block table row -> physical page
# ══════════════════════════════════════════════════════════════════════════
# Per-column curvatures tuned to avoid crossing
grid_to_table_rad = {0:  0.20, 1:  0.15, 2: -0.15, 3:  0.10, 4: -0.10, 5: -0.20}
table_to_page_rad = {0:  0.20, 1:  0.10, 2: -0.15, 3:  0.25, 4: -0.10, 5: -0.25}

for c in range(NC):
    if not any((r, c) in SELECTED for r in range(NR)):
        continue
    col_c  = C_PREFIX_D if c < NP else C_NEW_D
    gx_mid = GX + c * CW + CW / 2
    trow_y = TY + (NC - 1 - c) * TH + TH / 2
    phys   = PHI[c]
    pgx    = PX0 + phys * (PW + PGAP) + PW / 2

    # Grid → Block table
    arr(ax, gx_mid, GY, TX, trow_y, col_c, lw=1.8,
        rad=grid_to_table_rad[c])
    # Block table → Physical page
    arr(ax, TX + 2 * TCW, trow_y, pgx, PY0 + PH, col_c, lw=1.8,
        rad=table_to_page_rad[c])


# ══════════════════════════════════════════════════════════════════════════
# Isomorphism equation banner (bottom)
# ══════════════════════════════════════════════════════════════════════════
banner = mpatches.FancyBboxPatch(
    (0.50, -0.60), 19.0, 0.84,
    boxstyle='round,pad=0.1', facecolor='#EEF2FF',
    edgecolor='#5C6BC0', linewidth=2.0, zorder=1)
ax.add_patch(banner)
ax.text(10.0, -0.18,
        r'Attention Block $(r,\,c)$  $\longleftrightarrow$  '
        r'Physical KV Page $\phi(c)$'
        r'        [same index $c$ — no gather, no data reorganisation]',
        ha='center', va='center', fontsize=12, fontweight='bold',
        color='#1A237E', zorder=2)


# ══════════════════════════════════════════════════════════════════════════
# Legend (top)
# ══════════════════════════════════════════════════════════════════════════
items = [
    (C_SEL_L,    C_SEL_D,    'Selected block  (r, c)'),
    (C_PREFIX_L, C_PREFIX_D, 'Prefix KV  (cached, shared)'),
    (C_NEW_L,    C_NEW_D,    'New-token KV  (current request)'),
    (C_SKIP_L,   C_SKIP_D,   'Unaccessed page  (skipped by kernel)'),
]
lx, ly = 0.50, 12.55
for i, (fc, ec, lbl) in enumerate(items):
    bx = lx + i * 4.8
    box(ax, bx, ly - 0.30, 0.38, 0.28, fc, ec, 1.5, rad=0.05)
    ax.text(bx + 0.52, ly - 0.16, lbl,
            ha='left', va='center', fontsize=8.5, color=C_TXT)


plt.savefig('isomorphism.png',
            dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved isomorphism.png")
