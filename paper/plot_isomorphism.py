"""
Recreate the PagedAttention--Block-Sparsity Isomorphism diagram
with generous spacing — no label overlap.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Colour palette ──
C_PREFIX     = '#D6EAF8'
C_PREFIX_E   = '#85C1E9'
C_NEW        = '#FDEBD0'
C_NEW_E      = '#F0B27A'
C_SELECTED   = '#F1948A'
C_SELECTED_E = '#E74C3C'
C_UNACCESSED = '#F2F3F4'
C_UNACCESSED_E = '#BDC3C7'
C_NAVY       = '#1B2631'
C_DARK       = '#2C3E50'
C_BG         = '#FAFBFD'

fig, ax = plt.subplots(1, 1, figsize=(26, 16))
ax.set_xlim(-1.5, 26)
ax.set_ylim(-3.5, 13.5)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

# ════════════════════════════════════════════════════════════════
#  LEGEND
# ════════════════════════════════════════════════════════════════
legend_y = 13.0
legend_items = [
    (C_SELECTED, C_SELECTED_E, 'Selected block $(r,c)$'),
    (C_PREFIX,   C_PREFIX_E,   'Prefix KV (cached, shared)'),
    (C_NEW,      C_NEW_E,      'New-token KV (current request)'),
    (C_UNACCESSED, C_UNACCESSED_E, 'Unaccessed page (skipped)'),
]
lx = 1.5
for fc, ec, label in legend_items:
    ax.add_patch(FancyBboxPatch((lx, legend_y - 0.25), 0.55, 0.55,
                 boxstyle='round,pad=0.02', fc=fc, ec=ec, lw=1.3))
    ax.text(lx + 0.8, legend_y + 0.02, label,
            fontsize=11, va='center', fontfamily='sans-serif')
    lx += 5.8

# ════════════════════════════════════════════════════════════════
#  ATTENTION BLOCK GRID
# ════════════════════════════════════════════════════════════════
grid_x0, grid_y0 = 0, 5.0
bw, bh = 1.4, 1.6
n_cols, n_rows = 6, 3
prefix_cols = 4

# Title
ax.text(grid_x0 + n_cols * bw / 2, grid_y0 + n_rows * bh + 1.3,
        'Attention Block Grid', fontsize=16, fontweight='bold',
        ha='center', va='center', fontfamily='sans-serif')

# Column colour bars + labels
for c in range(n_cols):
    cx = grid_x0 + c * bw + bw / 2
    bar_color = C_PREFIX_E if c < prefix_cols else C_NEW_E
    ax.add_patch(FancyBboxPatch((grid_x0 + c * bw + 0.06, grid_y0 + n_rows * bh + 0.2),
                 bw - 0.12, 0.4, boxstyle='round,pad=0.02',
                 fc=bar_color, ec='none', alpha=0.5))
    ax.text(cx, grid_y0 + n_rows * bh + 0.4, f'$c$={c}',
            fontsize=10, ha='center', va='center', fontfamily='sans-serif',
            fontweight='bold')

# Bracket labels
prefix_mid = grid_x0 + prefix_cols * bw / 2
new_mid = grid_x0 + (prefix_cols + n_cols) * bw / 2
ax.text(prefix_mid, grid_y0 + n_rows * bh + 0.95,
        '$n_p$: prefix columns', fontsize=10, ha='center', va='center',
        color='#2980B9', style='italic', fontfamily='sans-serif')
ax.text(new_mid, grid_y0 + n_rows * bh + 0.95,
        '$n_q$: new cols', fontsize=10, ha='center', va='center',
        color='#E67E22', style='italic', fontfamily='sans-serif')

# Row labels
for r in range(n_rows):
    ry = grid_y0 + (n_rows - 1 - r) * bh + bh / 2
    ax.text(grid_x0 - 0.35, ry, f'$r$={r}',
            fontsize=10, ha='right', va='center', fontfamily='sans-serif',
            fontweight='bold')

# Y-axis label
ax.text(grid_x0 - 1.1, grid_y0 + n_rows * bh / 2,
        'Query rows ($n_q$)', fontsize=11, ha='center', va='center',
        rotation=90, fontfamily='sans-serif')

# Selected blocks
selected = {(0, 0), (0, 3), (0, 5),
            (1, 1), (1, 3), (1, 4),
            (2, 2), (2, 4), (2, 5)}

for r in range(n_rows):
    for c in range(n_cols):
        x = grid_x0 + c * bw
        y = grid_y0 + (n_rows - 1 - r) * bh

        if (r, c) in selected:
            fc, ec = C_SELECTED, C_SELECTED_E
        elif c < prefix_cols:
            fc, ec = C_PREFIX, C_PREFIX_E
        else:
            fc, ec = C_NEW, C_NEW_E

        ax.add_patch(FancyBboxPatch((x + 0.05, y + 0.05), bw - 0.1, bh - 0.1,
                     boxstyle='round,pad=0.03', fc=fc, ec=ec, lw=1))

        if (r, c) in selected:
            ax.text(x + bw / 2, y + bh / 2, f'({r},{c})',
                    fontsize=9, ha='center', va='center',
                    fontweight='bold', color=C_DARK, fontfamily='sans-serif')

# ════════════════════════════════════════════════════════════════
#  CONNECTING ARROW:  Grid  →  Block Table
# ════════════════════════════════════════════════════════════════
grid_right = grid_x0 + n_cols * bw
bt_x0 = 12.5         # generous gap
arrow_mid_y = grid_y0 + n_rows * bh / 2

ax.annotate('', xy=(bt_x0 - 0.2, arrow_mid_y),
            xytext=(grid_right + 0.2, arrow_mid_y),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#95A5A6',
                            connectionstyle='arc3,rad=0'))
ax.text((grid_right + bt_x0) / 2, arrow_mid_y + 0.5,
        'column $c$  $\\longrightarrow$  lookup $\\varphi(c)$',
        fontsize=10, ha='center', va='bottom', color='#7F8C8D',
        fontfamily='sans-serif', style='italic')

# ════════════════════════════════════════════════════════════════
#  BLOCK TABLE φ
# ════════════════════════════════════════════════════════════════
bt_y0 = 5.0
bt_w = 3.5
bt_row_h = 0.9

# Title — well above header
ax.text(bt_x0 + bt_w / 2, bt_y0 + n_cols * bt_row_h + 1.5,
        'Block Table $\\varphi$', fontsize=16, fontweight='bold',
        ha='center', va='center', fontfamily='sans-serif')

# Header row
hdr_y = bt_y0 + n_cols * bt_row_h
ax.add_patch(FancyBboxPatch((bt_x0, hdr_y),
             bt_w / 2, bt_row_h, boxstyle='round,pad=0.02',
             fc=C_NAVY, ec=C_NAVY, lw=1))
ax.text(bt_x0 + bt_w / 4, hdr_y + bt_row_h / 2,
        'Logical', fontsize=10, ha='center', va='center',
        color='white', fontweight='bold', fontfamily='sans-serif')
ax.add_patch(FancyBboxPatch((bt_x0 + bt_w / 2, hdr_y),
             bt_w / 2, bt_row_h, boxstyle='round,pad=0.02',
             fc=C_NAVY, ec=C_NAVY, lw=1))
ax.text(bt_x0 + 3 * bt_w / 4, hdr_y + bt_row_h / 2,
        'Physical $\\varphi(c)$', fontsize=10, ha='center', va='center',
        color='white', fontweight='bold', fontfamily='sans-serif')

# Data rows
bt_map = {0: 5, 1: 2, 2: 7, 3: 1, 4: 6, 5: 9}

for i, (logical, physical) in enumerate(bt_map.items()):
    row_y = bt_y0 + (n_cols - 1 - i) * bt_row_h

    bg = C_PREFIX if logical < prefix_cols else C_NEW
    edge = C_PREFIX_E if logical < prefix_cols else C_NEW_E

    ax.add_patch(FancyBboxPatch((bt_x0 + 0.03, row_y + 0.03),
                 bt_w / 2 - 0.06, bt_row_h - 0.06,
                 boxstyle='round,pad=0.02', fc=bg, ec=edge, lw=0.8))
    ax.text(bt_x0 + bt_w / 4, row_y + bt_row_h / 2, str(logical),
            fontsize=12, ha='center', va='center',
            fontweight='bold', fontfamily='sans-serif')

    ax.add_patch(FancyBboxPatch((bt_x0 + bt_w / 2 + 0.03, row_y + 0.03),
                 bt_w / 2 - 0.06, bt_row_h - 0.06,
                 boxstyle='round,pad=0.02', fc=bg, ec=edge, lw=0.8))
    ax.text(bt_x0 + 3 * bt_w / 4, row_y + bt_row_h / 2, str(physical),
            fontsize=12, ha='center', va='center',
            fontweight='bold', fontfamily='sans-serif')

# ════════════════════════════════════════════════════════════════
#  PHYSICAL KV PAGE POOL  (bottom — generous spacing)
# ════════════════════════════════════════════════════════════════
pool_y = 0.3
page_w, page_h = 1.8, 1.6
pool_pages = [1, 2, 3, 4, 5, 6, 7, 8, 9]
n_pages = len(pool_pages)

pool_total_w = 24.0
pool_x0 = 0.5
page_gap = (pool_total_w - n_pages * page_w) / (n_pages - 1)

prefix_pages = {5, 2, 7, 1}
new_pages = {6, 9}

# Title
ax.text(pool_x0 + pool_total_w / 2, pool_y + page_h + 1.35,
        'Physical KV Page Pool (GPU DRAM)', fontsize=16, fontweight='bold',
        ha='center', va='center', fontfamily='sans-serif')
ax.text(pool_x0 + pool_total_w / 2, pool_y + page_h + 0.8,
        'non-contiguous — managed by PagedAttention block table',
        fontsize=10, ha='center', va='center', color='#7F8C8D',
        style='italic', fontfamily='sans-serif')

for idx, page_num in enumerate(pool_pages):
    px = pool_x0 + idx * (page_w + page_gap)

    if page_num in prefix_pages:
        fc, ec = C_PREFIX, C_PREFIX_E
    elif page_num in new_pages:
        fc, ec = C_NEW, C_NEW_E
    else:
        fc, ec = C_UNACCESSED, C_UNACCESSED_E

    ax.add_patch(FancyBboxPatch((px, pool_y), page_w, page_h,
                 boxstyle='round,pad=0.05', fc=fc, ec=ec, lw=1.3))

    ax.text(px + page_w / 2, pool_y + page_h / 2,
            f'Page\n{page_num}', fontsize=11, ha='center', va='center',
            fontweight='bold', fontfamily='sans-serif', color=C_DARK,
            linespacing=1.3)

    if page_num not in prefix_pages and page_num not in new_pages:
        ax.add_patch(FancyBboxPatch((px, pool_y), page_w, page_h,
                     boxstyle='round,pad=0.05', fc='none', ec=C_UNACCESSED_E,
                     lw=1.5, ls='--'))

# ════════════════════════════════════════════════════════════════
#  ARROWS: Block Table → Physical Pages
# ════════════════════════════════════════════════════════════════
arrow_colors = {
    0: '#2980B9', 1: '#2980B9', 2: '#2980B9', 3: '#2980B9',
    4: '#E67E22', 5: '#E67E22',
}

# Curvatures tuned per arrow so they don't cross
arrow_rads = {0: 0.25, 1: 0.12, 2: -0.20, 3: 0.30, 4: -0.08, 5: -0.25}

for logical, physical in bt_map.items():
    row_idx = logical
    start_y = bt_y0 + (n_cols - 1 - row_idx) * bt_row_h
    start_x = bt_x0 + 3 * bt_w / 4

    page_idx = pool_pages.index(physical)
    end_x = pool_x0 + page_idx * (page_w + page_gap) + page_w / 2
    end_y = pool_y + page_h

    color = arrow_colors[logical]
    rad = arrow_rads[logical]

    ax.annotate('',
                xy=(end_x, end_y + 0.08),
                xytext=(start_x, start_y - 0.08),
                arrowprops=dict(
                    arrowstyle='->', lw=1.8,
                    color=color, alpha=0.55,
                    connectionstyle=f'arc3,rad={rad}',
                    shrinkA=3, shrinkB=3,
                ))

# ════════════════════════════════════════════════════════════════
#  BOTTOM CAPTION
# ════════════════════════════════════════════════════════════════
caption_y = -1.6
cap_w, cap_h = 20, 1.2
cap_x = (26 - cap_w) / 2 - 0.5

ax.add_patch(FancyBboxPatch((cap_x, caption_y - cap_h / 2),
             cap_w, cap_h, boxstyle='round,pad=0.1',
             fc='#EBF5FB', ec='#85C1E9', lw=1.5))
ax.text(cap_x + cap_w / 2, caption_y,
        'Attention Block $(r, c)$  $\\longleftrightarrow$  '
        'Physical KV Page $\\varphi(c)$'
        '    [same index $c$ — no gather, no data reorganisation]',
        fontsize=13, ha='center', va='center',
        fontweight='bold', fontfamily='sans-serif', color=C_DARK)

plt.tight_layout()
plt.savefig('/Users/helloawepoworld/Downloads/paper/tex/images/isomorphism.png',
            dpi=200, bbox_inches='tight', facecolor=C_BG, pad_inches=0.3)
plt.savefig('/Users/helloawepoworld/Downloads/paper/tex/images/isomorphism.pdf',
            bbox_inches='tight', facecolor=C_BG, pad_inches=0.3)
print("Saved isomorphism.png and isomorphism.pdf")
