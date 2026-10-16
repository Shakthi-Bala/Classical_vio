"""Shared plot colours and axes style (validated categorical palette, light theme)."""

TEXT = '#52514e'
TITLE = '#0b0b0b'
GRID = '#e6e5e1'
SPINE = '#d4d3cf'
SURFACE = '#fcfcfb'
NEUTRAL = '#8a8984'   # ground truth, reference lines, "before"

# One colour per method, the same in every figure.
PYTHON = '#2a78d6'
CPP_MSCKF = '#eb6834'
CPP_SWO = '#1baf7a'
VINS = '#eda100'
FAIL = '#d03b3b'


def style_axes(ax, grid='both'):
    """Recessive grid and spines, muted tick labels."""
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(SPINE)
    ax.tick_params(colors=TEXT)
