"""Shared chart settings for the site (palette slots from the validated reference palette)."""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

DATA = Path(__file__).parent / "data"
LEAGUE = {"NBA": "#2a78d6", "WNBA": "#eb6834"}          # categorical slots 1-2 (validated pair)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # first four slots, fixed order
SEQ_BLUE = [[0, "#cde2fb"], [0.33, "#6da7ec"], [0.66, "#2a78d6"], [1, "#0d366b"]]

pio.templates["hoops"] = go.layout.Template(layout=dict(
    font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13, color="#0b0b0b"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor="#e4e3df", zeroline=False, linecolor="rgba(0,0,0,0)", automargin=True),
    yaxis=dict(gridcolor="#e4e3df", zeroline=False, automargin=True),
    hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", y=1.08, x=0),
))
pio.templates.default = "hoops"


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


def line(fig: go.Figure, x, y, name: str, color: str, **kw) -> None:
    fig.add_scatter(x=x, y=y, name=name, mode="lines+markers", line=dict(color=color, width=2),
                    marker=dict(size=8, color=color, line=dict(width=2, color="rgba(252,252,251,1)")), **kw)


def show(fig: go.Figure) -> None:
    """Print a chart as a raw HTML block (cells run with `output: asis`). Emitting it as Jupyter HTML would
    make Quarto add require.js, which stops plotly.js (and Quarto's search) from setting their globals."""
    html = fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})
    print("\n```{=html}\n" + html + "\n```\n")


def raw_html(html: str) -> None:
    print("\n```{=html}\n" + html + "\n```\n")


def md(text: str) -> None:
    print("\n" + text + "\n")


def end_labels(fig: go.Figure, ends: dict, x, min_gap: float) -> None:
    """Direct labels at line ends, spread apart by at least min_gap and re-centred on the
    lines' own end values, so each label stays as close to its line as possible."""
    items = sorted(ends.items(), key=lambda kv: kv[1])
    ys = [y for _, y in items]
    placed = []
    for y in ys:
        placed.append(max(y, placed[-1] + min_gap) if placed else y)
    shift = (sum(ys) - sum(placed)) / len(ys)
    placed = [p + shift for p in placed]
    for (name, _), y2 in zip(items, placed):
        fig.add_annotation(x=x, y=y2, text=name, showarrow=False, xanchor="left", xshift=10, font=dict(size=12))
