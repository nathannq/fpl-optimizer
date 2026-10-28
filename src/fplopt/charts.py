"""README result charts (static PNGs, light and dark), drawn from saved evaluation outputs.

* ``differentials-{light,dark}.png`` - how the top-10 differentials did on the test seasons:
  return rate and mean points per pick, model vs baselines (two small multiples, one measure each).
* ``backtest-{light,dark}.png`` - cumulative actual points when the optimizer is driven by the
  model vs the last-3 baseline, one panel per replayed season.

Colours follow a validated two-slot categorical palette (blue, orange); text stays in ink tones.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from fplopt.config import REPO_ROOT, Config  # noqa: E402
from fplopt.evaluate import differential_metrics  # noqa: E402
from fplopt.models.report import evaluation_frame  # noqa: E402

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834"], "neutral": "#898781"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "series": ["#3987e5", "#d95926"], "neutral": "#898781"},
}
OUT_DIR = REPO_ROOT / "docs" / "images"
DPI = 200


def _style(fig, axes, t: dict) -> None:
    fig.patch.set_facecolor(t["surface"])
    for ax in axes:
        ax.set_facecolor(t["surface"])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(t["axis"])
        ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0, labelsize=9)


def _rounded_hbar(ax, y: float, width: float, height: float, color: str, radius_px: float = 4) -> None:
    """Horizontal bar with a rounded data end and a square baseline end."""
    fig = ax.figure
    fig.canvas.draw()
    (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (1, 1)])
    px_per_x, px_per_y = abs(x1 - x0), abs(y1 - y0)
    r = min(radius_px / px_per_x, width / 2)
    ax.add_patch(FancyBboxPatch((0, y - height / 2), width, height, boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=px_per_x / px_per_y, linewidth=0, facecolor=color))
    ax.add_patch(Rectangle((0, y - height / 2), min(width, 2 * r), height, linewidth=0, facecolor=color))


def differential_numbers(oos: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return rate and mean points of each ranking's weekly top-k differentials on the test seasons."""
    ev = evaluation_frame(oos, {"haul_weight": 0.0, "edge_weight": 0.0, "price_weight": 0.0}, cfg)
    test = ev[ev["split"] == "test"]
    k, own = cfg.evaluation.top_k, cfg.differential.ownership_max_pct
    ret, haul = cfg.targets.return_points, cfg.targets.haul_points
    rows = []
    for label, col in [("LightGBM model", "lgbm_points"), ("Ridge model", "ridge_points"),
                       ("Last-3 average", "base_last3")]:
        m = differential_metrics(test, col, k, own, ret, haul)
        rows.append({"label": label, "return_rate": 100 * m[f"precision@{k}"], "mean_points": m["mean_points"],
                     "gameweeks": m["gameweeks"]})
    m = differential_metrics(test, "lgbm_points", k, own, ret, haul)
    rows.append({"label": "Any differential", "return_rate": 100 * m["pool_return_rate"],
                 "mean_points": m["pool_mean_points"], "gameweeks": m["gameweeks"]})
    return pd.DataFrame(rows)


def differentials_chart(numbers: pd.DataFrame, cfg: Config, theme: str, path: Path) -> Path:
    t = THEMES[theme]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), dpi=DPI)
    _style(fig, axes, t)
    ys = list(range(len(numbers)))[::-1]
    panels = [("return_rate", f"Returned {cfg.targets.return_points}+ points", lambda v: f"{v:.0f}%"),
              ("mean_points", "Mean points per pick", lambda v: f"{v:.2f}")]
    for ax, (col, title, fmt) in zip(axes, panels):
        top = numbers[col].max()
        ax.set_xlim(0, top * 1.22)
        ax.set_ylim(-0.6, len(numbers) - 0.4)
        ax.xaxis.grid(True, color=t["grid"], linewidth=1)
        ax.set_axisbelow(True)
        ax.set_yticks(ys, numbers["label"] if ax is axes[0] else [""] * len(ys))
        ax.tick_params(axis="x", labelbottom=False)
        for y, (_, row) in zip(ys, numbers.iterrows()):
            color = t["series"][0] if row["label"].startswith("LightGBM") else t["neutral"]
            _rounded_hbar(ax, y, row[col], 0.42, color)
            ax.text(row[col] + top * 0.02, y, fmt(row[col]), va="center", ha="left", fontsize=9.5,
                    color=t["ink"], fontweight="bold" if row["label"].startswith("LightGBM") else "normal")
        ax.set_title(title, loc="left", fontsize=10.5, color=t["ink"], fontweight="bold", pad=8)
    gws = int(numbers["gameweeks"].iloc[0])
    fig.suptitle(f"Top {cfg.evaluation.top_k} differentials each gameweek, owned by under "
                 f"{cfg.differential.ownership_max_pct:g}%", x=0.012, ha="left", fontsize=12.5,
                 color=t["ink"], fontweight="bold", y=0.99)
    fig.text(0.012, 0.875, f"Unseen test seasons ({', '.join(cfg.model.test_seasons)}), {gws} gameweeks; "
             "'any differential' is the average of everyone under the threshold.",
             fontsize=9, color=t["ink2"], ha="left")
    fig.subplots_adjust(left=0.15, right=0.985, top=0.72, bottom=0.05, wspace=0.08)
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)
    return path


def backtest_chart(weeks: pd.DataFrame, theme: str, path: Path) -> Path:
    t = THEMES[theme]
    seasons = sorted(weeks["season"].unique())
    fig, axes = plt.subplots(1, len(seasons), figsize=(10, 3.8), dpi=DPI, sharey=True)
    axes = list(axes) if len(seasons) > 1 else [axes]
    _style(fig, axes, t)
    sources = [("model", "Model projections"), ("last3", "Last-3 average")]
    top = weeks.groupby(["season", "source"])["points"].sum().max()
    for ax, season in zip(axes, seasons):
        part = weeks[weeks["season"] == season]
        ax.yaxis.grid(True, color=t["grid"], linewidth=1)
        ax.set_axisbelow(True)
        for (src, label), color in zip(sources, t["series"]):
            s = part[part["source"] == src].sort_values("gw")
            if s.empty:
                continue
            cum = s["points"].cumsum()
            ax.plot(s["gw"], cum, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                    label=label)
            ax.scatter([s["gw"].iloc[-1]], [cum.iloc[-1]], s=60, color=color, edgecolors=t["surface"],
                       linewidths=2, zorder=3)
            ax.annotate(f"{cum.iloc[-1]:,.0f}", (s["gw"].iloc[-1], cum.iloc[-1]), xytext=(7, 0),
                        textcoords="offset points", va="center", fontsize=9.5, color=t["ink"],
                        fontweight="bold" if src == "model" else "normal")
        ax.set_title(season, loc="left", fontsize=10.5, color=t["ink"], fontweight="bold", pad=8)
        ax.set_xlim(part["gw"].min() - 0.5, part["gw"].max() + 5)
        ax.set_xticks([2, 10, 20, 30, 38])
        ax.set_xlabel("Gameweek", fontsize=9, color=t["muted"])
        ax.set_ylim(0, top * 1.08)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.1f}K".replace(".0K", "K") if v else "0"))
    legend = axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(t["ink2"])
    fig.suptitle("Season replay: actual points when the optimizer follows each projection", x=0.012, ha="left",
                 fontsize=12.5, color=t["ink"], fontweight="bold", y=0.99)
    fig.text(0.012, 0.875, "Same optimizer and rules (transfers, -4 hits, selling prices, autosubs); "
             "only the projections differ. Out-of-sample, no chips.", fontsize=9, color=t["ink2"], ha="left")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.75, bottom=0.14, wspace=0.08)
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)
    return path


def make_charts(cfg: Config, out_dir: Path = OUT_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    oos = pd.read_parquet(cfg.paths.oos_predictions)
    numbers = differential_numbers(oos, cfg)
    weeks_file = cfg.paths.processed / "backtest_weeks.parquet"
    weeks = pd.read_parquet(weeks_file) if weeks_file.exists() else None
    for theme in THEMES:
        paths.append(differentials_chart(numbers, cfg, theme, out_dir / f"differentials-{theme}.png"))
        if weeks is not None:
            paths.append(backtest_chart(weeks, theme, out_dir / f"backtest-{theme}.png"))
    return paths
