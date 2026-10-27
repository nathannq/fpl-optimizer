"""Small helpers for writing results as Markdown."""
from __future__ import annotations

import pandas as pd


def _fmt(value) -> str:
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        return f"{value:,.0f}" if value.is_integer() else f"{value:.3f}"
    return str(value)


def markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *body])
