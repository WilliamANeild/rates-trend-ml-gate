import pandas as pd
import numpy as np
from pathlib import Path

def build_trade_tickets(current_weights: pd.Series,
                        target_weights: pd.Series) -> pd.DataFrame:
    """
    Create a trade ticket table showing deltas between current and target weights.

    Parameters
    ----------
    current_weights : pd.Series
        Current portfolio weights (index = tickers).
    target_weights : pd.Series
        Target portfolio weights (index = tickers).

    Returns
    -------
    pd.DataFrame
        Table with columns: ['Ticker', 'TradeDelta', 'Direction'].
    """
    # Align tickers and compute deltas
    aligned = pd.concat([current_weights, target_weights], axis=1).fillna(0)
    aligned.columns = ["Current", "Target"]
    aligned["TradeDelta"] = aligned["Target"] - aligned["Current"]

    # Round for readability
    aligned["TradeDelta"] = aligned["TradeDelta"].round(4)

    # Label direction
    aligned["Direction"] = np.where(
        aligned["TradeDelta"] > 0, "Buy",
        np.where(aligned["TradeDelta"] < 0, "Sell", "Hold")
    )

    trade_df = aligned[["TradeDelta", "Direction"]].reset_index().rename(columns={"index": "Ticker"})
    return trade_df


def write_weights(weights_df: pd.DataFrame, path: str) -> None:
    """
    Save the full weights dataframe to a CSV.

    Parameters
    ----------
    weights_df : pd.DataFrame
        DataFrame of portfolio weights indexed by date.
    path : str
        Path to save CSV file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights_df.round(4).to_csv(path, index=True)
    print(f"✅ Saved weights to {path}")


def write_recommendation(weights_df: pd.DataFrame,
                         trade_df: pd.DataFrame,
                         rationale_dict: dict,
                         risk_flags_df: pd.DataFrame,
                         out_dir: str,
                         charts: list | None = None) -> None:
    """
    Generate a Markdown recommendation report summarizing allocations, rationale, and trades.

    Parameters
    ----------
    weights_df : pd.DataFrame
        DataFrame of target weights (indexed by date).
    trade_df : pd.DataFrame
        Trade delta table from build_trade_tickets().
    rationale_dict : dict
        Dictionary of rationale text per date.
    risk_flags_df : pd.DataFrame
        Boolean DataFrame of risk control flags.
    out_dir : str
        Directory to save markdown file and optional charts.
    charts : list, optional
        List of image paths to embed at bottom of markdown.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latest_date = weights_df.index[-1]
    weights = weights_df.loc[latest_date]
    rationale = rationale_dict.get(latest_date, "No rationale provided.")
    flags = risk_flags_df.loc[latest_date]

    # --- Build markdown text ---
    md_text = f"### Weekly Allocation Update — {latest_date:%Y-%m-%d}\n\n"
    md_text += f"**Rationale:** {rationale}\n\n"

    # Allocation table
    md_text += "| Ticker | Weight |\n|:--|--:|\n"
    for ticker, w in weights.items():
        md_text += f"| {ticker} | {w:.2%} |\n"
    md_text += "\n"

    # Risk flags
    active_flags = [k for k, v in flags.items() if v]
    if active_flags:
        md_text += f"**Active Risk Flags:** {', '.join(active_flags)}\n\n"
    else:
        md_text += "**Active Risk Flags:** None\n\n"

    # Trade table
    md_text += "#### Trade Tickets\n"
    md_text += trade_df.to_markdown(index=False)
    md_text += "\n\n"

    # Optional charts
    if charts:
        md_text += "#### Charts\n"
        for c in charts:
            md_text += f"![]({c})\n\n"

    # Write file
    out_path = out_dir / "recommendation.md"
    with open(out_path, "w") as f:
        f.write(md_text)

    print(f"✅ Saved recommendation markdown to {out_path}")
