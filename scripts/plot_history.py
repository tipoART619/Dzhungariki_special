"""График метрик по чекпойнтам: runs/<имя>/metrics_history.csv -> metrics_history.png

    python scripts/plot_history.py runs/full/metrics_history.csv
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot(csv_path):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(df.step, df.test_pcc, "o-", label="тест PCC")
    ax[0].plot(df.step, df.val_pcc, "s--", alpha=0.6, label="валидация PCC (обучение)")
    if df.ref_pcc.notna().any():
        ref = df.ref_pcc.dropna().iloc[0]
        ax[0].axhline(ref, color="gray", ls=":", label=f"опубликованная модель {ref:.3f}")
    bad = df[df.status != "ok"]
    ax[0].scatter(bad.step, bad.test_pcc, color="red", zorder=5, s=60, label="деградация")
    ax[0].set_title("PCC")
    ax[1].plot(df.step, df.test_ssim, "o-", label="тест SSIM")
    ax[1].set_title("SSIM")
    for a in ax:
        a.set_xlabel("шаг")
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    out = csv_path.with_suffix(".png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("график:", out)
    return out


if __name__ == "__main__":
    plot(sys.argv[1] if len(sys.argv) > 1 else "runs/main/metrics_history.csv")
