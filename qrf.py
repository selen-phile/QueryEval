import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_main_attention(attn_csv, outdir):

    df = pd.read_csv(attn_csv)

    # Mean absolute attention change
    agg = (
        df.groupby(["layer", "head"])["query_mass_delta"]
          .apply(lambda x: x.abs().mean())
          .reset_index(name="mean_delta")
    )

    top = (
        agg.sort_values("mean_delta", ascending=False)
           .head(20)
    )

    labels = [
    f"L{int(l)}H{int(h)}"
    for l, h in zip(top["layer"], top["head"])
]

    plt.figure(figsize=(7,6))

    plt.barh(
        labels[::-1],
        top["mean_delta"].values[::-1]
    )

    plt.xlabel("Mean |Attention Mass Change|")
    plt.ylabel("Attention Head")
    plt.title("Top-20 Perturbation-sensitive Attention Heads")

    plt.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)

    plt.savefig(
        os.path.join(outdir, "main_attention_heads.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved main_attention_heads.png")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--attn_csv",
        default="attention_results/grand_attention.csv"
    )

    parser.add_argument(
        "--outdir",
        default="plots_main"
    )

    args = parser.parse_args()

    plot_main_attention(
        args.attn_csv,
        args.outdir
    )


if __name__ == "__main__":
    main()