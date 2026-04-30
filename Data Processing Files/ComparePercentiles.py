import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from scipy.stats import fisher_exact
import random


def col_to_index(col):
    """Convert Excel-style column letters (A, B, C) to zero-based index."""
    if isinstance(col, int):
        return col
    col = str(col).strip().upper()
    if col.isdigit():
        return int(col)
    return ord(col[0]) - ord("A")


def clean_gene_list(df, col="A"):
    """
    Extract and clean gene IDs from one Excel column.
    Assumes:
      - row 0 contains the label/name (e.g. C4)
      - rows 1 onward contain genes
    Returns:
      (gene_set, label)
    """
    df = df.copy()
    idx = col_to_index(col)

    if idx >= df.shape[1]:
        raise ValueError(f"Column {col} is out of range for this sheet.")

    label = str(df.iloc[0, idx]).strip()
    genes = df.iloc[1:, idx].dropna().astype(str).str.strip()
    genes = genes[genes != ""]

    return set(genes), label


def pairwise_enrichment_test(name1, set1, name2, set2, universe):
    """
    Fisher's exact test:
    Is set1 enriched in set2 relative to the universe?
    """
    universe = set(universe)
    set1 = set(set1) & universe
    set2 = set(set2) & universe

    a = len(set1 & set2)              # in both
    b = len(set1 - set2)              # in set1, not set2
    c = len(set2 - set1)              # in set2, not set1
    d = len(universe - set1 - set2)   # in neither

    table = [[a, b], [c, d]]

    try:
        odds_ratio, p_value = fisher_exact(table, alternative="greater")
    except Exception:
        odds_ratio, p_value = float("nan"), float("nan")

    return {
        "Comparison": f"{name1} vs {name2}",
        "in_both": a,
        f"{name1}_only": b,
        f"{name2}_only": c,
        "in_neither": d,
        "Odds_ratio": odds_ratio,
        "P_value": p_value,
        "Table": str(table),
    }


def create_venn_diagram(set1, set2, set3, names, save_path, title):
    """Create and save a Venn diagram."""
    plt.figure(figsize=(10, 8))
    venn3([set1, set2, set3], set_labels=names)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 70)
    print("GENE OVERLAP ANALYSIS")
    print("=" * 70)

    excel_path = input("\nEnter INPUT Excel file path: ").strip()

    print("\n--- Gene Set Sheets ---")
    gene_sheet1_All = input("\nEnter the large sheet name containing ALL YOUR GENES ranked: ").strip()
    gene_sheet1 = input("\nEnter Gene Sheet name (MLP307): ").strip()
    gene_sheet2 = input("\nEnter Gene Sheet name (core): ").strip()
    gene_sheet3 = input("Enter Gene Sheet name (random): ").strip()
    gene_sheet4 = input("Enter Gene Sheet name (Hallmark): ").strip()

    gene_col = input("\nEnter gene column letter (default 'A'): ").strip() or "A"

    # Read with header=None so the first row is treated as data, not as the column name
    print("\nLoading gene sets...")
    gene_df1_All = pd.read_excel(excel_path, sheet_name=gene_sheet1_All, header=None)
    gene_df1 = pd.read_excel(excel_path, sheet_name=gene_sheet1, header=None)
    gene_df2 = pd.read_excel(excel_path, sheet_name=gene_sheet2, header=None)
    gene_df3 = pd.read_excel(excel_path, sheet_name=gene_sheet3, header=None)
    gene_df4 = pd.read_excel(excel_path, sheet_name=gene_sheet4, header=None)

    genes1, label1 = clean_gene_list(gene_df1, gene_col)
    genes2, label2 = clean_gene_list(gene_df2, gene_col)
    genes3, label3 = clean_gene_list(gene_df3, gene_col)
    genes4, label4 = clean_gene_list(gene_df4, gene_col)
    genes1_All, label_all = clean_gene_list(gene_df1_All, gene_col)

    print(f"  {gene_sheet1_All}: {len(genes1_All)} genes  | label: {label_all}")
    print(f"  {gene_sheet1}: {len(genes1)} genes        | label: {label1}")
    print(f"  {gene_sheet2}: {len(genes2)} genes        | label: {label2}")
    print(f"  {gene_sheet3}: {len(genes3)} genes        | label: {label3}")
    print(f"  {gene_sheet4}: {len(genes4)} genes        | label: {label4}")

    all_genes_set = genes1_All

    # Collect all pairwise results across thresholds
    all_gene_pairwise = []

    print("\n" + "=" * 70)
    print("GENE OVERLAP ANALYSIS")
    print("=" * 70)

    total_genes = len(all_genes_set)

    for threshold in [99, 95, 90, 75, 50, 25]:
        print(f"\n{'=' * 70}")
        print(f"THRESHOLD: {threshold}")
        print(f"{'=' * 70}")

        fraction = 1 - (threshold / 100)
        num_to_take = int(total_genes * fraction)

        # +1 because row 0 is the label, rows 1:n are genes
        threshold_df = gene_df1_All.iloc[: num_to_take + 1]
        genes_threshold, _ = clean_gene_list(threshold_df, gene_col)

        print(f"\nLoaded:")
        print(f"  Threshold {threshold} (top {fraction*100:.0f}%): {len(genes_threshold)} genes")
        print(f"  {gene_sheet2}: {len(genes2)} genes")
        print(f"  {gene_sheet3}: {len(genes3)} genes")

        only_1 = genes_threshold - genes2 - genes3
        only_2 = genes2 - genes_threshold - genes3
        only_3 = genes3 - genes_threshold - genes2

        overlap_1_2 = (genes_threshold & genes2) - genes3
        overlap_1_3 = (genes_threshold & genes3) - genes2
        overlap_2_3 = (genes2 & genes3) - genes_threshold

        overlap_all = genes_threshold & genes2 & genes3

        print("\n" + "-" * 70)
        print("GENE OVERLAP STATISTICS")
        print("-" * 70)
        print(f"\nOnly in Threshold {threshold}: {len(only_1)}")
        print(f"Only in {gene_sheet2}: {len(only_2)}")
        print(f"Only in {gene_sheet3}: {len(only_3)}")
        print(f"\nIn Threshold {threshold} ∩ {gene_sheet2} only: {len(overlap_1_2)}")
        print(f"In Threshold {threshold} ∩ {gene_sheet3} only: {len(overlap_1_3)}")
        print(f"In {gene_sheet2} ∩ {gene_sheet3} only: {len(overlap_2_3)}")
        print(f"\nIn all three sets: {len(overlap_all)}")

        print("\n" + "-" * 70)
        print("CREATING GENE VENN DIAGRAM")
        print("-" * 70)

        gene_venn_path = f"gene_venn_diagram_threshold_{threshold}.png"
        create_venn_diagram(
            genes_threshold,
            genes2,
            genes3,
            [f"Top {threshold}%", label2, label3],
            gene_venn_path,
            f"Gene Set Overlaps (Threshold {threshold})",
        )
        print(f"Gene Venn diagram saved to: {gene_venn_path}")

        print("\n" + "-" * 70)
        print("GENE PAIRWISE ENRICHMENT TESTS")
        print("-" * 70)

        gene_pairwise_results = [
            pairwise_enrichment_test(f"Threshold {threshold}", genes_threshold, gene_sheet2, genes2, all_genes_set),
            pairwise_enrichment_test(f"Threshold {threshold}", genes_threshold, gene_sheet3, genes3, all_genes_set),
        ]
        gene_pairwise_df = pd.DataFrame(gene_pairwise_results)
        gene_pairwise_df["Threshold"] = threshold
        print(gene_pairwise_df[["Comparison", "Odds_ratio", "P_value"]].to_string(index=False))
        all_gene_pairwise.append(gene_pairwise_df)

    mine_vs_core = pairwise_enrichment_test(gene_sheet1, genes1, gene_sheet2, genes2, all_genes_set)
    mine_vs_random = pairwise_enrichment_test(gene_sheet1, genes1, gene_sheet3, genes3, all_genes_set)

    hallmark_vs_core = pairwise_enrichment_test(gene_sheet4, genes4, gene_sheet2, genes2, all_genes_set)
    hallmark_vs_random = pairwise_enrichment_test(gene_sheet4, genes4, gene_sheet3, genes3, all_genes_set)

    print("\n" + "=" * 70)
    print("SUMMARY: GENE P-VALUES ACROSS ALL THRESHOLDS")
    print("=" * 70)
    summary_gene_df = pd.concat(all_gene_pairwise, ignore_index=True)
    print(summary_gene_df[["Threshold", "Comparison", "Odds_ratio", "P_value"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("SUMMARY: MLP307 AND HALLMARK VS CORE & RANDOM")
    print("=" * 70)
    summary_final_df = pd.DataFrame([mine_vs_core, mine_vs_random, hallmark_vs_core, hallmark_vs_random])
    print(summary_final_df[["Comparison", "Odds_ratio", "P_value"]].to_string(index=False))

    create_venn_diagram(
        genes1,
        genes2,
        genes3,
        [label1, label2, label3],
        "venn_mlp307_core_random.png",
        f"Venn Diagram: {label1} vs {label2} vs {label3}",
    )

    create_venn_diagram(
        genes4,
        genes2,
        genes3,
        [label4, label2, label3],
        "venn_hallmark_core_random.png",
        f"Venn Diagram: {label4} vs {label2} vs {label3}",
    )

    overlap_hallmark = set(genes4) & set(genes2)
    with open("OverlapHallmark.txt", "w") as f:
        f.write(str(overlap_hallmark))

    print("\n" + "=" * 70)
    print("SUMMARY: RANDOM VS CORE & RANDOM")
    print("=" * 70)

    random_vs_results = []
    for i in range(1, 6):
        random_genes_set = set(random.sample(list(all_genes_set), min(307, len(all_genes_set))))
        rand_vs_core = pairwise_enrichment_test(f"Random307_{i}", random_genes_set, gene_sheet2, genes2, all_genes_set)
        rand_vs_random = pairwise_enrichment_test(f"Random307_{i}", random_genes_set, gene_sheet3, genes3, all_genes_set)
        random_vs_results.append(rand_vs_core)
        random_vs_results.append(rand_vs_random)

    summary_random_df = pd.DataFrame(random_vs_results)
    print(summary_random_df[["Comparison", "Odds_ratio", "P_value"]].to_string(index=False))


    print("\n" + "=" * 70)
    print("AVERAGE OVERLAP: RANDOM307 vs MLP307 & CORE307")
    print("=" * 70)

    overlap_mlp_list = []
    overlap_core_list = []

    num_iterations = 100  # increase for better estimate

    for i in range(num_iterations):
        rand = set(random.sample(list(all_genes_set), min(307, len(all_genes_set))))

        overlap_mlp = len(rand & genes1)   # Random vs MLP307
        overlap_core = len(rand & genes2)  # Random vs Core307

        overlap_mlp_list.append(overlap_mlp)
        overlap_core_list.append(overlap_core)

    avg_mlp = sum(overlap_mlp_list) / len(overlap_mlp_list)
    avg_core = sum(overlap_core_list) / len(overlap_core_list)

    print(f"Average overlap (Random307 vs {label1}): {avg_mlp:.2f}")
    print(f"Average overlap (Random307 vs {label2}): {avg_core:.2f}")

if __name__ == "__main__":
    main()