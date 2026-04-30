import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3


def main():
    excel_path = input("\nEnter INPUT Excel file path: ").strip()
    sheet = input("Enter sheet name: ").strip()

    cols_input = input("Enter gene set columns (comma-separated, e.g. C4,C6,Hallmark): ").strip()
    columns = [c.strip() for c in cols_input.split(",") if c.strip()]

    sheet_df = pd.read_excel(excel_path, sheet_name=sheet)

    gene_sets = []
    labels = []

    for col in columns:
        if col not in sheet_df.columns:
            print(f"Column '{col}' not found. Skipping.")
            continue

        genes = set(sheet_df[col].dropna().astype(str).str.strip())
        gene_sets.append(genes)
        labels.append(col)

        print(f"{col} - {len(genes)}")

    if not gene_sets:
        print("No valid columns provided.")
        return

    # Intersection across all sets
    union = set.union(*gene_sets)
    intersection = sorted(set.intersection(*gene_sets))
    everything_else = union - set(intersection)


    print(f"\nIntersection across {len(gene_sets)} sets: {len(intersection)}\n")
    print("\t".join(sorted(intersection)))


    # for gene in everything_else:
    #     print(gene, sep="\t")
    # print("\n")

    if len(gene_sets) == 2:
        venn2(gene_sets, set_labels=labels)
        plt.title("Venn Diagram (2 sets)")
        plt.savefig("Overlap.png", dpi=300, bbox_inches='tight')
        plt.close()

    elif len(gene_sets) == 3:
        venn3(gene_sets, set_labels=labels)
        plt.title("Venn Diagram (3 sets)")
        plt.savefig("Overlap.png", dpi=300, bbox_inches='tight')
        plt.close()

    else:
        print("\nVenn diagram only supports 2 or 3 sets.")


if __name__ == "__main__":
    main()