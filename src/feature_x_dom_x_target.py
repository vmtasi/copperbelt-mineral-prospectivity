import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

audit_csv = ROOT / "figures" / "audit" / "feature_by_target_summary.csv"
out_path = ROOT / "figures" / "audit" / "domain_delta_effect_sizes.csv"

df = pd.read_csv(audit_csv)

# Keep only Daly-domain summaries
domain_df = df[df["Grouping_Type"] == "Daly_domain"].copy()

# Required columns
required_cols = [
    "Name",
    "Deposits",
    "Non_Deposits",
    "distance_to_fault_deposit_median",
    "distance_to_fault_nondeposit_median",
    "distance_to_lithology_contact_deposit_median",
    "distance_to_lithology_contact_nondeposit_median",
    "bouguer_deposit_median",
    "bouguer_nondeposit_median",
]

missing = [c for c in required_cols if c not in domain_df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# --------------------------------------------------
# Median differences:
# Delta = Median(X | deposit) - Median(X | non-deposit)
# --------------------------------------------------

domain_df["Fault_Delta_km"] = (
    domain_df["distance_to_fault_deposit_median"]
    - domain_df["distance_to_fault_nondeposit_median"]
)

domain_df["Lithology_Delta_km"] = (
    domain_df["distance_to_lithology_contact_deposit_median"]
    - domain_df["distance_to_lithology_contact_nondeposit_median"]
)

domain_df["Gravity_Delta"] = (
    domain_df["bouguer_deposit_median"]
    - domain_df["bouguer_nondeposit_median"]
)

# --------------------------------------------------
# Relative median ratios
# --------------------------------------------------

domain_df["Fault_Median_Ratio"] = (
    domain_df["distance_to_fault_deposit_median"]
    / domain_df["distance_to_fault_nondeposit_median"]
)

domain_df["Lithology_Median_Ratio"] = (
    domain_df["distance_to_lithology_contact_deposit_median"]
    / domain_df["distance_to_lithology_contact_nondeposit_median"]
)

domain_df["Gravity_Median_Ratio"] = (
    domain_df["bouguer_deposit_median"]
    / domain_df["bouguer_nondeposit_median"]
)

# Final table
output_cols = [
    "Name",
    "Deposits",
    "Non_Deposits",
    "Fault_Delta_km",
    "Lithology_Delta_km",
    "Gravity_Delta",
    "Fault_Median_Ratio",
    "Lithology_Median_Ratio",
    "Gravity_Median_Ratio",
]

delta_table = (
    domain_df[output_cols]
    .rename(columns={"Name": "Domain"})
    .sort_values("Domain")
)

# Save the final table to the audit folder as a CSV
delta_table.to_csv(out_path, index=False)

# Print using standard string formatting instead of markdown
print(delta_table.round(3).to_string(index=False))
print(f"\nSaved to: {out_path}")