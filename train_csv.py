import sys
import json
import pandas as pd

# Input CSV file
path = sys.argv[1] if len(sys.argv) > 1 else "predictions.csv"

# ---------- 1. Load and clean ----------
df = pd.read_csv(path)

# Normalize column names
df.columns = [
    c.strip().lower().replace(" ", "_")
    for c in df.columns
]

# Required columns for YOUR CSV
required = [
    "student_id",
    "group",
    "attendance_pct",
    "shortage_risk_pct",
    "risk_level"
]

for col in required:
    if col not in df.columns:
        sys.exit(
            f"Missing column '{col}'. "
            f"Available columns: {df.columns.tolist()}"
        )

# Remove duplicate students
df = df.drop_duplicates("student_id").copy()

# Convert percentage columns to numbers
df["attendance_pct"] = pd.to_numeric(
    df["attendance_pct"], errors="coerce"
)

df["shortage_risk_pct"] = pd.to_numeric(
    df["shortage_risk_pct"], errors="coerce"
)

# Remove rows with missing percentages
df = df.dropna(
    subset=["attendance_pct", "shortage_risk_pct"]
)

# Keep percentages within 0-100
df["attendance_pct"] = df["attendance_pct"].clip(0, 100)
df["shortage_risk_pct"] = df["shortage_risk_pct"].clip(0, 100)

# Normalize risk labels
df["risk_level"] = (
    df["risk_level"]
    .astype(str)
    .str.strip()
    .str.title()
)

# ---------- 2. Recommendations ----------
def get_recommendation(row):
    attendance = row["attendance_pct"]
    risk = row["risk_level"]

    if risk == "High":
        return "High shortage risk: review attendance and provide support"

    if risk == "Medium":
        return "Monitor attendance and encourage regular participation"

    if attendance >= 75:
        return "Attendance is on track: continue attending regularly"

    return "Review attendance and encourage regular participation"


df["recommendation"] = df.apply(
    get_recommendation, axis=1
)

# Exact classes needed cannot be calculated because
# total_classes and classes_attended are not in this CSV.
df["classes_needed_for_75"] = None

# ---------- 3. Prepare dashboard output ----------
out = df[
    [
        "student_id",
        "group",
        "attendance_pct",
        "shortage_risk_pct",
        "risk_level",
        "classes_needed_for_75",
        "recommendation"
    ]
].copy()

# Highest risk first
out = out.sort_values(
    "shortage_risk_pct",
    ascending=False
)

# Do not overwrite the input predictions.csv
out.to_csv("dashboard_predictions.csv", index=False)

# Create JavaScript data for the dashboard
json_data = out.to_json(orient="records")

with open("predictions.js", "w", encoding="utf-8") as f:
    f.write("window.PRED = " + json_data + ";\n")

print(f"Loaded {len(out)} students")
print("Risk levels:")
print(out["risk_level"].value_counts())

print("\nSaved dashboard_predictions.csv")
print("Saved predictions.js")
print("\nNote: Existing risk scores were used.")
print("No new ML model was trained.")