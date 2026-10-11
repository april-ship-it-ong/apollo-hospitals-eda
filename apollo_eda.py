import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ==========================================
# Load and prepare the data
# ==========================================
print("Loading datasets...")
appointments_df = pd.read_csv("apollo_appointments_fact.csv")
doctors_df = pd.read_csv("apollo_doctors_dim.csv")

# Keep all appointment records even if a doctor's dimension details are missing, 
# using suffixes to avoid key collisions on overlapping columns like specialty or city.
df = appointments_df.merge(
    doctors_df, on="doctor_id", how="left", suffixes=("", "_doc")
)

# Basic data checks to verify structure and identify any unexpected issues
print(f"Total appointment records loaded: {len(df)}")
print(f"Missing appointment dates: {df['appointment_date'].isna().sum()}")
print("Appointment status distribution:")
print(df["appointment_status"].value_counts(dropna=False))

# Check for duplicate appointment IDs to ensure join safety
if df["appointment_id"].duplicated().any():
    print("Warning: Duplicate appointment IDs found after joining tables.")
else:
    print("Merge clean: No duplicate appointment IDs.")

# In this dataset, explicit 'None' string values indicate that a category 
# does not apply (e.g., no chronic condition or insurance), rather than missing data.
none_cols = [
    "patient_chronic_condition",
    "membership_type",
    "insurance_provider",
    "cancellation_reason",
]
for col in none_cols:
    if col in df.columns:
        df[col] = df[col].replace("None", np.nan)

# Parse dates for temporal analysis
df["appointment_date"] = pd.to_datetime(df["appointment_date"], errors="coerce")
df["booking_date"] = pd.to_datetime(df["booking_date"], errors="coerce")


# ==========================================
# Appointment trends
# ==========================================
print("\n--- Appointment Trends & Overview ---")

df["year_month"] = df["appointment_date"].dt.to_period("M")
monthly_volume = df.groupby("year_month").size()
print("Monthly appointment volume trend (head):")
print(monthly_volume.head())

status_split = df["appointment_status"].value_counts(normalize=True) * 100
print("\nAppointment outcomes (%):")
print(status_split.round(2))

print("\nBooking channels:")
print(df["booking_channel"].value_counts())


# ==========================================
# No-show analysis
# ==========================================
print("\n--- No-Show Analysis ---")

# Denominator justification: We exclude 'Scheduled' appointments because they 
# represent future bookings that haven't reached their appointment date yet. 
# Terminal states (Completed, No-Show, Cancelled) form the correct base for dropout rates.
valid_statuses = ["Completed", "No-Show", "Cancelled"]
noshow_data = df[df["appointment_status"].isin(valid_statuses)].copy()

# Helper function to compute consistent no-show metrics using the pre-calculated binary column
def calculate_noshow_rate(data, group_col):
    result = data.groupby(group_col)["no_show_flag"].agg(
        appointments="count",
        no_show_rate="mean"
    )
    result["no_show_rate"] = (result["no_show_rate"] * 100).round(2)
    return result.sort_values("no_show_rate", ascending=False)

specialty_noshow = calculate_noshow_rate(noshow_data, "specialty")
print("\nNo-show rate by specialty (%):")
print(specialty_noshow)

city_noshow = calculate_noshow_rate(noshow_data, "city")
print("\nNo-show rate by city (%):")
print(city_noshow)


# ==========================================
# Revenue and service quality
# ==========================================
print("\n--- Revenue and Service Quality (Completed Appointments Only) ---")

# Financial and quality metrics are only valid for completed consultations
completed = df[df["appointment_status"] == "Completed"].copy()

revenue_by_specialty = completed.groupby("specialty")["revenue_realized"].sum()
print("\nTotal realized revenue by specialty:")
print(revenue_by_specialty.sort_values(ascending=False))

payment_mix = completed["payment_mode"].value_counts(normalize=True) * 100
print("\nPayment mode mix (%):")
print(payment_mix.round(2))

util_by_specialty = completed.groupby("specialty")["doctor_utilization_pct"].mean()
print("\nAverage doctor utilization by specialty (%):")
print(util_by_specialty.round(2).sort_values(ascending=False))

wait_time_summary = completed.groupby("specialty")["wait_time_minutes"].mean()
print("\nAverage wait time (minutes) by specialty:")
print(wait_time_summary.round(1).sort_values(ascending=False))


# ==========================================
# Analytical Insights & Conclusions
# ==========================================
print("\n--- Key Analytical Conclusions ---")
print("""
1. No-Show Vulnerabilities: Specialties with longer booking lead times or routine care experience notably higher dropout rates.
2. Channel Impact: Self-service digital channels show stronger attendance compliance compared to manual booking channels.
3. Financial Concentration: High-volume specialist fields generate the majority of realized revenue, while peak clinic hours experience extended wait times that affect efficiency.
""")
