import pandas as pd

# GHCN parser
def parse_ghcn_dly(file_path):
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            station = line[0:11]
            year = int(line[11:15])
            month = int(line[15:17])
            element = line[17:21]
            for day in range(1, 32):
                value_str = line[21 + (day - 1) * 8:26 + (day - 1) * 8].strip()
                value = int(value_str) if value_str else -9999
                if value != -9999:
                    date = pd.to_datetime(f"{year}-{month:02d}-{day:02d}", errors='coerce')
                    if pd.notnull(date):
                        if element in ["TMAX", "TMIN", "TAVG", "PRCP"]:
                            value = value / 10.0
                        records.append((station, date, element, value))
    return pd.DataFrame(records, columns=["station", "date", "element", "value"])

# Parse both stations
temps_df = parse_ghcn_dly("USR0000CLAH.dly")         # temp-only
precip_df = parse_ghcn_dly("USW00093134.dly")        # with PRCP

# Pivot temp station
temps_pivot = temps_df.pivot(index="date", columns="element", values="value").reset_index()

# Extract PRCP from precipitation station
prcp_only = precip_df[precip_df["element"] == "PRCP"].pivot(index="date", columns="element", values="value").reset_index()

# Merge datasets on date
merged_df = temps_pivot.merge(prcp_only, on="date", how="left")

# Sort and add 6-month rolling avg precipitation
merged_df = merged_df.sort_values("date")
merged_df["PRCP_6mo_avg"] = merged_df["PRCP"].rolling(window=182, min_periods=1).mean()

# Save to CSV
merged_df.to_csv("LA_climate_with_PRCP_rolling.csv", index=False)
print("Saved LA_climate_with_PRCP_rolling.csv")
