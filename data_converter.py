import pandas as pd

def parse_ghcn_dly(file_path):
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            station = line[0:11]
            year = int(line[11:15])
            month = int(line[15:17])
            element = line[17:21]  # e.g., TMAX, TMIN, PRCP
            for day in range(1, 32):
                value_str = line[21 + (day - 1) * 8 : 26 + (day - 1) * 8].strip()
                value = int(value_str) if value_str else -9999
                if value != -9999:
                    date = pd.to_datetime(f"{year}-{month:02d}-{day:02d}", errors='coerce')
                    if pd.notnull(date):
                        # GHCN values are in tenths; PRCP is mm, temps are °C
                        if element in ["TMAX", "TMIN", "TAVG"]:
                            value = value / 10.0
                        elif element == "PRCP":
                            value = value / 10.0  # Convert tenths of mm to mm
                        records.append((station, date, element, value))
    return pd.DataFrame(records, columns=["station", "date", "element", "value"])

# Parse your .dly file
df = parse_ghcn_dly("USR0000CLAH.dly")

# Pivot so each element is its own column
df_pivot = df.pivot(index="date", columns="element", values="value").reset_index()

# Save with precipitation included
df_pivot.to_csv("LA_climate_with_precip.csv", index=False)
