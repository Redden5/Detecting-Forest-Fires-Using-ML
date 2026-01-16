import requests

station_id = 'USR0000CLAH'
url = f"https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/all/{station_id}.dly"

with open(f"{station_id}.dly", "wb") as f:
    f.write(requests.get(url).content)