# # import pickle
# # import numpy as np

# # poly_path = "my_saved_poly_v3.pkl"
# # filename = "my_saved_model_v3.sav"

# # # [start, manu_avg_fin, manu_avg_fin_track, manu_track_delta, avg_fin_track]
# # x = np.array([[0.103, 0.558, 0.280, 0.473, 0.0]], dtype=float)

# # loaded_model = pickle.load(open(filename, "rb"))
# # poly = pickle.load(poly_path)
# # x_poly = poly.transform(x)
# # result = loaded_model.predict(x_poly)
# # print(result[0])

# import pickle
# import numpy as np

# poly_path = "my_saved_poly_v3.pkl"
# model_path = "my_saved_model_v3.sav"

# with open(poly_path, "rb") as f:
#     poly = pickle.load(f)

# with open(model_path, "rb") as f:
#     model = pickle.load(f)

# # [start, manu_avg_fin, manu_avg_fin_track, manu_track_delta, avg_fin_track]
# x = np.array([[0.103, 0.558, 0.280, 0.473, 0.0]], dtype=float)

# x_poly = poly.transform(x)  # not fit_transform
# result = model.predict(x_poly)

# print(float(result[0]))


import pickle
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = Path(
    "/workspaces/2026SE_MLOPSTask_Kevin.T/2.Model_Development/2.1.Data_Wrangling/NASCAR 2017-2024 Full Race  Points Data - Cup.csv"
)

poly_path = BASE / "my_saved_poly_v3.pkl"
model_path = BASE / "my_saved_model_v3.sav"
scaler_path = BASE / "my_saved_scaler.pkl"

with open(poly_path, "rb") as f:
    poly = pickle.load(f)
with open(model_path, "rb") as f:
    model = pickle.load(f)
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# Build lookup tables from raw data
df = pd.read_csv(DATA)

manu_avg_fin_track_lookup = df.groupby(["manu", "track"])["fin"].mean()
manu_avg_fin_lookup = df.groupby("manu")["fin"].mean()
avg_fin_track_lookup = df.groupby("track")["fin"].mean()


def predict(manufacturer: str, track: str, start: int) -> float:
    try:
        raw_manu_avg_fin_track = manu_avg_fin_track_lookup[(manufacturer, track)]
    except KeyError:
        print(f"No data for {manufacturer} at {track}, using manufacturer average")
        raw_manu_avg_fin_track = manu_avg_fin_lookup[manufacturer]

    raw_manu_avg_fin = manu_avg_fin_lookup[manufacturer]
    raw_avg_fin_track = avg_fin_track_lookup[track]
    raw_manu_track_delta = raw_avg_fin_track - raw_manu_avg_fin_track
    raw_start = float(start)

    # Use DataFrame with column names matching what scaler was fitted on
    raw = pd.DataFrame(
        [
            [
                raw_manu_avg_fin_track,
                raw_manu_avg_fin,
                raw_avg_fin_track,
                raw_manu_track_delta,
                raw_start,
                0.0,
            ]
        ],
        columns=[
            "manu_avg_fin_track",
            "manu_avg_fin",
            "avg_fin_track",
            "manu_track_delta",
            "start",
            "fin",
        ],
    )

    scaled = scaler.transform(raw)

    # Model expects: [start, manu_avg_fin, manu_avg_fin_track, manu_track_delta, avg_fin_track]
    x = np.array(
        [[scaled[0][4], scaled[0][1], scaled[0][0], scaled[0][3], scaled[0][2]]]
    )

    x_poly = poly.transform(x)
    result = model.predict(x_poly)

    # Unscale predicted finish (fin is index 5 in scaler)
    fin_min = scaler.data_min_[5]
    fin_max = scaler.data_max_[5]
    predicted_finish = result[0] * (fin_max - fin_min) + fin_min

    return round(predicted_finish, 1)


# --- Test it ---
manufacturer = "Toyota"
track = "Daytona"
start = 30

prediction = predict(manufacturer, track, start)
print(f"{manufacturer} starting P{start} at {track} → predicted finish: P{prediction}")
