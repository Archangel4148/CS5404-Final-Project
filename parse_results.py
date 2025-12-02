import csv
import json

def map_blur_level(blur):
    if blur == 0 or blur == 0.0:
        return "none"
    elif blur <= 5:
        return "low"
    else:
        return "high"

def map_noise_level(noise):
    if noise == 0 or noise == 0.0:
        return "none"
    elif noise <= 100:
        return "low"
    else:
        return "high"

def map_exposure_level(exp):
    if exp == 1.0:
        return "none"
    elif exp <= 4.0:
        return "low"
    else:
        return "high"

def parse_results(json_path, csv_path):
    # Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []
    tau_values = set()

    # --- First pass: collect all tau values ---
    for item_name, item_data in data.items():
        for obj_id, obj_data in item_data.items():
            for img in obj_data["images"]:
                for dist in img["distortions"]:
                    for ev in dist["evaluations"]:
                        tau_values.add(ev["tau"])

    tau_values = sorted(tau_values)

    # --- CSV Header ---
    header = [
        "object",
        "blurLevel",
        "exposureLevel",
        "noiseLevel",
        "chamferDistance",
    ] + [f"F{tau}" for tau in tau_values]

    # --- Second pass: build rows ---
    for item_name, item_data in data.items():
        for obj_id, obj_data in item_data.items():
            for img in obj_data["images"]:
                for dist in img["distortions"]:

                    distortion = dist["distortion"]
                    if str(distortion["blur"]) == "0.0":
                        continue
                    blur_level = map_blur_level(distortion["blur"])
                    exposure_level = map_exposure_level(distortion["exposure"])
                    noise_level = map_noise_level(distortion["noise"])

                    # Chamfer distance (same per distortion)
                    chamfer = dist["evaluations"][0]["metrics"]["chamfer_distance"]

                    row = {
                        "object": item_name,
                        "blurLevel": blur_level,
                        "exposureLevel": exposure_level,
                        "noiseLevel": noise_level,
                        "chamferDistance": chamfer,
                    }

                    # Insert F-scores per-tau
                    for ev in dist["evaluations"]:
                        tau = ev["tau"]
                        row[f"F{tau}"] = ev["metrics"]["fscore"]

                    rows.append(row)

    # --- Write CSV ---
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    parse_results(
        # r"C:\Users\joshu\PycharmProjects\CS5404-Final-Project\datasets\omniobject3d\spar3d_outputs\pipeline_results_20251201_225815.json",
        # r"C:\Users\joshu\PycharmProjects\CS5404-Final-Project\datasets\omniobject3d\spar3d_outputs\pipeline_results_20251201_231844.json",
        r"C:\Users\joshu\PycharmProjects\CS5404-Final-Project\pipeline_results_re-evaluated.json",
        "parsed_results-re-evaluated.csv"
    )
