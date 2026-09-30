from pathlib import Path
import csv
import hashlib
import math
import random

GROUP_CODE = "AI-G06" 
NUM_ROWS = 300
SEED = 3513

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "data" / "AI_A1_G06.csv"

FIELDS = [
    "record_id",
    "plot_area_ha",
    "rainfall_mm",
    "soil_ph",
    "seed_kg",
    "distance_km",
    "arrival_hour",
    "actual_yield_kg",
    "dispatch_attention",
]


def generate_dataset():
    rng = random.Random(SEED)
    rows = []

    for i in range(1, NUM_ROWS + 1):
        plot_area = round(rng.uniform(0.3, 4.0), 2)
        rainfall = round(max(25, min(180, rng.gauss(90, 28))), 1)
        soil_ph = round(max(4.5, min(7.2, rng.gauss(5.8, 0.45))), 2)
        seed_kg = round(
            max(40, plot_area * rng.uniform(130, 190) + rng.gauss(0, 18)),
            1,
        )
        distance = round(rng.uniform(1.0, 30.0), 1)
        arrival_hour = rng.randint(6, 17)

        rainfall_effect = max(0, rainfall - 45) * 1.15
        ph_effect = max(0, 1 - abs(soil_ph - 5.8) / 1.5) * 75
        seed_effect = seed_kg * 1.65
        area_effect = plot_area * 105

        actual_yield = round(
            max(
                80,
                area_effect
                + rainfall_effect
                + ph_effect
                + seed_effect
                + rng.gauss(0, 55),
            ),
            1,
        )

        risk_score = (
            -2.8
            + 0.075 * distance
            + 0.20 * max(0, arrival_hour - 11)
            + 0.018 * max(0, rainfall - 100)
            + 0.35 * max(0, 5.3 - soil_ph)
            + rng.gauss(0, 0.35)
        )

        probability = 1 / (1 + math.exp(-risk_score))
        dispatch_attention = int(rng.random() < probability)

        rows.append(
            {
                "record_id": f"R{i:04d}",
                "plot_area_ha": plot_area,
                "rainfall_mm": rainfall,
                "soil_ph": soil_ph,
                "seed_kg": seed_kg,
                "distance_km": distance,
                "arrival_hour": arrival_hour,
                "actual_yield_kg": actual_yield,
                "dispatch_attention": dispatch_attention,
            }
        )

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    fingerprint = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()

    print(f"Created: {CSV_PATH}")
    print(f"Rows: {len(rows)}")
    print(f"Columns: {len(FIELDS)}")
    print(f"Random seed: {SEED}")
    print(f"SHA-256: {fingerprint}")


if __name__ == "__main__":
    generate_dataset()
