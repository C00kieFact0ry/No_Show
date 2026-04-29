from pathlib import Path

import pandas as pd

# Repo root = .../No_Show (this file is at src/noshow/preprocessing/filtering.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW = PROJECT_ROOT / "data" / "raw"

INPUT_CSV = DATA_RAW / "no show extractie v20260417.csv"
OUTPUT_CSV = DATA_RAW / "poliafspraken_no_show.csv"


def main() -> None:
    print(f"Reading:  {INPUT_CSV}")
    appointments_df = pd.read_csv(INPUT_CSV)

    # Change column names to match this codebase
    appointments_df = appointments_df.rename(
        columns={
            "arrived": "gearriveerd",
            "cancelationReason_code": "mutationReason_code",
            "cancelationReason_display": "mutationReason_display",
        }
    )

    # Clean invalid postal codes (keep only rows with exactly 4 digits)
    column = "address_postalCodeNumbersNL"
    pc = appointments_df[column].astype("string").str.strip()
    valid = pc.str.fullmatch(r"\d{4}")
    cleaned = appointments_df.loc[valid].copy()
    cleaned[column] = cleaned[column].astype(int)

if __name__ == "__main__":
    main()