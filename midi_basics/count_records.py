import pandas as pd
from datasets import load_dataset


def calculate_notes_and_duration(split):
    total_notes = 0
    total_duration = 0
    for record in split:
        total_notes += len(record["notes"]["pitch"])
        total_duration += max(record["notes"]["end"])
    return total_notes, total_duration / 3600  # Convert duration to hours


def main():
    # Load the dataset
    dataset = load_dataset("roszcz/maestro-sustain-v2")

    # Initialize dictionaries to store calculated values
    split_info = {"train": {}, "validation": {}, "test": {}}

    # Calculate notes and duration for each split
    for split_name in split_info.keys():
        split = dataset[split_name]
        total_notes, total_duration = calculate_notes_and_duration(split)
        split_info[split_name]["Performances"] = len(split)
        split_info[split_name]["Duration (hours)"] = total_duration
        split_info[split_name]["Size (GB)"] = None  # Size not provided in dataset
        split_info[split_name]["Number of notes (millions)"] = total_notes / 1e6

    # Summarize the total
    total_info = {
        "Performances": sum([split["Performances"] for split in split_info.values()]),
        "Duration (hours)": sum([split["Duration (hours)"] for split in split_info.values()]),
        "Number of notes (millions)": sum([split["Number of notes (millions)"] for split in split_info.values()]),
    }

    # Create a DataFrame
    data = {
        "Split": ["Train", "Validation", "Test", "Total"],
        "Performances": [
            split_info["train"]["Performances"],
            split_info["validation"]["Performances"],
            split_info["test"]["Performances"],
            total_info["Performances"],
        ],
        "Duration (hours)": [
            split_info["train"]["Duration (hours)"],
            split_info["validation"]["Duration (hours)"],
            split_info["test"]["Duration (hours)"],
            total_info["Duration (hours)"],
        ],
        "Notes (millions)": [
            split_info["train"]["Notes (millions)"],
            split_info["validation"]["Notes (millions)"],
            split_info["test"]["Notes (millions)"],
            total_info["Number of notes (millions)"],
        ],
    }

    df = pd.DataFrame(data)
    print(df)


if __name__ == "__main__":
    main()
