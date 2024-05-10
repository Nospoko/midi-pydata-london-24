from datasets import load_dataset


def calculate_average_nps():
    dataset = load_dataset(
        path="roszcz/maestro-sustain-v2",
        split="train+validation+test",
    )

    total_notes = 0
    total_time = 0.0

    for record in dataset:
        total_notes += len(record["notes"]["pitch"])
        total_time += max(record["notes"]["start"])

    return total_notes / total_time


if __name__ == "__main__":
    avg_nps = calculate_average_nps()
    print(avg_nps)
