import argparse
import numpy as np


def summarize(lengths):
    arr = np.array(lengths)
    n = len(arr)
    print(f"valid_count = {n}")
    print(f"min = {int(arr.min())}")
    print(f"mean = {arr.mean():.2f}")
    for p in [50, 75, 90, 95, 99]:
        print(f"p{p} = {int(np.percentile(arr, p))}")
    print(f"max = {int(arr.max())}")
    for t in [128, 256, 512, 1024, 2048, 3072, 4096, 8192]:
        c = int((arr > t).sum())
        print(f">{t}: {c} / {n} = {100*c/n:.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Utility helper for summarizing token length arrays.")
    p.parse_args()
    print(
        "This is a utility skeleton. Use your dataset loader + processor chat template "
        "to generate token lengths, then pass the list into summarize()."
    )
