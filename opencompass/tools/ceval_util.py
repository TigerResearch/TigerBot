import pandas as pd
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='file path', type=str)
    parser.add_argument('--key', help='score column name', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    df = pd.read_csv(args.path)
    print(df.shape)
    score = []
    for idx, row in df.iterrows():
        print(row.to_dict())
        if row[args.key] == "-":
            continue
        if "ceval" not in row["dataset"]:
            continue
        score.append(float(row[args.key]))
    print(f"score: {score}, sum: {sum(score)}, length: {len(score)}")
    print(f"average score: {sum(score) / len(score)}")


if __name__ == "__main__":
    main()
