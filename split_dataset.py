#!/usr/bin/env python3
# coding: utf-8

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    p = argparse.ArgumentParser(description="Split dataset CSV into train/val/test sets")
    p.add_argument('--input_csv', required=True, help='Path to input CSV file containing sequence,label')
    p.add_argument('--train_ratio', type=float, default=0.8, help='Train set ratio (default: 0.8)')
    p.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio (default: 0.1)')
    p.add_argument('--test_ratio', type=float, default=0.1, help='Test set ratio (default: 0.1)')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--out_prefix', default="dataset", help='Output prefix for CSV files')
    args = p.parse_args()

    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    print("Loading:", args.input_csv)
    df = pd.read_csv(args.input_csv)

   
    train_df, temp_df = train_test_split(
        df,
        test_size=1 - args.train_ratio,
        random_state=args.seed,
        shuffle=True,
        stratify=df['label']   
    )

  
    val_ratio_adj = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - val_ratio_adj,
        random_state=args.seed,
        shuffle=True,
        stratify=temp_df['label']
    )

  
    train_path = f"{args.out_prefix}_train.csv"
    val_path   = f"{args.out_prefix}_val.csv"
    test_path  = f"{args.out_prefix}_test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Done!")
    print("Train:", len(train_df))
    print("Val:  ", len(val_df))
    print("Test: ", len(test_df))
    print(f"Saved:\n  {train_path}\n  {val_path}\n  {test_path}")

if __name__ == '__main__':
    main()
