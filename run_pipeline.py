"""
Main Pipeline Runner
====================
Runs the full meta-learning pipeline end-to-end:
  Stage 1: Generate synthetic datasets
  Stage 2: Extract meta-features
  Stage 3: Benchmark clustering algorithms
  Stage 4: Train & evaluate meta-learner

Usage:
  python run_pipeline.py          # Run all stages
  python run_pipeline.py --stage 2  # Run from stage 2 onwards
"""

import argparse
import time
import sys


def main():
    parser = argparse.ArgumentParser(description="Meta-Learning Clustering Pipeline")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Start from this stage (default: 1)")
    parser.add_argument("--n-datasets", type=int, default=None,
                        help="Override number of datasets to generate")
    args = parser.parse_args()

    total_start = time.time()

    if args.stage <= 1:
        print("\n" + "=" * 60)
        print("STAGE 1: Synthetic Dataset Generation")
        print("=" * 60)
        from stage1_generate import generate_all_datasets
        from config import N_DATASETS
        n = args.n_datasets or N_DATASETS
        generate_all_datasets(n_datasets=n)

    if args.stage <= 2:
        print("\n" + "=" * 60)
        print("STAGE 2: Meta-Feature Extraction")
        print("=" * 60)
        from stage2_metafeatures import extract_metafeatures_all_datasets
        extract_metafeatures_all_datasets()

    if args.stage <= 3:
        print("\n" + "=" * 60)
        print("STAGE 3: Clustering Algorithm Benchmarking")
        print("=" * 60)
        from stage3_benchmark import benchmark_all_datasets
        benchmark_all_datasets()

    if args.stage <= 4:
        print("\n" + "=" * 60)
        print("STAGE 4: Meta-Learner Training & Evaluation")
        print("=" * 60)
        from stage4_metalearner import (
            load_meta_dataset,
            train_and_evaluate,
            validate_on_real_data,
        )
        meta_df = load_meta_dataset()
        results, model, le, scaler, feat_cols = train_and_evaluate(meta_df)
        validate_on_real_data(model, le, scaler, feat_cols)

    elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
