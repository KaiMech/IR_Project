import argparse
import finetuning_random_setting as train_basic
import finetuning_batch_setting as train_advanced

def parse_arguments():
    parser = argparse.ArgumentParser(description="DPR Fine-Tuning Launcher")

    parser.add_argument(
        "--mode", 
        type=str, 
        required=True, 
        choices=["basic", "advanced"],
        help="Select 'basic' for RAM-heavy/Random Negatives or 'advanced' for SQLite/Hard Negatives/Mixed Precision"
    )

    return parser.parse_args()

def main():
    args = parse_arguments()

    if args.mode == "basic":
        train_basic.main()
    
    elif args.mode == "advanced":
        train_advanced.main()

if __name__ == "__main__":
    main()