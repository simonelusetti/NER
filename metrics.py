import argparse
from NER_cadec import compute_metrics 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("pos_weight_first", type=int)
    parser.add_argument("pos_weight_second", type=int)
    parser.add_argument("num_runs", type=int)
    parser.add_argument("metrics_file", type=str)
    parser.add_argument("best_model_path", type=str)
    args = parser.parse_args()

    compute_metrics(args.dataset_dir, list(range(args.pos_weight_first,args.pos_weight_second)), args.num_runs, args.metrics_file, args.best_model_path)
