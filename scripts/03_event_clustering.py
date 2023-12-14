import warnings
from os.path import isfile, exists, dirname

import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

from src.utils.NewsArticle import NewsArticle
from src.utils.NewsEvent import NewsEvent
from src.utils.NewsEventMonitor import NewsEventMonitor

# import conditions
from src.utils.strategy.Multilingual import MultilingualStrategy

# import models
from src.models.SBERT import SBERT

# ================================================
# Data loader functions
# ================================================


def create_dataframe(event_monitor):
    """Store all articles into the dataframe"""

    event_monitor.assign_cluster_ids_to_articles()

    df = pd.DataFrame(
        data=[
            article.to_array()
            for event in tqdm(event_monitor.events, desc="Output data prep")
            for article in event.articles
        ],
        columns=["id", "title", "body", "lang", "date_time", "cluster_id"],
    )
    return df


def load_events(input_file, run_as_test):
    df = pd.read_csv(
        input_file,
        dtype={
            "id": "int",
            "title": "str",
            "body": "str",
            "lang": "str",
            "date_time": "str",
            "cluster_id": "str",
        },
        parse_dates=["date_time"],
        index_col=False,
    )
    df = df.sort_values(by="date_time")

    cluster_ids = df["cluster_id"].unique()
    cluster_ids = cluster_ids[:1000] if run_as_test else cluster_ids

    # create the news events list
    events = [
        NewsEvent(
            articles=[
                NewsArticle(a)
                for a in df[df["cluster_id"] == cluster_id].to_dict("records")
            ]
        )
        for cluster_id in tqdm(cluster_ids, desc="Input data prep")
    ]
    return events


# ================================================
# Cluster functions
# ================================================


def cluster_and_save_events(input_file, output_file, run_as_test, use_gpu, args):
    if use_gpu and not torch.cuda.is_available():
        warnings.warn("GPU not available, using CPU")
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # initialize LM
    if args.lm is not None:
        NewsArticle.embed_model = SBERT(model_name=args.lm, device=device).eval()

    strategy = MultilingualStrategy(
        rank_th=args.rank_th,
        time_std=args.time_std,
        w_reg=args.w_reg,
        w_nit=args.w_nit,
        device=device,
    )

    event_monitor = NewsEventMonitor(strategy=strategy)

    events = load_events(input_file, run_as_test)
    for event in tqdm(events, desc="Event clustering"):
        # specify where we compare the articles
        event_monitor.update(event, device=device)

    df = create_dataframe(event_monitor)
    df.to_csv(output_file, encoding="utf-8", index=False)


# ================================================
# Main function
# ================================================


def main(args):
    if not isfile(args.input_file):
        raise FileNotFoundError(f"File not found: {args.input_file}")

    # create the results directory
    Path(dirname(args.output_file)).mkdir(parents=True, exist_ok=True)
    if exists(args.output_file) and not args.override:
        warnings.warn(f"File already exists: {args.output_file}")
        return

    cluster_and_save_events(
        input_file=args.input_file,
        output_file=args.output_file,
        run_as_test=args.test,
        use_gpu=args.use_gpu,
        args=args,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file", default=None, type=str)
    parser.add_argument("--output_file", default=None, type=str)
    parser.add_argument("-gpu", "--use_gpu", action="store_true")
    parser.add_argument("-o", "--override", action="store_true")
    parser.add_argument("-t", "--test", action="store_true")
    parser.add_argument("--rank_th", default=0.7, type=float)
    parser.add_argument("--time_std", default=3.0, type=float)
    parser.add_argument("--w_reg", default=0.1, type=float)
    parser.add_argument("--w_nit", default=10, type=int)
    parser.add_argument(
        "--lm",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        type=str,
    )

    args = parser.parse_args()

    main(args)
