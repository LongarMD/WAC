import warnings
from pathlib import Path
from argparse import ArgumentParser
from os.path import isfile, exists, dirname

import torch
import pandas as pd
from tqdm import tqdm

from wac.utils.NewsArticle import NewsArticle
from wac.utils.NewsEvent import NewsEvent
from wac.utils.NewsEventMonitor import NewsEventMonitor

# import conditions
from wac.utils.strategy.ArticleStrategy import ArticleStrategy

# import models
from wac.models.SBERT import SBERT
from wac.models.MNER import MNER

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


def load_articles(input_file, run_as_test):
    df = pd.read_csv(
        input_file,
        dtype={
            "id": "int",
            "title": "str",
            "body": "str",
            "lang": "str",
            "date_time": "str",
        },
        parse_dates=["date_time"],
        index_col=False,
    )
    df = df[:1000] if run_as_test else df
    df = df.sort_values(by="date_time")
    # create the news article list
    articles = [
        NewsArticle(article)
        for article in tqdm(df.to_dict("records"), desc="Input data prep")
    ]
    return articles


# ================================================
# Cluster functions
# ================================================


def cluster_and_save_articles(input_file, output_file, run_as_test, use_gpu, args):
    # setup the device for GPU use
    if use_gpu and not torch.cuda.is_available():
        warnings.warn("GPU not available, using CPU")
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # initialize LM
    NewsArticle.rep_model = SBERT(model_name=args.lm, device=device).eval()
    if args.ner != "" and args.ents_th > 0.0:
        NewsArticle.ner_model = MNER(model_name=args.ner, device=device).eval()
        NewsEvent.use_ne = True

    strategy = ArticleStrategy(
        rank_th=args.rank_th,
        ents_th=args.ents_th,
        time_std=args.time_std,
        multilingual=args.multilingual,
    )
    event_monitor = NewsEventMonitor(strategy=strategy)

    articles = load_articles(input_file, run_as_test)
    for article in tqdm(articles, desc="Article clustering"):
        # specify where we compare the articles
        event_monitor.update(article, device=device)

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
    cluster_and_save_articles(
        input_file=args.input_file,
        output_file=args.output_file,
        run_as_test=args.test,
        use_gpu=args.use_gpu,
        args=args,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Clusters the articles into event clusters")

    parser.add_argument(
        "--input_file",
        default=None,
        type=str,
        required=True,
        help="The input file containing the articles",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        type=str,
        required=True,
        help="The output file to store the clustered articles",
    )
    parser.add_argument(
        "--rank_th",
        default=0.5,
        type=float,
        help="The clustering rank threshold (default: 0.5)",
    )
    parser.add_argument(
        "--time_std",
        default=3.0,
        type=float,
        help="The clustering time standard deviation (default: 3.0)",
    )
    parser.add_argument(
        "--ents_th",
        default=0.2,
        type=float,
        help="The named entity similarity threshold (default: 0.2)",
    )

    parser.add_argument(
        "--lm",
        default="sentence-transformers/distiluse-base-multilingual-cased-v2",
        type=str,
        help="The language model to use (default: 'sentence-transformers/distiluse-base-multilingual-cased-v2')",
    )
    parser.add_argument(
        "--ner",
        default="tomaarsen/span-marker-mbert-base-multinerd",
        type=str,
        help="The SpanMarker model to use (default: 'tomaarsen/span-marker-mbert-base-multinerd')",
    )
    parser.add_argument(
        "--multilingual",
        action="store_true",
        help="If set, perform multilingual clustering. Else, performs monolingual clustering",
    )
    parser.add_argument(
        "-gpu",
        "--use_gpu",
        action="store_true",
        help="If true, use GPU if available (default: False)",
    )
    parser.add_argument(
        "-o",
        "--override",
        action="store_true",
        help="Override the output file if it already exists (default: False)",
    )
    parser.add_argument(
        "-t", "--test", action="store_true", help="Run in test mode (default: False)"
    )
    args = parser.parse_args()

    main(args)
