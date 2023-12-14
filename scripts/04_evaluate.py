import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm import tqdm

# ================================================
# Data loading functions
# ================================================


def load_file(input_file, type: str = None, run_as_test: bool = False):
    names = ["id", "title", "body", "lang", "date_time", "cluster_id"]
    dtype = {
        "id": "Int64",
        "title": "str",
        "body": "str",
        "lang": "str",
        "date_time": "str",
        "cluster_id": "str",
    }

    if type == "label":
        names.insert(5, "event_id")
        dtype["event_id"] = "str"

    df = pd.read_csv(
        input_file,
        names=names,
        dtype=dtype,
        parse_dates=["date_time"],
        on_bad_lines="warn",
        engine="python",
        skiprows=1,
    )
    df = df[:1000] if run_as_test else df
    return df


def create_dataframe(results):
    """Store all articles into the dataframe"""

    def format_result(result):
        return [
            result[0][0],  # article rank_th
            result[0][1],  # article time_std
            result[1][0],  # event rank_th
            result[1][1],  # event time_std
            result[2],  # languages (None is all)
            result[3]["standard"]["F1"] + result[3]["bcubed"]["F1"],  # sum of F1
            result[3]["standard"]["F1"],  # standard F1
            result[3]["standard"]["P"],  # standard P
            result[3]["standard"]["R"],  # standard R
            result[3]["bcubed"]["F1"],  # bcubed F1
            result[3]["bcubed"]["P"],  # bcubed P
            result[3]["bcubed"]["R"],  # bcubed R
            result[3]["clusters"],  # number of clusters
        ]

    df = pd.DataFrame(
        data=[format_result(r) for r in tqdm(results, desc="Output data prep")],
        columns=[
            "rank_th (article)",
            "time_std (article)",
            "rank_th (event)",
            "time_std (event)",
            "language",
            "SUM F1",
            "F1 (standard)",
            "P (standard)",
            "R (standard)",
            "F1 (bcubed)",
            "P (bcubed)",
            "R (bcubed)",
            "clusters",
        ],
    )
    return df


# ================================================
# Evaluation functions
# ================================================


def prepare_predicts(true_df, pred_df):
    true_cls_ids = {
        p["id"]: set([p["event_id"], p["cluster_id"]])
        for p in true_df.to_dict("records")
    }
    pred_cls_ids = {p["id"]: set([p["cluster_id"]]) for p in pred_df.to_dict("records")}
    return [
        {"true_id": true_cls_ids[key], "pred_id": pred_cls_ids[key]}
        for key in pred_cls_ids.keys()
    ]


# -------------------------------------
# Standard F1 calculations
# -------------------------------------


def calculate_standard_metrics(articles):
    """Measures the performance of the clustering algorithm"""

    # get the following statistics
    # tp - number of correctly clustered-together article pairs
    # fp - number of incorrectly clustered-together article pairs
    # fn - number of incorrectly not-clustered-together article pairs
    # tn - number of correctly not-clustered-together article pairs
    tp, fp, fn, tn = 0, 0, 0, 0
    for i, ai in enumerate(articles):
        for aj in articles[i + 1 :]:
            if (
                len(ai["pred_id"] & aj["pred_id"]) > 0
                and len(ai["true_id"] & aj["true_id"]) > 0
            ):
                tp += 1
            elif (
                len(ai["pred_id"] & aj["pred_id"]) > 0
                and len(ai["true_id"] & aj["true_id"]) == 0
            ):
                fp += 1
            elif (
                len(ai["pred_id"] & aj["pred_id"]) == 0
                and len(ai["true_id"] & aj["true_id"]) > 0
            ):
                fn += 1
            else:
                tn += 1

    # get the precision, recall and F1 scores
    P = tp / (tp + fp) if (tp + fp) > 0 else 0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0
    F1 = 2 * (P * R) / (P + R) if (P + R) > 0 else 0

    # return the metrics
    return {"F1": F1, "P": P, "R": R}


# -------------------------------------
# BCubed F1 calculations
# -------------------------------------

# Simple extended BCubed implementation in Python for clustering evaluation
# Copyright 2015 Hugo Hromic
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Extended BCubed algorithm taken from:
# Amigo, Enrique, et al. "A comparison of extrinsic clustering evaluation metrics
# based on formal constraints." Information retrieval 12.4 (2009): 461-486.

# The BCubed metrics were modified to work with the data set used in this script.


def is_not_empty(x):
    return 1 if len(x) > 0 else 0


def bcubed_mult_precision(el1, el2, cdict, ldict):
    """Computes the multiplicity precision for two elements."""
    return min(
        is_not_empty(cdict[el1] & cdict[el2]), is_not_empty(ldict[el1] & ldict[el2])
    ) / float(is_not_empty(cdict[el1] & cdict[el2]))


def bcubed_mult_recall(el1, el2, cdict, ldict):
    """Computes the multiplicity recall for two elements."""
    return min(
        is_not_empty(cdict[el1] & cdict[el2]), is_not_empty(ldict[el1] & ldict[el2])
    ) / float(is_not_empty(ldict[el1] & ldict[el2]))


def bcubed_precision(cdict, ldict):
    """Computes overall extended BCubed precision for the C and L dicts.

    Parameters
    ==========
    cdict: dict(item: set(cluster-ids))
        The cluster assignments to be evaluated
    ldict: dict(item: set(cluster-ids))
        The ground truth clustering
    """
    return np.mean(
        [
            np.mean(
                [
                    bcubed_mult_precision(el1, el2, cdict, ldict)
                    for el2 in cdict
                    if cdict[el1] & cdict[el2]
                ]
            )
            for el1 in cdict
        ]
    )


def bcubed_recall(cdict, ldict):
    """Computes overall extended BCubed recall for the C and L dicts.

    Parameters
    ==========
    cdict: dict(item: set(cluster-ids))
        The cluster assignments to be evaluated
    ldict: dict(item: set(cluster-ids))
        The ground truth clustering
    """
    return np.mean(
        [
            np.mean(
                [
                    bcubed_mult_recall(el1, el2, cdict, ldict)
                    for el2 in cdict
                    if ldict[el1] & ldict[el2]
                ]
            )
            for el1 in cdict
        ]
    )


def calculate_bcubed_metrics(articles):
    ldict = {f"item{idx+1}": a["true_id"] for idx, a in enumerate(articles)}
    cdict = {f"item{idx+1}": a["pred_id"] for idx, a in enumerate(articles)}
    P = bcubed_precision(cdict, ldict)
    R = bcubed_recall(cdict, ldict)
    F1 = 2 * (P * R) / (P + R) if (P + R) > 0 else 0
    return {"F1": F1, "P": P, "R": R}


def evaluate_performance(true_df, pred_df, lang: str = None):
    """Evaluates the performance of the clustering algorithm"""
    pdf = pred_df[pred_df["lang"] == lang] if lang is not None else pred_df
    articles = prepare_predicts(true_df, pdf)
    standard = calculate_standard_metrics(articles)
    bcubed = calculate_bcubed_metrics(articles)
    cls_n = len(pdf["cluster_id"].unique())
    return {"standard": standard, "bcubed": bcubed, "clusters": cls_n}


# ================================================
# Main function
# ================================================


def main(args):
    results = []
    label_df = load_file(args.label_file_path, type="label")

    pred_files = [
        f
        for f in os.listdir(args.pred_file_dir)
        if os.path.isfile(os.path.join(args.pred_file_dir, f))
    ]

    for f in tqdm(pred_files, desc="Files"):
        # get parameters from the file name
        params = [
            p for p in f.split(".csv")[0].split("_") if "article" in p or "event" in p
        ]
        monop = [p.split("=")[1] for p in params if "article" in p] + [None, None]
        multip = [p.split("=")[1] for p in params if "event" in p] + [None, None]

        pred_df = load_file(
            os.path.join(args.pred_file_dir, f), type="pred", run_as_test=args.test
        )
        langs = pred_df["lang"].unique() if args.monolingual else [None]
        for lang in tqdm(langs, desc="Languages"):
            performance = evaluate_performance(label_df, pred_df, lang)
            results.append((monop, multip, lang, performance))

    df = create_dataframe(results)
    df.to_csv(args.output_file, encoding="utf-8", index=False)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Evaluates the performance of the clustering algorithm"
    )
    parser.add_argument(
        "--label_file_path",
        type=str,
        required=True,
        help="The file that contains the true labels",
    )
    parser.add_argument(
        "--pred_file_dir",
        type=str,
        required=True,
        help="The directory that contains the files with predicted labels",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="The output file to store the results",
    )
    parser.add_argument(
        "--monolingual",
        action="store_true",
        help="Whether to evaluate for each language separately (default: False)",
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Whether to run in test mode (default: False)",
    )
    args = parser.parse_args()
    main(args)
