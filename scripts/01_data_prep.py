import warnings
from pathlib import Path
from argparse import ArgumentParser
from os.path import isfile, exists, dirname


import pandas as pd

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
    # load the raw articles
    df = pd.read_json(args.input_file)
    df.rename(
        columns={"date": "date_time", "text": "body", "cluster": "cluster_id"},
        inplace=True,
    )

    # FUTURE REFERENCE: this creates the data for the final evaluation
    # The algorithm will need only the title, body, lang and date_time
    # The event_id and cluster_id will be used only for the final evaluation
    df = df[["id", "title", "body", "lang", "date_time", "event_id", "cluster_id"]]
    df.to_csv(args.output_file, encoding="utf-8", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file", default=None, type=str)
    parser.add_argument("--output_file", default=None, type=str)
    parser.add_argument("-o", "--override", action="store_true")
    args = parser.parse_args()
    main(args)
