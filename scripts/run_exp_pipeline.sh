#!/bin/bash

# stop whole script when CTRL+C
trap "exit" INT

# ===============================================
# Prepare project environment
# ===============================================

if [ -d "./venv" ]; then
    source ./venv/bin/activate
else
    echo "Python environemnt does not exist"
    exit
fi;

# ===============================================
# Prepare dataset names
# ===============================================

# format is rank_th;time_std
declare -a ARTICLE_CLUSTER_PARAMS=(
    #"0.7;5"
    #"0.7;3"
    #"0.7;2"
    #"0.7;1"
    #"0.6;5"
    "0.6;3"
    #"0.6;2"
    #"0.6;1"
    #"0.5;5"
    "0.5;3"
    #"0.5;2"
    #"0.5;1"
    #"0.4;5"
    #"0.4;3"
    #"0.4;2"
    #"0.4;1"
)

# format is rank_th;time_std
declare -a EVENT_CLUSTER_PARAMS=(
    #"0.9;3"
    #"0.9;2"
    #"0.9;1"
    #"0.8;3"
    #"0.8;2"
    #"0.8;1"
    "0.7;3"
    #"0.7;2"
    #"0.7;1"
    #"0.6;3"
    #"0.6;2"
    #"0.6;1"
)

DATA_TYPE="test"

# define the target file
RAW_FILE="dataset.${DATA_TYPE}.json"
TARGET_FILE="dataset.${DATA_TYPE}.multi.wac.csv"

FOLDER_DIR="./data/processed.${DATA_TYPE}.multi.wac"

# define the folders used for the experiments
RAW_INPUT_DIR="./data/raw"
ARTICLE_INPUT_DIR=$FOLDER_DIR
ARTICLE_OUTPUT_DIR="$FOLDER_DIR/article_clusters"
EVENT_OUTPUT_DIR="$FOLDER_DIR/event_clusters"
EVAL_OUTPUT_DIR="./results"


# prepare the datasets for the experiments
python ./scripts/01_data_prep.py \
    --input_file $RAW_INPUT_DIR/$RAW_FILE \
    --output_file $ARTICLE_INPUT_DIR/$TARGET_FILE \
    --override


for ARTICLE_CLUSTER_PARAM in "${ARTICLE_CLUSTER_PARAMS[@]}"; do

    # turn e.g. "0.8;1" into
    # array ["0.8", "1"]
    IFS=";" read -r -a ARTICLE_PARAMS <<< "${ARTICLE_CLUSTER_PARAM}"
    ARTICLE_RANK_TH="${ARTICLE_PARAMS[0]}"
    ARTICLE_TIME_STD="${ARTICLE_PARAMS[1]}"

    ARTICLE_DATASET_FILE="dataset_articler=${ARTICLE_RANK_TH}_articlet=${ARTICLE_TIME_STD}.csv"

    echo "Start article clustering for: $TARGET_FILE (rank_th=$ARTICLE_RANK_TH; time_std=$ARTICLE_TIME_STD)"

    # perform article clustering
    python ./scripts/02_article_clustering.py \
        --input_file $ARTICLE_INPUT_DIR/$TARGET_FILE \
        --output_file $ARTICLE_OUTPUT_DIR/$ARTICLE_DATASET_FILE \
        --rank_th $ARTICLE_RANK_TH \
        --time_std $ARTICLE_TIME_STD \
        --multilingual \
        --ents_th 0.0 \
        -gpu


    for EVENT_CLUSTER_PARAM in "${EVENT_CLUSTER_PARAMS[@]}"; do

        IFS=";" read -r -a EVENT_PARAMS <<< "${EVENT_CLUSTER_PARAM}"
        EVENT_RANK_TH="${EVENT_PARAMS[0]}"
        EVENT_TIME_STD="${EVENT_PARAMS[1]}"

        EVENT_DATASET_FILE="dataset_articler=${ARTICLE_RANK_TH}_articlet=${ARTICLE_TIME_STD}_eventr=${EVENT_RANK_TH}_eventt=${EVENT_TIME_STD}.csv"

        if [ ! -f "$EVENT_OUTPUT_DIR/$EVENT_DATASET_FILE" ]; then
            echo "Creating event clusters for: $ARTICLE_DATASET_FILE (rank_th=$EVENT_RANK_TH; time_std=$EVENT_TIME_STD)"

            # perform event clustering
            python ./scripts/03_event_clustering.py \
                --input_file $ARTICLE_OUTPUT_DIR/$ARTICLE_DATASET_FILE \
                --output_file $EVENT_OUTPUT_DIR/$EVENT_DATASET_FILE \
                --rank_th $EVENT_RANK_TH \
                --time_std $EVENT_TIME_STD \
                -gpu

        else
            echo "Event clusters already exist: $ARTICLE_DATASET_FILE ($EVENT_RANK_TH; $EVENT_TIME_STD)"
        fi;
    done;
done;

# perform the evaluation
python ./scripts/04_evaluate.py \
    --label_file_path $ARTICLE_INPUT_DIR/$TARGET_FILE \
    --pred_file_dir $EVENT_OUTPUT_DIR \
    --output_file $EVAL_OUTPUT_DIR/$TARGET_FILE