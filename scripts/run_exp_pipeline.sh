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
declare -a MONO_CLUSTER_PARAMS=(
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
declare -a MULTI_CLUSTER_PARAMS=(
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
TARGET_FILE="dataset.${DATA_TYPE}.csv"

# define the folders used for the experiments
RAW_INPUT_DIR="./data/raw"
MONO_INPUT_DIR="./data/processed"
MONO_OUTPUT_DIR="./data/processed/mono"
MULTI_OUTPUT_DIR="./data/processed/multi"
EVAL_OUTPUT_DIR="./results"


# prepare the datasets for the experiments
python ./scripts/01_data_prep.py \
    --input_file $RAW_INPUT_DIR/$RAW_FILE \
    --output_file $MONO_INPUT_DIR/$TARGET_FILE


for MONO_CLUSTER_PARAM in "${MONO_CLUSTER_PARAMS[@]}"; do

    # turn e.g. "0.8;1" into
    # array ["0.8", "1"]
    IFS=";" read -r -a MONO_PARAMS <<< "${MONO_CLUSTER_PARAM}"
    MONO_RANK_TH="${MONO_PARAMS[0]}"
    MONO_TIME_STD="${MONO_PARAMS[1]}"

    MONO_DATASET_FILE="dataset_monor=${MONO_RANK_TH}_monot=${MONO_TIME_STD}.csv"

    echo "Start monolingual clustering for: $TARGET_FILE (rank_th=$MONO_RANK_TH; time_std=$MONO_TIME_STD)"

    # perform monolingual clustering
    python ./scripts/02_article_clustering.py \
        --input_file $MONO_INPUT_DIR/$TARGET_FILE \
        --output_file $MONO_OUTPUT_DIR/$MONO_DATASET_FILE \
        --rank_th $MONO_RANK_TH \
        --time_std $MONO_TIME_STD \
        -gpu


    for MULTI_CLUSTER_PARAM in "${MULTI_CLUSTER_PARAMS[@]}"; do

        IFS=";" read -r -a MULTI_PARAMS <<< "${MULTI_CLUSTER_PARAM}"
        MULTI_RANK_TH="${MULTI_PARAMS[0]}"
        MULTI_TIME_STD="${MULTI_PARAMS[1]}"

        MULTI_DATASET_FILE="dataset_monor=${MONO_RANK_TH}_monot=${MONO_TIME_STD}_multir=${MULTI_RANK_TH}_multit=${MULTI_TIME_STD}.csv"

        if [ ! -f "$MULTI_OUTPUT_DIR/$MULTI_DATASET_FILE" ]; then
            echo "Creating multilingual clusters for: $MONO_DATASET_FILE (rank_th=$MULTI_RANK_TH; time_std=$MULTI_TIME_STD)"

            # perform multilingual clustering
            python ./scripts/03_event_clustering.py \
                --input_file $MONO_OUTPUT_DIR/$MONO_DATASET_FILE \
                --output_file $MULTI_OUTPUT_DIR/$MULTI_DATASET_FILE \
                --rank_th $MULTI_RANK_TH \
                --time_std $MULTI_TIME_STD \
                -gpu

        else
            echo "Multilingual clusters already exist: $MONO_DATASET_FILE ($MULTI_RANK_TH; $MULTI_TIME_STD)"
        fi;
    done;
done;

# perform the evaluation
python ./scripts/04_evaluate.py \
    --label_file_path $MONO_INPUT_DIR/$TARGET_FILE \
    --pred_file_dir $MULTI_OUTPUT_DIR \
    --output_file $EVAL_OUTPUT_DIR/$TARGET_FILE