# WAC: **W**asserstein distance-based news **A**rticle **C**lustering

This project contains the implementation of the **W**asserstein distance-based news **A**rticle **C**lustering algorithm.
The algorithm is an unsupervised two-step online clustering algorithm that uses the Wasserstein distance (and distances
similar to it). The two steps are (1) monolingual clustering of news articles and (2) multilingual clustering of events into clusters.

The articles and events are represented using an SBERT language model, which are fine-tunned for clustering tasks.

The remainder of the project contains the instructions for running the experiments.

## 📚 Papers

In case you use any of the components for your research, please refer to (and cite) the papers:

**TODO**

## ☑️ Requirements

Before starting the project make sure these requirements are available:

- [python]. For setting up your research environment and python dependencies (version 3.8 or higher).
- [git]. For versioning your code.

## 🛠️ Setup

### Create a python environment

First create the virtual environment where all the modules will be stored.

#### Using venv

Using the `venv` command, run the following commands:

```bash
# create a new virtual environment
python -m venv venv

# activate the environment (UNIX)
source ./venv/bin/activate

# activate the environment (WINDOWS)
./venv/Scripts/activate

# deactivate the environment (UNIX & WINDOWS)
deactivate
```

### Install

To install the requirements run:

```bash
pip install -e .
```

## 🗃️ Data

The data used in the experiments are a currated set of news articles retrieved from the Event Registry and prepared
for the scientific paper[^1].

To download the data run:

```bash
bash scripts/00_download_data.sh
```

This will download the data files and store them in the `data/raw` folder.

## ⚗️ Experiments

To run the experiments, run the folowing command:

```bash
# run the experiments
bash scripts/run_exp_pipeline.sh
```

The command above will perform a series of experiments by executing the following steps:

```bash
# prepare the data examples for the experiment
python scripts/01_prepare_data.py \
    --input_file ./data/raw/dataset.test.json \
    --output_file ./data/processed/dataset.test.csv

# create the monolingual event clusters (merging articles)
python scripts/02_article_clustering.py \
    --input_file ./data/processed/dataset.test.csv \
    --output_file ./data/processed/mono/dataset.test.csv \
    --rank_th 0.5 \ # threshold for deciding if an article should be added to the cluster (optional)
    --time_std 3 # the std for temporal similarity between the article and event (optional)
    --gpu # to use GPU for event clustering (optional)

# create the multilingual event clusters (merging events)
python scripts/03_event_clustering.py \
    --input_file ./data/processed/mono/dataset.test.csv \
    --output_file ./data/processed/multi/dataset.test.csv \
    --rank_th 0.7 \ # threshold for deciding if events should be merged (optional)
    --time_std 3 \ # the std for temporal similarity between an events (optional)
    --w_reg 0.1 \ # the regularization factor for the wasserstein distance (optional)
    --w_nit 10 \ # the number of iterations for the wasserstein distance (optional)
    --gpu # to use GPU for event clustering (optional)

# evaluate the clusters
python scripts/04_evaluate.py \
    --label_file_path ./data/processed/dataset.test.csv \
    --pred_file_dir ./data/processed/multi \
    --output_file ./results/dataset.test.csv

```

The results will be stored in the `results` folder.

### Results

the hyper-parameters were selected by evaluating the performance of the clustering algorithm on the dev set. We performed a grid-search across the following hyper-parameters:

| Clustering Type | Parameter | Grid Search          | Description                                                         |
| :-------------- | :-------- | :------------------- | :------------------------------------------------------------------ |
| Monolingual     | rank_th   | [0.4, 0.5, 0.6, 0.7] | threshold for deciding if an article should be added to the cluster |
| Monolingual     | time_std  | [1, 2, 3, 5]         | the std for temporal similarity between the article and event       |
| Multilingual    | rank_th   | [0.6, 0.7, 0.8, 0.9] | threshold for deciding if events should be merged                   |
| Multilingual    | time_std  | [1, 2, 3]            | the std for temporal similarity between an events                   |

The best performance is obtained with the following parameters:

<table>
  <tr>
    <th style="text-align:center;" colspan="2">Monolingual</th>
    <th style="text-align:center;" colspan="2">Multilingual</th>
    <th style="text-align:center;" colspan="3">Standard</th>
    <th style="text-align:center;" colspan="3">BCubed</th>
    <th></th>
  </tr>
  <tr>
    <th style="text-align:center;">rank_th</th>
    <th style="text-align:center;">time_std</th>
    <th style="text-align:center;">rank_th</th>
    <th style="text-align:center;">time_std</th>
    <th style="text-align:center;">F1</th>
    <th style="text-align:center;">P</th>
    <th style="text-align:center;">R</th>
    <th style="text-align:center;">F1</th>
    <th style="text-align:center;">P</th>
    <th style="text-align:center;">R</th>
    <th style="text-align:center;">clusters</th>
  </tr>
  <tr>
    <td style="text-align:center;">0.5</td>
    <td style="text-align:center;">3</td>
    <td style="text-align:center;">0.7</td>
    <td style="text-align:center;">3</td>
    <td style="text-align:center;">87.00</td>
    <td style="text-align:center;">98.45</td>
    <td style="text-align:center;">77.95</td>
    <td style="text-align:center;">85.42</td>
    <td style="text-align:center;">93.04</td>
    <td style="text-align:center;">78.95</td>
    <td style="text-align:center;">1066</td>
  </tr>
  <tr>
    <td style="text-align:center;">0.6</td>
    <td style="text-align:center;">3</td>
    <td style="text-align:center;">0.7</td>
    <td style="text-align:center;">3</td>
    <td style="text-align:center;">69.50</td>
    <td style="text-align:center;">98.71</td>
    <td style="text-align:center;">53.63</td>
    <td style="text-align:center;">81.08</td>
    <td style="text-align:center;">94.14</td>
    <td style="text-align:center;">71.20</td>
    <td style="text-align:center;">1108</td>
  </tr>
</table>

## 📣 Acknowledgments

This work is developed by [Department of Artificial Intelligence][ailab] at [Jozef Stefan Institute][ijs].

This work was supported by the Slovenian Research Agency, and the European Union's Horizon 2020 project Humane AI Net [H2020-ICT-952026].

[python]: https://www.python.org/
[git]: https://git-scm.com/
[ailab]: http://ailab.ijs.si/
[ijs]: https://www.ijs.si/

[^1]: S. Miranda, A. Znotiņš, S. B. Cohen, and G. Barzdins, “Multilingual clustering of streaming news” in Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, 2018, pp. 4535–4544.
