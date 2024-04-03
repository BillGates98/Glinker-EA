# Glinker :dizzy:

> Install Python :snake: (>=3.8) and dependencies.

`python -m pip install -r ./requirements.txt`

> Important to do :

    - Unzip the data in the 'inputs' directory and move them into the root of 'inputs' directory

    - If you don't want to re-run evaluations, just unzip the data.zip file in 'outputs' directory

> :nerd_face: <strong>General Parameters</strong>

- $dataset : the name of the directory (in the 'inputs' directory) having 3 files containing the expressions (source, target, valid_same_as);
- $model : the name of the model, it can be : 'r2v', 'RESCAL', 'TransE', 'DistMult';
- $alpha : the threshold for accepting pairs of structurally similar candidate entities;
- $beta : the maximum similarity acceptance threshold between candidate entity literals;
- $string_measure : the name of the similarity measure to adopt for the linking process, it can be : 'jaro_winkler', 'hpp'.

## 1. Evaluations of HPP similarity measure:

> 1.1. Reproduce the comparison between HPP and Jaro-Winkler in the paper :

    `$ python ./similarity_measures_comparison.py`

> 1.2. Playing with the HPP similarity measure

    `$ python ./hpp_similarity.py --value1 'Conference' --value2 'conferance'`

> 1.3. Comparison between Jaro-Winkler and HPP Results

    ---------------------------------------------------------
    | N° | String 1    | String 2    | Jaro-Winkler | HPP   |
    --------------------------------------------------------
    | 01 | shackleford | shackelford | 0.983        | 0.893 |
    | 02 | cunningham  | cunnigham   | 0.983        | 0.871 |
    | 03 | campell     | campbell    | 0.975        | 0.911 |
    | 04 | nichleson   | nichulson   | 0.955        | 0.898 |
    | 05 | massey      | massie      | 0.933        | 0.888 |
    | 06 | galloway    | calloway    | 0.916        | 0.998 |
    | 07 | lampley     | campley     | 0.904        | 0.928 |
    | 08 | frederick   | frederic    | 0.992        | 0.991 |
    | 09 | michele     | michelle    | 0.983        | 0.999 |
    | 10 | jesse       | jessie      | 0.966        | 0.997 |
    | 11 | marhta      | martha      | 0.961        | 0.880 |
    | 12 | jonathon    | jonathan    | 0.966        | 0.994 |
    | 13 | julies      | juluis      | 0.922        | 0.805 |
    | 14 | jeraldine   | geraldine   | 0.925        | 0.981 |
    | 15 | yvette      | yevett      | 0.899        | 0.991 |
    | 16 | tanya       | tonya       | 0.880        | 0.900 |
    | 17 | dwayne      | duane       | 0.840        | 0.545 |
    | 18 | jon         | john        | 0.933        | 0.71  |
    | 19 | hardin      | martinez    | 0.722        | 0.571 |
    | 20 | itman       | smith       | 0.466        | 0.500 |
    ---------------------------------------------------------

## 2. Glinker execution on datasets

> 2.1. running on a single dataset

    `$ python ./main.py --suffix $dataset --embedding_name $model --dimension $dimension --alpha $alpha --beta $beta --similarity_measure $string_measure`

> 2.2. running for all the datasets

    `$ sh ./general.sh`

## 3. Glinker optimization at ablation level

> 3.1. optimization on a single dataset

    `$ python ./optimization_ablation.py --suffix $dataset --embedding_name $model --similarity_measure $string_measure`

> 3.2. Glinker optimization on all the datasets

    `$ sh ./general_optimization_on_ablation.sh`

## 4. Glinker optimization without ablation

> 4.1. optimization on a single dataset

    `$ python ./optimization.py --suffix $dataset --embedding_name $model --similarity_measure $string_measure`

> 4.2. Glinker optimization on all the datasets

    `$ sh ./general_optimization_without_ablation.sh`
