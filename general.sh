for model in 'r2v' 'RESCAL' 'TransE' 'DistMult'; do
    for dataset in 'person' 'doremus' 'SPIMBENCH_small-2019'; do
        for stringmeasure in 'jaro_winkler' 'hpp_sim'; do
            python3.8 ./main.py --suffix $dataset --embedding_name $model --dimension 300 --alpha 0.95 --beta 0.95 --similarity_measure $stringmeasure
        done
    done
done

# 0.95 0.95
# 0.0 0.95