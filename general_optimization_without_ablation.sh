for model in  'RESCAL' 'TransE' 'DistMult' 'r2v'; do
    for dataset in 'person' 'doremus' 'SPIMBENCH_small-2019';do
        for stringmeasure in 'hpp_sim' 'jaro_winkler'; do
            python3.8 ./optimization.py --suffix $dataset --embedding_name $model --similarity_measure $stringmeasure
        done
    done
done


