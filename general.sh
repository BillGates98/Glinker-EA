for model in  'r2v' 'RESCAL' 'TransE' 'DistMult' 'ComplEx' 'ConvE' 'RotatE' 'TuckER' 'TransR'; do
    for dataset in  'doremus' 'person' 'restaurant' 'SPIMBENCH_small-2019' 'SPIMBENCH_large-2019'; do
        # echo $model
        for stringm in 'bill_sim' 'jaro_winkler'; do
            python3.8 ./main.py --suffix $dataset --embedding_name $model --dimension 300 --alpha 0.95 --beta 0.9 --similarity_measure $stringm
        done
    done
done
