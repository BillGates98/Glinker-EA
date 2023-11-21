for model in  'RESCAL' 'TransE' 'DistMult' 'ComplEx' 'ConvE' 'RotatE' 'TuckER'; do 
    # echo $model
    # python3 ./general.py --input_path ./inputs/ --output_path ./outputs/ --suffix doremus --threshold 0.97 --model $model --dimension 20
    # python3 ./general.py --input_path ./inputs/ --output_path ./outputs/ --suffix person --threshold 0.99 --model $model --dimension 20
    # python3 ./general.py --input_path ./inputs/ --output_path ./outputs/ --suffix restaurant --threshold 0.7 --model $model --dimension 20
    # python3 ./general.py --input_path ./inputs/ --output_path ./outputs/ --suffix SPIMBENCH_small-2016 --threshold 1.0 --model $model --dimension 20
    python3 ./general.py --input_path ./inputs/ --output_path ./outputs/ --suffix SPIMBENCH_large-2016 --threshold 1.0 --model $model --dimension 20
done

# python3 ./tests.py --input_path ./inputs/ --output_path ./outputs/ --suffix spaten_hobbit
# list of models :
#   

# source_directory="./inputs/freebase"

# for target_directory in "$source_directory"/*; do
#     if [ -d "$target_directory" ]; then
#         name=$(basename "$target_directory")
#         python3 ./tests.py --input_path "$source_directory/" --output_path "./outputs/freebase/" --suffix "$name" --threshold 0.3
#     fi
#     exit
# done



