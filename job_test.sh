# each suffix have 3 files : source, target, valid_same_as
#
#
# python3 ./test.py --input_path ./inputs/ --output_path ./outputs/ --suffix spaten_hobbit


# python3 ./test.py --input_path ./inputs/ --output_path ./outputs/ --suffix person --threshold 0.99
# python3 ./test.py --input_path ./inputs/ --output_path ./outputs/ --suffix restaurant --threshold 0.7
# python3 ./test.py --input_path ./inputs/ --output_path ./outputs/ --suffix sider-dailymed --threshold 0.98
# python3 ./test.py --input_path ./inputs/ --output_path ./outputs/ --suffix sider-drugbank --threshold 0.98
# python3 ./test.py --input_path ./inputs/ --output_path ./outputs/ --suffix diseasome-sider --threshold 0.98
# python3 ./test.py --input_path ./inputs/ --output_path ./outputs/ --suffix SPIMBENCH_large-2016 --threshold 1.0
# python3 ./test.py --input_path ./inputs/ --output_path ./outputs/ --suffix SPIMBENCH_small-2016 --threshold 1.0
# python3 ./test.py --input_path ./inputs/ --output_path ./outputs/ --suffix UOBM_small-2016 --threshold 0.95
# python3 ./test.py --input_path ./inputs/ --output_path ./outputs/ --suffix doremus --threshold 0.97

source_directory="./inputs/freebase-large/"
# source_directory="./inputs/freebase/"

for target_directory in "$source_directory"/*; do
    if [ -d "$target_directory" ]; then
        name=$(basename "$target_directory")
        python3 ./test.py --input_path ./inputs/freebase-large/ --output_path ./outputs/freebase-large/ --suffix "$name" --threshold 0.3
    fi
done


