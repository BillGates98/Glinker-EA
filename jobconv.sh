#!/bin/bash
source_directory="./inputs/freebase-large"

for target_directory in "$source_directory"/*; do
    if [ -d "$target_directory" ]; then
        python3 ./convert_xml_to_graph.py --input_data "$target_directory/refalign.rdf" --output_path "$target_directory/"
    fi
done


