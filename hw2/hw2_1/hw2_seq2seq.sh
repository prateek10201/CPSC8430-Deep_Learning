#!/bin/bash
# Usage: ./hw2_seq2seq.sh <data_directory> <output_file>
# Example: ./hw2_seq2seq.sh testing_data/feat result_output_testset.txt

python3 test_seq2seq.py $1 $2
