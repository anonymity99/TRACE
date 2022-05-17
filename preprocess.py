"""
Preprocess script.
"""

import os
import glob
import argparse

from trace.args import parse_args
from trace.data.dataset import Dataset
from trace.data.fields.field import BPETextField

FILE_NAME = 'train.json'


def main():
    parser = argparse.ArgumentParser()
    Dataset.add_cmdline_argument(parser)
    BPETextField.add_cmdline_argument(parser)
    args = parse_args(parser)

    bpe = BPETextField(args)
    build_examples_fn = bpe.build_examples_multi_turn
    build_score_matrix_fn = bpe.build_score_matrix
    build_score_matrix_multiprocessing_fn = bpe.build_score_matrix_multiprocessing
    data_paths = list(
        os.path.dirname(c) for c in sorted(glob.glob(args.data_dir + '/**/' + FILE_NAME, recursive=True)))
    data_paths = bpe.filter_data_path(data_paths=data_paths)

    for mode in ['train', 'valid', 'test']:
        for data_path in data_paths:
            input_file = os.path.join(data_path, f'{mode}.json')
            output_file = os.path.join(data_path, f'{mode}.{args.tokenizer_type}.jsonl')
            output_score_file = os.path.join(data_path, f'{mode}.Score.npy')
            if os.path.exists(input_file) and not os.path.exists(output_file):
                examples = build_examples_fn(input_file, data_type=mode)
                if examples:
                    bpe.save_examples(examples, output_file)
                else:
                    continue
            if os.path.exists(output_file) and not os.path.exists(output_score_file) and \
                    not args.dynamic_score and 'UniDA' in data_path:
                examples = bpe.load_examples(output_file)
                if args.num_process >= 2:
                    score_matrix = build_score_matrix_multiprocessing_fn(examples)
                else:
                    score_matrix = build_score_matrix_fn(examples)
                bpe.save_examples(score_matrix, output_score_file)


if __name__ == "__main__":
    main()
