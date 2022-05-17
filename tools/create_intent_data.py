import csv
import json
import os

import numpy as np
from tqdm import tqdm


_HEADER = ["text", "category"]


def create_examples(input_file, output_file, data_name):
    examples = {}
    data_dirname = os.path.dirname(input_file)
    intent_vocab_path = os.path.join(data_dirname, "categories.json")
    intent_names = json.load(open(intent_vocab_path))
    intent_label_to_idx = dict((label, idx) for idx, label in enumerate(intent_names))

    with open(input_file, "r") as data_file:
        reader = list(csv.reader(data_file))
        header = reader[0]
        assert header == _HEADER
        input_data = np.array(reader[1:])

    for dialog_id, (utt, intent) in enumerate(tqdm(input_data)):
        example = {
            "turns": [],
            "extra_info": ""
        }

        turn = {
            "turn_id": 0,
            "text": utt,
            "role": "user",
            "label": {'DEFAULT_DOMAIN': {intent: {}}},
            "extra_info": {'intent_label': intent_label_to_idx[intent]}
        }
        example["turns"].append(turn)
        examples[f'{data_name}-{dialog_id}'] = example

    with open(output_file, "w") as f:
        json.dump(examples, f, indent=2, separators=(",", ": "), sort_keys=True)


if __name__ == '__main__':
    data_names = ['banking', 'clinc', 'hwu']

    for data_name in data_names:
        for mode in ["train", "valid", "test"]:
            input_file = f'../data/{data_name}/{mode}.csv'
            output_file = f"../data/pre_train/UniDA2.0/single_turn/{data_name}/{mode}.json"
            assert os.path.exists(input_file), f"{input_file} isn't exist"
            assert not os.path.exists(output_file), f"{output_file} is already existed"
            create_examples(input_file=input_file, output_file=output_file, data_name=data_name)
            print(f'{data_name}-{mode}: saved to {output_file}')
        for mode in ["train_5", "train_10"]:
            input_file = f'../data/{data_name}/{mode}.csv'
            output_file = f"../data/pre_train/UniDA2.0/single_turn/{data_name}/{mode}.json"
            assert os.path.exists(input_file), f"{input_file} isn't exist"
            assert not os.path.exists(output_file), f"{output_file} is already existed"
            create_examples(input_file=input_file, output_file=output_file, data_name=data_name)
            print(f'{data_name}-{mode}: saved to {output_file}')
