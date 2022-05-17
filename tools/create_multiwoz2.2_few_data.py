import json


def create_examples(input_file, output_file, mode):
    examples = {}
    ids = []

    with open(f"../data/pre_train/UniDA2.0/multi_turn/MultiWOZ2.2/{mode}.json", "r", encoding='utf-8') as reader:
        all_examples = json.load(reader)

    if mode == 'train':
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)
        for dialog_id in input_data:
            id = f"MultiWOZ2.2-{dialog_id.replace('.json', '')}"
            if id not in ids:
                ids.append(id)
        for dialog_id, dialog in all_examples.items():
            if dialog_id in ids:
                examples[dialog_id] = dialog
        assert len(examples) == len(input_data)
    else:
        examples = all_examples

    with open(output_file, "w") as f:
        json.dump(examples, f, indent=2, separators=(",", ": "), sort_keys=True)


if __name__ == '__main__':
    data_name = 'MultiWOZ2.2_few'
    for mode in ["train", "valid", "test"]:
        input_file = f'../data/MultiWOZ2.1_few/{mode}_dials.json'
        output_file = f"../data/pre_train/UniDA2.0/multi_turn/{data_name}/{mode}.json"
        create_examples(input_file=input_file, output_file=output_file, mode=mode)
        print(f'{mode}: saved to {output_file}')
