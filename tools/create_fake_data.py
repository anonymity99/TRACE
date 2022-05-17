import json
from tqdm import tqdm


def create_examples(input_file, output_file, data_name):
    examples = {}

    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)

    count = 0
    for dialog_id in tqdm(input_data):
        count += 1
        if count > 100:
            break
        example = {
            "turns": [],
            "extra_info": ""
        }
        entry = input_data[dialog_id]
        utterances = entry['log']
        for turn_id, utt in enumerate(utterances):
            span_info = utt["span_info"]
            dialog_act = utt["dialog_act"]
            span, label = {}, {}
            for act, tuple in dialog_act.items():
                label[act] = {}
                for slot, value in tuple:
                    if slot == 'none':
                        continue
                    if slot in label[act]:
                        assert value not in ['?', 'none']
                        label[act][slot].append(value)
                    else:
                        if value not in ['?', 'none']:
                            label[act][slot] = [value]
                        else:
                            label[act][slot] = []
            for tuple in span_info:
                assert len(tuple) == 5
                act, slot, value, start, exclusive_end = tuple
                if slot in span:
                    span[slot].append({"start": start, "exclusive_end": exclusive_end})
                else:
                    span[slot] = [{"start": start, "exclusive_end": exclusive_end}]

            turn = {
                "turn_id": turn_id,
                "text": utt['text'],
                "role": "system" if utt['metadata'] else "user",
                "label": label,
                "span": span,
                "domain": "",
                "extra_info": ""
            }

            example["turns"].append(turn)
        examples[f'{data_name}-{dialog_id.replace(".json", "")}'] = example

    with open(output_file, "w") as f:
        json.dump(examples, f, indent=2, separators=(",", ": "), sort_keys=True)


if __name__ == '__main__':
    input_file = '../data/MultiWOZ2.2/train_dials.json'
    for data_name in ['DemoDataset01', 'DemoDataset02', 'DemoDataset03', 'DemoDataset04']:
        output_file = f'./train_{data_name}.json'
        create_examples(input_file=input_file, output_file=output_file, data_name=data_name)
