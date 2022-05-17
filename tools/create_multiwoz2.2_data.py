import json
from collections import defaultdict

from tqdm import tqdm


def refactor_dialog_act(raw_dialog_act):
    dialog_act = defaultdict(dict)
    for domain_act, value in raw_dialog_act.items():
        assert '-' in domain_act
        domain, act = domain_act.split("-")
        assert act not in dialog_act[domain]
        dialog_act[domain][act] = value
    return dialog_act


def construct_span(value, text):
    text = text.lower()
    value = value.lower()
    if value in text:
        start = text.index(value)
        exclusive_end = start + len(value)
        return [start, exclusive_end]
    else:
        """
        1. refer
        2. number (6 <-> six)
        3. yes/no
        4. don't care
        5. label map
        """
        # print(f'Slot value: {value}, text: {text}')
        return None


def construct_domain_label(dialog_act, text):
    label = {}

    for act, tuple in dialog_act.items():
        label[act] = {}
        for slot, value in tuple:
            if slot == 'none':
                continue
            if slot in label[act]:
                assert value != '?'
                if value == 'none':
                    print(f"Type-1 text: {text}, dialog_act: {dialog_act}")
                    assert slot == 'choice'
                label[act][slot].append({"span": construct_span(value, text), 'value': value})
            else:
                if value != '?':
                    if value == 'none':
                        if slot in ['entrancefee', 'choice']:
                            print(f"Type-2 text: {text}, dialog_act: {dialog_act}")
                            label[act][slot] = [{"span": construct_span(value, text), 'value': value}]
                        else:
                            print(f"Type-3 text: {text}, dialog_act: {dialog_act}")
                            assert slot == 'phone'
                            label[act][slot] = []
                    else:
                        label[act][slot] = [{"span": construct_span(value, text), 'value': value}]
                else:
                    label[act][slot] = []

    return label


def construct_label(dialog_act, utt):
    label = {}
    for domain, act_info in dialog_act.items():
        domain_label = construct_domain_label(dialog_act=act_info, text=utt['text'])
        label[domain] = domain_label
    return label


def create_examples(input_file, output_file, data_name):
    examples = {}

    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)

    for dialog_id in tqdm(input_data):
        example = {
            "turns": [],
            "extra_info": ""
        }
        entry = input_data[dialog_id]
        utterances = entry['log']

        for turn_id, utt in enumerate(utterances):
            dialog_act = refactor_dialog_act(raw_dialog_act=utt["dialog_act"])
            label = construct_label(dialog_act=dialog_act, utt=utt)
            turn = {
                "turn_id": turn_id,
                "text": utt['text'],
                "role": "system" if utt['metadata'] else "user",
                "label": label,
                "extra_info": ""
            }

            example["turns"].append(turn)
        examples[f'{data_name}-{dialog_id.replace(".json", "")}'] = example

    with open(output_file, "w") as f:
        json.dump(examples, f, indent=2, separators=(",", ": "), sort_keys=True)


if __name__ == '__main__':
    data_name = 'MultiWOZ2.2'
    with_label = True
    for mode in ["train", "valid", "test"]:
        input_file = f'../data/{data_name}/{mode}_dials.json'
        output_file = f"../data/pre_train/{'UniDA2.0' if with_label else 'UnDial2.0'}/{data_name}/{mode}.json"
        create_examples(input_file=input_file, output_file=output_file, data_name=data_name)
        print(f'{mode}: saved to {output_file}')
