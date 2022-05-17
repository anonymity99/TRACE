from collections import defaultdict


class Schema:
    def __init__(self):
        self.schema = defaultdict(lambda: {"intents": set(), "slots": set()})

    def add_schema(self, dialogs):
        # train/valid/test
        for dialog in dialogs:
            for dial_id, dial in dialog.items():
                for turn in dial['turns']:
                    for domain in turn['label']:
                        for intent, slots in turn['label'][domain].items():
                            self.schema[domain]['intents'].add(intent)
                            self.schema[domain]['slots'].update(set(slots))

    def get_schema(self):
        return {
            "has_usr_label": True,  # 自行修改
            "has_sys_label": True,  # 自行修改
            "has_slot_label": True,  # 自行修改
            "schema": [
                {
                    "domain": domain,
                    "description": None,
                    "intents": [{"name": ii, "description": None} for ii in self.schema[domain]['intents']],
                    "slots": [{"name": ii, "description": None} for ii in self.schema[domain]['slots']]
                }
                for domain in self.schema
            ]
        }


if __name__ == '__main__':
    schema = Schema()
    pass
