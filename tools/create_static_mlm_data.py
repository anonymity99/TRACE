import collections
import json
import pickle
import random

seed = 11
random.seed(seed)

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, vocab_words, rng, do_whole_word_mask=False,
                                 masked_lm_prob=0.15, max_predictions_per_seq=20):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (do_whole_word_mask and len(cand_indexes) >= 1 and
                token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return output_tokens, masked_lm_positions, masked_lm_labels


def __getitem__(self, item):

    t = self.lines[item]
    t1_random, t1_label = self.random_word(t)

    # [CLS] tag = SOS tag, [SEP] tag = EOS tag
    mlm_input = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]  # 3，1，2
    mlm_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]

    return mlm_input, mlm_label


def random_word(chars, bpe):
    output_label = []
    output_chars = []

    for i, char in enumerate(chars):
        if char in [bpe.bos_id, bpe.eos_id]:
            output_chars.append(char)
            output_label.append(bpe.pad_id)
            continue

        prob = random.random()
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_chars.append(bpe.mask_id)

            # 10% randomly change token to random token
            elif prob < 0.9:
                output_chars.append(random.randint(1, bpe.vocab_size - 1))  # start from 1, to exclude pad_id

            # 10% randomly change token to current token
            else:
                output_chars.append(char)

            output_label.append(char)

        else:
            output_chars.append(char)
            output_label.append(bpe.pad_id)

    return output_chars, output_label


def add_mask(sample, bpe):
    src = sample['src']
    src_mlm = []
    src_mlm_label = []
    for sentence in src:
        mlm, mlm_label = random_word(chars=sentence, bpe=bpe)
        assert len(sentence) == len(mlm) == len(mlm_label)
        src_mlm.append(mlm)
        src_mlm_label.append(mlm_label)
    sample['src_mlm'] = src_mlm
    sample['src_mlm_label'] = src_mlm_label
    return sample


def create_data(input_data, output_data, bpe):

    samples = []
    with open(input_data, 'r', encoding='utf-8') as fin:
        for line in fin:
            sample = json.loads(line)
            sample = add_mask(sample=sample, bpe=bpe)
            samples.append(sample)

    with open(output_data, 'w') as fw:
        for sample in samples:
            line = json.dumps(sample)
            fw.write(line)
            fw.write('\n')


if __name__ == '__main__':

    with open('../model/bpe.pt', 'rb') as fin:
        bpe = pickle.load(fin)

    # input_data = '../data/pre_train/train_label_data_ALL.Bert.jsonl'
    # output_data = '../data/pre_train/train_label_data_ALL_MLM.Bert.jsonl'
    input_data = '../data/pre_train/train_nolabel_data_ALL.Bert.jsonl'
    output_data = '../data/pre_train/train_nolabel_data_ALL_MLM.Bert.jsonl'

    create_data(input_data=input_data, output_data=output_data, bpe=bpe)

