import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PAD_ID = 0

DEFAULT_SEEDS = [2026]


def run_tag(model, data_path, seeds):
    return (f"{model.split('/')[-1]}_{Path(data_path).stem}"
            f"_seed{'-'.join(str(s) for s in seeds)}")


def load_bbq(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def split_dev_test(examples, dev_ratio=0.1, seed=0):
    rng = np.random.RandomState(seed)
    dev_indices = set(rng.choice(len(examples), int(len(examples) * dev_ratio),
                                 replace=False).tolist())
    dev = [e for i, e in enumerate(examples) if i in dev_indices]
    test = [e for i, e in enumerate(examples) if i not in dev_indices]
    return dev, test


def sample_examples(examples, n, seed):
    indices = np.sort(np.random.RandomState(seed).choice(len(examples), n, replace=False))
    return [examples[i] for i in indices]


def build_prompt(instruction, example):
    options = ', '.join(example[f'ans{i}'] for i in range(len(example['answer_info'])))
    return (f"{instruction} Context: {example['context']} "
            f" Question: {example['question']} "
            f" Answer options: {options}  Your answer: ")


class BBQDataset(Dataset):

    def __init__(self, examples, instruction, tokenizer):
        assert tokenizer.pad_token_id == PAD_ID, \
            f'PAD_ID is {PAD_ID} but the tokenizer pads with {tokenizer.pad_token_id}'
        n_candidates = len(examples[0]['answer_info'])
        prompts = [build_prompt(instruction, e) for e in examples]
        options = [e[f'ans{i}'] for e in examples for i in range(n_candidates)]

        self.input_ids = tokenizer(prompts).input_ids
        option_ids = tokenizer(options).input_ids
        self.candidate_ids = [option_ids[i:i + n_candidates]
                              for i in range(0, len(option_ids), n_candidates)]
        self.option_texts = [options[i:i + n_candidates]
                             for i in range(0, len(options), n_candidates)]
        self.labels = np.array([e['label'] for e in examples])
        self.context_conditions = np.array([e['context_condition'] for e in examples])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx]),
            'candidate_ids': [torch.tensor(ids) for ids in self.candidate_ids[idx]],
            'label': int(self.labels[idx]),
        }


def pad(sequences, max_len):
    return torch.stack([
        torch.cat([s, torch.full((max_len - len(s),), PAD_ID, dtype=torch.long)])
        for s in sequences
    ])


def collate_fn(samples):
    input_ids = pad([s['input_ids'] for s in samples],
                    max(len(s['input_ids']) for s in samples))
    candidate_max_len = max(len(ids) for s in samples for ids in s['candidate_ids'])

    return {
        'input_ids': input_ids,
        'attention_mask': (input_ids != PAD_ID).long(),
        'candidate_ids': torch.stack([pad(s['candidate_ids'], candidate_max_len)
                                      for s in samples]),
        'label': torch.tensor([s['label'] for s in samples]),
    }
