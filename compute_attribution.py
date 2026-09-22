
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from attribution import BiasAttributionScorer
from data import (BBQDataset, DEFAULT_SEEDS, collate_fn, load_bbq, run_tag, sample_examples,
                  split_dev_test)
from instructions import BBQ_INSTRUCTIONS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='google/flan-t5-base')
    parser.add_argument('--data', default='data/BBQ/SES.jsonl')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                        help='one attribution trial per seed; the trials are averaged')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--skill_lambda', type=float, default=1.0,
                        help='weight on the subtracted golden-label (skill) term')
    parser.add_argument('--max_instructions', type=int, default=-1,
                        help='use only the first N instructions; for quick checks only')
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', default=None)
    return parser.parse_args()


def mean_scores(score_dicts):
    return {name: torch.stack([s[name] for s in score_dicts]).mean(dim=0)
            for name in score_dicts[0]}


def main():
    args = parse_args()
    dataset_id = Path(args.data).stem
    output = args.output or f'outputs/bias_attribution_{run_tag(args.model, args.data, args.seeds)}.pt'

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    scorer = BiasAttributionScorer(model, args.device, skill_lambda=args.skill_lambda)

    instructions = (BBQ_INSTRUCTIONS if args.max_instructions == -1
                    else BBQ_INSTRUCTIONS[:args.max_instructions])
    dev_examples, _ = split_dev_test(load_bbq(args.data), seed=args.split_seed)
    print(f'{dataset_id}: {len(dev_examples)} development instances, '
          f'{args.num_samples} sampled per trial, {len(args.seeds)} trial(s)', flush=True)

    trials = []
    for seed in args.seeds:
        examples = sample_examples(dev_examples, args.num_samples, seed)
        ambiguous = sum(e['context_condition'] == 'ambig' for e in examples)
        print(f'seed {seed}: {ambiguous}/{len(examples)} ambiguous instances', flush=True)

        per_instruction = []
        for i, instruction in enumerate(instructions):
            dataset = BBQDataset(examples, instruction, tokenizer)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_fn)
            per_instruction.append(scorer.attribution_for_dataloader(loader))
            print(f'  instruction {i + 1}/{len(instructions)} done', flush=True)

        trials.append(mean_scores(per_instruction))

    scores = mean_scores(trials)
    scorer.close()

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'scores': scores, 'model': args.model, 'data': args.data,
                'num_samples': args.num_samples, 'seeds': args.seeds,
                'split_seed': args.split_seed, 'skill_lambda': args.skill_lambda}, output)
    print(f'{sum(s.numel() for s in scores.values())} neuron scores saved to {output}')


if __name__ == '__main__':
    main()
