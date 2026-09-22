
import argparse
import csv
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from attribution import collect_target_modules
from data import DEFAULT_SEEDS, BBQDataset, load_bbq, run_tag, split_dev_test
from evaluate import Evaluator
from instructions import BBQ_INSTRUCTIONS
from pruning import NeuronPruner

GOLDEN_RATIO = (5 ** 0.5 - 1) / 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='google/flan-t5-base')
    parser.add_argument('--data', default='data/BBQ/SES.jsonl')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--eval', choices=['scoring', 'generation'], default='generation')
    parser.add_argument('--attribution', default=None)
    parser.add_argument('--output_prefix', default=None)
    parser.add_argument('--max_neurons', type=int, default=500)
    parser.add_argument('--search_tolerance', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_new_tokens', type=int, default=16)
    parser.add_argument('--max_dev', type=int, default=-1)
    parser.add_argument('--max_test', type=int, default=-1)
    parser.add_argument('--max_instructions', type=int, default=-1,
                        help='use only the first N instructions; for quick checks only')
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def build_datasets(examples, tokenizer, limit, instructions):
    examples = examples if limit == -1 else examples[:limit]
    return [BBQDataset(examples, instruction, tokenizer) for instruction in instructions]


def percent(value):
    return round(value * 100, 2)


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_dev_evaluator(evaluator, pruner, ranking, datasets):
    rows = {}

    def accuracy(n_neurons):
        if n_neurons not in rows:
            pruner.prune(ranking, n_neurons)
            result = evaluator.evaluate(datasets)
            rows[n_neurons] = {'n_neurons': n_neurons,
                               'mean_acc': percent(result['mean_acc']),
                               'std_acc': percent(result['std_acc'])}
            print(f"[dev] {n_neurons:>4} neurons  acc {rows[n_neurons]['mean_acc']:.2f} "
                  f"(std {rows[n_neurons]['std_acc']:.2f})", flush=True)
        return rows[n_neurons]['mean_acc']

    return accuracy, rows


def golden_section_search(accuracy, lo, hi, tolerance):
    assert tolerance >= 1, 'search_tolerance must be at least 1 to terminate'
    while hi - lo > tolerance:
        c = hi - round((hi - lo) * GOLDEN_RATIO)
        d = lo + round((hi - lo) * GOLDEN_RATIO)
        if accuracy(c) < accuracy(d):
            lo = c
        else:
            hi = d

    for n_neurons in range(lo, hi + 1):
        accuracy(n_neurons)


def evaluate_on_test(evaluator, pruner, ranking, datasets, best_n):
    rows = []
    for method, n_neurons in [('original', 0), ('CRISPR', best_n)]:
        if rows and n_neurons == rows[0]['n_neurons']:
            rows.append({**rows[0], 'method': method})
        else:
            pruner.prune(ranking, n_neurons)
            result = evaluator.evaluate(datasets)
            row = {'method': method, 'n_neurons': n_neurons,
                   'mean_acc': percent(result['mean_acc']),
                   'std_acc': percent(result['std_acc'])}
            row.update({key: percent(value) for key, value in result.items()
                        if key.startswith('mean_') and key != 'mean_acc'})
            row['acc_per_instruction'] = ' '.join(f'{percent(a):.2f}'
                                                  for a in result['acc_per_instruction'])
            rows.append(row)
        print(f"[test] {method:<8} {n_neurons:>4} neurons  acc {rows[-1]['mean_acc']:.2f} "
              f"(std {rows[-1]['std_acc']:.2f})", flush=True)
    return rows


def main():
    args = parse_args()
    tag = run_tag(args.model, args.data, args.seeds)
    attribution_path = args.attribution or f'outputs/bias_attribution_{tag}.pt'
    output_prefix = args.output_prefix or f'results/{tag}'

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).eval().to(args.device)
    evaluator = Evaluator(model, tokenizer, mode=args.eval, batch_size=args.batch_size,
                          device=args.device, max_new_tokens=args.max_new_tokens)

    checkpoint = torch.load(attribution_path, weights_only=True)
    assert checkpoint['model'] == args.model, \
        f"attribution model is {checkpoint['model']}, evaluation uses {args.model}"
    assert Path(checkpoint['data']).stem == Path(args.data).stem, \
        f"attribution data is {checkpoint['data']}, evaluation uses {args.data}"
    assert checkpoint['split_seed'] == args.split_seed, \
        f"attribution split_seed is {checkpoint['split_seed']}, evaluation uses {args.split_seed}"
    assert checkpoint['seeds'] == args.seeds, \
        f"attribution seeds are {checkpoint['seeds']}, evaluation uses {args.seeds}"

    scores = {name: score.to(args.device) for name, score in checkpoint['scores'].items()}
    dev_examples, test_examples = split_dev_test(load_bbq(args.data), seed=args.split_seed)
    instructions = (BBQ_INSTRUCTIONS if args.max_instructions == -1
                    else BBQ_INSTRUCTIONS[:args.max_instructions])
    dev_datasets = build_datasets(dev_examples, tokenizer, args.max_dev, instructions)
    test_datasets = build_datasets(test_examples, tokenizer, args.max_test, instructions)
    print(f'{tag}: {len(dev_datasets[0])} dev / {len(test_datasets[0])} test instances '
          f'x {len(instructions)} instructions, eval: {args.eval}', flush=True)

    pruner = NeuronPruner(collect_target_modules(model))
    ranking = NeuronPruner.rank_neurons(scores, top_k=args.max_neurons)

    accuracy, dev_rows = make_dev_evaluator(evaluator, pruner, ranking, dev_datasets)
    golden_section_search(accuracy, 0, args.max_neurons, args.search_tolerance)

    best_n = max(dev_rows, key=lambda n: dev_rows[n]['mean_acc'])
    print(f'\nbest number of bias neurons: {best_n} ({len(dev_rows)} evaluations)\n', flush=True)
    if best_n in (0, args.max_neurons):
        print(f'warning: best value is at a boundary of [0, {args.max_neurons}]\n', flush=True)

    test_rows = evaluate_on_test(evaluator, pruner, ranking, test_datasets, best_n)

    write_csv(f'{output_prefix}_dev_search.csv', [dev_rows[n] for n in sorted(dev_rows)])
    write_csv(f'{output_prefix}_test.csv', test_rows)
    print(f'saved {output_prefix}_dev_search.csv and {output_prefix}_test.csv')


if __name__ == '__main__':
    main()
