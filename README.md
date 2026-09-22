# CRISPR — Bias Neuron Elimination

Implementation of **CRISPR** from *Mitigating Biases for Instruction-following Language Models via Bias Neurons Elimination* (Yang et al., ACL 2024) [PDF](https://aclanthology.org/2024.acl-long.490/).
No training is involved: every neuron is scored by its contribution to a biased answer, the top-ranked neurons are zeroed out, and the model is evaluated zero-shot with ten synonymous instructions.

The code in this repository has been repackaged, and a binary search algorithm has been added to determine the optimal number of neurons.

## Setup

```bash
pip install -r requirements.txt
python download_data.py
```

## Run

```bash
python compute_attribution.py
python run_crispr.py
```

The defaults are flan-t5-base on BBQ-SES and the output will be saved in the `results` folder.

On a SLURM cluster, `submit.sh` runs both steps (set `PYTHON` if `python` is not the
interpreter you want):

```bash
sbatch --export=ALL submit.sh
```


## Options

| Flag | Meaning |
| --- | --- |
| `--model` | any Flan-T5 checkpoint |
| `--data` | `SES.jsonl` or `Age.jsonl` |
| `--seeds` | one attribution trial per seed, averaged |
| `--num_samples` | development instances per trial (attribution only) |
| `--skill_lambda` | weight on the subtracted skill term (attribution only) |
| `--eval` | `generation` (greedy decode, match the output string to an option) or `scoring` (highest option probability) |
| `--max_neurons` | upper bound of the neuron-count search |
| `--search_tolerance` | interval width at which the search becomes exhaustive |
| `--batch_size` | attribution keeps gradients, so it needs a smaller value than evaluation |
| `--split_seed` | dev/test split; must match between the two scripts |
| `--max_dev`, `--max_test`, `--max_instructions` | caps for quick checks only |

Run either script with `--help` for the defaults.


## Files

| File | Contents |
| --- | --- |
| `instructions.py` | The ten synonymous BBQ instructions (paper Table 6) |
| `data.py` | Loading, dev/test split, prompt template, batching, `DEFAULT_SEEDS` |
| `attribution.py` | Bias attribution of every neuron (paper Eq. 1–5) |
| `pruning.py` | Neuron ranking and structured zero-pruning (paper Section 3.4) |
| `evaluate.py` | `Evaluator` — generation or scoring, per-instruction metrics |
| `compute_attribution.py` | Step 1 — attribution, averaged over instructions (Eq. 6) and seeds |
| `run_crispr.py` | Step 2 — neuron search on dev, evaluation on test |
| `download_data.py` | Fetches and checksums the BBQ files |
| `submit.sh` | SLURM entry point running both steps |
| `results/` | Where `run_crispr.py` writes its result CSVs |

## Citation and license

```bibtex
@inproceedings{yang2024mitigating,
  title={Mitigating biases for instruction-following language models via bias neurons elimination},
  author={Yang, Nakyeong and Kang, Taegwan and Choi, Stanley Jungkyu and Lee, Honglak and Jung, Kyomin},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={9061--9073},
  year={2024}
}
```

BBQ is distributed by NYU MLL under CC-BY-4.0 and is downloaded rather than
redistributed here. The code in this repository is released under the MIT license
(`LICENSE`).
