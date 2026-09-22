import numpy as np
import torch
from torch.utils.data import DataLoader

from data import PAD_ID, collate_fn


def normalize(text):
    return text.strip().strip('.').lower()


class Evaluator:

    def __init__(self, model, tokenizer, mode='scoring', batch_size=32, device='cuda',
                 max_new_tokens=16):
        assert mode in ('scoring', 'generation')
        self.model = model
        self.tokenizer = tokenizer
        self.mode = mode
        self.batch_size = batch_size
        self.device = device
        self.max_new_tokens = max_new_tokens

    def _loader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=collate_fn)

    @torch.no_grad()
    def _choices_scoring(self, dataset):
        choices = []
        for batch in self._loader(dataset):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            candidate_ids = batch['candidate_ids'].to(self.device)

            option_scores = []
            for k in range(candidate_ids.size(1)):
                label_ids = candidate_ids[:, k].contiguous()
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                    labels=label_ids).logits
                probs = torch.softmax(logits, dim=-1).gather(
                    -1, label_ids.unsqueeze(-1)).squeeze(-1)
                option_scores.append(probs.masked_fill(label_ids == PAD_ID, 1.0).prod(dim=-1))

            choices.append(torch.stack(option_scores, dim=-1).argmax(dim=-1).cpu())
        return torch.cat(choices).numpy()

    @torch.no_grad()
    def _choices_generation(self, dataset):
        texts = []
        for batch in self._loader(dataset):
            generated = self.model.generate(input_ids=batch['input_ids'].to(self.device),
                                            attention_mask=batch['attention_mask'].to(self.device),
                                            max_new_tokens=self.max_new_tokens,
                                            do_sample=False, num_beams=1)
            texts.extend(self.tokenizer.batch_decode(generated.cpu(), skip_special_tokens=True))

        choices, exact = [], []
        for text, options, label in zip(texts, dataset.option_texts, dataset.labels):
            normalized = [normalize(o) for o in options]
            choices.append(normalized.index(normalize(text))
                           if normalize(text) in normalized else -1)
            exact.append(text.strip() == options[label].strip())
        return np.array(choices), np.array(exact)

    def report(self, dataset):
        extra = {}
        if self.mode == 'scoring':
            choices = self._choices_scoring(dataset)
        else:
            choices, exact = self._choices_generation(dataset)
            extra = {'exact_acc': float(exact.mean()),
                     'unmatched': float((choices == -1).mean())}

        correct = choices == dataset.labels
        ambiguous = dataset.context_conditions == 'ambig'
        return {'acc': float(correct.mean()),
                'ambig_acc': float(correct[ambiguous].mean()),
                'disambig_acc': float(correct[~ambiguous].mean()),
                **extra}

    def evaluate(self, datasets):
        reports = [self.report(dataset) for dataset in datasets]
        accuracies = [r['acc'] for r in reports]

        result = {'acc_per_instruction': accuracies,
                  'mean_acc': float(np.mean(accuracies)),
                  'std_acc': float(np.std(accuracies))}
        result.update({f'mean_{key}': float(np.mean([r[key] for r in reports]))
                       for key in reports[0] if key != 'acc'})
        return result
