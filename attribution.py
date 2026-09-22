import torch
from torch import nn

from data import PAD_ID

ENCODER_SIDE_SUFFIXES = ('EncDecAttention.k', 'EncDecAttention.v')


def collect_target_modules(model):
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and 'lm_head' not in name and 'shared' not in name
    }


def is_encoder_side(name):
    return name.startswith('encoder') or name.endswith(ENCODER_SIDE_SUFFIXES)


def sequence_log_prob(token_probs, mask):
    return torch.log(token_probs).masked_fill(~mask, 0.0).sum(dim=-1)


def sequence_prob(token_probs, mask):
    return token_probs.masked_fill(~mask, 1.0).prod(dim=-1)


class NeuronHook:
    def __init__(self, module):
        self.output = None
        self.grad = None
        self._handles = [
            module.register_forward_hook(lambda m, i, o: setattr(self, 'output', o)),
            module.register_full_backward_hook(lambda m, gi, go: setattr(self, 'grad', go[0])),
        ]

    def remove(self):
        for handle in self._handles:
            handle.remove()


class BiasAttributionScorer:

    def __init__(self, model, device='cuda', skill_lambda=1.0):
        self.model = model.eval().to(device)
        self.device = device
        self.skill_lambda = skill_lambda
        self.modules = collect_target_modules(self.model)
        self.hooks = {name: NeuronHook(module) for name, module in self.modules.items()}

    def close(self):
        for hook in self.hooks.values():
            hook.remove()

    def _token_probabilities(self, input_ids, attention_mask, label_ids):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=label_ids).logits
        token_probs = torch.softmax(logits, dim=-1).gather(-1, label_ids.unsqueeze(-1)).squeeze(-1)
        return token_probs, label_ids != PAD_ID

    def _attribution(self, rows, input_mask, label_mask, weights):
        scores = {}
        for name, hook in self.hooks.items():
            activation = hook.output.detach()[rows]
            gradient = hook.grad.detach()[rows]
            mask = (input_mask if is_encoder_side(name) else label_mask)[rows]

            attribution = weights.view(-1, 1, 1) * activation * gradient
            attribution = attribution.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            scores[name] = attribution.max(dim=1).values.sum(dim=0).cpu()
        return scores

    def batch_attribution(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        candidate_ids = batch['candidate_ids'].to(self.device)
        labels = batch['label'].to(self.device)

        batch_size, n_candidates, n_label_tokens = candidate_ids.shape
        n_distractors = n_candidates - 1
        rows = torch.arange(batch_size, device=self.device)

        golden_ids = candidate_ids[rows, labels]
        keep = torch.ones(batch_size, n_candidates, dtype=torch.bool, device=self.device)
        keep[rows, labels] = False
        distractor_ids = candidate_ids[keep].view(batch_size, n_distractors, n_label_tokens)

        flat_attention_mask = attention_mask.repeat_interleave(n_distractors, dim=0)
        token_probs, label_mask = self._token_probabilities(
            input_ids.repeat_interleave(n_distractors, dim=0),
            flat_attention_mask,
            distractor_ids.reshape(-1, n_label_tokens))

        log_probs = sequence_log_prob(token_probs, label_mask).view(batch_size, n_distractors)
        probs = sequence_prob(token_probs, label_mask).view(batch_size, n_distractors)

        biased = log_probs.argmax(dim=-1)
        confusion = probs[rows, biased].detach()

        self.model.zero_grad(set_to_none=True)
        log_probs[rows, biased].sum().backward()
        bias_scores = self._attribution(rows * n_distractors + biased,
                                        flat_attention_mask.bool(), label_mask, confusion)

        token_probs, label_mask = self._token_probabilities(input_ids, attention_mask, golden_ids)
        self.model.zero_grad(set_to_none=True)
        sequence_log_prob(token_probs, label_mask).sum().backward()
        skill_scores = self._attribution(rows, attention_mask.bool(), label_mask, confusion)

        self.model.zero_grad(set_to_none=True)
        return {name: bias_scores[name] - self.skill_lambda * skill_scores[name].clamp(min=0.0)
                for name in bias_scores}

    def attribution_for_dataloader(self, dataloader):
        totals = None
        n_instances = 0

        for batch in dataloader:
            batch_scores = self.batch_attribution(batch)
            totals = batch_scores if totals is None else \
                {name: totals[name] + batch_scores[name] for name in totals}
            n_instances += len(batch['label'])

        return {name: score / n_instances for name, score in totals.items()}
