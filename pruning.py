import torch


class NeuronPruner:

    def __init__(self, modules):
        self.modules = modules
        self._pruned = {}

    @staticmethod
    def rank_neurons(scores, top_k=None):
        names = list(scores.keys())
        sizes = [scores[name].numel() for name in names]
        order = torch.argsort(torch.cat([scores[name] for name in names]), descending=True)
        order = order if top_k is None else order[:top_k]

        offsets = torch.tensor([0] + sizes, device=order.device).cumsum(0)
        layers = torch.bucketize(order, offsets[1:], right=True)
        neurons = order - offsets[layers]
        return [(names[l], int(n)) for l, n in zip(layers.tolist(), neurons.tolist())]

    def restore(self):
        with torch.no_grad():
            for (name, neuron), (weight, bias) in self._pruned.items():
                self.modules[name].weight[neuron] = weight
                if bias is not None:
                    self.modules[name].bias[neuron] = bias
        self._pruned.clear()

    def prune(self, ranking, n_neurons):
        self.restore()
        with torch.no_grad():
            for name, neuron in ranking[:n_neurons]:
                module = self.modules[name]
                bias = None if module.bias is None else module.bias[neuron].detach().clone()
                self._pruned[(name, neuron)] = (module.weight[neuron].detach().clone(), bias)
                module.weight[neuron] = 0.0
                if module.bias is not None:
                    module.bias[neuron] = 0.0
