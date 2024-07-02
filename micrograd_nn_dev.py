import random
from typing import List, Union
from itertools import chain
from micrograd.engine import Value

class Module:
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0

    def parameters(self) -> List[Value]:
        return []

class Neuron(Module):
    def __init__(self, nin: int, nonlin: bool = True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin

    def __call__(self, x: List[Value]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self) -> List[Value]:
        return self.w + [self.b]

    def __repr__(self) -> str:
        n_weights = len(self.w)
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({n_weights})"

class Layer(Module):
    def __init__(self, nin: int, nout: int, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x: List[Value]) -> Union[Value, List[Value]]:
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> List[Value]:
        return list(chain.from_iterable(n.parameters() for n in self.neurons))

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    def __init__(self, nin: int, nouts: List[int]):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i != len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x: List[Value]) -> List[Value]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Value]:
        return list(chain.from_iterable(layer.parameters() for layer in self.layers))

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
