from core import Value
import random

class Module:
    def zero_grad(self):
        for param in self.parameters():
            if param.grad is not None:
                param.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, n_in):
        self.weights = [Value(random.uniform(-1,1)) for __ in range(n_in)]
        self.bias = Value(random.uniform(-1,1))

    def __call__(self, ins):
        pre_act_value: Value = sum(w * x for w, x in list(zip(self.weights, ins))) + self.bias
        out = pre_act_value.tanh()
        return out
    
    def parameters(self):
        return self.weights + [self.bias]
    
class Layer(Module):
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for __ in range(n_out)]

    def __call__(self, ins):
        return [neuron(ins) for neuron in self.neurons]
    
    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]
    
class MLP(Module):
    def __init__(self, n_in, n_outs):
        n_neurons_per_layer = [n_in] + n_outs
        self.layers = [Layer(n_neurons_per_layer[i], n_neurons_per_layer[i+1]) for i in range(len(n_outs))]

    def __call__(self, ins):
        for layer in self.layers:
            ins = layer(ins)

        return ins
    
    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]
    
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets
n = MLP(len(xs[0]),[4, 4, 1])
ypred = [n(x) for x in xs]
print(ypred)
