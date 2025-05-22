from core import Value
import random

class Neuron:
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1,1)) for __ in range(n_in)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, ins):
        pre_act_value: Value = sum(w * x for w, x in list(zip(self.w, ins))) + self.b
        out = pre_act_value.tanh()
        return out
    
class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for __ in range(n_out)]

    def __call__(self, ins):
        outs = [neuron(ins) for neuron in self.neurons]
        return outs
    
class MLP:
    def __init__(self, n_in, n_outs):
        n_neurons_per_layer = [n_in] + n_outs
        self.layers = [Layer(n_neurons_per_layer[i], n_neurons_per_layer[i+1]) for i in range(len(n_outs))]

    def __call__(self, ins):
        for layer in self.layers:
            ins = layer(ins)
        
        return ins
    
ins = [2.0, 3.0, -1.0]
nn = MLP(len(ins), [4, 4, 1])
print(nn(ins))