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

    def train(self, ins_matrix, ygts, epochs=20, lr=0.01):
        for epoch in range(epochs):
            # forward pass to get outputs for each input set from the input set matrix
            ypreds = [self(ins) for ins in ins_matrix]
            loss = sum((ypred - ygt)**2 for ypred, ygt in zip(ypreds, ygts))

            # backward pass to update gradients across each layers' nodes
            for param in self.parameters():
                param.grad = 0.0
            loss.backward()

            # update and reduce loss
            for param in self.parameters():
                param.data += -lr * param.grad

            