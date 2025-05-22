import math

class Value:
    def __init__(self, data, _prev=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_prev)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other: 'Value'):
        out = Value(self.data + other.data, (self, other), '+')
        self.grad = 1.0 * other.grad
        other.grad = 1.0 * self.grad
        return out
    
    def __sub_(self, other: 'Value'):
        out = Value(self.data - other.data, (self, other), '-')
        return out
    
    def __mul_(self, other: 'Value'):
        out = Value(self.data * other.data, (self, other), '*')
        self.grad = other.data * out.grad
        other.grad = self.data * out.grad
        return out
    
    def __div_(self, other: 'Value'):
        out = Value(self.data / other.data, (self, other), '/')
        return out
    
    def __tanh(self):
        tanh = (math.exp(2 * self.data) - 1) / ((math.exp(2 * self.data)) + 1)
        out = Value(tanh, (self, ), 'tanh')
        self.grad = (1 - tanh**2) * out.grad
        return out