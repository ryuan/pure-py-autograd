import math

class Value:
    def __init__(self, data, _prev=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev: set[Value] = set(_prev)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other: 'Value'):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        self._backward = _backward

        return out
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other: 'Value'):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __mul__(self, other: 'Value'):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        self._backward = _backward

        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self, ), f'**{self.data}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        self._backward = _backward

        return out
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1
    
    def tanh(self):
        tanh = (math.exp(2 * self.data) - 1) / ((math.exp(2 * self.data)) + 1)
        out = Value(tanh, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - tanh**2) * out.grad
        self._backward = _backward

        return out
    
    def backward(self):
        self.grad = 1.0
        stack = []
        visited = set()

        def topo_sort(v: 'Value'):
            if v not in visited:
                visited.add(v)
                for u in v._prev:
                    topo_sort(u)
                stack.append(v)
        topo_sort(self)

        topo_sorted: list['Value'] = stack.reverse()
        for node in topo_sorted:
            node._backward()

