import torch
from core import Value

def main():
    test_sanity_check()
    test_more_ops()

def test_sanity_check():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.tanh() + z * x
    h = (z * z).tanh()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.tanh() + z * x
    h = (z * z).tanh()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    print(f"ymg = {ymg.data}; ypt = {ypt.data.item()}")
    assert ymg.data == ypt.data.item()
    # backward pass went well
    print(f"xmg = {xmg.grad}; xpt = {xpt.grad.item()}")
    assert xmg.grad == xpt.grad.item()

def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).tanh()
    d += 3 * d + (b - a).tanh()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).tanh()
    d = d + 3 * d + (b - a).tanh()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    print(f"gmg = {gmg.data}; gpt = {gpt.data.item()}")
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    print(f"amg = {amg.grad}; apt = {apt.grad.item()}")
    assert abs(amg.grad - apt.grad.item()) < tol
    print(f"bmg = {bmg.grad}; bpt = {bpt.grad.item()}")
    assert abs(bmg.grad - bpt.grad.item()) < tol

if __name__ == "__main__":
    main()