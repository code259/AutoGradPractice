import math
import random

import numpy as np


def _as_array(value):
    return np.array(value, dtype=float)


def _unbroadcast(grad, shape):
    grad = np.asarray(grad, dtype=float)

    if shape == ():
        return np.array(grad.sum(), dtype=float)

    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad


class Scalar:
    def __init__(self, value, _children=(), _op=""):
        self.value = float(value)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Scalar(value={self.value:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.value + other.value, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.value * other.value, (self, other), "*")

        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return Scalar(other) / self

    def __pow__(self, power):
        assert isinstance(power, (int, float))
        out = Scalar(self.value**power, (self,), f"**{power}")

        def _backward():
            self.grad += power * (self.value ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.value)
        out = Scalar(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Scalar(max(0.0, self.value), (self,), "relu")

        def _backward():
            self.grad += (self.value > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()


class Tensor:
    def __init__(self, data, _children=(), _op="", requires_grad=True):
        self.data = _as_array(data)
        self.grad = np.zeros_like(self.data, dtype=float)
        self.requires_grad = requires_grad
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, data={self.data}, grad={self.grad})"

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = np.zeros_like(self.data, dtype=float)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return Tensor(other, requires_grad=False) / self

    def __pow__(self, power):
        assert isinstance(power, (int, float))
        out = Tensor(self.data**power, (self,), f"**{power}")

        def _backward():
            if self.requires_grad:
                self.grad += power * (self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def exp(self):
        data = np.exp(self.data)
        out = Tensor(data, (self,), "exp")

        def _backward():
            if self.requires_grad:
                self.grad += data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            if self.requires_grad:
                self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        data = np.tanh(self.data)
        out = Tensor(data, (self,), "tanh")

        def _backward():
            if self.requires_grad:
                self.grad += (1 - data * data) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        data = np.maximum(0, self.data)
        out = Tensor(data, (self,), "relu")

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        data = 1 / (1 + np.exp(-self.data))
        out = Tensor(data, (self,), "sigmoid")

        def _backward():
            if self.requires_grad:
                self.grad += data * (1 - data) * out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            if not self.requires_grad:
                return
            grad = out.grad
            if axis is not None and not keepdims:
                axes = axis if isinstance(axis, tuple) else (axis,)
                for ax in sorted(axes):
                    grad = np.expand_dims(grad, ax)
            self.grad += np.ones_like(self.data) * grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            denom = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            denom = np.prod([self.data.shape[ax] for ax in axes])
        return self.sum(axis=axis, keepdims=keepdims) / denom

    def backward(self):
        topo = []
        visited = set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)
        self.grad = np.ones_like(self.data, dtype=float)

        for node in reversed(topo):
            node._backward()


class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()


class Neuron(Module):
    def __init__(self, input_size, activation="tanh"):
        scale = math.sqrt(2 / input_size)
        self.w = Parameter(np.random.randn(input_size, 1) * scale)
        self.b = Parameter(np.zeros((1,)))
        self.activation = activation

    def __call__(self, x):
        out = x @ self.w + self.b
        if self.activation == "relu":
            return out.relu()
        if self.activation == "sigmoid":
            return out.sigmoid()
        if self.activation == "tanh":
            return out.tanh()
        return out

    def parameters(self):
        return [self.w, self.b]


class Layer(Module):
    def __init__(self, input_size, output_size, activation="tanh"):
        scale = math.sqrt(2 / input_size)
        self.w = Parameter(np.random.randn(input_size, output_size) * scale)
        self.b = Parameter(np.zeros((1, output_size)))
        self.activation = activation

    def __call__(self, x):
        out = x @ self.w + self.b
        if self.activation == "relu":
            return out.relu()
        if self.activation == "sigmoid":
            return out.sigmoid()
        if self.activation == "tanh":
            return out.tanh()
        return out

    def parameters(self):
        return [self.w, self.b]


class MLP(Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation="tanh"):
        sizes = [input_size] + list(hidden_sizes) + [output_size]
        self.layers = []

        for i in range(len(sizes) - 1):
            is_last = i == len(sizes) - 2
            layer_activation = "linear" if is_last else activation
            self.layers.append(Layer(sizes[i], sizes[i + 1], layer_activation))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [parameter for layer in self.layers for parameter in layer.parameters()]


class GraphConv(Module):
    def __init__(self, input_size, output_size, activation="relu"):
        scale = math.sqrt(2 / input_size)
        self.w = Parameter(np.random.randn(input_size, output_size) * scale)
        self.b = Parameter(np.zeros((1, output_size)))
        self.activation = activation

    def __call__(self, x, adjacency):
        # adjacency is usually fixed data, but making it a Tensor keeps the math tidy.
        if not isinstance(adjacency, Tensor):
            adjacency = Tensor(adjacency, requires_grad=False)
        out = adjacency @ x @ self.w + self.b
        if self.activation == "relu":
            return out.relu()
        if self.activation == "tanh":
            return out.tanh()
        return out

    def parameters(self):
        return [self.w, self.b]


class SimpleGNN(Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.conv1 = GraphConv(input_size, hidden_size, activation="relu")
        self.conv2 = GraphConv(hidden_size, output_size, activation="linear")

    def __call__(self, x, adjacency):
        h = self.conv1(x, adjacency)
        return self.conv2(h, adjacency)

    def parameters(self):
        return self.conv1.parameters() + self.conv2.parameters()


def mse_loss(prediction, target):
    target = target if isinstance(target, Tensor) else Tensor(target, requires_grad=False)
    return ((prediction - target) ** 2).mean()


def softmax_cross_entropy(logits, targets, mask=None):
    targets = np.asarray(targets, dtype=int)
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    sample_losses = -np.log(probs[np.arange(len(targets)), targets] + 1e-12)
    if mask is None:
        weights = np.ones_like(sample_losses)
    else:
        weights = np.asarray(mask, dtype=float)

    denom = max(weights.sum(), 1.0)
    loss_value = (sample_losses * weights).sum() / denom
    out = Tensor(loss_value, (logits,), "cross_entropy")

    def _backward():
        if not logits.requires_grad:
            return
        grad = probs.copy()
        grad[np.arange(len(targets)), targets] -= 1
        grad *= weights[:, None] / denom
        logits.grad += grad * out.grad

    out._backward = _backward
    return out


def accuracy(logits, targets, mask=None):
    targets = np.asarray(targets, dtype=int)
    predictions = np.argmax(logits.data, axis=1)
    correct = predictions == targets
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        return float(correct[mask].mean())
    return float(correct.mean())


def sgd_step(parameters, lr):
    for parameter in parameters:
        parameter.data -= lr * parameter.grad


def train_test_split(x, y, test_ratio=0.25, seed=0):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    test_count = int(len(x) * test_ratio)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
