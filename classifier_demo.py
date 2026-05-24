import numpy as np

from autograd_numpy import (
    MLP,
    Tensor,
    accuracy,
    set_seed,
    sgd_step,
    softmax_cross_entropy,
    train_test_split,
)


def make_classifier_data(seed=11):
    try:
        from sklearn.datasets import load_iris

        iris = load_iris()
        x = iris.data.astype(float)
        y = iris.target.astype(int)
    except ImportError:
        rng = np.random.default_rng(seed)
        centers = np.array([[-2.0, 0.0], [2.0, 0.5], [0.0, 2.5]])
        x_parts = []
        y_parts = []
        for label, center in enumerate(centers):
            x_parts.append(center + rng.normal(scale=0.55, size=(120, 2)))
            y_parts.append(np.full(120, label))
        x = np.vstack(x_parts)
        y = np.concatenate(y_parts)

    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
    return x, y


def main():
    set_seed(9)
    x, y = make_classifier_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_ratio=0.25, seed=9)

    n_classes = int(y.max()) + 1
    model = MLP(input_size=x.shape[1], hidden_sizes=[16, 12], output_size=n_classes, activation="relu")
    lr = 0.06
    epochs = 300

    for epoch in range(1, epochs + 1):
        logits = model(Tensor(x_train, requires_grad=False))
        loss = softmax_cross_entropy(logits, y_train)

        model.zero_grad()
        loss.backward()
        sgd_step(model.parameters(), lr)

        if epoch == 1 or epoch % 50 == 0:
            test_logits = model(Tensor(x_test, requires_grad=False))
            train_acc = accuracy(logits, y_train)
            test_acc = accuracy(test_logits, y_test)
            print(
                f"epoch {epoch:03d} | loss {loss.data.item():.4f} | "
                f"train acc {train_acc:.3f} | test acc {test_acc:.3f}"
            )

    final_logits = model(Tensor(x_test, requires_grad=False))
    print("\nfinal test accuracy:", round(accuracy(final_logits, y_test), 3))
    print("first ten predictions:", np.argmax(final_logits.data[:10], axis=1).tolist())
    print("first ten targets:    ", y_test[:10].tolist())


if __name__ == "__main__":
    main()
