import numpy as np

from autograd_numpy import (
    SimpleGNN,
    Tensor,
    accuracy,
    set_seed,
    sgd_step,
    softmax_cross_entropy,
)


def normalize_adjacency(adjacency):
    adjacency = adjacency + np.eye(adjacency.shape[0])
    degree = adjacency.sum(axis=1)
    inv_sqrt_degree = np.diag(1.0 / np.sqrt(degree + 1e-8))
    return inv_sqrt_degree @ adjacency @ inv_sqrt_degree


def make_toy_graph(seed=3):
    rng = np.random.default_rng(seed)
    n_per_group = 25
    n_nodes = n_per_group * 2
    labels = np.array([0] * n_per_group + [1] * n_per_group)

    # Two soft clusters. The feature signal helps, and the graph structure nudges it.
    x_left = rng.normal(loc=[-1.0, 0.0, 0.5], scale=0.45, size=(n_per_group, 3))
    x_right = rng.normal(loc=[1.0, 0.0, -0.5], scale=0.45, size=(n_per_group, 3))
    features = np.vstack([x_left, x_right])

    adjacency = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            same_group = labels[i] == labels[j]
            edge_prob = 0.22 if same_group else 0.03
            if rng.random() < edge_prob:
                adjacency[i, j] = 1
                adjacency[j, i] = 1

    train_mask = np.zeros(n_nodes, dtype=float)
    train_mask[:8] = 1
    train_mask[n_per_group : n_per_group + 8] = 1
    test_mask = 1 - train_mask

    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    return features, normalize_adjacency(adjacency), labels, train_mask, test_mask


def main():
    set_seed(13)
    features, adjacency, labels, train_mask, test_mask = make_toy_graph()

    model = SimpleGNN(input_size=features.shape[1], hidden_size=10, output_size=2)
    lr = 0.08
    epochs = 250

    x = Tensor(features, requires_grad=False)
    graph = Tensor(adjacency, requires_grad=False)

    for epoch in range(1, epochs + 1):
        logits = model(x, graph)
        loss = softmax_cross_entropy(logits, labels, mask=train_mask)

        model.zero_grad()
        loss.backward()
        sgd_step(model.parameters(), lr)

        if epoch == 1 or epoch % 50 == 0:
            train_acc = accuracy(logits, labels, mask=train_mask)
            test_acc = accuracy(logits, labels, mask=test_mask)
            print(
                f"epoch {epoch:03d} | loss {loss.data.item():.4f} | "
                f"train acc {train_acc:.3f} | test acc {test_acc:.3f}"
            )

    final_logits = model(x, graph)
    print("\nfinal masked test accuracy:", round(accuracy(final_logits, labels, mask=test_mask), 3))
    print("node predictions:", np.argmax(final_logits.data, axis=1).tolist())


if __name__ == "__main__":
    main()
