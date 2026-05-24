import numpy as np

from autograd_numpy import MLP, Tensor, mse_loss, set_seed, sgd_step, train_test_split


def make_regression_data(n_samples=600, seed=7):
    try:
        from sklearn.datasets import make_regression

        x, y = make_regression(
            n_samples=n_samples,
            n_features=4,
            n_informative=4,
            noise=12.0,
            random_state=seed,
        )
        y = y.reshape(-1, 1)
    except ImportError:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n_samples, 4))
        y = (
            2.5 * x[:, [0]]
            - 1.7 * x[:, [1]]
            + 0.8 * x[:, [2]] * x[:, [3]]
            + rng.normal(scale=0.2, size=(n_samples, 1))
        )

    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
    y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-8)
    return x, y


def main():
    set_seed(4)
    x, y = make_regression_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_ratio=0.25, seed=4)

    model = MLP(input_size=x.shape[1], hidden_sizes=[24, 16], output_size=1, activation="tanh")
    lr = 0.04
    epochs = 350

    for epoch in range(1, epochs + 1):
        predictions = model(Tensor(x_train, requires_grad=False))
        loss = mse_loss(predictions, y_train)

        model.zero_grad()
        loss.backward()
        sgd_step(model.parameters(), lr)

        if epoch == 1 or epoch % 50 == 0:
            test_predictions = model(Tensor(x_test, requires_grad=False))
            test_loss = mse_loss(test_predictions, y_test)
            print(
                f"epoch {epoch:03d} | train mse {loss.data.item():.4f} | "
                f"test mse {test_loss.data.item():.4f}"
            )

    sample_predictions = model(Tensor(x_test[:5], requires_grad=False)).data.ravel()
    print("\nfirst five normalized predictions vs targets")
    for pred, target in zip(sample_predictions, y_test[:5].ravel()):
        print(f"pred {pred: .3f} | target {target: .3f}")


if __name__ == "__main__":
    main()
