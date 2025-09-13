import pandas as pd
import pytest

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from architecture.classifier import DeepBinaryClassifier
from architecture.nodes.ripper import make_ripper_node


@pytest.fixture(scope="module")
def dataset():
    dataset_df = pd.read_csv("./test_dataset.csv")
    input_names = [c for c in dataset_df.columns if c != "target"]
    input_values = dataset_df[input_names].to_numpy(bool)
    target_values = dataset_df["target"].to_numpy(bool)

    return train_test_split(input_values, target_values, test_size=0.2, random_state=42, stratify=target_values)


@pytest.fixture
def trained_net(dataset):
    input_values_train, input_values_test, target_values_train, target_values_test = dataset
    config = dict(layer_node_counts=[8] * 3 + [1], layer_bit_counts=[4] * 4, seed=42)
    net = DeepBinaryClassifier(**config, node_factory=make_ripper_node, jobs=4)
    net.fit(input_values_train, target_values_train)
    return net, (input_values_test, target_values_test)


def test_pruning_preserves_accuracy(trained_net, epsilon: float = 1e-6):
    net, (input_values_test, target_values_test) = trained_net
    acc_before = accuracy_score(target_values_test, net.predict(input_values_test))

    net.prune()
    acc_after = accuracy_score(target_values_test, net.predict(input_values_test))

    assert abs(acc_after - acc_before) <= epsilon, "Pruning did not preserve accuracy"


def test_reduction_and_pruning_preserves_accuracy(trained_net, epsilon: float = 1e-6):
    net, (input_values_test, target_values_test) = trained_net
    acc_before = accuracy_score(target_values_test, net.predict(input_values_test))

    for layer in net.layers:
        for node in layer:
            if hasattr(node, "reduce_expression"):
                node.reduce_expression()
    net.prune()

    acc_after = accuracy_score(target_values_test, net.predict(input_values_test))

    assert abs(acc_after - acc_before) <= epsilon, "Reduction and pruning did not preserve accuracy"