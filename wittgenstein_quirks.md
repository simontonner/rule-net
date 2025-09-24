# Wittgenstein Quirks


The `wittgenstein` library handles some edge cases in ways that may be unexpected. This document addresses a few issues
and how they are affect the implementation.

There are specific tests in place to ensure that these quirks do not affect the correctness of the network
implementation.


#### HOMOGENEOUS LABELS

If `wittgenstein` encounters a homogeneous target column, it writes the following warnings:

```
ripper.py: .fit: RuntimeWarning: 
No positive samples. Existing target labels=[False].

ripper.py: .fit | base.py: ._check_allpos_allneg: RuntimeWarning: 
Ruleset is empty. All predictions it makes with method .predict will be negative. It may be untrained or was trained on
a dataset split lacking positive examples.
```

The same warning is produced if all labels are `[True]`.

However, homogeneous labels can only occur in contrived scenarios. Typically each node sees the global targets anyways
and the dataset should have both classes present anyways.


**Handling**

The algorithm can't grow a rule based on information gain in this case, it falls back to a default rule. In the case of
`wittgenstein`, this default rule always predicts `False` despite the official RIPPER behaviour which should choose the
majority class.

The default rule does not use any features, so all backlinks are cut. Furthermore, this node is itself a homogeneous
input feature for any subsequent node. RIPPER removes homogeneous input features since they carry no information. This
implies that such a node is completely isolated and does not contribute to the final output.

The warnings are not explicitly ignored because they indicate something fundamentally wrong with the data.


#### LOW BIT COUNTS

If a node has very few input bits, it may not be able to grow any meaningful rule. In this case it falls back to the
default rule as described above.


**Handling**

Same as above.


#### NOISY LABELS

In theory, RIPPER cannot create any rule if the target is pure noise or any other case where the target is uncorrelated
with the input features (e.g., parity). However, in practice we have observed that `wittgenstein` will still find some rule that somehow partitions the data.


**Handling**

Since the network behaves stochastically anyways, a little extra rule is not problematic. Should it really find no rule,
then the node will self-isolate as described above.