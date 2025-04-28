# Deep Binary Classifier using Rule-Based Nodes


This project is part of the course Practical Work in AI (Master) at Johannes Kepler University Linz. It explores the use of rule-based nodes in a deep network for binary prediction tasks.

The architecture is based on the work ["Learning and Memorization" by Satrajit Chatterjee (2018)](https://proceedings.mlr.press/v80/chatterjee18a.html), which introduces deep classifiers built entirely from Lookup Tables (LUTs). [Gstrein (2022)](https://cca.informatik.uni-freiburg.de/gstrein/) reimplemented this concept in his Master's Thesis "Tuning the Learning of Circuit-Based Classifiers", expanding on training strategies.

This project builds on both implementations and investigates whether individual LUT nodes can be effectively replaced by rule-based classifiers. It uses RIPPER for logic rule extraction.


## Dataset

The experiments are based on a binary version of the MNIST dataset. Download these files from [LeCun's website](http://yann.lecun.com/exdb/mnist/),
extract them and put them in a folder called `data/mnist`:

- [train-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz)
- [train-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz)
- [t10k-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz)
- [t10k-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz)

REVIEW THIS


## Licence & Copyright

© 2025, Simon Tonner

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
See the [LICENSE](./LICENSE) file for details.