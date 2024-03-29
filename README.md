# config4ml
Basic configuration utilities and API for ML projects

NOTE: This repo is in pre-alpha and under development!!!

# Change-log

## [v0.0.1]

- [data] Transform config & registry
- [data] Dataset base config
- [data] MNIST built-in config (also as tutorial)
- [lightning] Lightning trainer config with callbacks support

## [v0.0.2]
- [lightning] Support for Neptune, Tensorboard logger configs
- [lightning] Support for Console logging

## [v0.0.3]
- [lightning] Trainer with console logging override option
- [lightning] More arguments for Early Stopping callback
- [lightning] Weights and biases logger (Wand) support
- [torch] Support for torch LR schedulers config [stepLR, exponentialLR, reduceOnPlateauLR]
- [torch] Support for torch Optimizers config [Adam, SGD]

## [v0.0.3b]
- [lightning] Trainer support for "resume_from_checkpoint" argument (default: None)

## [v0.0.3c]
- [data] Transform config separately for train / val
