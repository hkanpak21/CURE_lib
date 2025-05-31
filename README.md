# Split Learning with Homomorphic Encryption

This project demonstrates a split learning approach for neural networks using homomorphic encryption with the CKKS scheme.

## Project Structure

```
.
├── cmd/
│   └── split_training_demo/
│       └── main.go        ← CLI handling
└── split_training/
    ├── data.go            ← MNIST I/O
    ├── params.go          ← Constants & global flags
    ├── hecontext.go       ← HE context initialization
    ├── models.go          ← Model structures
    ├── utils.go           ← Helper functions
    ├── forward.go         ← Forward pass
    ├── backward.go        ← Backward pass
    ├── training.go        ← Training loops
    └── evaluation.go      ← Model evaluation
```

## Prerequisites

- Go 1.18 or later
- MNIST dataset (will be downloaded automatically with `go generate`)

## Building and Running

### Downloading MNIST Data

To download the MNIST dataset:

```bash
cd split_training
go generate
```

### Training a Model

To train a new model with default settings:

```bash
make run
```

Or manually:

```bash
go run ./cmd/split_training_demo train
```

Training with fully homomorphic backpropagation (slower but more secure):

```bash
make train-full-he
```

Or manually:

```bash
go run ./cmd/split_training_demo train --he --batches 10
```

### Evaluating a Model

To evaluate a previously trained model:

```bash
make eval
```

Or manually:

```bash
go run ./cmd/split_training_demo eval client_model.txt server_model.txt
```

### Advanced Options

```
go run ./cmd/split_training_demo train [options]
  --batches <num>     - Train with specified number of batches
  --he                - Use fully homomorphic backpropagation
  --save              - Save trained models
  --client <path>     - Client model filename
  --server <path>     - Server model filename
  --batch <size>      - Mini-batch size (<= CKKS slots)
```

## Cleaning Up

```bash
make clean
```