# Split Training with Homomorphic Encryption

This document provides a detailed overview of the `split_training` library, which implements a secure split learning approach for neural networks using homomorphic encryption (HE).

## Overview

Split learning is a privacy-preserving machine learning technique where a neural network is divided into multiple segments, each trained by different parties without sharing raw data. This implementation uses homomorphic encryption to enhance privacy guarantees by allowing computations on encrypted data.

The library demonstrates training a neural network on the MNIST dataset with the network split between a client and server, where the server processes encrypted inputs and gradients without having access to the actual values.

## Architecture

### Neural Network Structure

The neural network consists of 3 layers:
- **Layer 1** (Server-side): Input → Hidden Layer 1 (784 → 128 neurons)
- **Layer 2** (Client-side): Hidden Layer 1 → Hidden Layer 2 (128 → 32 neurons)
- **Layer 3** (Client-side): Hidden Layer 2 → Output (32 → 10 neurons)

### Key Components

The library is organized into several modules, each handling specific aspects of the split learning process:

#### Core Data Structures

- `ClientModel`: Holds the weights and biases for the client-side layers (layers 2 and 3)
- `ServerModel`: Holds the weights and biases for the server-side layer (layer 1)
- `HEServerModel`: Encrypted version of the server model
- `HEServerPacked`: Optimized version of the encrypted server model for SIMD operations
- `HEContext`: Holds all cryptographic components for homomorphic encryption

#### Parameters

- `InputDim`: Input dimension (784 for MNIST)
- `HiddenDim1`: First hidden layer dimension (128)
- `HiddenDim2`: Second hidden layer dimension (32)
- `OutputDim`: Output dimension (10 for MNIST digit classification)
- `NeuronsPerCT`: Number of neurons packed per ciphertext (64)
- `BatchSize`: Mini-batch size for training

#### Training Modes

The library supports two training modes:
1. **Standard Homomorphic Mode**: Server forward pass is encrypted, but weights are updated after decrypting gradients
2. **Fully Homomorphic Mode**: Both forward and backward passes are performed homomorphically

## Workflow

### 1. Data Flow in Split Learning

```
Client                          Server
+-------------------------+     +------------------------+
| 1. Prepare & encrypt    |---->| 2. Forward pass on     |
|    input data (images)  |     |    encrypted inputs    |
+-------------------------+     +------------------------+
                                          |
                                          v
+-------------------------+     +------------------------+
| 3. Decrypt activations  |<----| Return encrypted       |
|    Forward through      |     | activations (layer 1)  |
|    layers 2 & 3         |     +------------------------+
|    Compute loss         |
|    Backward pass        |
|    Update client weights|
+-------------------------+
          |
          v
+-------------------------+     +------------------------+
| 4. Encrypt gradients    |---->| 5. Backward pass &     |
|    for server layer     |     |    weight updates      |
+-------------------------+     +------------------------+
```

### 2. Homomorphic Encryption Implementation

The library uses the Lattigo library to implement homomorphic encryption with the CKKS scheme:

- **Encryption Parameters**: Configured for sufficient multiplicative depth (LogN=12, LogQ=[40, 40, 40, 40])
- **SIMD Operations**: Packing multiple values into a single ciphertext to parallelize operations
- **Rotation Keys**: Generated to support slot rotations for efficient vector operations

### 3. Optimizations

The library implements several optimizations to improve performance:

- **Parallel Processing**: Multi-threaded operations for forward and backward passes
- **Neuron Packing**: Multiple neurons are packed into a single ciphertext (NeuronsPerCT = 64)
- **Batch Packing**: Multiple examples from a batch are processed in parallel
- **Efficient Slot Rotation**: Optimized rotation operations for summing slots and gradient computation

## Key Functions

### Data Handling

- `readMNISTData()`: Loads MNIST dataset from files
- `readMNISTImages()`: Helper to read image data
- `readMNISTLabels()`: Helper to read label data

### Model Initialization

- `initHE()`: Initializes homomorphic encryption context
- `initClientModel()`: Creates and initializes client-side model
- `initServerModel()`: Creates and initializes server-side model

### Forward Pass

- `clientPrepareAndEncryptBatch()`: Encrypts a batch of images
- `serverForwardPass()`: Performs forward pass on encrypted inputs
- `clientForwardAndBackward()`: Client-side forward and backward pass

### Backward Pass and Weight Updates

- `serverBackwardAndUpdate()`: Updates server weights homomorphically
- `packedUpdate()`: Updates weights using packed ciphertexts for SIMD operations
- `innerSumSlots()`: Sums values across slots in a ciphertext
- `chunkSum()`: Reduces slots belonging to each neuron

### Model Evaluation

- `evaluateModel()`: Evaluates model accuracy on test data
- `clientEvaluateForwardPass()`: Client-side forward pass for evaluation

### Utility Functions

- `convertToPacked()`: Converts a server model to a packed homomorphic version
- `scalarPlain()`: Encodes a constant replicated in every slot
- `maskFirst()`: Creates a mask to keep only the first slot in each chunk
- `repeat()`: Creates a repeated value array

## Homomorphic Encryption Techniques

### 1. Encrypted Matrix-Vector Multiplication

The server performs matrix-vector multiplication on encrypted inputs:

```
encZ1[i] = bias[i] + Σ(encInput[j] * W1[j][i]) for j=0 to inputDim-1
```

### 2. Encrypted ReLU Approximation

ReLU activation is approximated using Chebyshev polynomials:

```
ReLU(x) ≈ 0.32 + 0.5*x + 0.23*x²
```

### 3. Encrypted Gradient Computation

Gradients are computed and packed efficiently:

```
encGradBlk[b] = pack(dA1_Transposed[b*NeuronsPerCT:(b+1)*NeuronsPerCT])
```

### 4. Homomorphic Weight Updates

Weights are updated fully homomorphically:

```
W ← W + η * (X ⊙ gradients)
```

where η is the learning rate and ⊙ represents element-wise multiplication.

## Security Considerations

- **Data Privacy**: Raw data never leaves the client
- **Model Privacy**: Server model weights can be kept encrypted during training
- **Gradient Privacy**: Gradients are encrypted during transmission
- **Threat Model**: Protects against honest-but-curious adversaries

## Performance Considerations

- **Computational Overhead**: Homomorphic operations are computationally intensive
- **Memory Usage**: Ciphertexts require significantly more memory than plaintexts
- **Trade-offs**: The library balances security with performance through SIMD optimizations

## Limitations

- Approximate activation functions are used instead of exact ReLU
- Limited multiplicative depth restricts network complexity
- Performance is significantly slower than plaintext training

## Conclusion

The `split_training` library demonstrates how homomorphic encryption can be used to implement privacy-preserving split learning. By keeping data encrypted during computation, it provides strong privacy guarantees while still allowing effective model training.

The implementation showcases practical techniques for working with homomorphic encryption in machine learning, including SIMD operations, parallel processing, and efficient algorithms for common neural network operations. 