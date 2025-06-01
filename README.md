# CURE_lib: Configurable Split Learning with Homomorphic Encryption

CURE_lib is a Go library for implementing privacy-preserving split learning with homomorphic encryption. The library enables training neural networks where part of the computation is performed on encrypted data, preserving privacy while still enabling effective training.

## Key Features

### Configurable Network Architecture

- **Arbitrary Network Depth**: Support for an arbitrary number of layers on both server and client sides
- **Flexible Split Point**: Ability to configure where the network is split between server and client
- **Full Backpropagation**: Complete backward pass through all server layers with proper gradient propagation

### Homomorphic Encryption Integration

- **CKKS Scheme**: Uses the Lattigo CKKS scheme for approximate homomorphic encryption
- **SIMD Operations**: Efficiently packs multiple values into ciphertext slots for parallel processing
- **Optimized HE Operations**: Minimizes the number of expensive operations while maintaining security

## Implementation Details

### Multi-Layer Server Processing

The implementation supports configurable neural networks with:

1. **Multiple Server Layers**: 
   - Each server layer processes encrypted inputs homomorphically
   - All intermediate activations are cached for backward pass

2. **Backward Pass Through Server Layers**:
   - Computes and applies weight/bias updates in the encrypted domain
   - Propagates gradients backward through all server layers
   - Uses ReLU derivative approximation for non-linear activations

3. **Parallelized Processing**:
   - Uses goroutines for parallel processing of neurons and examples
   - SIMD packing for processing multiple samples simultaneously

### Client-Side Processing

1. **Client Forward Pass**:
   - Decrypts server outputs
   - Performs forward pass through client layers
   - Computes loss and final outputs

2. **Client Backward Pass**:
   - Computes gradients for client weights and biases
   - Updates client model parameters
   - Prepares encrypted gradients to send back to server

## Performance

- Training a model with architecture [784-128-64-32-10] where the server has 3 layers (784→128→64→32) and the client has 1 layer (32→10)
- Processing time for a batch of 4 examples:
  - Encryption: ~3.6s
  - Server forward pass: ~75s
  - Client computation: ~10ms
  - Server backward pass: ~28s

## Usage Example

```go
// Create model configuration with configurable architecture
config := &split.ModelConfig{
    Arch:     []int{784, 128, 64, 32, 10}, // Architecture with 3 server layers
    SplitIdx: 3,                            // Server has 3 layers (784->128->64->32), client has 1 (32->10)
}

// Initialize client and server models
clientModel := split.InitClientModel(config)
serverModel := split.InitServerModel(config)

// Forward pass with layer input caching
layerInputs, encActivations, err := split.ServerForwardPassWithLayerInputs(heContext, serverModel, encPixels)

// Client forward and backward pass
encGradients, err := split.ClientForwardAndBackward(heContext, clientModel, encActivations, labels, batchIndices)

// Server backward pass with multi-layer gradient propagation
err = split.ServerBackwardAndUpdate(heContext, serverModel, encGradients, layerInputs, learningRate)
```

## Future Work

- Implement support for convolutional layers
- Add checkpointing for long-running training processes
- Optimize homomorphic operations for faster training
- Add support for different activation functions