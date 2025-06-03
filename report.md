# Vertical Feature Approach in CURE_lib Split Learning

## Overview

In CURE_lib's split learning implementation, the framework uses a "vertical feature approach" for matrix multiplication and dot products with homomorphic encryption. This approach differs from traditional sample-per-vector embeddings by:

1. Placing **feature points vertically** into entries of vectors
2. Arranging **multiple samples horizontally** across vector slots

This document explains how dot products are performed in this approach and why it's effective for homomorphic encryption-based split learning.

## Data Representation

### Traditional Approach vs. Vertical Approach

**Traditional (Sample-per-Vector):**
- Each vector represents one sample/image
- Each element in the vector is a feature (e.g., pixel)
- Process one sample at a time

**Vertical Approach (CURE_lib):**
- Each ciphertext slot at position `i` holds the same feature from different samples
- Multiple samples are packed horizontally in SIMD fashion
- This enables parallel processing of multiple samples in a single ciphertext

## Encryption Process

The encryption process (in `clientPrepareAndEncryptBatch`) demonstrates this approach:

1. For each image in the batch:
   - Create a ciphertext that contains all the pixels of one image
   - The result is an array of ciphertexts, one per image

```go
func clientPrepareAndEncryptBatch(he *HEContext, imgs [][]float64, idx []int) ([]*rlwe.Ciphertext, error) {
    // Create a ciphertext for each image in the batch
    encInputs := make([]*rlwe.Ciphertext, batch)
    
    // For each image in the batch
    for b := 0; b < batch; b++ {
        // Create a buffer for encoding
        vec := make([]float64, slots)
        
        // Copy the image pixels into the vector
        for i := 0; i < pixelsPerImage; i++ {
            vec[i] = img[i]
        }
        
        // Encode and encrypt
        // ...
        
        encInputs[b] = ct
    }
}
```

## Dot Product Implementation

The core of the dot product calculation is implemented in the `serverForwardPass` function. Here's how it works:

### 1. Vector-Matrix Multiplication in Homomorphic Domain

For each layer in the server model, the function processes input ciphertexts to produce output ciphertexts:

```go
// Process each layer
for l := 0; l < numLayers; l++ {
    // Get the dimensions of the current layer
    inputDim := serverModel.GetLayerInputDim(l)
    outputDim := serverModel.GetLayerOutputDim(l)
    
    // Create outputs for this layer
    var layerOutputs []*rlwe.Ciphertext
    
    // For each output neuron
    for j := 0; j < outputDim; j++ {
        // Initialize a zero ciphertext for accumulation
        outputCipher, err := he.encryptor.EncryptNew(zeroPlaintext)
        
        // Compute weighted sum: W[l][i][j] * encInputs[i] + b[l][j]
        for i := 0; i < inputDim; i++ {
            // Get current input
            current := layerInputs[l][i]
            
            // Multiply by weight (homomorphically)
            weight := serverModel.GetWeight(l, i, j)
            weightPlaintext := createPlaintextWithValue(weight)
            
            // Multiply input by weight
            weighted := current.CopyNew()
            he.evaluator.Mul(weighted, weightPlaintext, weighted)
            
            // Add to the output
            he.evaluator.Add(outputCipher, weighted, outputCipher)
        }
        
        // Add bias
        // ...
        
        layerOutputs = append(layerOutputs, outputCipher)
    }
}
```

### 2. SIMD-Packed Implementation

For efficient processing, the system leverages SIMD (Single Instruction, Multiple Data) parallelism using ciphertext "blocks":

```go
// Example from the forward pass implementation
numBlocks := (outputDim + neuronsPerCT - 1) / neuronsPerCT
layerOutputs := make([]*rlwe.Ciphertext, numBlocks)

// For each block of output neurons
for blockIdx := 0; blockIdx < numBlocks; blockIdx++ {
    // Compute matrix-vector product for this block
    layerOutputs[blockIdx], err = heLayerMultiplyAndAddBias(
        heContext, serverModel, l, prev, blockIdx,
    )
}
```

### 3. Rotation-Based Dot Product

For more optimized dot products, the framework uses a rotation-based approach (`dotPacked` function):

```go
func dotPacked(he *HEContext, encImg *rlwe.Ciphertext, ptWRow []*rlwe.Plaintext, slotsPerPixel int) (*rlwe.Ciphertext, error) {
    // Create the accumulator as a copy of the first product
    acc := encImg.CopyNew()
    
    // Multiply with the first weight
    he.evaluator.Mul(acc, ptWRow[0], acc)
    
    // For each remaining pixel
    for i := 1; i < len(ptWRow); i++ {
        // Rotate the input vector
        rotated := encImg.CopyNew()
        he.evaluator.Rotate(encImg, i*slotsPerPixel, rotated)
        
        // Multiply by the corresponding weight
        he.evaluator.Mul(rotated, ptWRow[i], rotated)
        
        // Add to accumulator
        he.evaluator.Add(acc, rotated, acc)
    }
    
    return acc, nil
}
```

## Gradient Computation and Backpropagation

During backpropagation, the framework must handle gradients in the same vertical format:

1. The client decrypts the encrypted activations from the server
2. Each activation is arranged so that `a1PerBlock[b][j][nInBlock]` represents the activation of server-neuron (block b, offset nInBlock) for example j
3. After computing gradients, they are re-packed into ciphertexts in the same vertical format

```go
// Example from clientForwardAndBackward
for b := 0; b < numBlocks; b++ {
    // Pack neurons for this block
    startNeuron := b * neuronsPerCT
    endNeuron := min(startNeuron+neuronsPerCT, inputDim)
    
    // Pack each neuron's gradients for all examples in the batch
    for n := startNeuron; n < endNeuron; n++ {
        neuronOffset := (n - startNeuron) * batchSize
        
        // Copy this neuron's gradients for all examples
        for i := 0; i < batchSize; i++ {
            scratch[neuronOffset+i] = inputGradients[n][i]
        }
    }
    
    // Encode and encrypt
    encGradBlk[b] = encGrad
}
```

## Advantages of the Vertical Approach

1. **SIMD Parallelism**: Process multiple samples in parallel within a single ciphertext
2. **Reduced Ciphertext Count**: When processing batches, fewer ciphertexts are needed compared to one-per-sample
3. **Efficient Weight Updates**: During backpropagation, weight updates can be computed in parallel across samples
4. **Optimized Memory Usage**: Better utilization of the available slots in each ciphertext
5. **Homomorphic-Friendly**: This approach is more compatible with the constraints of homomorphic encryption operations

## Conclusion

The vertical feature approach implemented in CURE_lib provides an efficient way to perform dot products and matrix multiplications under homomorphic encryption. By arranging features vertically and samples horizontally, the framework can leverage SIMD parallelism to process multiple samples simultaneously, making the split learning process more efficient in the encrypted domain. 