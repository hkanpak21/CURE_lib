# Batch Size Scaling in Homomorphic Encryption Split Learning

This document compares the performance of the CURE_lib homomorphic encryption split learning framework when processing different batch sizes: 8, 64, and 1024 samples.

## Experiment Setup

- **Model Architecture**: 
  - Server: 3 layers (784→128→64→32)
  - Client: 1 layer (32→10)
- **Environment**: MacBook Air
- **Encryption**: CKKS homomorphic encryption via Lattigo v6
- **Data**: Synthetic MNIST-like data (784 pixels per image)
- **Configuration**: 1 batch, 1 epoch

## Performance Results

| Metric | 8 Samples | 64 Samples | 1024 Samples | Ratio (1024/8) |
|--------|-----------|------------|--------------|----------------|
| **Total Training Time** | 1m43.086s | 1m40.783s | 1m43.523s | 1.00 |
| **Encryption Time** | 1.026s | 0.994s | 1.029s | 1.00 |
| **Server Forward Pass** | 1m15.009s | 1m15.179s | 1m15.196s | 1.00 |
| **Client Compute** | 9.400ms | 9.604ms | 76.600ms | 8.15 |
| **Server Backward** | 27.041s | 24.599s | 27.222s | 1.01 |

## Analysis

### 1. Total Training Time
The total training time remains remarkably consistent across all batch sizes, with less than 3% variation between the smallest (8 samples) and largest (1024 samples) batches. This demonstrates exceptional scaling efficiency of the homomorphic encryption framework.

### 2. Encryption Time
Encryption time is nearly identical across all batch sizes (approximately 1 second). This confirms that the encryption process is primarily dependent on the number of features (784 pixels) rather than the number of samples.

### 3. Server Forward Pass
The server forward pass time shows minimal variation (all approximately 1m15s) despite the 128x increase in batch size from 8 to 1024 samples. This is a strong validation of the SIMD (Single Instruction, Multiple Data) approach, where homomorphic operations process multiple samples in parallel within the same ciphertext.

### 4. Client Compute
Client-side computation is the only component that shows significant scaling with batch size, increasing from ~9ms with 8 samples to ~77ms with 1024 samples. This is expected because:
- Client operations are performed on plaintext (decrypted) data
- These operations scale linearly with the number of samples
- Even with 1024 samples, client compute remains a negligible portion of the overall pipeline (less than 0.1% of total time)

### 5. Server Backward Pass
The server backward pass time remains consistent across all batch sizes (approximately 25-27 seconds). This again demonstrates the effectiveness of SIMD parallelism in homomorphic operations, where gradient computation and weight updates can be performed on multiple samples simultaneously.

## Implications

1. **Exceptional SIMD Utilization**: The framework shows near-perfect SIMD utilization, with virtually no performance penalty when increasing batch size from 8 to 1024 samples (a 128x increase).

2. **Optimal Batch Size Strategy**: Since training time doesn't increase with batch size, users should maximize batch size to fully utilize the available slots in ciphertexts. This leads to more efficient training by processing more samples in the same amount of time.

3. **Client-Server Balance**: The negligible client computation time (~0.1% of total) confirms that this split learning approach successfully offloads the computational burden to the server while minimizing client-side work.

4. **Computational Bottlenecks**: The server forward pass (~73% of total time) and backward pass (~26% of total time) dominate the computation pipeline. Any optimization efforts should focus on these components.

5. **Efficiency at Scale**: The framework demonstrates remarkable efficiency even with 1024 samples, making it suitable for practical machine learning applications where large batch sizes are desirable.

## Conclusion

The CURE_lib's implementation of vertical feature packing for homomorphic encryption shows exceptional scaling properties. The ability to process 128x more data with essentially no increase in computation time validates the effectiveness of the SIMD-based approach. 

This scaling behavior makes the framework highly suitable for privacy-preserving machine learning applications where throughput and efficiency are critical. The vertical feature approach successfully leverages the parallel processing capabilities of homomorphic encryption, overcoming one of the traditional bottlenecks in encrypted machine learning. 