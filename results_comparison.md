# Performance Comparison: 8 Samples vs 64 Samples in CURE_lib

This document compares the performance of the CURE_lib homomorphic encryption split learning framework when processing different batch sizes.

## Experiment Setup

- **Model Architecture**: 
  - Server: 3 layers (784→128→64→32)
  - Client: 1 layer (32→10)
- **Environment**: MacBook Air
- **Encryption**: CKKS homomorphic encryption via Lattigo v6
- **Data**: Synthetic MNIST-like data (784 pixels per image)

## Performance Results

| Metric | 8 Samples | 64 Samples | Ratio (64/8) |
|--------|-----------|------------|--------------|
| **Total Training Time** | 1m43.086s | 1m40.783s | 0.98 |
| **Encryption Time** | 1.026s | 0.994s | 0.97 |
| **Server Forward Pass** | 1m15.009s | 1m15.179s | 1.00 |
| **Client Compute** | 9.400ms | 9.604ms | 1.02 |
| **Server Backward** | 27.041s | 24.599s | 0.91 |

## Analysis

1. **Total Training Time**: 
   - The 64-sample batch was slightly faster overall (by about 2%)
   - This suggests that larger batch sizes can be more efficient

2. **Encryption Phase**:
   - Similar performance between 8 and 64 samples
   - The pixel-wise encryption approach means the encryption time is primarily determined by the number of pixels (784), not the batch size

3. **Server Forward Pass**:
   - Nearly identical performance despite 8x more samples
   - This demonstrates the efficiency of the SIMD (Single Instruction, Multiple Data) parallelism in the homomorphic operations
   - The computational cost is primarily driven by the number of ciphertexts and operations, not by the number of samples

4. **Client Compute**:
   - Minimal difference despite processing 8x more data
   - Client-side computation is very efficient since it operates on plaintext (decrypted) data

5. **Server Backward Pass**:
   - 64-sample batch was about 9% faster
   - This indicates that the homomorphic gradient computation and weight updates become more efficient with larger batch sizes

## Conclusion

The vertical feature packing approach employed by CURE_lib demonstrates excellent scaling with batch size. Despite increasing the batch size by 8x (from 8 to 64 samples), the overall training time remained almost the same, with some phases even showing improved performance.

This confirms that the SIMD-packed approach efficiently utilizes the available slots in each ciphertext, allowing the framework to process more samples with minimal additional computational cost. The primary bottleneck remains the homomorphic operations themselves, rather than the batch size.

For practical applications, this suggests that using larger batch sizes (up to the available slots in the ciphertexts) is beneficial for training efficiency in homomorphic encryption-based split learning. 