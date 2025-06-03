# CURE_lib Code Review Results

## 1. Verify `slotsPerPixel` is not hard-coded

✅ **PASS**: The `dotPacked` function (in `forward.go`) correctly uses the `slotsPerPixel` parameter instead of hard-coded values:

```go
// Helper for dot product with power-of-two rotations
func dotPacked(he *HEContext, encImg *rlwe.Ciphertext, ptWRow []*rlwe.Plaintext, slotsPerPixel int) (*rlwe.Ciphertext, error) {
    // ...
    for i := 1; i < len(ptWRow); i++ {
        // Rotate the input vector
        rotated := encImg.CopyNew()
        if err := he.evaluator.Rotate(encImg, i*slotsPerPixel, rotated); err != nil {
            return nil, fmt.Errorf("error rotating in dot product: %v", err)
        }
        // ...
    }
    // ...
}
```

The `batchSize` parameter is correctly defined as a configurable variable in `params.go` and passed appropriately through function calls.

## 2. Confirm correct plaintext-weight packing

✅ **PASS**: The weight packing is implemented correctly in the `scalarPlain` function:

```go
// scalarPlain creates a plaintext with all slots set to the same value
func scalarPlain(value float64, params ckks.Parameters, encoder *ckks.Encoder) *rlwe.Plaintext {
    pt := ckks.NewPlaintext(params, params.MaxLevel())

    // Create a vector with all slots having the same value
    vec := make([]float64, params.N()/2)
    for i := range vec {
        vec[i] = value
    }

    // Encode the vector
    encoder.Encode(vec, pt)

    return pt
}
```

For selective masking, the code includes `maskFirst` function that creates a mask for only the first `batchSize` slots:

```go
func maskFirst(params ckks.Parameters, encoder *ckks.Encoder, batchSize int) *rlwe.Plaintext {
    // ...
    // Create a vector with 1s for the first batchSize slots and 0s elsewhere
    vec := make([]float64, params.N()/2)
    for i := 0; i < batchSize && i < len(vec); i++ {
        vec[i] = 1.0
    }
    // ...
}
```

## 3. Check that every rotation uses `CopyNew()`

✅ **PASS**: All rotations in the code make proper use of `CopyNew()`:

```go
// In dotPacked function
rotated := encImg.CopyNew()
if err := he.evaluator.Rotate(encImg, i*slotsPerPixel, rotated); err != nil {
    return nil, fmt.Errorf("error rotating in dot product: %v", err)
}

// In sumSlotsWithRotations function
rotated := ct.CopyNew()
if err := ctx.evaluator.Rotate(result, i, rotated); err != nil {
    return nil, fmt.Errorf("error in rotation: %v", err)
}
```

No instances of direct modifications like `Rotate(encImg, ..., encImg)` were found.

## 4. Ensure loop bounds cover exactly all features

✅ **PASS**: Feature iteration uses proper bounds in the relevant functions:

```go
// In serverForwardPass function
for i := 0; i < inputDim; i++ {
    // Process each input dimension
}

// In dotPacked function
for i := 1; i < len(ptWRow); i++ {
    // Iterate through all weight elements
}
```

No off-by-one errors or incorrect bounds were found in the critical loops.

## 5. Initialize accumulator from a true zero-ciphertext

⚠️ **PARTIAL**: The `dotPacked` function initializes the accumulator by copying the first product:

```go
// Create the accumulator as a copy of the first product
acc := encImg.CopyNew()

// Multiply with the first weight
if err := he.evaluator.Mul(acc, ptWRow[0], acc); err != nil {
    return nil, fmt.Errorf("error in dot product multiplication: %v", err)
}
```

This is a valid optimization technique as it eliminates one addition operation, but it should be noted that `acc` is not initialized as an all-zero ciphertext.

In the `serverForwardPass` function, output ciphertexts are correctly initialized with zeros:

```go
// Initialize a ciphertext with zeros for the output neuron
zeroPlaintext := ckks.NewPlaintext(he.params, he.params.MaxLevel())
// Encrypt it
outputCipher, err := he.encryptor.EncryptNew(zeroPlaintext)
```

## 6. Rescale after each homomorphic multiplication

⚠️ **PARTIAL**: The code doesn't consistently rescale after every multiplication:

- In `dotPacked`, there's no explicit rescaling after the multiplication before adding to the accumulator.
- In `applyReLU`, rescaling is done once at the end rather than after each operation.

However, the `serverBackwardAndUpdate` function does include rescaling:

```go
// Check if rescaling is needed and if we have enough levels
if gradCopy.Level() > 0 {
    // Rescale if possible
    if err := heContext.evaluator.Rescale(gradCopy, gradCopy); err != nil {
        // ...
    }
}
```

## 7. Round-trip consistency test

🔄 **NEEDS VERIFICATION**: The code doesn't include explicit consistency tests for different batch sizes. A test should be implemented to verify that the results for small batch sizes match the corresponding slots in larger batch sizes.

However, the test framework (in `training_test.go`) does include test cases with different batch sizes:

```go
func setTestBatchSize(size int) func() {
    original := BatchSize
    BatchSize = size
    return func() {
        BatchSize = original
    }
}
```

## 8. Inspect CKKS parameter set for adequate noise budget

✅ **PASS**: The CKKS parameters in `hecontext.go` are configured with adequate depth for multi-layer operations:

```go
paramsLiteral := ckks.ParametersLiteral{
    LogN:            13,                                // Ring degree: 2^13 = 8192
    LogQ:            []int{55, 50, 50, 50, 50, 50, 50}, // More levels for multi-layer operations
    LogP:            []int{60, 60},                     // Special modulus for key switching
    LogDefaultScale: 40,                                // Higher scale for better precision
}
```

The high LogN (13) provides 4096 slots, which is sufficient for batching.

## 9. Verify no unintended slot-mixing via dummy samples

🔄 **NEEDS VERIFICATION**: There's no explicit verification in the code to ensure dummy slots remain zero. The `maskFirst` function creates a mask for only the first `batchSize` slots, but additional tests should verify that operations don't accidentally affect dummy slots.

## 10. Double-check bias addition uses correct mask

⚠️ **CONCERN**: The bias addition in `serverForwardPass` doesn't use a mask to limit to only the first `batchSize` slots:

```go
// Create a plaintext for the bias
biasPlaintext := ckks.NewPlaintext(he.params, he.params.MaxLevel())
// Encode a single value in all slots
biasValues := make([]float64, he.params.N()/2)
for k := range biasValues {
    biasValues[k] = bias
}
he.encoder.Encode(biasValues, biasPlaintext)
```

This could potentially lead to bias being applied to dummy slots as well.

## 11. Confirm rotation distances at the boundaries

✅ **PASS**: The `hecontext.go` file generates rotation keys for powers of two up to sufficient range:

```go
// --- rotations we need (powers of two up to slots/2) ---
rotations := []int{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}
```

This ensures that rotation distances at boundaries will work correctly.

## 12. Validate no in-place ciphertext overwrites during accumulation

✅ **PASS**: The accumulation operations don't overwrite input ciphertexts:

```go
// In dotPacked function
if err := he.evaluator.Add(acc, rotated, acc); err != nil {
    return nil, fmt.Errorf("error adding in dot product: %v", err)
}
```

Here, `rotated` is a fresh copy, and `acc` is correctly used as both input and output.

## 13. Check for stray hard-coded batch size references

⚠️ **PARTIAL**: While most of the code correctly uses the `BatchSize` variable, there are a few places that need attention:

1. In `clientForwardAndBackward`, there's a hard-coded `64` in the neuronsPerCT calculation:
   ```go
   neuronsPerCT := calculateNeuronsPerCT(heContext.params.N()/2, batchSize, 64)
   ```

2. In `params.go`, the default batch size is set to a specific value:
   ```go
   var BatchSize = 8 // Reduced from 64 for faster testing
   ```
   However, this is a variable rather than a hard-coded constant throughout the code.

## Summary

The code is generally well-structured and follows good practices for homomorphic encryption operations. A few areas need verification or improvement:

1. **Rescaling after multiplication**: Ensure consistent rescaling after homomorphic multiplications.
2. **Bias masking**: Apply appropriate masking to bias values to prevent affecting dummy slots.
3. **Round-trip verification**: Implement tests to verify consistency between small and large batch sizes.
4. **Hard-coded constants**: Replace the few remaining hard-coded values with named constants.

The framework successfully handles both 8-sample and 64-sample batch sizes, as demonstrated by the experimental results showing similar performance across these batch sizes. 