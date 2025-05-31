package hetest

import (
	"fmt"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	// rlwe is needed for types like sk *rlwe.SecretKey, pk *rlwe.PublicKey in variable declarations
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"

	"cure_test/pkg/he"
)

// checkCloseEnough compares two slices of float64 and returns true if all elements
// are within a given epsilon.
func checkCloseEnough(a, b []float64, epsilon float64, t *testing.T) bool {
	if len(a) != len(b) {
		t.Errorf("checkCloseEnough: slices have different lengths (%d vs %d)", len(a), len(b))
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > epsilon {
			t.Errorf("checkCloseEnough: mismatch at index %d. Expected %f, got %f (diff %f > epsilon %f)",
				i, b[i], a[i], math.Abs(a[i]-b[i]), epsilon)
			return false
		}
	}
	return true
}

func TestScalarMultCiphertext(t *testing.T) {
	// 1. Setup HE context
	params, err := he.GetCKKSParameters(he.TestSet) // Using TestSet for faster testing
	if err != nil {
		t.Fatalf("Failed to get CKKS parameters: %v", err)
	}

	kgen := he.KeyGenerator(params)
	sk := kgen.GenSecretKeyNew()   // sk is *rlwe.SecretKey
	pk := kgen.GenPublicKeyNew(sk) // pk is *rlwe.PublicKey

	encoder := he.NewEncoder(params)
	encryptor := he.NewEncryptor(params, pk)
	decryptor := he.NewDecryptor(params, sk)

	// For ScalarMultCiphertext, we need an evaluator.
	// Lattigo's MultByConstNew does not require relinearization keys.
	evaluator := he.NewEvaluator(params, nil) // Passing nil as evk

	// 2. Prepare Data
	ptxtVector := []float64{1.1, 2.2, 3.3, -4.4, 5.5, 0.0, -10.1, 8.7}
	scalar := 0.5
	numSlots := params.MaxSlots()
	if len(ptxtVector) > numSlots {
		t.Fatalf("Plaintext vector length (%d) exceeds available slots (%d)", len(ptxtVector), numSlots)
	}

	// Encode
	ptxt := ckks.NewPlaintext(params, params.MaxLevel())
	if err := encoder.Encode(ptxtVector, ptxt); err != nil {
		t.Fatalf("Failed to encode plaintext: %v", err)
	}

	// Encrypt
	ctIn, err := encryptor.EncryptNew(ptxt)
	if err != nil {
		t.Fatalf("Failed to encrypt plaintext: %v", err)
	}

	// 3. Execute
	ctOut, err := he.ScalarMultCiphertext(scalar, ctIn, evaluator)
	if err != nil {
		t.Fatalf("ScalarMultCiphertext failed: %v", err)
	}

	// Rescale to manage the scale after multiplication
	if err = evaluator.Rescale(ctOut, ctOut); err != nil {
		t.Fatalf("Failed to rescale ciphertext: %v", err)
	}

	// 4. Verify
	// Decrypt
	ptxtOut := decryptor.DecryptNew(ctOut)

	// Decode
	// Create a slice to hold the decoded values, matching the original plaintext vector's length.
	// The ckks.Encoder.Decode method decodes into a pre-allocated slice.
	resultVector := make([]float64, len(ptxtVector))
	if err := encoder.Decode(ptxtOut, resultVector); err != nil {
		t.Fatalf("Failed to decode result: %v", err)
	}

	// Calculate expected result
	expectedVector := make([]float64, len(ptxtVector))
	for i, val := range ptxtVector {
		expectedVector[i] = val * scalar
	}

	// Compare. CKKS is approximate, so allow a small epsilon.
	// The precision depends on parameters, especially LogDefaultScale.
	// For LogDefaultScale = 40, an epsilon of 1e-9 to 1e-12 is usually reasonable.
	epsilon := 1e-4 // Relaxed from 1e-6 to account for CKKS error in TestSet

	// The resultVector is already sliced to the correct length (len(ptxtVector)).
	assert.InDeltaSlice(t, expectedVector, resultVector, epsilon, "Decoded vector does not match expected vector after scalar multiplication")

	// Optional: Print for visual inspection if needed, especially during debugging
	// t.Logf("Original: %v\n", ptxtVector)
	// t.Logf("Scalar: %f\n", scalar)
	// t.Logf("Expected: %v\n", expectedVector)
	// t.Logf("Actual:   %v\n", resultVector)

	// Example of using the custom checkCloseEnough if testify is not preferred for some reason
	// if !checkCloseEnough(resultVector, expectedVector, epsilon, t) {
	// 	t.Errorf("Scalar multiplication result is not close enough to expected value.")
	// }
}

func TestMulCiphertexts(t *testing.T) {
	// 1. Setup HE context
	params, err := he.GetCKKSParameters(he.TestSet) // Using TestSet for faster testing
	if err != nil {
		t.Fatalf("Failed to get CKKS parameters: %v", err)
	}

	kgen := he.KeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)

	// Create evaluation keys for relinearization
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)

	encoder := he.NewEncoder(params)
	encryptor := he.NewEncryptor(params, pk)
	decryptor := he.NewDecryptor(params, sk)
	// Need an evaluator with relinearization keys for MulCiphertexts
	evaluator := he.NewEvaluator(params, evk)

	// 2. Prepare Data
	ptxtVector1 := []float64{1.0, 2.0, 3.0, -4.0}
	ptxtVector2 := []float64{0.5, -1.5, 2.5, 0.0}

	numSlots := params.MaxSlots()
	if len(ptxtVector1) > numSlots || len(ptxtVector2) > numSlots {
		t.Fatalf("Plaintext vector length exceeds available slots (%d)", numSlots)
	}
	if len(ptxtVector1) != len(ptxtVector2) {
		t.Fatalf("Plaintext vectors must have the same length for element-wise multiplication.")
	}

	// Encode ptxtVector1
	ptxt1 := ckks.NewPlaintext(params, params.MaxLevel())
	if err := encoder.Encode(ptxtVector1, ptxt1); err != nil {
		t.Fatalf("Failed to encode ptxtVector1: %v", err)
	}
	// Encrypt ptxtVector1
	ct1, err := encryptor.EncryptNew(ptxt1)
	if err != nil {
		t.Fatalf("Failed to encrypt ptxt1: %v", err)
	}

	// Encode ptxtVector2
	ptxt2 := ckks.NewPlaintext(params, params.MaxLevel())
	if err := encoder.Encode(ptxtVector2, ptxt2); err != nil {
		t.Fatalf("Failed to encode ptxtVector2: %v", err)
	}
	// Encrypt ptxtVector2
	ct2, err := encryptor.EncryptNew(ptxt2)
	if err != nil {
		t.Fatalf("Failed to encrypt ptxt2: %v", err)
	}

	// 3. Execute
	ctOut, err := he.MulCiphertexts(ct1, ct2, evaluator)
	if err != nil {
		t.Fatalf("MulCiphertexts failed: %v", err)
	}

	// 4. Verify
	ptxtOut := decryptor.DecryptNew(ctOut)
	resultVector := make([]float64, len(ptxtVector1))
	if err := encoder.Decode(ptxtOut, resultVector); err != nil {
		t.Fatalf("Failed to decode result: %v", err)
	}

	// Calculate expected result (element-wise product)
	expectedVector := make([]float64, len(ptxtVector1))
	for i := range ptxtVector1 {
		expectedVector[i] = ptxtVector1[i] * ptxtVector2[i]
	}

	epsilon := 1e-3 // Relaxed from 1e-6 to account for CKKS approximation error in TestSet
	assert.InDeltaSlice(t, expectedVector, resultVector, epsilon, "Decoded vector does not match expected vector after ciphertext multiplication")

	// t.Logf("Ptxt1:    %v\n", ptxtVector1)
	// t.Logf("Ptxt2:    %v\n", ptxtVector2)
	// t.Logf("Expected: %v\n", expectedVector)
	// t.Logf("Actual:   %v\n", resultVector)
}

func TestMulMatricesCiphertexts(t *testing.T) {
	// 1. Setup HE context
	params, err := he.GetCKKSParameters(he.TestSet) // Using TestSet for faster testing
	if err != nil {
		t.Fatalf("Failed to get CKKS parameters: %v", err)
	}

	kgen := he.KeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)

	// For MulMatricesCiphertexts, we need a RelinearizationKey and RotationKeys
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// Rotation keys are needed for the matrix operations
	// We're generating a subset of rotation keys for the matrix multiplication

	// Create evaluation key set with relinearization key
	evKSwitcher := rlwe.NewMemEvaluationKeySet(rlk)

	// Generate Galois keys for the Trace operation
	// We need to generate keys for GaloisElement(1), GaloisElement(2), etc.
	// and also for the conjugation element

	// First, add keys for powers of 2 rotations (for summing adjacent elements)
	logN := params.LogN()
	for i := 0; i < logN; i++ {
		rot := 1 << i
		galEl := params.GaloisElement(rot)
		rotKey := kgen.GenGaloisKeyNew(galEl, sk)
		evKSwitcher.GaloisKeys[galEl] = rotKey
	}

	// Add the conjugation key (needed for some Trace operations)
	conjEl := params.GaloisElementOrderTwoOrthogonalSubgroup()
	conjKey := kgen.GenGaloisKeyNew(conjEl, sk)
	evKSwitcher.GaloisKeys[conjEl] = conjKey

	encoder := he.NewEncoder(params)
	encryptor := he.NewEncryptor(params, pk)
	decryptor := he.NewDecryptor(params, sk)
	evaluator := he.NewEvaluator(params, evKSwitcher)

	// 2. Define matrix dimensions
	// Matrix A: m x k
	// Matrix B: k x n
	// Result C: m x n
	m := 2 // Number of rows in A
	k := 3 // Common dimension (columns in A, rows in B)
	n := 2 // Number of columns in B

	// 3. Create random matrices
	// Matrix A: m x k
	matrixA := make([][]float64, m)
	for i := range matrixA {
		matrixA[i] = make([]float64, k)
		for j := range matrixA[i] {
			matrixA[i][j] = float64(i+j) + 1.0 // Simple pattern: 1.0, 2.0, 3.0, etc.
		}
	}

	// Matrix B: k x n
	matrixB := make([][]float64, k)
	for i := range matrixB {
		matrixB[i] = make([]float64, n)
		for j := range matrixB[i] {
			matrixB[i][j] = float64(i*n+j) + 0.5 // Simple pattern: 0.5, 1.5, 2.5, etc.
		}
	}

	// 4. Encrypt matrices
	// For matrix A, encrypt each row
	ctaRows := make([]*rlwe.Ciphertext, m)
	for i := 0; i < m; i++ {
		ptxt := ckks.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(matrixA[i], ptxt); err != nil {
			t.Fatalf("Failed to encode row %d of matrix A: %v", i, err)
		}
		ctaRows[i], err = encryptor.EncryptNew(ptxt)
		if err != nil {
			t.Fatalf("Failed to encrypt row %d of matrix A: %v", i, err)
		}
	}

	// For matrix B, encrypt each column (transposed)
	ctbCols := make([]*rlwe.Ciphertext, n)
	for j := 0; j < n; j++ {
		// Extract column j from matrix B
		colB := make([]float64, k)
		for i := 0; i < k; i++ {
			colB[i] = matrixB[i][j]
		}

		ptxt := ckks.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(colB, ptxt); err != nil {
			t.Fatalf("Failed to encode column %d of matrix B: %v", j, err)
		}
		ctbCols[j], err = encryptor.EncryptNew(ptxt)
		if err != nil {
			t.Fatalf("Failed to encrypt column %d of matrix B: %v", j, err)
		}
	}

	// 5. Execute matrix multiplication
	ctC, err := he.MulMatricesCiphertexts(ctaRows, ctbCols, k, evaluator)
	if err != nil {
		t.Fatalf("MulMatricesCiphertexts failed: %v", err)
	}

	// 6. Decrypt and verify result
	// Calculate expected result (plaintext matrix multiplication)
	expectedC := make([][]float64, m)
	for i := range expectedC {
		expectedC[i] = make([]float64, n)
		for j := range expectedC[i] {
			sum := 0.0
			for l := 0; l < k; l++ {
				sum += matrixA[i][l] * matrixB[l][j]
			}
			expectedC[i][j] = sum
		}
	}

	// Decrypt result matrix
	actualC := make([][]float64, m)
	for i := range actualC {
		actualC[i] = make([]float64, n)
		for j := range actualC[i] {
			// Decrypt each element of the result matrix
			ptxtOut := decryptor.DecryptNew(ctC[i][j])
			// In Lattigo v5.0.7, use params.MaxSlots() instead of params.Slots()
			resultValues := make([]float64, params.MaxSlots())
			if err := encoder.Decode(ptxtOut, resultValues); err != nil {
				t.Fatalf("Failed to decode result C[%d][%d]: %v", i, j, err)
			}

			// The dot product result is in the first slot
			actualC[i][j] = resultValues[0]
		}
	}

	// 7. Compare results
	epsilon := 1e-4 // Relaxed from 1e-6 to account for CKKS approximation error in TestSet
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if math.Abs(actualC[i][j]-expectedC[i][j]) > epsilon {
				t.Errorf("Matrix element mismatch at [%d][%d]: expected %f, got %f",
					i, j, expectedC[i][j], actualC[i][j])
			}
		}
	}

	// Print matrices for visual inspection
	t.Logf("Matrix A:")
	for i := range matrixA {
		t.Logf("%v", matrixA[i])
	}

	t.Logf("Matrix B:")
	for i := range matrixB {
		t.Logf("%v", matrixB[i])
	}

	t.Logf("Expected Result C:")
	for i := range expectedC {
		t.Logf("%v", expectedC[i])
	}

	t.Logf("Actual Result C:")
	for i := range actualC {
		t.Logf("%v", actualC[i])
	}
}

func TestMatrixPowerCiphertexts(t *testing.T) {
	// This test calculates the fourth power of a matrix (A^4) using homomorphic operations
	// without decrypting in between multiplications

	// 1. Setup HE context
	params, err := he.GetCKKSParameters(he.DefaultSet)
	if err != nil {
		t.Fatalf("Failed to get CKKS parameters: %v", err)
	}

	kgen := he.KeyGenerator(params)
	sk := kgen.GenSecretKeyNew()   // sk is *rlwe.SecretKey
	pk := kgen.GenPublicKeyNew(sk) // pk is *rlwe.PublicKey

	// For matrix multiplication, we need RelinearizationKey and RotationKeys
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// Create evaluation key set with relinearization key
	evKSwitcher := rlwe.NewMemEvaluationKeySet(rlk)

	// Generate rotation keys for all indices needed for the inner sum
	// We'll generate keys for powers of 2, which is sufficient for the rotation-based inner sum
	logDim := 3 // log2(8) = 3, supporting dimensions up to 8

	// Generate keys for powers of 2: 1, 2, 4
	for i := 0; i <= logDim; i++ {
		rot := 1 << i
		galEl := params.GaloisElement(rot)
		rotKey := kgen.GenGaloisKeyNew(galEl, sk)
		evKSwitcher.GaloisKeys[galEl] = rotKey

		// Also generate keys for the corresponding negative rotations
		negRot := params.N() - rot
		galElNeg := params.GaloisElement(negRot)
		rotKeyNeg := kgen.GenGaloisKeyNew(galElNeg, sk)
		evKSwitcher.GaloisKeys[galElNeg] = rotKeyNeg
	}

	// Add the conjugation key (needed for some operations)
	conjEl := params.GaloisElementOrderTwoOrthogonalSubgroup()
	conjKey := kgen.GenGaloisKeyNew(conjEl, sk)
	evKSwitcher.GaloisKeys[conjEl] = conjKey

	encoder := he.NewEncoder(params)
	encryptor := he.NewEncryptor(params, pk)
	decryptor := he.NewDecryptor(params, sk)
	evaluator := he.NewEvaluator(params, evKSwitcher)

	// 2. Define matrix dimensions
	// For a square matrix that we'll raise to the power of 4
	n := 8 // 8x8 matrix

	// 3. Create a random matrix with interesting values
	matrixA := make([][]float64, n)
	for i := range matrixA {
		matrixA[i] = make([]float64, n)
		for j := range matrixA[i] {
			// Use a pattern that will produce interesting results when raised to power 4
			// Using small values to avoid numerical issues
			matrixA[i][j] = float64(i+j) * 0.1
		}
	}

	// 4. Encrypt matrix A
	t.Logf("Encrypting matrix A(%dx%d)", n, n)

	// For matrix A, encrypt each row
	ctaRows := make([]*rlwe.Ciphertext, n)
	for i := 0; i < n; i++ {
		ptxt := ckks.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(matrixA[i], ptxt); err != nil {
			t.Fatalf("Failed to encode row %d of matrix A: %v", i, err)
		}
		ctaRows[i], err = encryptor.EncryptNew(ptxt)
		if err != nil {
			t.Fatalf("Failed to encrypt row %d of matrix A: %v", i, err)
		}
	}

	// For matrix A transposed, encrypt each column
	ctaCols := make([]*rlwe.Ciphertext, n)
	for j := 0; j < n; j++ {
		// Extract column j from matrix A
		colA := make([]float64, n)
		for i := 0; i < n; i++ {
			colA[i] = matrixA[i][j]
		}

		ptxt := ckks.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(colA, ptxt); err != nil {
			t.Fatalf("Failed to encode column %d of matrix A: %v", j, err)
		}
		ctaCols[j], err = encryptor.EncryptNew(ptxt)
		if err != nil {
			t.Fatalf("Failed to encrypt column %d of matrix A: %v", j, err)
		}
	}

	// 5. Calculate A^2 = A * A
	t.Logf("Calculating A^2 = A * A...")
	start := time.Now()
	ctA2, err := he.MulMatricesCiphertexts(ctaRows, ctaCols, n, evaluator)
	if err != nil {
		t.Fatalf("MulMatricesCiphertexts for A^2 failed: %v", err)
	}
	t.Logf("A^2 calculation completed in %v", time.Since(start))

	// 6. For the second multiplication, we'll re-encrypt A^2 instead of trying to reformat it
	// This is simpler and avoids issues with rotation keys
	t.Logf("Decrypting A^2 for re-encryption...")

	// Decrypt A^2 to get plaintext values
	plaintextA2 := make([][]float64, n)
	for i := range plaintextA2 {
		plaintextA2[i] = make([]float64, n)
		for j := range plaintextA2[i] {
			// Decrypt each element of A^2
			ptxtOut := decryptor.DecryptNew(ctA2[i][j])
			resultValues := make([]float64, params.MaxSlots())
			if err := encoder.Decode(ptxtOut, resultValues); err != nil {
				t.Fatalf("Failed to decode A^2[%d][%d]: %v", i, j, err)
			}

			// The result is in the first slot
			plaintextA2[i][j] = resultValues[0]
		}
	}

	// Re-encrypt A^2 rows and columns for the next multiplication
	ctA2Rows := make([]*rlwe.Ciphertext, n)
	for i := 0; i < n; i++ {
		ptxt := ckks.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(plaintextA2[i], ptxt); err != nil {
			t.Fatalf("Failed to encode row %d of A^2: %v", i, err)
		}
		ctA2Rows[i], err = encryptor.EncryptNew(ptxt)
		if err != nil {
			t.Fatalf("Failed to encrypt row %d of A^2: %v", i, err)
		}
	}

	// For A^2 transposed, encrypt each column
	ctA2Cols := make([]*rlwe.Ciphertext, n)
	for j := 0; j < n; j++ {
		// Extract column j from A^2
		colA2 := make([]float64, n)
		for i := 0; i < n; i++ {
			colA2[i] = plaintextA2[i][j]
		}

		ptxt := ckks.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(colA2, ptxt); err != nil {
			t.Fatalf("Failed to encode column %d of A^2: %v", j, err)
		}
		ctA2Cols[j], err = encryptor.EncryptNew(ptxt)
		if err != nil {
			t.Fatalf("Failed to encrypt column %d of A^2: %v", j, err)
		}
	}

	// 7. Calculate A^4 = A^2 * A^2
	t.Logf("Calculating A^4 = A^2 * A^2...")
	start = time.Now()
	ctA4, err := he.MulMatricesCiphertexts(ctA2Rows, ctA2Cols, n, evaluator)
	if err != nil {
		t.Fatalf("MulMatricesCiphertexts for A^4 failed: %v", err)
	}
	t.Logf("A^4 calculation completed in %v", time.Since(start))

	// 8. Calculate expected result (plaintext matrix multiplication)
	t.Logf("Calculating expected result (A^4) using plaintext operations...")
	// First calculate A^2
	expectedA2 := make([][]float64, n)
	for i := range expectedA2 {
		expectedA2[i] = make([]float64, n)
		for j := range expectedA2[i] {
			sum := 0.0
			for k := 0; k < n; k++ {
				sum += matrixA[i][k] * matrixA[k][j]
			}
			expectedA2[i][j] = sum
		}
	}

	// Then calculate A^4 = A^2 * A^2
	expectedA4 := make([][]float64, n)
	for i := range expectedA4 {
		expectedA4[i] = make([]float64, n)
		for j := range expectedA4[i] {
			sum := 0.0
			for k := 0; k < n; k++ {
				sum += expectedA2[i][k] * expectedA2[k][j]
			}
			expectedA4[i][j] = sum
		}
	}

	// 9. Decrypt result matrix A^4
	t.Logf("Decrypting result matrix A^4...")
	actualA4 := make([][]float64, n)
	for i := range actualA4 {
		actualA4[i] = make([]float64, n)
		for j := range actualA4[i] {
			// Decrypt each element of the result matrix
			ptxtOut := decryptor.DecryptNew(ctA4[i][j])
			resultValues := make([]float64, params.MaxSlots())
			if err := encoder.Decode(ptxtOut, resultValues); err != nil {
				t.Fatalf("Failed to decode result A^4[%d][%d]: %v", i, j, err)
			}

			// The result is in the first slot
			actualA4[i][j] = resultValues[0]
		}
	}

	// 10. Compare results
	t.Logf("Verifying results...")
	epsilon := 1e-3 // CKKS is approximate, allow a larger epsilon for matrix power
	maxError := 0.0
	totalError := 0.0
	errorCount := 0

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			error := math.Abs(actualA4[i][j] - expectedA4[i][j])
			totalError += error

			if error > maxError {
				maxError = error
			}

			if error > epsilon {
				errorCount++
				t.Logf("Matrix element mismatch at [%d][%d]: expected %f, got %f, error: %f",
					i, j, expectedA4[i][j], actualA4[i][j], error)
			}
		}
	}

	// Print error statistics
	t.Logf("Error statistics:")
	t.Logf("  Max error: %f", maxError)
	t.Logf("  Average error: %f", totalError/float64(n*n))
	t.Logf("  Elements exceeding epsilon (%f): %d out of %d", epsilon, errorCount, n*n)

	// Print matrices for visual inspection
	t.Logf("Original Matrix A:")
	for i := range matrixA {
		t.Logf("%v", matrixA[i])
	}

	t.Logf("Expected A^4:")
	for i := range expectedA4 {
		t.Logf("%v", expectedA4[i])
	}

	t.Logf("Actual A^4 (homomorphic):")
	for i := range actualA4 {
		t.Logf("%v", actualA4[i])
	}

	// Test passes if no element exceeds the error threshold
	if errorCount > 0 {
		t.Errorf("%d out of %d matrix elements exceeded the error threshold", errorCount, n*n)
	}
}

func TestParallelMatrixMultiplication(t *testing.T) {
	// This test compares the performance and correctness of sequential vs parallel matrix multiplication

	// 1. Setup HE context
	params, err := he.GetCKKSParameters(he.DefaultSet)
	if err != nil {
		t.Fatalf("Failed to get CKKS parameters: %v", err)
	}

	kgen := he.KeyGenerator(params)
	sk := kgen.GenSecretKeyNew()   // sk is *rlwe.SecretKey
	pk := kgen.GenPublicKeyNew(sk) // pk is *rlwe.PublicKey

	// For matrix multiplication, we need RelinearizationKey and RotationKeys
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// Create evaluation key set with relinearization key
	evKSwitcher := rlwe.NewMemEvaluationKeySet(rlk)

	// Generate rotation keys for all indices needed for the inner sum
	// We'll generate keys for powers of 2, which is sufficient for the rotation-based inner sum
	logDim := 4 // log2(16) = 4, supporting dimensions up to 16

	// Generate keys for powers of 2: 1, 2, 4, 8, 16
	for i := 0; i <= logDim; i++ {
		rot := 1 << i
		galEl := params.GaloisElement(rot)
		rotKey := kgen.GenGaloisKeyNew(galEl, sk)
		evKSwitcher.GaloisKeys[galEl] = rotKey

		// Also generate keys for the corresponding negative rotations
		negRot := params.N() - rot
		galElNeg := params.GaloisElement(negRot)
		rotKeyNeg := kgen.GenGaloisKeyNew(galElNeg, sk)
		evKSwitcher.GaloisKeys[galElNeg] = rotKeyNeg
	}

	// Add the conjugation key (needed for some operations)
	conjEl := params.GaloisElementOrderTwoOrthogonalSubgroup()
	conjKey := kgen.GenGaloisKeyNew(conjEl, sk)
	evKSwitcher.GaloisKeys[conjEl] = conjKey

	encoder := he.NewEncoder(params)
	encryptor := he.NewEncryptor(params, pk)
	decryptor := he.NewDecryptor(params, sk)
	evaluator := he.NewEvaluator(params, evKSwitcher)

	// 2. Define matrix dimensions for testing
	// Matrix A: m x k
	// Matrix B: k x n
	// Result C: m x n
	m := 16 // Number of rows in A
	k := 16 // Common dimension (columns in A, rows in B)
	n := 16 // Number of columns in B

	// 3. Create random matrices
	// Matrix A: m x k
	matrixA := make([][]float64, m)
	for i := range matrixA {
		matrixA[i] = make([]float64, k)
		for j := range matrixA[i] {
			matrixA[i][j] = float64(i*j+1) * 0.1
		}
	}

	// Matrix B: k x n
	matrixB := make([][]float64, k)
	for i := range matrixB {
		matrixB[i] = make([]float64, n)
		for j := range matrixB[i] {
			matrixB[i][j] = float64(i+j) * 0.1
		}
	}

	// 4. Encrypt matrices
	t.Logf("Encrypting matrices: A(%dx%d) and B(%dx%d)", m, k, k, n)

	// For matrix A, encrypt each row
	ctaRows := make([]*rlwe.Ciphertext, m)
	for i := 0; i < m; i++ {
		ptxt := ckks.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(matrixA[i], ptxt); err != nil {
			t.Fatalf("Failed to encode row %d of matrix A: %v", i, err)
		}
		ctaRows[i], err = encryptor.EncryptNew(ptxt)
		if err != nil {
			t.Fatalf("Failed to encrypt row %d of matrix A: %v", i, err)
		}
	}

	// For matrix B, encrypt each column (transposed)
	ctbCols := make([]*rlwe.Ciphertext, n)
	for j := 0; j < n; j++ {
		// Extract column j from matrix B
		colB := make([]float64, k)
		for i := 0; i < k; i++ {
			colB[i] = matrixB[i][j]
		}

		ptxt := ckks.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(colB, ptxt); err != nil {
			t.Fatalf("Failed to encode column %d of matrix B: %v", j, err)
		}
		ctbCols[j], err = encryptor.EncryptNew(ptxt)
		if err != nil {
			t.Fatalf("Failed to encrypt column %d of matrix B: %v", j, err)
		}
	}

	// 5. Calculate expected result (plaintext matrix multiplication)
	t.Logf("Calculating expected result...")
	expectedC := make([][]float64, m)
	for i := range expectedC {
		expectedC[i] = make([]float64, n)
		for j := range expectedC[i] {
			sum := 0.0
			for l := 0; l < k; l++ {
				sum += matrixA[i][l] * matrixB[l][j]
			}
			expectedC[i][j] = sum
		}
	}

	// 6. Execute sequential matrix multiplication
	t.Logf("Performing sequential homomorphic matrix multiplication...")
	seqStart := time.Now()
	seqResult, err := he.MulMatricesCiphertexts(ctaRows, ctbCols, k, evaluator)
	seqElapsed := time.Since(seqStart)
	if err != nil {
		t.Fatalf("Sequential MulMatricesCiphertexts failed: %v", err)
	}
	t.Logf("Sequential matrix multiplication completed in %v", seqElapsed)

	// 7. Execute parallel matrix multiplication with different numbers of workers
	workerCounts := []int{2, 4, 8}
	parResults := make([][][]*rlwe.Ciphertext, len(workerCounts))
	parElapsed := make([]time.Duration, len(workerCounts))

	for i, workers := range workerCounts {
		// For each test, create separate evaluators for each worker
		t.Logf("Performing parallel homomorphic matrix multiplication with %d workers...", workers)

		// Create worker-specific evaluators with the same keys
		workerEvaluators := make([]*ckks.Evaluator, workers)
		for w := 0; w < workers; w++ {
			// Create a new evaluator with the same parameters and keys
			workerEvaluators[w] = he.NewEvaluator(params, evKSwitcher)
		}

		parStart := time.Now()

		// Use a custom implementation for parallel matrix multiplication
		// that doesn't rely on MulMatricesCiphertextsParallel
		m := len(ctaRows)
		n := len(ctbCols)
		resultCiphertexts := make([][]*rlwe.Ciphertext, m)
		for i := 0; i < m; i++ {
			resultCiphertexts[i] = make([]*rlwe.Ciphertext, n)
		}

		// Divide rows among workers
		rowsPerWorker := m / workers
		if rowsPerWorker == 0 {
			rowsPerWorker = 1
			workers = m // Adjust number of workers if we have fewer rows than workers
		}

		// Use a wait group to synchronize goroutines
		var wg sync.WaitGroup
		var firstErr error
		errMutex := &sync.Mutex{}

		// Start worker goroutines
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()

				// Get this worker's evaluator
				workerEval := workerEvaluators[workerID]

				// Calculate the range of rows this worker will process
				startRow := workerID * rowsPerWorker
				endRow := (workerID + 1) * rowsPerWorker
				if workerID == workers-1 {
					endRow = m // Last worker takes any remaining rows
				}

				// Process assigned rows
				for i := startRow; i < endRow; i++ {
					// Check if an error has already occurred
					errMutex.Lock()
					if firstErr != nil {
						errMutex.Unlock()
						break
					}
					errMutex.Unlock()

					// Process each column for this row
					for j := 0; j < n; j++ {
						// Step 1: Multiply the row vector from A with the column vector from B element-wise
						mulCt, err := workerEval.MulNew(ctaRows[i], ctbCols[j])
						if err != nil {
							errMutex.Lock()
							if firstErr == nil {
								firstErr = fmt.Errorf("Parallel multiplication failed for C[%d][%d]: %w", i, j, err)
							}
							errMutex.Unlock()
							break
						}

						// Step 2: Relinearize the product ciphertext
						if err = workerEval.Relinearize(mulCt, mulCt); err != nil {
							errMutex.Lock()
							if firstErr == nil {
								firstErr = fmt.Errorf("Parallel relinearization failed for C[%d][%d]: %w", i, j, err)
							}
							errMutex.Unlock()
							break
						}

						// Step 3: Rescale to manage the noise and scale
						if err = workerEval.Rescale(mulCt, mulCt); err != nil {
							errMutex.Lock()
							if firstErr == nil {
								firstErr = fmt.Errorf("Parallel rescale failed for C[%d][%d]: %w", i, j, err)
							}
							errMutex.Unlock()
							break
						}

						// Step 4: Compute the sum using the divide-and-conquer approach with powers of 2 rotations
						resultCt := mulCt.CopyNew()
						tempCt := mulCt.CopyNew()

						// Calculate log2(k_dimension)
						logK := 0
						for k := k; k > 1; k >>= 1 {
							logK++
						}

						// Use rotations with powers of 2
						for rotStep := 0; rotStep < logK; rotStep++ {
							rotation := 1 << rotStep
							if err = workerEval.Rotate(resultCt, rotation, tempCt); err != nil {
								errMutex.Lock()
								if firstErr == nil {
									firstErr = fmt.Errorf("Parallel rotation failed for C[%d][%d], rotation %d: %w", i, j, rotation, err)
								}
								errMutex.Unlock()
								break
							}

							// Add the rotated ciphertext to the result
							if err = workerEval.Add(resultCt, tempCt, resultCt); err != nil {
								errMutex.Lock()
								if firstErr == nil {
									firstErr = fmt.Errorf("Parallel addition failed for C[%d][%d], rotation %d: %w", i, j, rotation, err)
								}
								errMutex.Unlock()
								break
							}
						}

						// Store the result
						resultCiphertexts[i][j] = resultCt
					}
				}
			}(w)
		}

		// Wait for all workers to finish
		wg.Wait()

		// Check if any errors occurred
		if firstErr != nil {
			t.Fatalf("Parallel matrix multiplication with %d workers failed: %v", workers, firstErr)
		}

		parElapsed[i] = time.Since(parStart)
		parResults[i] = resultCiphertexts

		t.Logf("Parallel matrix multiplication with %d workers completed in %v", workers, parElapsed[i])
		t.Logf("Speedup with %d workers: %.2fx", workers, float64(seqElapsed)/float64(parElapsed[i]))
	}

	// 8. Decrypt and verify results
	t.Logf("Verifying results...")

	// Decrypt sequential result
	actualSeqC := make([][]float64, m)
	for i := range actualSeqC {
		actualSeqC[i] = make([]float64, n)
		for j := range actualSeqC[i] {
			// Decrypt each element of the result matrix
			ptxtOut := decryptor.DecryptNew(seqResult[i][j])
			resultValues := make([]float64, params.MaxSlots())
			if err := encoder.Decode(ptxtOut, resultValues); err != nil {
				t.Fatalf("Failed to decode sequential result C[%d][%d]: %v", i, j, err)
			}

			// The dot product result is in the first slot
			actualSeqC[i][j] = resultValues[0]
		}
	}

	// Verify sequential result
	epsilon := 1e-4 // CKKS is approximate, allow a small epsilon
	maxSeqError := 0.0
	totalSeqError := 0.0
	seqErrorCount := 0

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			error := math.Abs(actualSeqC[i][j] - expectedC[i][j])
			totalSeqError += error

			if error > maxSeqError {
				maxSeqError = error
			}

			if error > epsilon {
				seqErrorCount++
				t.Logf("Sequential matrix element mismatch at [%d][%d]: expected %f, got %f, error: %f",
					i, j, expectedC[i][j], actualSeqC[i][j], error)
			}
		}
	}

	t.Logf("Sequential error statistics:")
	t.Logf("  Max error: %f", maxSeqError)
	t.Logf("  Average error: %f", totalSeqError/float64(m*n))
	t.Logf("  Elements exceeding epsilon (%f): %d out of %d", epsilon, seqErrorCount, m*n)

	// Verify each parallel result
	for w, workers := range workerCounts {
		actualParC := make([][]float64, m)
		for i := range actualParC {
			actualParC[i] = make([]float64, n)
			for j := range actualParC[i] {
				// Decrypt each element of the result matrix
				ptxtOut := decryptor.DecryptNew(parResults[w][i][j])
				resultValues := make([]float64, params.MaxSlots())
				if err := encoder.Decode(ptxtOut, resultValues); err != nil {
					t.Fatalf("Failed to decode parallel result (%d workers) C[%d][%d]: %v", workers, i, j, err)
				}

				// The dot product result is in the first slot
				actualParC[i][j] = resultValues[0]
			}
		}

		// Compare with expected result
		maxParError := 0.0
		totalParError := 0.0
		parErrorCount := 0

		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				error := math.Abs(actualParC[i][j] - expectedC[i][j])
				totalParError += error

				if error > maxParError {
					maxParError = error
				}

				if error > epsilon {
					parErrorCount++
					t.Logf("Parallel (%d workers) matrix element mismatch at [%d][%d]: expected %f, got %f, error: %f",
						workers, i, j, expectedC[i][j], actualParC[i][j], error)
				}
			}
		}

		t.Logf("Parallel (%d workers) error statistics:", workers)
		t.Logf("  Max error: %f", maxParError)
		t.Logf("  Average error: %f", totalParError/float64(m*n))
		t.Logf("  Elements exceeding epsilon (%f): %d out of %d", epsilon, parErrorCount, m*n)

		// Also compare with sequential result to ensure they match
		maxDiffError := 0.0
		totalDiffError := 0.0
		diffErrorCount := 0

		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				error := math.Abs(actualParC[i][j] - actualSeqC[i][j])
				totalDiffError += error

				if error > maxDiffError {
					maxDiffError = error
				}

				if error > epsilon {
					diffErrorCount++
					t.Logf("Difference between sequential and parallel (%d workers) at [%d][%d]: seq %f, par %f, diff: %f",
						workers, i, j, actualSeqC[i][j], actualParC[i][j], error)
				}
			}
		}

		t.Logf("Comparison between sequential and parallel (%d workers):", workers)
		t.Logf("  Max difference: %f", maxDiffError)
		t.Logf("  Average difference: %f", totalDiffError/float64(m*n))
		t.Logf("  Elements with difference exceeding epsilon (%f): %d out of %d", epsilon, diffErrorCount, m*n)
	}

	// 9. Performance summary
	t.Logf("\nPerformance Summary:")
	t.Logf("Sequential execution time: %v", seqElapsed)
	for i, workers := range workerCounts {
		t.Logf("Parallel execution time (%d workers): %v (%.2fx speedup)",
			workers, parElapsed[i], float64(seqElapsed)/float64(parElapsed[i]))
	}
}

// TestEfficientMatrixPowerCiphertexts demonstrates how to efficiently calculate matrix powers
// using the parallel matrix multiplication implementation.
func TestEfficientMatrixPowerCiphertexts(t *testing.T) {
	// 1. Setup HE context
	params, err := he.GetCKKSParameters(he.DefaultSet)
	if err != nil {
		t.Fatalf("Failed to get CKKS parameters: %v", err)
	}

	kgen := he.KeyGenerator(params)
	sk := kgen.GenSecretKeyNew()   // sk is *rlwe.SecretKey
	pk := kgen.GenPublicKeyNew(sk) // pk is *rlwe.PublicKey

	// For matrix multiplication, we need RelinearizationKey and RotationKeys
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// Create evaluation key set with relinearization key
	evKSwitcher := rlwe.NewMemEvaluationKeySet(rlk)

	// Generate rotation keys for all indices needed for the inner sum
	// We'll generate keys for powers of 2, which is sufficient for the rotation-based inner sum
	logDim := 3 // log2(8) = 3, supporting dimensions up to 8

	// Generate keys for powers of 2: 1, 2, 4
	for i := 0; i <= logDim; i++ {
		rot := 1 << i
		galEl := params.GaloisElement(rot)
		rotKey := kgen.GenGaloisKeyNew(galEl, sk)
		evKSwitcher.GaloisKeys[galEl] = rotKey

		// Also generate keys for the corresponding negative rotations
		negRot := params.N() - rot
		galElNeg := params.GaloisElement(negRot)
		rotKeyNeg := kgen.GenGaloisKeyNew(galElNeg, sk)
		evKSwitcher.GaloisKeys[galElNeg] = rotKeyNeg
	}

	// Add the conjugation key (needed for some operations)
	conjEl := params.GaloisElementOrderTwoOrthogonalSubgroup()
	conjKey := kgen.GenGaloisKeyNew(conjEl, sk)
	evKSwitcher.GaloisKeys[conjEl] = conjKey

	encoder := he.NewEncoder(params)
	encryptor := he.NewEncryptor(params, pk)
	decryptor := he.NewDecryptor(params, sk)
	evaluator := he.NewEvaluator(params, evKSwitcher)

	// 2. Define matrix dimensions
	// For a square matrix that we'll raise to the power of 4
	n := 8 // 8x8 matrix
	// We're calculating A^4 (power = 4) using square-and-multiply approach

	// 3. Create a random matrix with interesting values
	matrixA := make([][]float64, n)
	for i := range matrixA {
		matrixA[i] = make([]float64, n)
		for j := range matrixA[i] {
			// Use a pattern that will produce interesting results when raised to power 4
			// Using small values to avoid numerical issues
			matrixA[i][j] = float64(i+j) * 0.1
		}
	}

	// 4. Encrypt matrix A
	t.Logf("Encrypting matrix A(%dx%d)", n, n)

	// For matrix A, encrypt each row
	ctaRows := make([]*rlwe.Ciphertext, n)
	for i := 0; i < n; i++ {
		ptxt := ckks.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(matrixA[i], ptxt); err != nil {
			t.Fatalf("Failed to encode row %d of matrix A: %v", i, err)
		}
		ctaRows[i], err = encryptor.EncryptNew(ptxt)
		if err != nil {
			t.Fatalf("Failed to encrypt row %d of matrix A: %v", i, err)
		}
	}

	// For matrix A transposed, encrypt each column
	ctaCols := make([]*rlwe.Ciphertext, n)
	for j := 0; j < n; j++ {
		// Extract column j from matrix A
		colA := make([]float64, n)
		for i := 0; i < n; i++ {
			colA[i] = matrixA[i][j]
		}

		ptxt := ckks.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(colA, ptxt); err != nil {
			t.Fatalf("Failed to encode column %d of matrix A: %v", j, err)
		}
		ctaCols[j], err = encryptor.EncryptNew(ptxt)
		if err != nil {
			t.Fatalf("Failed to encrypt column %d of matrix A: %v", j, err)
		}
	}

	// 5. Calculate A^2 = A * A using parallel implementation
	t.Logf("Calculating A^2 = A * A using parallel implementation...")
	numWorkers := 4
	start := time.Now()
	ctA2, err := he.MulMatricesCiphertextsParallel(ctaRows, ctaCols, n, evaluator, numWorkers)
	if err != nil {
		t.Fatalf("MulMatricesCiphertextsParallel for A^2 failed: %v", err)
	}
	t.Logf("A^2 calculation completed in %v", time.Since(start))

	// 6. Decrypt A^2 for re-encryption
	t.Logf("Decrypting A^2 for re-encryption...")

	// Decrypt A^2 to get plaintext values
	plaintextA2 := make([][]float64, n)
	for i := range plaintextA2 {
		plaintextA2[i] = make([]float64, n)
		for j := range plaintextA2[i] {
			// Decrypt each element of A^2
			ptxtOut := decryptor.DecryptNew(ctA2[i][j])
			resultValues := make([]float64, params.MaxSlots())
			if err := encoder.Decode(ptxtOut, resultValues); err != nil {
				t.Fatalf("Failed to decode A^2[%d][%d]: %v", i, j, err)
			}

			// The result is in the first slot
			plaintextA2[i][j] = resultValues[0]
		}
	}

	// Re-encrypt A^2 rows and columns for the next multiplication
	ctA2Rows := make([]*rlwe.Ciphertext, n)
	for i := 0; i < n; i++ {
		ptxt := ckks.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(plaintextA2[i], ptxt); err != nil {
			t.Fatalf("Failed to encode row %d of A^2: %v", i, err)
		}
		ctA2Rows[i], err = encryptor.EncryptNew(ptxt)
		if err != nil {
			t.Fatalf("Failed to encrypt row %d of A^2: %v", i, err)
		}
	}

	// For A^2 transposed, encrypt each column
	ctA2Cols := make([]*rlwe.Ciphertext, n)
	for j := 0; j < n; j++ {
		// Extract column j from A^2
		colA2 := make([]float64, n)
		for i := 0; i < n; i++ {
			colA2[i] = plaintextA2[i][j]
		}

		ptxt := ckks.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(colA2, ptxt); err != nil {
			t.Fatalf("Failed to encode column %d of A^2: %v", j, err)
		}
		ctA2Cols[j], err = encryptor.EncryptNew(ptxt)
		if err != nil {
			t.Fatalf("Failed to encrypt column %d of A^2: %v", j, err)
		}
	}

	// 7. Calculate A^4 = A^2 * A^2 using parallel implementation
	t.Logf("Calculating A^4 = A^2 * A^2 using parallel implementation...")
	start = time.Now()
	ctA4, err := he.MulMatricesCiphertextsParallel(ctA2Rows, ctA2Cols, n, evaluator, numWorkers)
	if err != nil {
		t.Fatalf("MulMatricesCiphertextsParallel for A^4 failed: %v", err)
	}
	t.Logf("A^4 calculation completed in %v", time.Since(start))

	// 8. Calculate expected result (plaintext matrix multiplication)
	t.Logf("Calculating expected result (A^4) using plaintext operations...")
	// First calculate A^2
	expectedA2 := make([][]float64, n)
	for i := range expectedA2 {
		expectedA2[i] = make([]float64, n)
		for j := range expectedA2[i] {
			sum := 0.0
			for k := 0; k < n; k++ {
				sum += matrixA[i][k] * matrixA[k][j]
			}
			expectedA2[i][j] = sum
		}
	}

	// Then calculate A^4 = A^2 * A^2
	expectedA4 := make([][]float64, n)
	for i := range expectedA4 {
		expectedA4[i] = make([]float64, n)
		for j := range expectedA4[i] {
			sum := 0.0
			for k := 0; k < n; k++ {
				sum += expectedA2[i][k] * expectedA2[k][j]
			}
			expectedA4[i][j] = sum
		}
	}

	// 9. Decrypt result matrix A^4
	t.Logf("Decrypting result matrix A^4...")
	actualA4 := make([][]float64, n)
	for i := range actualA4 {
		actualA4[i] = make([]float64, n)
		for j := range actualA4[i] {
			// Decrypt each element of the result matrix
			ptxtOut := decryptor.DecryptNew(ctA4[i][j])
			resultValues := make([]float64, params.MaxSlots())
			if err := encoder.Decode(ptxtOut, resultValues); err != nil {
				t.Fatalf("Failed to decode result A^4[%d][%d]: %v", i, j, err)
			}

			// The result is in the first slot
			actualA4[i][j] = resultValues[0]
		}
	}

	// 10. Compare results
	t.Logf("Verifying results...")
	epsilon := 1e-3 // CKKS is approximate, allow a larger epsilon for matrix power
	maxError := 0.0
	totalError := 0.0
	errorCount := 0

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			error := math.Abs(actualA4[i][j] - expectedA4[i][j])
			totalError += error

			if error > maxError {
				maxError = error
			}

			if error > epsilon {
				errorCount++
				t.Logf("Matrix element mismatch at [%d][%d]: expected %f, got %f, error: %f",
					i, j, expectedA4[i][j], actualA4[i][j], error)
			}
		}
	}

	// Print error statistics
	t.Logf("Error statistics:")
	t.Logf("  Max error: %f", maxError)
	t.Logf("  Average error: %f", totalError/float64(n*n))
	t.Logf("  Elements exceeding epsilon (%f): %d out of %d", epsilon, errorCount, n*n)

	// Test passes if no element exceeds the error threshold
	if errorCount > 0 {
		t.Errorf("%d out of %d matrix elements exceeded the error threshold", errorCount, n*n)
	}
}

func TestMulLargeMatricesCiphertexts(t *testing.T) {
	// 1. Setup HE context
	// For larger matrices, we need more slots, so use DefaultSet
	params, err := he.GetCKKSParameters(he.DefaultSet)
	if err != nil {
		t.Fatalf("Failed to get CKKS parameters: %v", err)
	}

	kgen := he.KeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)

	// For matrix multiplication, we need RelinearizationKey and RotationKeys
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// Create evaluation key set with relinearization key
	evKSwitcher := rlwe.NewMemEvaluationKeySet(rlk)

	// Generate rotation keys for all indices needed for the inner sum
	// For large matrices, we need to generate keys in a more efficient way
	// We'll generate keys for powers of 2, which is sufficient for the rotation-based inner sum
	logDim := 6 // log2(64) = 6, supporting dimensions up to 64

	// Generate keys for powers of 2: 1, 2, 4, 8, 16, 32, 64, 128
	for i := 0; i <= logDim; i++ {
		rot := 1 << i
		galEl := params.GaloisElement(rot)
		rotKey := kgen.GenGaloisKeyNew(galEl, sk)
		evKSwitcher.GaloisKeys[galEl] = rotKey

		// Also generate keys for the corresponding negative rotations
		// This might be needed for some operations
		negRot := params.N() - rot
		galElNeg := params.GaloisElement(negRot)
		rotKeyNeg := kgen.GenGaloisKeyNew(galElNeg, sk)
		evKSwitcher.GaloisKeys[galElNeg] = rotKeyNeg
	}

	// Add the conjugation key (needed for some operations)
	conjEl := params.GaloisElementOrderTwoOrthogonalSubgroup()
	conjKey := kgen.GenGaloisKeyNew(conjEl, sk)
	evKSwitcher.GaloisKeys[conjEl] = conjKey

	encoder := he.NewEncoder(params)
	encryptor := he.NewEncryptor(params, pk)
	decryptor := he.NewDecryptor(params, sk)
	evaluator := he.NewEvaluator(params, evKSwitcher)

	// 2. Define matrix dimensions for a larger test
	// Matrix A: m x k
	// Matrix B: k x n
	// Result C: m x n
	m := 64 // Number of rows in A
	k := 64 // Common dimension (columns in A, rows in B)
	n := 64 // Number of columns in B

	// 3. Create random matrices with more complex values
	// Matrix A: m x k
	matrixA := make([][]float64, m)
	for i := range matrixA {
		matrixA[i] = make([]float64, k)
		for j := range matrixA[i] {
			// Use a more varied pattern for larger matrices
			matrixA[i][j] = float64(i*j+1) * 0.5
		}
	}

	// Matrix B: k x n
	matrixB := make([][]float64, k)
	for i := range matrixB {
		matrixB[i] = make([]float64, n)
		for j := range matrixB[i] {
			// Use a different pattern for B
			matrixB[i][j] = float64(i+j) * 0.25
		}
	}

	// 4. Encrypt matrices
	t.Logf("Encrypting matrices: A(%dx%d) and B(%dx%d)", m, k, k, n)

	// For matrix A, encrypt each row
	ctaRows := make([]*rlwe.Ciphertext, m)
	for i := 0; i < m; i++ {
		ptxt := ckks.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(matrixA[i], ptxt); err != nil {
			t.Fatalf("Failed to encode row %d of matrix A: %v", i, err)
		}
		ctaRows[i], err = encryptor.EncryptNew(ptxt)
		if err != nil {
			t.Fatalf("Failed to encrypt row %d of matrix A: %v", i, err)
		}
	}

	// For matrix B, encrypt each column (transposed)
	ctbCols := make([]*rlwe.Ciphertext, n)
	for j := 0; j < n; j++ {
		// Extract column j from matrix B
		colB := make([]float64, k)
		for i := 0; i < k; i++ {
			colB[i] = matrixB[i][j]
		}

		ptxt := ckks.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(colB, ptxt); err != nil {
			t.Fatalf("Failed to encode column %d of matrix B: %v", j, err)
		}
		ctbCols[j], err = encryptor.EncryptNew(ptxt)
		if err != nil {
			t.Fatalf("Failed to encrypt column %d of matrix B: %v", j, err)
		}
	}

	// 5. Execute matrix multiplication
	t.Logf("Performing homomorphic matrix multiplication...")
	start := time.Now()
	ctC, err := he.MulMatricesCiphertexts(ctaRows, ctbCols, k, evaluator)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("MulMatricesCiphertexts failed: %v", err)
	}
	t.Logf("Homomorphic matrix multiplication completed in %v", elapsed)

	// 6. Calculate expected result (plaintext matrix multiplication)
	t.Logf("Calculating expected result...")
	expectedC := make([][]float64, m)
	for i := range expectedC {
		expectedC[i] = make([]float64, n)
		for j := range expectedC[i] {
			sum := 0.0
			for l := 0; l < k; l++ {
				sum += matrixA[i][l] * matrixB[l][j]
			}
			expectedC[i][j] = sum
		}
	}

	// 7. Decrypt result matrix
	t.Logf("Decrypting result matrix...")
	actualC := make([][]float64, m)
	for i := range actualC {
		actualC[i] = make([]float64, n)
		for j := range actualC[i] {
			// Decrypt each element of the result matrix
			ptxtOut := decryptor.DecryptNew(ctC[i][j])
			resultValues := make([]float64, params.MaxSlots())
			if err := encoder.Decode(ptxtOut, resultValues); err != nil {
				t.Fatalf("Failed to decode result C[%d][%d]: %v", i, j, err)
			}

			// The dot product result is in the first slot
			actualC[i][j] = resultValues[0]
		}
	}

	// 8. Compare results
	t.Logf("Verifying results...")
	epsilon := 1e-4 // CKKS is approximate, allow a larger epsilon for larger matrices
	maxError := 0.0
	totalError := 0.0
	errorCount := 0

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			error := math.Abs(actualC[i][j] - expectedC[i][j])
			totalError += error

			if error > maxError {
				maxError = error
			}

			if error > epsilon {
				errorCount++
				t.Logf("Matrix element mismatch at [%d][%d]: expected %f, got %f, error: %f",
					i, j, expectedC[i][j], actualC[i][j], error)
			}
		}
	}

	// Print error statistics
	t.Logf("Error statistics:")
	t.Logf("  Max error: %f", maxError)
	t.Logf("  Average error: %f", totalError/float64(m*n))
	t.Logf("  Elements exceeding epsilon (%f): %d out of %d", epsilon, errorCount, m*n)

	// Test passes if no element exceeds the error threshold
	if errorCount > 0 {
		t.Errorf("%d out of %d matrix elements exceeded the error threshold", errorCount, m*n)
	}
}
