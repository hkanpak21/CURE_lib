package activation_he

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"

	"cure_test/pkg/he"
)

// Test 1: Plain polynomial accuracy
func TestChebyshevPlainAccuracy(t *testing.T) {
	const numSamples = 1000
	const tolerance = 2e-3

	// Test data: random values in [-1, 1]
	testData := make([]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		testData[i] = 2*rand.Float64() - 1 // Random value in [-1, 1]
	}

	// Test ReLU degree 3
	t.Run("ReLU_Degree3", func(t *testing.T) {
		mse := 0.0
		for _, x := range testData {
			approx := EvalReLUPlain3(x)
			reference := ReLUReference(x)
			diff := approx - reference
			mse += diff * diff
		}
		mse /= float64(numSamples)
		t.Logf("ReLU degree 3 MSE: %e", mse)
		assert.Less(t, mse, tolerance, "ReLU degree 3 MSE should be less than %e", tolerance)
	})

	// Test ReLU degree 5
	t.Run("ReLU_Degree5", func(t *testing.T) {
		mse := 0.0
		for _, x := range testData {
			approx := EvalReLUPlain5(x)
			reference := ReLUReference(x)
			diff := approx - reference
			mse += diff * diff
		}
		mse /= float64(numSamples)
		t.Logf("ReLU degree 5 MSE: %e", mse)
		assert.Less(t, mse, tolerance, "ReLU degree 5 MSE should be less than %e", tolerance)
	})

	// Test Sigmoid degree 3
	t.Run("Sigmoid_Degree3", func(t *testing.T) {
		mse := 0.0
		for _, x := range testData {
			approx := EvalSigmoidPlain3(x)
			reference := SigmoidReference(x)
			diff := approx - reference
			mse += diff * diff
		}
		mse /= float64(numSamples)
		t.Logf("Sigmoid degree 3 MSE: %e", mse)
		assert.Less(t, mse, tolerance, "Sigmoid degree 3 MSE should be less than %e", tolerance)
	})

	// Test Sigmoid degree 5
	t.Run("Sigmoid_Degree5", func(t *testing.T) {
		mse := 0.0
		for _, x := range testData {
			approx := EvalSigmoidPlain5(x)
			reference := SigmoidReference(x)
			diff := approx - reference
			mse += diff * diff
		}
		mse /= float64(numSamples)
		t.Logf("Sigmoid degree 5 MSE: %e", mse)
		assert.Less(t, mse, tolerance, "Sigmoid degree 5 MSE should be less than %e", tolerance)
	})
}

// Test 2: Ciphertext functional test
func TestChebyshevHEAccuracy(t *testing.T) {
	// Setup CKKS parameters using the existing parameter sets
	params, err := he.GetCKKSParameters(he.TestSet)
	assert.NoError(t, err, "Failed to create CKKS parameters")

	// Key generation
	kgen := he.KeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)

	// Create encoder, encryptor, decryptor, evaluator
	encoder := he.NewEncoder(params)
	encryptor := he.NewEncryptor(params, pk)
	decryptor := he.NewDecryptor(params, sk)
	evaluator := he.NewEvaluator(params, evk)

	// Test data: 32 random values in [-1, 1]
	const numSlots = 32
	testData := make([]float64, numSlots)
	for i := 0; i < numSlots; i++ {
		testData[i] = 2*rand.Float64() - 1
	}

	// Encrypt test data
	plaintext := ckks.NewPlaintext(params, params.MaxLevel())
	err = encoder.Encode(testData, plaintext)
	assert.NoError(t, err, "Failed to encode test data")

	ciphertext, err := encryptor.EncryptNew(plaintext)
	assert.NoError(t, err, "Failed to encrypt test data")

	const tolerance = 2e-3

	// Test ReLU degree 3
	t.Run("ReLU_Degree3_HE", func(t *testing.T) {
		result, err := EvalReLU3(ciphertext, evaluator)
		assert.NoError(t, err, "EvalReLU3 failed")

		// Decrypt and decode
		decryptedPt := decryptor.DecryptNew(result)
		resultData := make([]float64, numSlots)
		err = encoder.Decode(decryptedPt, resultData)
		assert.NoError(t, err, "Failed to decode result")

		// Compare with plain reference
		for i := 0; i < numSlots; i++ {
			expected := EvalReLUPlain3(testData[i])
			actual := resultData[i]
			assert.InDelta(t, expected, actual, tolerance, "ReLU3 HE result mismatch at index %d", i)
		}
	})

	// Test ReLU degree 5
	t.Run("ReLU_Degree5_HE", func(t *testing.T) {
		result, err := EvalReLU5(ciphertext, evaluator)
		assert.NoError(t, err, "EvalReLU5 failed")

		// Decrypt and decode
		decryptedPt := decryptor.DecryptNew(result)
		resultData := make([]float64, numSlots)
		err = encoder.Decode(decryptedPt, resultData)
		assert.NoError(t, err, "Failed to decode result")

		// Compare with plain reference
		for i := 0; i < numSlots; i++ {
			expected := EvalReLUPlain5(testData[i])
			actual := resultData[i]
			assert.InDelta(t, expected, actual, tolerance, "ReLU5 HE result mismatch at index %d", i)
		}
	})

	// Test Sigmoid degree 3
	t.Run("Sigmoid_Degree3_HE", func(t *testing.T) {
		result, err := EvalSigmoid3(ciphertext, evaluator)
		assert.NoError(t, err, "EvalSigmoid3 failed")

		// Decrypt and decode
		decryptedPt := decryptor.DecryptNew(result)
		resultData := make([]float64, numSlots)
		err = encoder.Decode(decryptedPt, resultData)
		assert.NoError(t, err, "Failed to decode result")

		// Compare with plain reference
		for i := 0; i < numSlots; i++ {
			expected := EvalSigmoidPlain3(testData[i])
			actual := resultData[i]
			assert.InDelta(t, expected, actual, tolerance, "Sigmoid3 HE result mismatch at index %d", i)
		}
	})

	// Test Sigmoid degree 5
	t.Run("Sigmoid_Degree5_HE", func(t *testing.T) {
		result, err := EvalSigmoid5(ciphertext, evaluator)
		assert.NoError(t, err, "EvalSigmoid5 failed")

		// Decrypt and decode
		decryptedPt := decryptor.DecryptNew(result)
		resultData := make([]float64, numSlots)
		err = encoder.Decode(decryptedPt, resultData)
		assert.NoError(t, err, "Failed to decode result")

		// Compare with plain reference
		for i := 0; i < numSlots; i++ {
			expected := EvalSigmoidPlain5(testData[i])
			actual := resultData[i]
			assert.InDelta(t, expected, actual, tolerance, "Sigmoid5 HE result mismatch at index %d", i)
		}
	})
}

// Test 3: Batch ranges
func TestAffineMapping(t *testing.T) {
	// Setup CKKS parameters using the existing parameter sets
	params, err := he.GetCKKSParameters(he.TestSet)
	assert.NoError(t, err, "Failed to create CKKS parameters")

	// Key generation
	kgen := he.KeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)

	// Create encoder, encryptor, decryptor, evaluator
	encoder := he.NewEncoder(params)
	encryptor := he.NewEncryptor(params, pk)
	decryptor := he.NewDecryptor(params, sk)
	evaluator := he.NewEvaluator(params, evk)

	// Test data: values in [-1, 1]
	const numSlots = 32
	testData := make([]float64, numSlots)
	for i := 0; i < numSlots; i++ {
		testData[i] = 2*rand.Float64() - 1
	}

	// Encrypt test data
	plaintext := ckks.NewPlaintext(params, params.MaxLevel())
	err = encoder.Encode(testData, plaintext)
	assert.NoError(t, err, "Failed to encode test data")

	ciphertext, err := encryptor.EncryptNew(plaintext)
	assert.NoError(t, err, "Failed to encrypt test data")

	testActivations := []struct {
		name     string
		evalFunc func(*rlwe.Ciphertext, *ckks.Evaluator) (*rlwe.Ciphertext, error)
		minOut   float64
		maxOut   float64
	}{
		{"ReLU3", EvalReLU3, -0.05, 1.05},
		{"ReLU5", EvalReLU5, -0.05, 1.05},
		{"Sigmoid3", EvalSigmoid3, -0.05, 1.05},
		{"Sigmoid5", EvalSigmoid5, -0.05, 1.05},
	}

	for _, test := range testActivations {
		t.Run(test.name, func(t *testing.T) {
			result, err := test.evalFunc(ciphertext, evaluator)
			assert.NoError(t, err, "%s evaluation failed", test.name)

			// Decrypt and decode
			decryptedPt := decryptor.DecryptNew(result)
			resultData := make([]float64, numSlots)
			err = encoder.Decode(decryptedPt, resultData)
			assert.NoError(t, err, "Failed to decode result for %s", test.name)

			// Check output range
			for i := 0; i < numSlots; i++ {
				assert.GreaterOrEqual(t, resultData[i], test.minOut,
					"%s output at index %d (%f) should be >= %f", test.name, i, resultData[i], test.minOut)
				assert.LessOrEqual(t, resultData[i], test.maxOut,
					"%s output at index %d (%f) should be <= %f", test.name, i, resultData[i], test.maxOut)
			}
		})
	}
}

// Benchmark tests
func BenchmarkReLU3Plain(b *testing.B) {
	x := 0.5
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = EvalReLUPlain3(x)
	}
}

func BenchmarkReLU5Plain(b *testing.B) {
	x := 0.5
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = EvalReLUPlain5(x)
	}
}

func BenchmarkSigmoid3Plain(b *testing.B) {
	x := 0.5
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = EvalSigmoidPlain3(x)
	}
}

func BenchmarkSigmoid5Plain(b *testing.B) {
	x := 0.5
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = EvalSigmoidPlain5(x)
	}
}
