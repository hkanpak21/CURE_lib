package split

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Helper function to find the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// innerSumSlots returns a ciphertext that contains the slot-sum of ct
// replicated in every slot. Uses Lattigo's more efficient InnerSum
// which provides an O(log n) tree reduction rather than linear rotations.
func innerSumSlots(ct *rlwe.Ciphertext, slots int, evaluator *ckks.Evaluator) (*rlwe.Ciphertext, error) {
	// Use Lattigo's optimized InnerSum for O(log n) reduction
	// This creates rotations in a tree pattern rather than linearly
	res := ct.CopyNew()
	if err := evaluator.InnerSum(res, 1, slots, res); err != nil {
		return nil, err
	}
	return res, nil
}

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

// repeat creates a slice with a value repeated n times
func repeat(value float64, n int) []float64 {
	result := make([]float64, n)
	for i := range result {
		result[i] = value
	}
	return result
}

// maskFirst creates a plaintext mask that keeps only the first few slots
// and zeros out the rest. This is useful for extracting just the first value
// from batch operations.
func maskFirst(params ckks.Parameters, encoder *ckks.Encoder, batchSize int) *rlwe.Plaintext {
	pt := ckks.NewPlaintext(params, params.MaxLevel())

	// Create a vector with 1s for the first batchSize slots and 0s elsewhere
	vec := make([]float64, params.N()/2)
	for i := 0; i < batchSize && i < len(vec); i++ {
		vec[i] = 1.0
	}

	// Encode the vector
	encoder.Encode(vec, pt)

	return pt
}

// chunkSum computes the sum across slots within specified chunks
// For example, it can sum across batch elements for each neuron
func chunkSum(ct *rlwe.Ciphertext, chunkSize int, evaluator *ckks.Evaluator) (*rlwe.Ciphertext, error) {
	// Use innerSumSlots for the efficient implementation
	return innerSumSlots(ct, chunkSize, evaluator)
}
