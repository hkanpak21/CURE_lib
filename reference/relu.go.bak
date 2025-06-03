package activation_he

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// EvalReLU3 evaluates a degree-3 ReLU approximation on a CKKS ciphertext.
// The input ciphertext should contain values in the domain [-1, 1].
// Returns a new ciphertext containing the ReLU approximation result.
func EvalReLU3(ct *rlwe.Ciphertext, eval *ckks.Evaluator) (*rlwe.Ciphertext, error) {
	return EvalChebyshev(ct, ReluDeg3, eval)
}

// EvalReLU5 evaluates a degree-5 ReLU approximation on a CKKS ciphertext.
// The input ciphertext should contain values in the domain [-1, 1].
// Returns a new ciphertext containing the ReLU approximation result.
func EvalReLU5(ct *rlwe.Ciphertext, eval *ckks.Evaluator) (*rlwe.Ciphertext, error) {
	return EvalChebyshev(ct, ReluDeg5, eval)
}
