package activation_he

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// EvalSigmoid3 evaluates a degree-3 sigmoid approximation on a CKKS ciphertext.
// The input ciphertext should contain values in the domain [-1, 1].
// Returns a new ciphertext containing the sigmoid approximation result.
func EvalSigmoid3(ct *rlwe.Ciphertext, eval *ckks.Evaluator) (*rlwe.Ciphertext, error) {
	return EvalChebyshev(ct, SigmDeg3, eval)
}

// EvalSigmoid5 evaluates a degree-5 sigmoid approximation on a CKKS ciphertext.
// The input ciphertext should contain values in the domain [-1, 1].
// Returns a new ciphertext containing the sigmoid approximation result.
func EvalSigmoid5(ct *rlwe.Ciphertext, eval *ckks.Evaluator) (*rlwe.Ciphertext, error) {
	return EvalChebyshev(ct, SigmDeg5, eval)
}
