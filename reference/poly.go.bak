package activation_he

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// EvalChebyshev evaluates a Chebyshev polynomial on a CKKS ciphertext
// using the iterative Chebyshev recurrence relation.
// coeffs contains coefficients c₀, c₁, ..., cₙ for Chebyshev polynomials T₀, T₁, ..., Tₙ
// The input ciphertext should contain values in the domain [-1, 1]
// Returns a new ciphertext containing the polynomial evaluation result
func EvalChebyshev(ct *rlwe.Ciphertext, coeffs []float64, eval *ckks.Evaluator) (*rlwe.Ciphertext, error) {
	if len(coeffs) == 0 {
		// Return zero ciphertext
		result, err := eval.MulNew(ct, 0.0)
		if err != nil {
			return nil, err
		}
		return result, nil
	}

	if len(coeffs) == 1 {
		// Return constant polynomial c₀ * T₀(x) = c₀ * 1 = c₀
		result, err := eval.MulNew(ct, 0.0) // Zero ciphertext
		if err != nil {
			return nil, err
		}
		result, err = eval.AddNew(result, coeffs[0]) // Add constant
		if err != nil {
			return nil, err
		}
		return result, nil
	}

	n := len(coeffs) - 1

	// Initialize result = c₀ * T₀(x) = c₀ * 1 = c₀
	result, err := eval.MulNew(ct, 0.0) // Zero ciphertext
	if err != nil {
		return nil, err
	}
	result, err = eval.AddNew(result, coeffs[0]) // Add constant c₀
	if err != nil {
		return nil, err
	}

	if n >= 1 {
		// Add c₁ * T₁(x) = c₁ * x
		t1, err := eval.MulNew(ct, coeffs[1])
		if err != nil {
			return nil, err
		}
		result, err = eval.AddNew(result, t1)
		if err != nil {
			return nil, err
		}
	}

	if n >= 2 {
		// For n >= 2, use the recurrence relation T_k = 2*x*T_{k-1} - T_{k-2}
		// We need to compute T₂, T₃, ..., Tₙ and accumulate c_k * T_k

		// T₀ = 1, T₁ = x (we already handled these)
		// We keep track of the two previous Chebyshev polynomials
		T_prev2, err := eval.MulNew(ct, 0.0) // T₀ = 1 (as a zero ciphertext + 1)
		if err != nil {
			return nil, err
		}
		T_prev2, err = eval.AddNew(T_prev2, 1.0)
		if err != nil {
			return nil, err
		}

		T_prev1 := ct.CopyNew() // T₁ = x

		// Compute T₂, T₃, ..., Tₙ and add c_k * T_k to the result
		for k := 2; k <= n; k++ {
			// T_k = 2*x*T_{k-1} - T_{k-2}

			// Compute 2*x*T_{k-1}
			temp1, err := eval.MulNew(ct, T_prev1)
			if err != nil {
				return nil, err
			}
			err = eval.Relinearize(temp1, temp1)
			if err != nil {
				return nil, err
			}
			temp1, err = eval.MulNew(temp1, 2.0)
			if err != nil {
				return nil, err
			}

			// Subtract T_{k-2}
			T_k, err := eval.SubNew(temp1, T_prev2)
			if err != nil {
				return nil, err
			}

			// Add c_k * T_k to the result
			if coeffs[k] != 0.0 {
				term, err := eval.MulNew(T_k, coeffs[k])
				if err != nil {
					return nil, err
				}
				result, err = eval.AddNew(result, term)
				if err != nil {
					return nil, err
				}
			}

			// Update for next iteration
			T_prev2 = T_prev1
			T_prev1 = T_k
		}
	}

	return result, nil
}
