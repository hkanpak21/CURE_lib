package activation_he

import "math"

// EvalChebyshevPlain evaluates a Chebyshev polynomial on a plain float64 value
// using Clenshaw's recurrence algorithm.
// coeffs contains coefficients c₀, c₁, ..., cₙ for Chebyshev polynomials T₀, T₁, ..., Tₙ
// x should be in the domain [-1, 1]
func EvalChebyshevPlain(x float64, coeffs []float64) float64 {
	if len(coeffs) == 0 {
		return 0.0
	}
	if len(coeffs) == 1 {
		return coeffs[0]
	}

	// Clenshaw's algorithm for Chebyshev polynomials
	// b_{n+1} = b_{n+2} = 0
	// b_k = c_k + 2*x*b_{k+1} - b_{k+2} for k = n, n-1, ..., 1
	// b_0 = c_0 + x*b_1 - b_2
	// Result = b_0

	n := len(coeffs) - 1
	if n == 0 {
		return coeffs[0]
	}

	b_k2 := 0.0 // b_{k+2}
	b_k1 := 0.0 // b_{k+1}

	// Start from the highest degree coefficient
	for k := n; k >= 1; k-- {
		b_k := coeffs[k] + 2*x*b_k1 - b_k2
		b_k2 = b_k1
		b_k1 = b_k
	}

	// Final step for k=0
	return coeffs[0] + x*b_k1 - b_k2
}

// EvalReLUPlain3 evaluates degree-3 ReLU approximation on a plain float64
func EvalReLUPlain3(x float64) float64 {
	return EvalChebyshevPlain(x, ReluDeg3)
}

// EvalReLUPlain5 evaluates degree-5 ReLU approximation on a plain float64
func EvalReLUPlain5(x float64) float64 {
	return EvalChebyshevPlain(x, ReluDeg5)
}

// EvalSigmoidPlain3 evaluates degree-3 sigmoid approximation on a plain float64
func EvalSigmoidPlain3(x float64) float64 {
	return EvalChebyshevPlain(x, SigmDeg3)
}

// EvalSigmoidPlain5 evaluates degree-5 sigmoid approximation on a plain float64
func EvalSigmoidPlain5(x float64) float64 {
	return EvalChebyshevPlain(x, SigmDeg5)
}

// Reference functions for testing accuracy

// ReLUReference computes the true ReLU function
func ReLUReference(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// SigmoidReference computes the true sigmoid function
func SigmoidReference(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
