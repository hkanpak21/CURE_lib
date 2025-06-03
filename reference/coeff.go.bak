// Package activation_he provides homomorphic evaluation of activation functions
// using Chebyshev polynomial approximations on CKKS ciphertexts.
package activation_he

// Chebyshev polynomial coefficients for ReLU approximation on domain [-1, 1]
// ReLU(x) ≈ max(0, x)

// ReluDeg3 contains coefficients for degree-3 ReLU approximation
// Coefficients: c₀, c₁, c₂, c₃ for T₀, T₁, T₂, T₃
// MSE ≤ 4×10⁻⁴ on [-1, 1]
var ReluDeg3 = []float64{0.32811094, 0.5, 0.23435157, 0.0}

// ReluDeg5 contains coefficients for degree-5 ReLU approximation
// Coefficients: c₀, c₁, c₂, c₃, c₄, c₅ for T₀, T₁, T₂, T₃, T₄, T₅
// MSE ≤ 4×10⁻⁴ on [-1, 1]
var ReluDeg5 = []float64{0.31495238, 0.5, 0.20509862, 0.0, -0.05125416, 0.0}

// Chebyshev polynomial coefficients for Sigmoid approximation on domain [-1, 1]
// Sigmoid(x) ≈ 1/(1+e^{-x})

// SigmDeg3 contains coefficients for degree-3 sigmoid approximation
// Coefficients: c₀, c₁, c₂, c₃ for T₀, T₁, T₂, T₃
// MSE ≤ 4×10⁻⁴ on [-1, 1]
var SigmDeg3 = []float64{0.5, 0.23551963, 0.0, -0.00468065}

// SigmDeg5 contains coefficients for degree-5 sigmoid approximation
// Coefficients: c₀, c₁, c₂, c₃, c₄, c₅ for T₀, T₁, T₂, T₃, T₄, T₅
// MSE ≤ 4×10⁻⁴ on [-1, 1]
var SigmDeg5 = []float64{0.5, 0.23557248, 0.0, -0.00461894, 0.0, 0.00011125}
