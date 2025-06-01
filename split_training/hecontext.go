package split

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// HEContext holds all the necessary objects for homomorphic encryption
type HEContext struct {
	params    ckks.Parameters
	encoder   *ckks.Encoder
	encryptor *rlwe.Encryptor
	decryptor *rlwe.Decryptor
	evaluator *ckks.Evaluator
	sk        *rlwe.SecretKey
	pk        *rlwe.PublicKey
	rlk       *rlwe.RelinearizationKey
	rtks      []*rlwe.GaloisKey // Rotation keys
}

// GetParams returns the CKKS parameters
func (he *HEContext) GetParams() ckks.Parameters {
	return he.params
}

// GetEncoder returns the CKKS encoder
func (he *HEContext) GetEncoder() *ckks.Encoder {
	return he.encoder
}

// GetEncryptor returns the RLWE encryptor
func (he *HEContext) GetEncryptor() *rlwe.Encryptor {
	return he.encryptor
}

// GetDecryptor returns the RLWE decryptor
func (he *HEContext) GetDecryptor() *rlwe.Decryptor {
	return he.decryptor
}

// GetEvaluator returns the CKKS evaluator
func (he *HEContext) GetEvaluator() *ckks.Evaluator {
	return he.evaluator
}

// GetSlots returns the number of slots available in the scheme
func (he *HEContext) GetSlots() int {
	return he.params.N() / 2
}

// Initialize HE parameters and generate keys
func initHE() (*HEContext, error) {
	// Use parameters with enough multiplicative depth for our configurable networks
	paramsLiteral := ckks.ParametersLiteral{
		LogN:            13,                                // Ring degree: 2^13 = 8192 (higher for deeper networks)
		LogQ:            []int{55, 50, 50, 50, 50, 50, 50}, // More levels for multi-layer operations
		LogP:            []int{60, 60},                     // Special modulus for key switching
		LogDefaultScale: 40,                                // Higher scale for better precision
	}

	// Create parameters from literal
	params, err := ckks.NewParametersFromLiteral(paramsLiteral)
	if err != nil {
		return nil, fmt.Errorf("error creating CKKS parameters: %v", err)
	}

	// Generate keys
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)

	// Generate relinearization keys (for relinearization)
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// --- rotations we need (powers of two up to slots/2) ---
	rotations := []int{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}

	// Generate rotation keys for all needed rotations
	var rotKeys []*rlwe.GaloisKey
	for _, rot := range rotations {
		galEl := params.GaloisElement(rot)
		rotKey := kgen.GenGaloisKeyNew(galEl, sk)
		rotKeys = append(rotKeys, rotKey)
	}

	// Collect everything in a single EvaluationKey set
	evk := rlwe.NewMemEvaluationKeySet(rlk, rotKeys...)

	return &HEContext{
		params:    params,
		encoder:   ckks.NewEncoder(params),
		encryptor: rlwe.NewEncryptor(params, pk),
		decryptor: rlwe.NewDecryptor(params, sk),
		evaluator: ckks.NewEvaluator(params, evk),
		sk:        sk,
		pk:        pk,
		rlk:       rlk,
		rtks:      rotKeys,
	}, nil
}
