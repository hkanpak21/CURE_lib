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

// Initialize HE parameters and generate keys
func initHE() (*HEContext, error) {
	// Use simple parameters with enough multiplicative depth for our operations
	paramsLiteral := ckks.ParametersLiteral{
		LogN:            12,                    // Ring degree: 2^12 = 4096 (smaller for faster testing)
		LogQ:            []int{40, 40, 40, 40}, // More conservative parameters
		LogP:            []int{45, 45},         // Special modulus for key switching
		LogDefaultScale: 30,                    // Scale 2^30
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
	rotations := []int{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}

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
