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
// replicated in every slot. Needs Pow-2 rotation keys.
func innerSumSlots(ct *rlwe.Ciphertext, slots int, evaluator *ckks.Evaluator) (*rlwe.Ciphertext, error) {
	res := ct.CopyNew()
	tmp := ct.CopyNew()

	// log₂(slots) rotations/additions
	for k := 1; k < slots; k <<= 1 {
		if err := evaluator.Rotate(res, k, tmp); err != nil {
			return nil, err
		}
		if err := evaluator.Add(res, tmp, res); err != nil {
			return nil, err
		}
	}
	return res, nil
}

// scalarPlain encodes a constant replicated in every slot.
func scalarPlain(val float64, params ckks.Parameters, encoder *ckks.Encoder) *rlwe.Plaintext {
	pt := ckks.NewPlaintext(params, params.MaxLevel())
	v := make([]float64, params.N()/2) // Number of slots is N/2 for CKKS
	for i := range v {
		v[i] = val
	}
	encoder.Encode(v, pt)
	return pt
}

// repeat(x, k) returns slice {x,x,…} (len=k)
func repeat(x float64, k int) []float64 {
	v := make([]float64, k)
	for i := range v {
		v[i] = x
	}
	return v
}

// chunkSum reduces the B slots that belong to each neuron
// (i.e. slots [k*B : k*B+B-1]) and replicates that sum on
// **every** slot of the same chunk.
func chunkSum(ct *rlwe.Ciphertext, B int, ev *ckks.Evaluator) (*rlwe.Ciphertext, error) {
	res := ct.CopyNew()
	tmp := ct.CopyNew()

	for k := 1; k < B; k <<= 1 { // log2(B) rotations
		if err := ev.Rotate(res, k, tmp); err != nil {
			return nil, err
		}
		if err := ev.Add(res, tmp, res); err != nil {
			return nil, err
		}
	}
	return res, nil
}

// maskFirst() → 1,0,…,0 every B slots (keeps slot-0 of each chunk)
func maskFirst(p ckks.Parameters, enc *ckks.Encoder, B int) *rlwe.Plaintext {
	v := make([]float64, p.N()/2)
	for i := 0; i < len(v); i += B {
		v[i] = 1
	}
	pt := ckks.NewPlaintext(p, p.MaxLevel())
	enc.Encode(v, pt)
	return pt
}
