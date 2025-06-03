package split

import (
	"sync"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Global buffer pools
var (
	float64Pool    sync.Pool
	ciphertextPool sync.Pool
)

// InitGlobalPools initializes the global pools with appropriate factory functions
func InitGlobalPools(params ckks.Parameters) {
	// Initialize float64 buffer pool
	float64Pool = sync.Pool{
		New: func() interface{} {
			// Create a buffer with the size of params.N()/2
			return make([]float64, params.N()/2)
		},
	}

	// Initialize ciphertext pool
	ciphertextPool = sync.Pool{
		New: func() interface{} {
			return ckks.NewCiphertext(params, 1, params.MaxLevel())
		},
	}
}

// GetFloat64Buffer retrieves a []float64 buffer from the pool.
// Remember to PutFloat64Buffer it back when done.
func GetFloat64Buffer() []float64 {
	return float64Pool.Get().([]float64)
}

// PutFloat64Buffer returns a []float64 buffer to the pool.
// Important: Clear the buffer before putting it back if sensitive data was used,
// or ensure it's always overwritten before next use.
func PutFloat64Buffer(buf []float64) {
	// Optional: Clear buffer before putting back
	// for i := range buf { buf[i] = 0 }
	float64Pool.Put(buf)
}

// GetCiphertext retrieves a *rlwe.Ciphertext from the pool.
// Remember to PutCiphertext it back when done.
// The ciphertext is initialized with degree 1, max level.
func GetCiphertext() *rlwe.Ciphertext {
	return ciphertextPool.Get().(*rlwe.Ciphertext)
}

// PutCiphertext returns a *rlwe.Ciphertext to the pool.
func PutCiphertext(ct *rlwe.Ciphertext) {
	ciphertextPool.Put(ct)
}
