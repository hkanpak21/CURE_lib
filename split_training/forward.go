package split

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Client prepares and encrypts a batch of images with SIMD optimization
// Packs an entire image into one ciphertext instead of one ciphertext per pixel
func clientPrepareAndEncryptBatch(he *HEContext, imgs [][]float64, idx []int) ([]*rlwe.Ciphertext, error) {
	slots := he.params.N() / 2

	// Instead of 784 ciphertexts, we'll use just one per image
	// Each image will be fully packed into its own ciphertext
	encInputs := make([]*rlwe.Ciphertext, 1)

	// Create a buffer for encoding
	vec := make([]float64, slots)

	// For now we process just one image at a time, later we can batch multiple images
	if len(idx) > 0 {
		// Get the first image
		imgIdx := idx[0]
		img := imgs[imgIdx]

		// Copy the image pixels into the vector
		// Make sure we don't exceed the number of slots
		pixelsToUse := min(len(img), slots)
		for i := 0; i < pixelsToUse; i++ {
			vec[i] = img[i]
		}

		// Encode the vector
		pt := ckks.NewPlaintext(he.params, he.params.MaxLevel())
		he.encoder.Encode(vec, pt)

		// Encrypt
		var err error
		encInputs[0], err = he.encryptor.EncryptNew(pt)
		if err != nil {
			return nil, fmt.Errorf("encryption error: %v", err)
		}
	}

	return encInputs, nil
}

// Server performs forward pass on encrypted inputs
func serverForwardPass(he *HEContext, serverModel *ServerModel, encInputs []*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error) {
	// Check if input is valid
	if len(encInputs) == 0 || encInputs[0] == nil {
		return nil, fmt.Errorf("invalid input: empty or nil encInputs")
	}

	// Since we've optimized to pack the entire image into a single ciphertext,
	// we need to implement a matrix-vector product differently.
	// We'll still create one ciphertext per neuron in the first hidden layer.

	// Create a packed server model for SIMD operations
	heServer, err := convertToPacked(serverModel, he)
	if err != nil {
		return nil, fmt.Errorf("failed to convert to packed model: %v", err)
	}

	// One ciphertext per neuron block in the first hidden layer
	blk := HiddenDim1 / NeuronsPerCT
	encActivations := make([]*rlwe.Ciphertext, blk)

	// For each neuron/group of neurons in the first hidden layer
	for b := 0; b < blk; b++ {
		// Initialize with the bias for this neuron group
		encActivations[b] = heServer.b[b].CopyNew()

		// Get the single packed input
		encInput := encInputs[0]

		// For each input dimension, multiply by weight and add to accumulator
		for i := 0; i < InputDim; i++ {
			// Create a temporary ciphertext for this multiplication
			temp := encInput.CopyNew()

			// Multiply the input by the weight for this neuron
			if err := he.evaluator.Mul(temp, heServer.W[i][b], temp); err != nil {
				return nil, fmt.Errorf("error in neuron %d input %d multiplication: %v", b, i, err)
			}

			// Relinearize
			if err := he.evaluator.Relinearize(temp, temp); err != nil {
				return nil, fmt.Errorf("error in neuron %d input %d relinearization: %v", b, i, err)
			}

			// Add to the accumulator
			if err := he.evaluator.Add(encActivations[b], temp, encActivations[b]); err != nil {
				return nil, fmt.Errorf("error adding to accumulator for neuron %d: %v", b, err)
			}
		}

		// Apply ReLU approximation to this neuron's activation
		// For simplicity, we'll use a degree-1 approximation max(0,x) ≈ 0.5x + 0.5
		halfPt := scalarPlain(0.5, he.params, he.encoder)

		// First multiply by 0.5
		if err := he.evaluator.Mul(encActivations[b], halfPt, encActivations[b]); err != nil {
			return nil, fmt.Errorf("error in ReLU scaling for neuron %d: %v", b, err)
		}

		// Then add 0.5
		if err := he.evaluator.Add(encActivations[b], halfPt, encActivations[b]); err != nil {
			return nil, fmt.Errorf("error in ReLU offset for neuron %d: %v", b, err)
		}
	}

	return encActivations, nil
}
