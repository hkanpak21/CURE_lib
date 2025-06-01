package split

import (
	"fmt"
	"sync"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Client prepares and encrypts a batch of images for SIMD optimization
// Each image is encrypted into a separate ciphertext
func clientPrepareAndEncryptBatch(he *HEContext, imgs [][]float64, idx []int) ([]*rlwe.Ciphertext, error) {
	slots := he.params.N() / 2
	batch := len(idx)

	if batch == 0 {
		return nil, fmt.Errorf("empty batch")
	}

	// Create a ciphertext for each image in the batch
	encInputs := make([]*rlwe.Ciphertext, batch)

	// Create a buffer for encoding
	vec := make([]float64, slots)

	// Process each image in the batch
	for b := 0; b < batch; b++ {
		imgIdx := idx[b]
		img := imgs[imgIdx]
		pixelsPerImage := len(img)

		if pixelsPerImage > slots {
			return nil, fmt.Errorf("image too large for slot capacity: %d pixels > %d slots", pixelsPerImage, slots)
		}

		// Clear the buffer
		for i := range vec {
			vec[i] = 0
		}

		// Copy the image pixels into the vector
		for i := 0; i < pixelsPerImage; i++ {
			vec[i] = img[i]
		}

		// Encode the vector
		pt := ckks.NewPlaintext(he.params, he.params.MaxLevel())
		he.encoder.Encode(vec, pt)

		// Encrypt
		var err error
		encInputs[b], err = he.encryptor.EncryptNew(pt)
		if err != nil {
			return nil, fmt.Errorf("encryption error for image %d: %v", b, err)
		}
	}

	return encInputs, nil
}

// Cache for ReLU approximation coefficients
var (
	reluCoeffsMu sync.Mutex
	reluCoeffsC0 map[*HEContext]*rlwe.Plaintext
	reluCoeffsC1 map[*HEContext]*rlwe.Plaintext
	reluCoeffsC2 map[*HEContext]*rlwe.Plaintext
	relu05       map[*HEContext]*rlwe.Plaintext
)

func init() {
	reluCoeffsC0 = make(map[*HEContext]*rlwe.Plaintext)
	reluCoeffsC1 = make(map[*HEContext]*rlwe.Plaintext)
	reluCoeffsC2 = make(map[*HEContext]*rlwe.Plaintext)
	relu05 = make(map[*HEContext]*rlwe.Plaintext)
}

// Get cached ReLU approximation constants (using Chebyshev degree 2)
func getReLUCoeffs(he *HEContext) (*rlwe.Plaintext, *rlwe.Plaintext, *rlwe.Plaintext, *rlwe.Plaintext) {
	reluCoeffsMu.Lock()
	defer reluCoeffsMu.Unlock()

	// Check if coefficients already exist for this context
	if _, ok := reluCoeffsC0[he]; !ok {
		// Degree-2 Chebyshev approximation of ReLU: max(0, x) ≈ C0 + C1*x + C2*x^2
		// Optimized coefficients for the range [-1, 1]
		c0 := scalarPlain(0.25, he.params, he.encoder)  // Constant term
		c1 := scalarPlain(0.5, he.params, he.encoder)   // Linear term
		c2 := scalarPlain(0.25, he.params, he.encoder)  // Quadratic term
		half := scalarPlain(0.5, he.params, he.encoder) // For 0.5 constant

		reluCoeffsC0[he] = c0
		reluCoeffsC1[he] = c1
		reluCoeffsC2[he] = c2
		relu05[he] = half
	}

	return reluCoeffsC0[he], reluCoeffsC1[he], reluCoeffsC2[he], relu05[he]
}

// Apply ReLU approximation using degree-2 Chebyshev polynomial
func applyReLU(he *HEContext, ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	// Get cached coefficients
	c0, c1, c2, _ := getReLUCoeffs(he)

	// Create a copy for x^2
	ctSq := ct.CopyNew()

	// Calculate x^2
	if err := he.evaluator.Mul(ct, ct, ctSq); err != nil {
		return nil, fmt.Errorf("error in ReLU square calculation: %v", err)
	}

	// Relinearize after squaring
	if err := he.evaluator.Relinearize(ctSq, ctSq); err != nil {
		return nil, fmt.Errorf("error in ReLU relinearization: %v", err)
	}

	// Apply C2*x^2
	if err := he.evaluator.Mul(ctSq, c2, ctSq); err != nil {
		return nil, fmt.Errorf("error in ReLU quadratic term: %v", err)
	}

	// Apply C1*x to the original ct and store in result
	result := ct.CopyNew()
	if err := he.evaluator.Mul(result, c1, result); err != nil {
		return nil, fmt.Errorf("error in ReLU linear term: %v", err)
	}

	// Add C2*x^2 to C1*x
	if err := he.evaluator.Add(result, ctSq, result); err != nil {
		return nil, fmt.Errorf("error adding ReLU terms: %v", err)
	}

	// Add C0 (constant term)
	if err := he.evaluator.Add(result, c0, result); err != nil {
		return nil, fmt.Errorf("error adding ReLU constant term: %v", err)
	}

	// Rescale once at the end to maintain precision
	if err := he.evaluator.Rescale(result, result); err != nil {
		return nil, fmt.Errorf("error rescaling ReLU result: %v", err)
	}

	return result, nil
}

// Helper for dot product with power-of-two rotations
func dotPacked(he *HEContext, encImg *rlwe.Ciphertext, ptWRow []*rlwe.Plaintext, slotsPerPixel int) (*rlwe.Ciphertext, error) {
	// Create the accumulator as a copy of the first product
	acc := encImg.CopyNew()

	// Multiply with the first weight
	if err := he.evaluator.Mul(acc, ptWRow[0], acc); err != nil {
		return nil, fmt.Errorf("error in dot product multiplication: %v", err)
	}

	// For each remaining pixel
	for i := 1; i < len(ptWRow); i++ {
		// Rotate the input vector
		rotated := encImg.CopyNew()
		if err := he.evaluator.Rotate(encImg, i*slotsPerPixel, rotated); err != nil {
			return nil, fmt.Errorf("error rotating in dot product: %v", err)
		}

		// Multiply by the corresponding weight
		if err := he.evaluator.Mul(rotated, ptWRow[i], rotated); err != nil {
			return nil, fmt.Errorf("error in dot product multiplication: %v", err)
		}

		// Add to accumulator
		if err := he.evaluator.Add(acc, rotated, acc); err != nil {
			return nil, fmt.Errorf("error adding in dot product: %v", err)
		}
	}

	return acc, nil
}

// Parallel execution helper
func parallelFor(start, end int, fn func(int)) {
	var wg sync.WaitGroup
	numWorkers := NumWorkers

	// Adjust worker count if range is small
	if end-start < numWorkers {
		numWorkers = end - start
	}

	if numWorkers <= 1 {
		// Just run sequentially
		for i := start; i < end; i++ {
			fn(i)
		}
		return
	}

	// Divide work among workers
	wg.Add(numWorkers)
	chunkSize := (end - start + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		go func(workerID int) {
			defer wg.Done()

			// Calculate this worker's range
			workerStart := start + workerID*chunkSize
			workerEnd := min(workerStart+chunkSize, end)

			// Process this worker's range
			for i := workerStart; i < workerEnd; i++ {
				fn(i)
			}
		}(w)
	}

	wg.Wait()
}

// Server performs forward pass on encrypted inputs with optimized parallel processing
// Each ciphertext contains a single image
func serverForwardPass(he *HEContext, serverModel *ServerModel, encInputs []*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error) {
	// Check if input is valid
	if len(encInputs) == 0 || encInputs[0] == nil {
		return nil, fmt.Errorf("invalid input: empty or nil encInputs")
	}

	// Create a packed server model for SIMD operations
	heServer, err := convertToPacked(serverModel, he)
	if err != nil {
		return nil, fmt.Errorf("failed to convert to packed model: %v", err)
	}

	// Number of images in the batch
	batchSize := len(encInputs)

	// Process each image separately
	resultOutputs := make([]*rlwe.Ciphertext, batchSize)

	// For each image in the batch
	for batchIdx := 0; batchIdx < batchSize; batchIdx++ {
		// Process each layer in the server model
		currentLayerOutput := encInputs[batchIdx : batchIdx+1] // Slice with single ciphertext

		for l := 0; l < len(serverModel.Weights); l++ {
			inputDim := serverModel.GetLayerInputDim(l)
			outputDim := serverModel.GetLayerOutputDim(l)

			// Calculate the number of blocks needed for the output neurons
			neuronsPerCT := heServer.NeuronsPerCT
			numBlocks := (outputDim + neuronsPerCT - 1) / neuronsPerCT

			// Prepare the next layer activations
			nextLayer := make([]*rlwe.Ciphertext, numBlocks)

			// Create a mutex for thread-safe access to shared resources
			var mutex sync.Mutex

			// Process each block of neurons in parallel
			parallelFor(0, numBlocks, func(b int) {
				// Initialize with bias for this neuron block
				blockResult := heServer.b[l][b].CopyNew()

				// Get the input for this image
				encInput := currentLayerOutput[0]

				// For each input dimension, multiply by weight and add to accumulator
				for i := 0; i < inputDim; i++ {
					// Create a temporary ciphertext for this multiplication
					temp := encInput.CopyNew()

					// Multiply the input by the weight for this neuron block
					he.evaluator.Mul(temp, heServer.W[l][i][b], temp)

					// Relinearize
					he.evaluator.Relinearize(temp, temp)

					// Add to the accumulator
					he.evaluator.Add(blockResult, temp, blockResult)
				}

				// Apply ReLU approximation to this block's activation
				activatedBlock, err := applyReLU(he, blockResult)
				if err != nil {
					fmt.Printf("Error in ReLU for block %d: %v\n", b, err)
					return
				}

				// Safely store the result
				mutex.Lock()
				nextLayer[b] = activatedBlock
				mutex.Unlock()
			})

			// Update current layer for the next iteration
			currentLayerOutput = nextLayer
		}

		// Store the final output for this image
		resultOutputs[batchIdx] = currentLayerOutput[0]
	}

	return resultOutputs, nil
}
