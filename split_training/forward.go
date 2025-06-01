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

	// Use mutex to protect access to the encInputs slice and error reporting
	var mu sync.Mutex
	var encError error

	// Process images in parallel
	parallelFor(0, batch, func(b int) {
		// Early return if an error was already encountered
		if encError != nil {
			return
		}

		imgIdx := idx[b]
		img := imgs[imgIdx]
		pixelsPerImage := len(img)

		if pixelsPerImage > slots {
			mu.Lock()
			encError = fmt.Errorf("image too large for slot capacity: %d pixels > %d slots", pixelsPerImage, slots)
			mu.Unlock()
			return
		}

		// Create a buffer for encoding
		vec := make([]float64, slots)

		// Copy the image pixels into the vector
		for i := 0; i < pixelsPerImage; i++ {
			vec[i] = img[i]
		}

		// Encode the vector
		pt := ckks.NewPlaintext(he.params, he.params.MaxLevel())
		he.encoder.Encode(vec, pt)

		// Encrypt
		ct, err := he.encryptor.EncryptNew(pt)
		if err != nil {
			mu.Lock()
			encError = fmt.Errorf("encryption error for image %d: %v", b, err)
			mu.Unlock()
			return
		}

		// Store the encrypted result
		mu.Lock()
		encInputs[b] = ct
		mu.Unlock()
	})

	if encError != nil {
		return nil, encError
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

// Server performs forward pass on encrypted input
func serverForwardPass(he *HEContext, serverModel *ServerModel, encInputs []*rlwe.Ciphertext) ([][]*rlwe.Ciphertext, []*rlwe.Ciphertext, error) {
	// Get number of layers in the server model
	numLayers := len(serverModel.Weights)

	// Create a slice to store the inputs for each layer (including the initial input)
	layerInputs := make([][]*rlwe.Ciphertext, numLayers+1)
	layerInputs[0] = encInputs // First layer's input is the original encrypted input

	// Check if input is valid
	if len(encInputs) == 0 {
		return nil, nil, fmt.Errorf("invalid input: empty encInputs")
	}

	// Process each layer
	for l := 0; l < numLayers; l++ {
		// Get the dimensions of the current layer
		inputDim := serverModel.GetLayerInputDim(l)
		outputDim := serverModel.GetLayerOutputDim(l)

		// Ensure encInputs has the right size
		if len(encInputs) != inputDim && l == 0 {
			return nil, nil, fmt.Errorf("input dimension mismatch: got %d, expected %d", len(encInputs), inputDim)
		}

		// Create a slice for the current layer's outputs
		var layerOutputs []*rlwe.Ciphertext

		// Process the inputs one at a time for this layer
		for j := 0; j < outputDim; j++ {
			// Initialize a ciphertext with zeros for the output neuron
			// Create a zero plaintext
			zeroPlaintext := ckks.NewPlaintext(he.params, he.params.MaxLevel())
			// Encrypt it
			outputCipher, err := he.encryptor.EncryptNew(zeroPlaintext)
			if err != nil {
				return nil, nil, fmt.Errorf("error initializing output: %v", err)
			}

			// Compute weighted sum: W[l][i][j] * encInputs[i] + b[l][j]
			for i := 0; i < inputDim; i++ {
				// Get current input
				current := layerInputs[l][i]

				// Multiply by weight
				weight := serverModel.GetWeight(l, i, j)

				// Create a plaintext for the weight
				weightPlaintext := ckks.NewPlaintext(he.params, he.params.MaxLevel())
				// Encode a single value in all slots
				values := make([]float64, he.params.N()/2)
				for k := range values {
					values[k] = weight
				}
				he.encoder.Encode(values, weightPlaintext)

				// Multiply input by weight
				weighted := current.CopyNew()
				if err := he.evaluator.Mul(weighted, weightPlaintext, weighted); err != nil {
					return nil, nil, fmt.Errorf("error in weight multiplication: %v", err)
				}

				// Add to the output
				if err := he.evaluator.Add(outputCipher, weighted, outputCipher); err != nil {
					return nil, nil, fmt.Errorf("error in accumulation: %v", err)
				}
			}

			// Add bias
			bias := serverModel.Biases[l][j]

			// Create a plaintext for the bias
			biasPlaintext := ckks.NewPlaintext(he.params, he.params.MaxLevel())
			// Encode a single value in all slots
			biasValues := make([]float64, he.params.N()/2)
			for k := range biasValues {
				biasValues[k] = bias
			}
			he.encoder.Encode(biasValues, biasPlaintext)

			if err := he.evaluator.Add(outputCipher, biasPlaintext, outputCipher); err != nil {
				return nil, nil, fmt.Errorf("error adding bias: %v", err)
			}

			// Apply ReLU activation if not the last layer
			if l < numLayers-1 {
				// Use the existing applyReLU function
				activatedCipher, err := applyReLU(he, outputCipher)
				if err != nil {
					return nil, nil, fmt.Errorf("error applying ReLU: %v", err)
				}
				outputCipher = activatedCipher
			}

			// Add the output neuron to the outputs
			layerOutputs = append(layerOutputs, outputCipher)
		}

		// Store this layer's outputs as the next layer's inputs
		layerInputs[l+1] = layerOutputs
	}

	// Return both the layer inputs (for backpropagation) and the final layer's output
	return layerInputs, layerInputs[numLayers], nil
}
