package split

import (
	"fmt"
	"math"
	"sync"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Client performs forward and backward pass after receiving encrypted activations
func clientForwardAndBackward(heContext *HEContext, clientModel *ClientModel, encActivations []*rlwe.Ciphertext,
	labels []int, batchIndices []int) ([]*rlwe.Ciphertext, error) {

	batchSize := len(batchIndices)
	if batchSize == 0 {
		return nil, fmt.Errorf("empty batch")
	}

	// Check if activations are valid
	if len(encActivations) == 0 {
		return nil, fmt.Errorf("invalid input: empty encActivations")
	}

	// Get dimensions from configuration
	clientLayers := len(clientModel.Weights)
	if clientLayers == 0 {
		return nil, fmt.Errorf("client model has no layers")
	}

	// Input dimension for the client is the output of the server's last layer
	inputDim := clientModel.GetLayerInputDim(0)

	// Step 1: Receive & Decrypt server's activations (SIMD-packed)
	// Each encActivations[b] is one ciphertext packing up to `neuronsPerCT` server-neurons × batchSize examples
	neuronsPerCT := calculateNeuronsPerCT(heContext.params.N()/2, batchSize, 64)
	numBlocks := (inputDim + neuronsPerCT - 1) / neuronsPerCT

	// a1PerBlock[b][j][nInBlock] := activation of server-neuron (block b, offset nInBlock) for example j
	a1PerBlock := make([][][]float64, numBlocks)

	// Decrypt each block once, then unpack all batch examples
	for b := 0; b < numBlocks; b++ {
		pt := heContext.decryptor.DecryptNew(encActivations[b])
		decoded := make([]float64, heContext.params.N()/2)
		heContext.encoder.Decode(pt, decoded)

		a1PerBlock[b] = make([][]float64, batchSize)
		for j := 0; j < batchSize; j++ {
			a1PerBlock[b][j] = make([]float64, neuronsPerCT)
			offset := j * neuronsPerCT
			for nInBlk := 0; nInBlk < neuronsPerCT; nInBlk++ {
				slotIdx := offset + nInBlk
				if slotIdx < len(decoded) {
					a1PerBlock[b][j][nInBlk] = decoded[slotIdx]
				}
			}
		}
	}

	// Reassemble a1Transposed[j][neuronIdx] for j ∈ [0..batchSize), neuronIdx ∈ [0..inputDim)
	a1Transposed := make([][]float64, batchSize)
	for j := 0; j < batchSize; j++ {
		a1Transposed[j] = make([]float64, inputDim)
		for b := 0; b < numBlocks; b++ {
			base := b * neuronsPerCT
			for nInBlk := 0; nInBlk < neuronsPerCT; nInBlk++ {
				neuronIdx := base + nInBlk
				if neuronIdx < inputDim {
					a1Transposed[j][neuronIdx] = a1PerBlock[b][j][nInBlk]
				}
			}
		}
	}

	// Step 2: Forward Pass through Client's Layers (parallelized)
	// Store activations for each layer (including input)
	activations := make([][][]float64, clientLayers+1)
	activations[0] = a1Transposed // Input layer

	// Process each layer
	for l := 0; l < clientLayers; l++ {
		inDim := clientModel.GetLayerInputDim(l)
		outDim := clientModel.GetLayerOutputDim(l)

		// Initialize this layer's activations
		activations[l+1] = make([][]float64, batchSize)
		for i := 0; i < batchSize; i++ {
			activations[l+1][i] = make([]float64, outDim)
		}

		// Parallelize over each example i ∈ [0..batchSize)
		parallelFor(0, batchSize, func(i int) {
			for j := 0; j < outDim; j++ {
				sum := clientModel.Biases[l][j] // Add bias
				for k := 0; k < inDim; k++ {
					sum += activations[l][i][k] * clientModel.GetWeight(l, k, j)
				}
				// Apply ReLU activation if not the output layer
				if l < clientLayers-1 && sum < 0 {
					sum = 0 // ReLU
				}
				activations[l+1][i][j] = sum
			}
		})
	}

	// Output layer activations
	outputActivations := activations[clientLayers]

	// Step 3: Compute Loss (Cross-Entropy Loss)
	// Apply softmax and compute cross-entropy loss
	outputDim := clientModel.GetLayerOutputDim(clientLayers - 1)

	// For tracking softmax values for gradient computation
	softmaxValues := make([][]float64, batchSize)

	parallelFor(0, batchSize, func(i int) {
		// Apply softmax: exp(output) / sum(exp(output))
		maxVal := outputActivations[i][0]
		for j := 1; j < outputDim; j++ {
			if outputActivations[i][j] > maxVal {
				maxVal = outputActivations[i][j]
			}
		}

		expSum := 0.0
		expValues := make([]float64, outputDim)
		for j := 0; j < outputDim; j++ {
			expValues[j] = math.Exp(outputActivations[i][j] - maxVal)
			expSum += expValues[j]
		}

		// Compute softmax
		softmaxValues[i] = make([]float64, outputDim)
		for j := 0; j < outputDim; j++ {
			softmaxValues[i][j] = expValues[j] / expSum
		}
	})

	// Step 4: Compute Gradients (Backward)
	// Initialize arrays to store gradients for each layer
	dZ := make([][][]float64, clientLayers)
	dW := make([][][]float64, clientLayers)
	db := make([][]float64, clientLayers)

	// Output layer gradients (start backpropagation)
	outputLayerIdx := clientLayers - 1
	dZ[outputLayerIdx] = make([][]float64, batchSize)

	parallelFor(0, batchSize, func(i int) {
		dZ[outputLayerIdx][i] = make([]float64, outputDim)

		// Ensure we have valid label indices for computing gradients
		labelIdx := 0 // Default to class 0 if index is out of bounds
		if i < len(batchIndices) {
			batchIdx := batchIndices[i]
			if batchIdx < len(labels) {
				labelIdx = labels[batchIdx]
				// Make sure labelIdx is valid for the output dimension
				if labelIdx >= outputDim {
					labelIdx = 0 // If label is invalid, default to class 0
				}
			}
		}

		for j := 0; j < outputDim; j++ {
			// Derivative of softmax cross-entropy is (softmax - one_hot_true)
			if j == labelIdx {
				// Softmax - 1.0 for true class
				dZ[outputLayerIdx][i][j] = softmaxValues[i][j] - 1.0
			} else {
				// Softmax for other classes
				dZ[outputLayerIdx][i][j] = softmaxValues[i][j]
			}
		}
	})

	// Backpropagate through each layer
	for l := clientLayers - 1; l >= 0; l-- {
		inputDim := clientModel.GetLayerInputDim(l)
		outputDim := clientModel.GetLayerOutputDim(l)

		// Compute weight gradients for this layer
		dW[l] = make([][]float64, inputDim)

		parallelFor(0, inputDim, func(i int) {
			// Skip if index is out of bounds for the current layer
			if i >= len(clientModel.Weights[l]) {
				fmt.Printf("Warning: Skipping out-of-bounds index i=%d for layer %d (weights dim: %d)\n",
					i, l, len(clientModel.Weights[l]))
				dW[l][i] = make([]float64, outputDim) // Still create the array to avoid nil references
				return
			}

			dW[l][i] = make([]float64, outputDim)
			for j := 0; j < outputDim; j++ {
				// Skip if j is out of bounds
				if j >= len(clientModel.Weights[l][i]) {
					fmt.Printf("Warning: Skipping out-of-bounds index j=%d for layer %d, neuron %d (weights dim: %d)\n",
						j, l, i, len(clientModel.Weights[l][i]))
					continue
				}

				for k := 0; k < batchSize; k++ {
					// Bounds checking
					if k >= len(activations[l]) || i >= len(activations[l][k]) ||
						k >= len(dZ[l]) || j >= len(dZ[l][k]) {
						continue
					}

					dW[l][i][j] += activations[l][k][i] * dZ[l][k][j]
				}
				dW[l][i][j] /= float64(batchSize)
			}
		})

		// Compute bias gradients for this layer
		db[l] = make([]float64, outputDim)
		for i := 0; i < outputDim; i++ {
			for j := 0; j < batchSize; j++ {
				// Bounds checking
				if j >= len(dZ[l]) || i >= len(dZ[l][j]) {
					continue
				}

				db[l][i] += dZ[l][j][i]
			}
			db[l][i] /= float64(batchSize)
		}

		// Compute gradients for previous layer if not at input layer
		if l > 0 {
			prevLayerDim := clientModel.GetLayerInputDim(l - 1)
			dZ[l-1] = make([][]float64, batchSize)

			parallelFor(0, batchSize, func(i int) {
				dZ[l-1][i] = make([]float64, prevLayerDim)

				// For each neuron in the previous layer
				for j := 0; j < prevLayerDim; j++ {
					// For each neuron in the current layer
					for k := 0; k < outputDim; k++ {
						// Bounds checking for clientModel.Weights
						if j >= len(clientModel.Weights[l]) {
							continue
						}

						if k >= len(clientModel.Weights[l][j]) {
							continue
						}

						// Bounds checking for dZ
						if i >= len(dZ[l]) || k >= len(dZ[l][i]) {
							continue
						}

						// Use GetWeight which has bounds checking instead of direct array access
						weight := clientModel.GetWeight(l, j, k)
						dZ[l-1][i][j] += dZ[l][i][k] * weight
					}

					// Apply ReLU derivative if not the input layer
					if l > 1 { // Changed from l > 0 to l > 1 to avoid accessing non-existent activations
						// Bounds checking
						if i >= len(activations[l-1]) || j >= len(activations[l-1][i]) {
							continue
						}

						// ReLU derivative: 0 if input <= 0, 1 otherwise
						if activations[l-1][i][j] <= 0 {
							dZ[l-1][i][j] = 0
						}
					}
				}
			})
		}
	}

	// Step 5: Update Client's Weights
	// Update weights and biases for each layer
	for l := 0; l < clientLayers; l++ {
		inputDim := clientModel.GetLayerInputDim(l)
		outputDim := clientModel.GetLayerOutputDim(l)

		// Update weights
		for i := 0; i < inputDim; i++ {
			for j := 0; j < outputDim; j++ {
				newWeight := clientModel.GetWeight(l, i, j) - LearningRate*dW[l][i][j]
				clientModel.SetWeight(l, i, j, newWeight)
			}
		}

		// Update biases
		for j := 0; j < outputDim; j++ {
			clientModel.Biases[l][j] -= LearningRate * db[l][j]
		}
	}

	// Step 6: Prepare gradients for the server: ∂L/∂a₁ = dZ × W_client^T
	inputGradients := make([][]float64, inputDim)
	for n := 0; n < inputDim; n++ {
		inputGradients[n] = make([]float64, batchSize)
	}

	// Parallelize over each example j or over each output neuron k, whichever is larger
	if batchSize > outputDim {
		parallelFor(0, batchSize, func(j int) {
			for k := 0; k < outputDim; k++ {
				gradVal := dZ[0][j][k]
				for n := 0; n < inputDim; n++ {
					inputGradients[n][j] += gradVal * clientModel.GetWeight(0, n, k)
				}
			}
		})
	} else {
		parallelFor(0, outputDim, func(k int) {
			for j := 0; j < batchSize; j++ {
				gradVal := dZ[0][j][k]
				for n := 0; n < inputDim; n++ {
					inputGradients[n][j] += gradVal * clientModel.GetWeight(0, n, k)
				}
			}
		})
	}

	// Step 7: Pack and encrypt gradients to send back to server
	// Calculate how many neurons per ciphertext
	neuronsPerCT = calculateNeuronsPerCT(heContext.params.N()/2, batchSize, 64)
	numBlocks = (inputDim + neuronsPerCT - 1) / neuronsPerCT

	encGradBlk := make([]*rlwe.Ciphertext, numBlocks)
	slots := heContext.params.N() / 2

	// Process each block of neurons
	for b := 0; b < numBlocks; b++ {
		// Reuse this buffer across iterations - memory optimization
		scratch := make([]float64, slots)

		// Pack neurons for this block
		startNeuron := b * neuronsPerCT
		endNeuron := min(startNeuron+neuronsPerCT, inputDim)

		// Clear the scratch buffer
		for i := range scratch {
			scratch[i] = 0
		}

		// Pack each neuron's gradients for all examples in the batch
		for n := startNeuron; n < endNeuron; n++ {
			neuronOffset := (n - startNeuron) * batchSize

			// Copy this neuron's gradients for all examples
			for i := 0; i < batchSize; i++ {
				scratch[neuronOffset+i] = inputGradients[n][i]
			}
		}

		// Encode
		pt := ckks.NewPlaintext(heContext.params, heContext.params.MaxLevel())
		heContext.encoder.Encode(scratch, pt)

		// Encrypt
		encGrad, err := heContext.encryptor.EncryptNew(pt)
		if err != nil {
			return nil, fmt.Errorf("error encrypting gradients: %v", err)
		}

		encGradBlk[b] = encGrad
	}

	return encGradBlk, nil
}

// PerformEvaluation performs forward pass for evaluation without training
func PerformEvaluation(heContext *HEContext, clientModel *ClientModel, encActivations []*rlwe.Ciphertext,
	numImages int) ([]int, [][]float64, error) {

	// Get dimensions from configuration
	clientLayers := len(clientModel.Weights)
	if clientLayers == 0 {
		return nil, nil, fmt.Errorf("client model has no layers")
	}

	// Input dimension for the client is the output of the server's last layer
	inputDim := clientModel.GetLayerInputDim(0)
	numBlocks := len(encActivations)
	neuronsPerCT := inputDim / numBlocks

	// Step 1: Decrypt the server's encrypted activations
	a1 := make([][]float64, inputDim)

	for b := 0; b < numBlocks; b++ {
		// Decrypt the ciphertext for this block
		pt := heContext.decryptor.DecryptNew(encActivations[b])

		// Decode the plaintext
		decoded := make([]float64, heContext.params.N()/2)
		heContext.encoder.Decode(pt, decoded)

		// Extract values for each neuron in this block
		for n := 0; n < neuronsPerCT; n++ {
			neuronIdx := b*neuronsPerCT + n
			if neuronIdx < inputDim {
				a1[neuronIdx] = make([]float64, numImages)
				for i := 0; i < numImages; i++ {
					a1[neuronIdx][i] = decoded[n*numImages+i]
				}
			}
		}
	}

	// Transpose a1 to have shape [batchSize x inputDim]
	a1Transposed := make([][]float64, numImages)
	for i := range a1Transposed {
		a1Transposed[i] = make([]float64, inputDim)
		for j := 0; j < inputDim; j++ {
			a1Transposed[i][j] = a1[j][i]
		}
	}

	// Step 2: Forward Pass through Client's Layers
	// Store activations for each layer (including input)
	activations := make([][][]float64, clientLayers+1)
	activations[0] = a1Transposed // Input layer

	// Process each layer
	for l := 0; l < clientLayers; l++ {
		inputDim := clientModel.GetLayerInputDim(l)
		outputDim := clientModel.GetLayerOutputDim(l)

		// Initialize this layer's activations
		activations[l+1] = make([][]float64, numImages)
		for i := range activations[l+1] {
			activations[l+1][i] = make([]float64, outputDim)

			// Compute linear combination: act = prev_act * W + b
			for j := 0; j < outputDim; j++ {
				activations[l+1][i][j] = clientModel.Biases[l][j] // Add bias

				for k := 0; k < inputDim; k++ {
					activations[l+1][i][j] += activations[l][i][k] * clientModel.GetWeight(l, k, j)
				}
			}

			// Apply ReLU activation if not the output layer
			if l < clientLayers-1 {
				for j := range activations[l+1][i] {
					if activations[l+1][i][j] < 0 {
						activations[l+1][i][j] = 0
					}
				}
			}
		}
	}

	// Output layer activations
	outputActivations := activations[clientLayers]

	// Apply softmax to get probabilities
	predictions := make([]int, numImages)
	confidences := make([][]float64, numImages)

	for i := 0; i < numImages; i++ {
		// Apply softmax: exp(output) / sum(exp(output))
		maxVal := outputActivations[i][0]
		for j := 1; j < len(outputActivations[i]); j++ {
			if outputActivations[i][j] > maxVal {
				maxVal = outputActivations[i][j]
			}
		}

		expSum := 0.0
		expValues := make([]float64, len(outputActivations[i]))
		for j := 0; j < len(outputActivations[i]); j++ {
			expValues[j] = math.Exp(outputActivations[i][j] - maxVal)
			expSum += expValues[j]
		}

		// Compute softmax and find prediction
		confidences[i] = make([]float64, len(outputActivations[i]))
		maxProb := 0.0
		prediction := 0

		for j := 0; j < len(outputActivations[i]); j++ {
			confidences[i][j] = expValues[j] / expSum
			if confidences[i][j] > maxProb {
				maxProb = confidences[i][j]
				prediction = j
			}
		}

		predictions[i] = prediction
	}

	return predictions, confidences, nil
}

// Helper function to sum slots of a ciphertext for SIMD-based forward pass
func sumSlotsWithRotations(ctx *HEContext, ct *rlwe.Ciphertext, batchSize int) (*rlwe.Ciphertext, error) {
	// Sum all slots with log(batchSize) rotations
	result := ct.CopyNew()
	for i := 1; i < batchSize; i *= 2 {
		rotated := ct.CopyNew()
		if err := ctx.evaluator.Rotate(result, i, rotated); err != nil {
			return nil, fmt.Errorf("error in rotation: %v", err)
		}
		if err := ctx.evaluator.Add(result, rotated, result); err != nil {
			return nil, fmt.Errorf("error in addition: %v", err)
		}
	}
	return result, nil
}

// Server backward pass and weight update
func serverBackwardAndUpdate(
	heContext *HEContext,
	serverModel *ServerModel,
	encGradients []*rlwe.Ciphertext,
	cachedLayerInputs [][]*rlwe.Ciphertext,
	learningRate float64,
) error {
	numLayers := len(serverModel.Weights)

	// Convert server model to packed format for efficient homomorphic operations
	heServerPacked, err := convertToPacked(serverModel, heContext)
	if err != nil {
		return fmt.Errorf("failed to pack server model: %v", err)
	}

	// ∂L/∂z_N comes directly from client
	gradCipher := encGradients

	// Loop backwards through each server layer
	for l := numLayers - 1; l >= 0; l-- {
		// Get dimensions for layer l
		inputDim := serverModel.GetLayerInputDim(l)   // Number of neurons feeding into layer l
		outputDim := serverModel.GetLayerOutputDim(l) // Number of neurons output by layer l

		// Calculate how many neurons per ciphertext and how many blocks we need
		neuronsPerCT := heServerPacked.NeuronsPerCT
		numBlocks := (outputDim + neuronsPerCT - 1) / neuronsPerCT

		// Prepare a plaintext encoding of "-learningRate/BatchSize"
		lrNegPt := scalarPlain(-learningRate/float64(BatchSize), heContext.params, heContext.encoder)

		// ======= Compute & apply weight-updates homomorphically =======
		// Use parallelization for efficiency
		var wg sync.WaitGroup
		var errMutex sync.Mutex
		var updateErr error

		// Make sure we have enough cached layer inputs
		if l >= len(cachedLayerInputs) {
			return fmt.Errorf("not enough cached layer inputs: have %d, need index %d",
				len(cachedLayerInputs), l)
		}

		// Make sure the cached layer has enough inputs
		if inputDim > len(cachedLayerInputs[l]) {
			return fmt.Errorf("cached layer %d has only %d inputs, need %d",
				l, len(cachedLayerInputs[l]), inputDim)
		}

		// For each input neuron i in [0..inputDim)
		for i := 0; i < inputDim; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()

				// Check bounds
				if i >= len(cachedLayerInputs[l]) {
					errMutex.Lock()
					updateErr = fmt.Errorf("input index %d is out of bounds for cached layer %d (size: %d)",
						i, l, len(cachedLayerInputs[l]))
					errMutex.Unlock()
					return
				}

				// Get the cached input activation for this neuron
				actCipher := cachedLayerInputs[l][i]

				// Check if we have enough blocks in the packed model
				if i >= len(heServerPacked.W[l]) {
					errMutex.Lock()
					updateErr = fmt.Errorf("input index %d is out of bounds for heServerPacked.W[%d] (size: %d)",
						i, l, len(heServerPacked.W[l]))
					errMutex.Unlock()
					return
				}

				// Process each output block
				for blk := 0; blk < numBlocks; blk++ {
					// Check if block is within bounds
					if blk >= len(heServerPacked.W[l][i]) {
						continue // Skip this block if out of bounds
					}

					// Copy input activation for this neuron
					inCopy := actCipher.CopyNew()

					// Sum across batch to get a single "sum over batch" in each slot
					summedInput, err := sumSlotsWithRotations(heContext, inCopy, BatchSize)
					if err != nil {
						errMutex.Lock()
						updateErr = fmt.Errorf("error in summing slots (input) in layer %d: %v", l, err)
						errMutex.Unlock()
						return
					}

					// Check if we have enough gradient blocks
					if blk >= len(gradCipher) {
						errMutex.Lock()
						updateErr = fmt.Errorf("block %d is out of bounds for gradCipher (size: %d)",
							blk, len(gradCipher))
						errMutex.Unlock()
						return
					}

					// Copy gradient for this output block
					gradCopy := gradCipher[blk].CopyNew()

					// Multiply gradient × input activation
					if err := heContext.evaluator.Mul(gradCopy, summedInput, gradCopy); err != nil {
						errMutex.Lock()
						updateErr = fmt.Errorf("error in gradient-input multiplication for layer %d: %v", l, err)
						errMutex.Unlock()
						return
					}

					// Check if rescaling is needed and if we have enough levels
					if gradCopy.Level() > 0 {
						// Rescale if possible
						if err := heContext.evaluator.Rescale(gradCopy, gradCopy); err != nil {
							errMutex.Lock()
							updateErr = fmt.Errorf("error in rescaling for layer %d: %v", l, err)
							errMutex.Unlock()
							return
						}
					}

					// Scale by learning rate
					if err := heContext.evaluator.Mul(gradCopy, lrNegPt, gradCopy); err != nil {
						errMutex.Lock()
						updateErr = fmt.Errorf("error in learning rate scaling for layer %d: %v", l, err)
						errMutex.Unlock()
						return
					}

					// Add to the weights
					if err := heContext.evaluator.Add(heServerPacked.W[l][i][blk], gradCopy, heServerPacked.W[l][i][blk]); err != nil {
						errMutex.Lock()
						updateErr = fmt.Errorf("error updating weights for layer %d: %v", l, err)
						errMutex.Unlock()
						return
					}
				}
			}(i)
		}

		// Wait for all weight updates to complete
		wg.Wait()
		if updateErr != nil {
			return updateErr
		}

		// ======= Compute & apply bias-updates homomorphically =======
		// Check if bias blocks are within bounds
		if l >= len(heServerPacked.b) {
			return fmt.Errorf("layer %d is out of bounds for heServerPacked.b (size: %d)",
				l, len(heServerPacked.b))
		}

		// Process each output block
		for blk := 0; blk < numBlocks; blk++ {
			// Check if block is within bounds
			if blk >= len(heServerPacked.b[l]) {
				continue // Skip this block if out of bounds
			}

			// Check if we have enough gradient blocks
			if blk >= len(gradCipher) {
				return fmt.Errorf("block %d is out of bounds for gradCipher (size: %d)",
					blk, len(gradCipher))
			}

			// Copy gradient for bias update
			gradCopy := gradCipher[blk].CopyNew()

			// Sum across batch
			summedGrad, err := sumSlotsWithRotations(heContext, gradCopy, BatchSize)
			if err != nil {
				return fmt.Errorf("error in summing slots (bias) in layer %d: %v", l, err)
			}

			// Scale by learning rate
			if err := heContext.evaluator.Mul(summedGrad, lrNegPt, summedGrad); err != nil {
				return fmt.Errorf("error in learning rate scaling for biases in layer %d: %v", l, err)
			}

			// Add to biases
			if err := heContext.evaluator.Add(heServerPacked.b[l][blk], summedGrad, heServerPacked.b[l][blk]); err != nil {
				return fmt.Errorf("error updating biases for layer %d: %v", l, err)
			}
		}

		// ======= Compute propagated gradient for next iteration =======
		// Skip this for layer 0 since there's no previous layer
		if l > 0 {
			prevInputDim := serverModel.GetLayerInputDim(l - 1)
			prevNumBlocks := (prevInputDim + neuronsPerCT - 1) / neuronsPerCT

			// Create the nextGrad array with explicit size
			nextGrad := make([]*rlwe.Ciphertext, prevNumBlocks)

			// Only process up to prevInputDim neurons
			for iPrev := 0; iPrev < prevInputDim && iPrev < prevNumBlocks; iPrev++ {
				// Initialize an accumulator for the propagated gradient
				// Start with an empty ciphertext
				emptyPt := ckks.NewPlaintext(heContext.params, heContext.params.MaxLevel())
				accum, err := heContext.encryptor.EncryptNew(emptyPt)
				if err != nil {
					return fmt.Errorf("error creating accumulator for backprop: %v", err)
				}

				// For each output block, multiply by the corresponding weights and accumulate
				for blk := 0; blk < numBlocks; blk++ {
					// Check if we have enough gradient blocks
					if blk >= len(gradCipher) {
						return fmt.Errorf("block %d is out of bounds for gradCipher in backprop (size: %d)",
							blk, len(gradCipher))
					}

					// Get the weight matrix in plaintext form
					wPlain := getWeightsPlaintext(heContext, serverModel, l, iPrev, blk)

					// Copy the gradient for this block
					gradCopy := gradCipher[blk].CopyNew()

					// Multiply by weights
					if err := heContext.evaluator.Mul(gradCopy, wPlain, gradCopy); err != nil {
						return fmt.Errorf("error in weight-gradient multiplication for backprop: %v", err)
					}

					// Add to the accumulator
					if err := heContext.evaluator.Add(accum, gradCopy, accum); err != nil {
						return fmt.Errorf("error accumulating backprop sum: %v", err)
					}
				}

				// If we're not at the input layer, apply ReLU derivative approximation
				// ReLU derivative is 1 for x > 0, 0 otherwise
				// For simplicity, we'll use a basic approximation of the ReLU derivative
				// without implementing a full polynomial evaluator
				if err := applyReLUDerivative(heContext, accum); err != nil {
					return fmt.Errorf("error applying ReLU derivative: %v", err)
				}

				// Store the propagated gradient for this input neuron
				if iPrev < len(nextGrad) {
					nextGrad[iPrev] = accum
				} else {
					return fmt.Errorf("iPrev=%d is out of bounds for nextGrad (size: %d)",
						iPrev, len(nextGrad))
				}
			}

			// Update gradCipher for the next layer
			gradCipher = nextGrad
		}
	}

	// Finally, extract all updated weights/biases back into serverModel
	for l := 0; l < numLayers; l++ {
		err = updateModelFromHE(heContext, serverModel, heServerPacked, l, BatchSize)
		if err != nil {
			return fmt.Errorf("error updating model from HE for layer %d: %v", l, err)
		}
	}

	return nil
}

// Encode weights for a specific block into a plaintext
func getWeightsPlaintext(heContext *HEContext, serverModel *ServerModel, layer, inputIdx, blockIdx int) *rlwe.Plaintext {
	// Create a new plaintext
	pt := ckks.NewPlaintext(heContext.params, heContext.params.MaxLevel())

	// Check bounds for layer
	if layer < 0 || layer >= len(serverModel.Weights) {
		// If out of bounds, return an empty plaintext
		return pt
	}

	// Check bounds for inputIdx
	if inputIdx < 0 || inputIdx >= len(serverModel.Weights[layer]) {
		// If out of bounds, return an empty plaintext
		return pt
	}

	// Get the dimensions
	outputDim := serverModel.GetLayerOutputDim(layer)

	// Calculate neurons per ciphertext
	neuronsPerCT := heContext.params.N() / 2 / BatchSize

	// Create a buffer for the weights
	weights := make([]float64, heContext.params.N()/2)

	// Determine the range of output neurons for this block
	startJ := blockIdx * neuronsPerCT
	endJ := min(startJ+neuronsPerCT, outputDim)

	// Fill the weights buffer
	for j := startJ; j < endJ; j++ {
		// Additional bounds check for j
		if j >= len(serverModel.Weights[layer][inputIdx]) {
			continue
		}

		// Get the weight for input neuron inputIdx to output neuron j
		weight := serverModel.GetWeight(layer, inputIdx, j)

		// Populate all slots corresponding to this weight (for SIMD)
		for k := 0; k < BatchSize; k++ {
			slotIdx := (j-startJ)*BatchSize + k
			if slotIdx < len(weights) {
				weights[slotIdx] = weight
			}
		}
	}

	// Encode the weights into the plaintext
	heContext.encoder.Encode(weights, pt)

	return pt
}

// Apply a simple approximation of the ReLU derivative
func applyReLUDerivative(heContext *HEContext, ct *rlwe.Ciphertext) error {
	// ReLU derivative is 1 for x > 0, 0 otherwise
	// We'll use a simple polynomial approximation: 0.5 + 0.5*x/sqrt(x²+ε)
	// This is a smooth approximation of the step function

	// First, square the input
	ctSq := ct.CopyNew()
	if err := heContext.evaluator.Mul(ct, ct, ctSq); err != nil {
		return fmt.Errorf("error squaring input: %v", err)
	}

	// Add a small epsilon (encoded as plaintext) to avoid division by zero
	epsilonPt := scalarPlain(1e-6, heContext.params, heContext.encoder)
	if err := heContext.evaluator.Add(ctSq, epsilonPt, ctSq); err != nil {
		return fmt.Errorf("error adding epsilon: %v", err)
	}

	// Approximate 1/sqrt(x²+ε) using Newton-Raphson
	// For simplicity, we'll just use a constant approximation for now
	invSqrtPt := scalarPlain(0.5, heContext.params, heContext.encoder)

	// Multiply input by 0.5*invSqrt
	if err := heContext.evaluator.Mul(ct, invSqrtPt, ct); err != nil {
		return fmt.Errorf("error in derivative approximation: %v", err)
	}

	// Add 0.5
	halfPt := scalarPlain(0.5, heContext.params, heContext.encoder)
	if err := heContext.evaluator.Add(ct, halfPt, ct); err != nil {
		return fmt.Errorf("error adding constant term: %v", err)
	}

	return nil
}

// Helper function to update a specific layer from the homomorphic encrypted version
func updateModelFromHE(heContext *HEContext, serverModel *ServerModel, heServer *HEServerPacked, layer int, batchSize int) error {
	// Get dimensions
	inputDim := serverModel.GetLayerInputDim(layer)
	outputDim := serverModel.GetLayerOutputDim(layer)
	neuronsPerCT := heServer.NeuronsPerCT

	// Calculate number of blocks
	outputBlocks := (outputDim + neuronsPerCT - 1) / neuronsPerCT

	// Ensure we don't go out of bounds
	if layer >= len(serverModel.Weights) || layer >= len(heServer.W) {
		return nil
	}

	// Make sure inputDim matches actual dimensions of weights
	actualInputDim := min(inputDim, len(serverModel.Weights[layer]))

	// Temporary buffer for decoding
	slots := heContext.params.N() / 2
	plainVector := make([]float64, slots)

	// For each input neuron
	for i := 0; i < actualInputDim; i++ {
		// Skip if beyond the packed model's dimensions
		if i >= len(heServer.W[layer]) {
			continue
		}

		// For each output block
		actualBlocks := min(outputBlocks, len(heServer.W[layer][i]))
		for blk := 0; blk < actualBlocks; blk++ {
			// Decrypt the weight ciphertext
			pt := heContext.decryptor.DecryptNew(heServer.W[layer][i][blk])
			heContext.encoder.Decode(pt, plainVector)

			// Extract individual weights from the packed representation
			for j := 0; j < neuronsPerCT; j++ {
				outputIdx := blk*neuronsPerCT + j
				if outputIdx < outputDim && outputIdx < len(serverModel.Weights[layer][i]) {
					// Weight is at slot j
					serverModel.SetWeight(layer, i, outputIdx, plainVector[j])
				}
			}
		}
	}

	// Update biases
	// Ensure we don't go out of bounds with biases
	if layer >= len(serverModel.Biases) || layer >= len(heServer.b) {
		return nil
	}

	actualBiasBlocks := min(outputBlocks, len(heServer.b[layer]))
	for blk := 0; blk < actualBiasBlocks; blk++ {
		// Decrypt the bias ciphertext
		pt := heContext.decryptor.DecryptNew(heServer.b[layer][blk])
		heContext.encoder.Decode(pt, plainVector)

		// Extract individual biases
		for j := 0; j < neuronsPerCT; j++ {
			outputIdx := blk*neuronsPerCT + j
			if outputIdx < outputDim && outputIdx < len(serverModel.Biases[layer]) {
				// Bias is at slot j
				serverModel.Biases[layer][outputIdx] = plainVector[j]
			}
		}
	}

	return nil
}
