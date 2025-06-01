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

	// Step 1: Receive & Decrypt server's activations
	// Each ciphertext represents a single image
	a1Transposed := make([][]float64, batchSize)

	// Process each image in the batch
	for i := 0; i < batchSize; i++ {
		// Decrypt the ciphertext for this image
		pt := heContext.decryptor.DecryptNew(encActivations[i])

		// Decode the plaintext - we get values in slots
		decoded := make([]float64, heContext.params.N()/2)
		heContext.encoder.Decode(pt, decoded)

		// Extract values for each neuron
		a1Transposed[i] = make([]float64, inputDim)
		for j := 0; j < inputDim; j++ {
			a1Transposed[i][j] = decoded[j]
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
		activations[l+1] = make([][]float64, batchSize)
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

	// Step 3: Compute Loss (Cross-Entropy Loss)
	// Apply softmax and compute cross-entropy loss
	outputDim := clientModel.GetLayerOutputDim(clientLayers - 1)

	// For tracking softmax values for gradient computation
	softmaxValues := make([][]float64, batchSize)

	for i := 0; i < batchSize; i++ {
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

		// Cross-entropy loss computation is commented out as it's not used in the algorithm
		// but kept for reference and future logging purposes
		// trueLabel := labels[batchIndices[i]]
		// loss += -math.Log(softmaxValues[i][trueLabel])
	}
	// loss /= float64(batchSize)

	// Step 4: Compute Gradients (Backward)
	// Initialize arrays to store gradients for each layer
	dZ := make([][][]float64, clientLayers)
	dW := make([][][]float64, clientLayers)
	db := make([][]float64, clientLayers)

	// Output layer gradients (start backpropagation)
	outputLayerIdx := clientLayers - 1
	dZ[outputLayerIdx] = make([][]float64, batchSize)

	for i := range dZ[outputLayerIdx] {
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
	}

	// Backpropagate through each layer
	for l := clientLayers - 1; l >= 0; l-- {
		inputDim := clientModel.GetLayerInputDim(l)
		outputDim := clientModel.GetLayerOutputDim(l)

		// Compute weight gradients for this layer
		dW[l] = make([][]float64, inputDim)
		for i := 0; i < inputDim; i++ {
			// Skip if index is out of bounds for the current layer
			if i >= len(clientModel.Weights[l]) {
				fmt.Printf("Warning: Skipping out-of-bounds index i=%d for layer %d (weights dim: %d)\n",
					i, l, len(clientModel.Weights[l]))
				dW[l][i] = make([]float64, outputDim) // Still create the array to avoid nil references
				continue
			}

			dW[l][i] = make([]float64, outputDim)
			for j := 0; j < outputDim; j++ {
				for k := 0; k < batchSize; k++ {
					dW[l][i][j] += activations[l][k][i] * dZ[l][k][j]
				}
				dW[l][i][j] /= float64(batchSize)
			}
		}

		// Compute bias gradients for this layer
		db[l] = make([]float64, outputDim)
		for i := 0; i < outputDim; i++ {
			for j := 0; j < batchSize; j++ {
				db[l][i] += dZ[l][j][i]
			}
			db[l][i] /= float64(batchSize)
		}

		// Compute gradients for previous layer if not at input layer
		if l > 0 {
			prevLayerDim := clientModel.GetLayerInputDim(l - 1)
			dZ[l-1] = make([][]float64, batchSize)

			for i := range dZ[l-1] {
				dZ[l-1][i] = make([]float64, prevLayerDim)

				// For each neuron in the previous layer
				for j := 0; j < prevLayerDim; j++ {
					// For each neuron in the current layer
					for k := 0; k < outputDim; k++ {
						// Skip if k is out of bounds for the current layer
						if k >= len(clientModel.Weights[l][0]) {
							continue
						}

						// Skip if j is out of bounds for the current layer's weights
						if j >= len(clientModel.Weights[l]) {
							continue
						}

						// Use GetWeight which has bounds checking instead of direct array access
						weight := clientModel.GetWeight(l, j, k)
						dZ[l-1][i][j] += dZ[l][i][k] * weight
					}

					// Apply ReLU derivative if not the input layer
					if l > 1 { // Changed from l > 0 to l > 1 to avoid accessing non-existent activations
						// ReLU derivative: 0 if input <= 0, 1 otherwise
						if j < len(activations[l-1][i]) && activations[l-1][i][j] <= 0 {
							dZ[l-1][i][j] = 0
						}
					}
				}
			}
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

	// Step 6: Prepare gradients for the server (gradients for the input layer)
	// First, convert dA1 to the format needed by the server
	inputGradients := make([][]float64, inputDim)
	for i := 0; i < inputDim; i++ {
		inputGradients[i] = make([]float64, batchSize)
		for j := 0; j < batchSize; j++ {
			inputGradients[i][j] = dZ[0][j][i]
		}
	}

	// Step 7: Pack and encrypt gradients to send back to server
	// Calculate how many neurons per ciphertext
	neuronsPerCT := calculateNeuronsPerCT(heContext.params.N()/2, batchSize, 64)
	numBlocks := (inputDim + neuronsPerCT - 1) / neuronsPerCT

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

// ServerBackwardAndUpdate performs server backward pass and weight updates
func serverBackwardAndUpdate(heContext *HEContext, serverModel *ServerModel, encGradients []*rlwe.Ciphertext,
	cachedInputs []*rlwe.Ciphertext, learningRate float64) error {
	// Delegate to packed update for efficiency
	return packedUpdate(heContext, serverModel, cachedInputs, encGradients, learningRate, len(cachedInputs))
}

// Packed update using SIMD optimizations
func packedUpdate(heContext *HEContext, serverModel *ServerModel, encInputs []*rlwe.Ciphertext,
	encGradients []*rlwe.Ciphertext, learningRate float64, batchSize int) error {

	// Convert server model to packed format
	heServerPacked, err := convertToPacked(serverModel, heContext)
	if err != nil {
		return fmt.Errorf("error converting to packed model: %v", err)
	}

	// Process each layer in the server model
	for l := 0; l < len(serverModel.Weights); l++ {
		inputDim := serverModel.GetLayerInputDim(l)
		outputDim := serverModel.GetLayerOutputDim(l)

		// Calculate number of blocks for the output dimension
		neuronsPerCT := heServerPacked.NeuronsPerCT
		outputBlocks := (outputDim + neuronsPerCT - 1) / neuronsPerCT

		// Use min to ensure we don't go out of bounds
		maxInputIdx := min(inputDim, len(encInputs))
		maxOutputIdx := min(outputBlocks, len(encGradients))

		// Use a WaitGroup to parallelize the update process
		var wg sync.WaitGroup

		// Error channel to collect errors from goroutines
		errCh := make(chan error, maxInputIdx*maxOutputIdx)

		// For each input dimension (limited by available inputs)
		for i := 0; i < maxInputIdx; i++ {
			// Capture loop variable
			inputIdx := i

			// For each output block (limited by available gradients)
			for j := 0; j < maxOutputIdx; j++ {
				// Capture loop variable
				outputIdx := j

				// Increment WaitGroup counter
				wg.Add(1)

				// Launch goroutine for each weight update
				go func(i, j int) {
					defer wg.Done()

					// Compute outer product of input and gradient
					// Expand input: duplicate each element to match with corresponding gradients
					expandedInput := encInputs[i].CopyNew()

					// Extract the j-th block of gradients
					gradBlock := encGradients[j].CopyNew()

					// Multiply expanded input with gradient block (element-wise)
					product := expandedInput.CopyNew()
					if err := heContext.evaluator.Mul(expandedInput, gradBlock, product); err != nil {
						errCh <- fmt.Errorf("error computing gradient product: %v", err)
						return
					}

					// Rescale product
					if err := heContext.evaluator.Rescale(product, product); err != nil {
						errCh <- fmt.Errorf("error rescaling product: %v", err)
						return
					}

					// Scale by learning rate (negative since we're doing gradient descent)
					scaled := product.CopyNew()
					if err := heContext.evaluator.Mul(product, -learningRate/float64(batchSize), scaled); err != nil {
						errCh <- fmt.Errorf("error scaling by learning rate: %v", err)
						return
					}

					// Add to current weights
					if err := heContext.evaluator.Add(heServerPacked.W[l][i][j], scaled, heServerPacked.W[l][i][j]); err != nil {
						errCh <- fmt.Errorf("error updating weights: %v", err)
						return
					}
				}(inputIdx, outputIdx)
			}
		}

		// Wait for all weight updates to complete
		wg.Wait()

		// Check if there were any errors
		select {
		case err := <-errCh:
			return err
		default:
			// No errors, continue
		}

		// Update biases (limited by available gradients) - also in parallel
		maxBiasIdx := min(len(heServerPacked.b[l]), len(encGradients))

		// Reset WaitGroup
		wg = sync.WaitGroup{}

		for j := 0; j < maxBiasIdx; j++ {
			// Capture loop variable
			biasIdx := j

			// Increment WaitGroup counter
			wg.Add(1)

			// Launch goroutine for each bias update
			go func(j int) {
				defer wg.Done()

				// Scale gradient by learning rate (negative for gradient descent)
				scaledGrad := encGradients[j].CopyNew()
				if err := heContext.evaluator.Mul(encGradients[j], -learningRate/float64(batchSize), scaledGrad); err != nil {
					errCh <- fmt.Errorf("error scaling bias gradient: %v", err)
					return
				}

				// Add to current biases
				if err := heContext.evaluator.Add(heServerPacked.b[l][j], scaledGrad, heServerPacked.b[l][j]); err != nil {
					errCh <- fmt.Errorf("error updating biases: %v", err)
					return
				}
			}(biasIdx)
		}

		// Wait for all bias updates to complete
		wg.Wait()

		// Check if there were any errors
		select {
		case err := <-errCh:
			return err
		default:
			// No errors, continue
		}
	}

	// Extract updated weights back to the normal server model
	for l := 0; l < len(serverModel.Weights); l++ {
		updateModelFromHE(heContext, serverModel, heServerPacked, l, batchSize)
	}

	return nil
}

// Helper function to update a specific layer from the homomorphic encrypted version
func updateModelFromHE(heContext *HEContext, serverModel *ServerModel, heServer *HEServerPacked, layer int, batchSize int) {
	// Get dimensions
	inputDim := serverModel.GetLayerInputDim(layer)
	outputDim := serverModel.GetLayerOutputDim(layer)
	neuronsPerCT := heServer.NeuronsPerCT

	// Calculate number of blocks
	outputBlocks := (outputDim + neuronsPerCT - 1) / neuronsPerCT

	// Ensure we don't go out of bounds
	if layer >= len(serverModel.Weights) || layer >= len(heServer.W) {
		return
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
		return
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
}
