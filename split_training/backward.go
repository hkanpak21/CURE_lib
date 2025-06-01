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
		for j := 0; j < outputDim; j++ {
			// Derivative of softmax cross-entropy is (softmax - one_hot_true)
			if j == labels[batchIndices[i]] {
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
				for j := 0; j < prevLayerDim; j++ {
					// Skip if index is out of bounds for the current layer
					if j >= len(clientModel.Weights[l]) {
						fmt.Printf("Warning: Skipping out-of-bounds index j=%d for layer %d (weights dim: %d)\n",
							j, l, len(clientModel.Weights[l]))
						continue
					}

					for k := 0; k < outputDim; k++ {
						dZ[l-1][i][j] += dZ[l][i][k] * clientModel.GetWeight(l, j, k)
					}

					// Apply ReLU derivative if not the input layer
					if l > 0 {
						// ReLU derivative: 0 if input <= 0, 1 otherwise
						if activations[l][i][j] <= 0 {
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

	// Step 3: Compute Predictions
	predictions := make([]int, numImages)
	confidences := make([][]float64, numImages)

	for i := 0; i < numImages; i++ {
		// Apply softmax
		outputDim := clientModel.GetLayerOutputDim(clientLayers - 1)
		maxVal := outputActivations[i][0]

		for j := 1; j < outputDim; j++ {
			if outputActivations[i][j] > maxVal {
				maxVal = outputActivations[i][j]
			}
		}

		expValues := make([]float64, outputDim)
		expSum := 0.0
		for j := 0; j < outputDim; j++ {
			expValues[j] = math.Exp(outputActivations[i][j] - maxVal)
			expSum += expValues[j]
		}

		// Find the class with highest probability
		confidences[i] = make([]float64, outputDim)
		maxProb := 0.0
		for j := 0; j < outputDim; j++ {
			confidences[i][j] = expValues[j] / expSum
			if confidences[i][j] > maxProb {
				maxProb = confidences[i][j]
				predictions[i] = j
			}
		}
	}

	return predictions, confidences, nil
}

// sumSlotsWithRotations implements efficient summation across slots
func sumSlotsWithRotations(ctx *HEContext, ct *rlwe.Ciphertext, batchSize int) (*rlwe.Ciphertext, error) {
	result := ct.CopyNew()

	// Use rotation-and-add technique to sum up values across slots
	for i := 1; i < batchSize; i *= 2 {
		rotated := result.CopyNew()
		if err := ctx.evaluator.Rotate(result, i, rotated); err != nil {
			return nil, fmt.Errorf("error rotating in innerSum: %v", err)
		}

		if err := ctx.evaluator.Add(result, rotated, result); err != nil {
			return nil, fmt.Errorf("error adding in innerSum: %v", err)
		}
	}

	return result, nil
}

// serverBackwardAndUpdate performs the backward pass and updates server model weights
func serverBackwardAndUpdate(heContext *HEContext, serverModel *ServerModel, encGradients []*rlwe.Ciphertext,
	cachedInputs []*rlwe.Ciphertext, learningRate float64) error {

	// Process with the packed update method
	return packedUpdate(heContext, serverModel, cachedInputs, encGradients, learningRate, BatchSize)
}

// packedUpdate performs weight updates using packed ciphertexts
func packedUpdate(heContext *HEContext, serverModel *ServerModel, encInputs []*rlwe.Ciphertext,
	encGradients []*rlwe.Ciphertext, learningRate float64, batchSize int) error {

	// Create a packed server model for SIMD operations
	heServer, err := convertToPacked(serverModel, heContext)
	if err != nil {
		return fmt.Errorf("failed to convert to packed model: %v", err)
	}

	// Process only the first (last) layer of the server model
	// In future versions, this could be extended to all server layers
	l := len(serverModel.Weights) - 1
	inputDim := serverModel.GetLayerInputDim(l)
	outputDim := serverModel.GetLayerOutputDim(l)

	// Prepare LR plaintext
	lrPt := scalarPlain(-1.0*learningRate/float64(batchSize), heContext.params, heContext.encoder)

	// Calculate how many neurons per ciphertext
	neuronsPerCT := heServer.NeuronsPerCT
	numBlocks := (outputDim + neuronsPerCT - 1) / neuronsPerCT

	// Check if we have the right number of gradient blocks
	if len(encGradients) != numBlocks {
		return fmt.Errorf("gradient blocks mismatch: got %d, expected %d", len(encGradients), numBlocks)
	}

	// Process each block of neurons in parallel
	var wg sync.WaitGroup
	var errMutex sync.Mutex
	var packedErr error

	for b := 0; b < numBlocks; b++ {
		wg.Add(1)
		go func(blockIdx int) {
			defer wg.Done()

			// For each input dimension
			for i := 0; i < inputDim; i++ {
				// 1. Create a copy of the input
				inputCopy := encInputs[0].CopyNew()

				// 2. Perform inner sum across the batch dimension
				summedInput, err := sumSlotsWithRotations(heContext, inputCopy, batchSize)
				if err != nil {
					errMutex.Lock()
					packedErr = fmt.Errorf("error in inner sum for block %d: %v", blockIdx, err)
					errMutex.Unlock()
					return
				}

				// 3. Create a copy of the gradient
				gradCopy := encGradients[blockIdx].CopyNew()

				// 4. Multiply gradient with input
				if err := heContext.evaluator.Mul(gradCopy, summedInput, gradCopy); err != nil {
					errMutex.Lock()
					packedErr = fmt.Errorf("error in gradient-input multiplication for block %d: %v", blockIdx, err)
					errMutex.Unlock()
					return
				}

				// 5. Scale by learning rate
				if err := heContext.evaluator.Mul(gradCopy, lrPt, gradCopy); err != nil {
					errMutex.Lock()
					packedErr = fmt.Errorf("error in learning rate scaling for block %d: %v", blockIdx, err)
					errMutex.Unlock()
					return
				}

				// 6. Add to the weights
				if err := heContext.evaluator.Add(heServer.W[l][i][blockIdx], gradCopy, heServer.W[l][i][blockIdx]); err != nil {
					errMutex.Lock()
					packedErr = fmt.Errorf("error updating weights for block %d: %v", blockIdx, err)
					errMutex.Unlock()
					return
				}
			}

			// Update biases
			// 1. Create a copy of the gradient
			gradCopy := encGradients[blockIdx].CopyNew()

			// 2. Sum across batch dimension
			summedGrad, err := sumSlotsWithRotations(heContext, gradCopy, batchSize)
			if err != nil {
				errMutex.Lock()
				packedErr = fmt.Errorf("error in inner sum for biases in block %d: %v", blockIdx, err)
				errMutex.Unlock()
				return
			}

			// 3. Scale by learning rate
			if err := heContext.evaluator.Mul(summedGrad, lrPt, summedGrad); err != nil {
				errMutex.Lock()
				packedErr = fmt.Errorf("error in learning rate scaling for biases in block %d: %v", blockIdx, err)
				errMutex.Unlock()
				return
			}

			// 4. Add to the biases
			if err := heContext.evaluator.Add(heServer.b[l][blockIdx], summedGrad, heServer.b[l][blockIdx]); err != nil {
				errMutex.Lock()
				packedErr = fmt.Errorf("error updating biases for block %d: %v", blockIdx, err)
				errMutex.Unlock()
				return
			}
		}(b)
	}

	wg.Wait()

	if packedErr != nil {
		return packedErr
	}

	// Update the plaintext model from the homomorphic model
	// This is necessary because we need to pass the updated weights back to the caller
	updateModelFromHE(heContext, serverModel, heServer, l, batchSize)

	return nil
}

// updateModelFromHE updates the plaintext server model from the homomorphic model
func updateModelFromHE(heContext *HEContext, serverModel *ServerModel, heServer *HEServerPacked, layer int, batchSize int) {
	outputDim := serverModel.GetLayerOutputDim(layer)
	inputDim := serverModel.GetLayerInputDim(layer)
	neuronsPerCT := heServer.NeuronsPerCT
	numBlocks := (outputDim + neuronsPerCT - 1) / neuronsPerCT

	// For each block of neurons
	for b := 0; b < numBlocks; b++ {
		// Get the neurons in this block
		startNeuron := b * neuronsPerCT
		endNeuron := min(startNeuron+neuronsPerCT, outputDim)

		// For each input dimension
		for i := 0; i < inputDim; i++ {
			// Decrypt the weights
			pt := heContext.decryptor.DecryptNew(heServer.W[layer][i][b])

			// Decode
			values := make([]float64, heContext.params.N()/2)
			heContext.encoder.Decode(pt, values)

			// Update the server model weights
			for n := startNeuron; n < endNeuron; n++ {
				if n < outputDim {
					newWeight := values[(n-startNeuron)*batchSize]
					serverModel.SetWeight(layer, i, n, newWeight)
				}
			}
		}

		// Decrypt the biases
		pt := heContext.decryptor.DecryptNew(heServer.b[layer][b])

		// Decode
		values := make([]float64, heContext.params.N()/2)
		heContext.encoder.Decode(pt, values)

		// Update the server model biases
		for n := startNeuron; n < endNeuron; n++ {
			if n < outputDim {
				serverModel.Biases[layer][n] = values[(n-startNeuron)*batchSize]
			}
		}
	}
}
