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

	// With our optimized server forward pass, we're currently handling one example at a time
	batchSize := 1

	// Check if activations are valid
	if len(encActivations) == 0 {
		return nil, fmt.Errorf("invalid input: empty encActivations")
	}

	// Step 1: Receive & Decrypt
	// With our optimized approach, we have blk ciphertexts, each containing NeuronsPerCT neurons
	neurons_per_block := HiddenDim1 / NeuronsPerCT
	a1 := make([][]float64, HiddenDim1)

	// Process each block of neurons
	for b := 0; b < neurons_per_block; b++ {
		// Decrypt the ciphertext for this block
		pt := heContext.decryptor.DecryptNew(encActivations[b])

		// Decode the plaintext - in our optimized version, we get values in slots
		decoded := make([]float64, heContext.params.N()/2)
		heContext.encoder.Decode(pt, decoded)

		// Extract values for each neuron in this block
		for n := 0; n < NeuronsPerCT; n++ {
			neuronIdx := b*NeuronsPerCT + n
			if neuronIdx < HiddenDim1 {
				// Just extract the first value for now
				a1[neuronIdx] = []float64{decoded[n*BatchSize]}
			}
		}
	}

	// Transpose a1 to have shape [batchSize x hiddenDim1]
	a1Transposed := make([][]float64, batchSize)
	for i := range a1Transposed {
		a1Transposed[i] = make([]float64, HiddenDim1)
		for j := 0; j < HiddenDim1; j++ {
			a1Transposed[i][j] = a1[j][0]
		}
	}

	// Step 2: Forward Pass through Client's Layers
	// Compute a2 = ReLU(a1 * W2 + b2)
	a2 := make([][]float64, batchSize)
	for i := range a2 {
		a2[i] = make([]float64, HiddenDim2)

		// Compute linear combination
		for j := 0; j < HiddenDim2; j++ {
			a2[i][j] = clientModel.b2[j]
			for k := 0; k < HiddenDim1; k++ {
				a2[i][j] += a1Transposed[i][k] * clientModel.W2[k][j]
			}
		}

		// Apply ReLU
		for j := range a2[i] {
			if a2[i][j] < 0 {
				a2[i][j] = 0
			}
		}
	}

	// Compute a3 = a2 * W3 + b3
	a3 := make([][]float64, batchSize)
	for i := range a3 {
		a3[i] = make([]float64, OutputDim)

		// Compute linear combination
		for j := 0; j < OutputDim; j++ {
			a3[i][j] = clientModel.b3[j]
			for k := 0; k < HiddenDim2; k++ {
				a3[i][j] += a2[i][k] * clientModel.W3[k][j]
			}
		}
	}

	// Step 3: Compute Loss (Cross-Entropy Loss)
	// Apply softmax and compute cross-entropy loss
	loss := 0.0

	// For tracking softmax values for gradient computation
	softmaxValues := make([][]float64, batchSize)

	for i := 0; i < batchSize; i++ {
		// Apply softmax: exp(a3) / sum(exp(a3))
		maxVal := a3[i][0]
		for j := 1; j < OutputDim; j++ {
			if a3[i][j] > maxVal {
				maxVal = a3[i][j]
			}
		}

		expSum := 0.0
		expValues := make([]float64, OutputDim)
		for j := 0; j < OutputDim; j++ {
			expValues[j] = math.Exp(a3[i][j] - maxVal)
			expSum += expValues[j]
		}

		// Compute softmax
		softmaxValues[i] = make([]float64, OutputDim)
		for j := 0; j < OutputDim; j++ {
			softmaxValues[i][j] = expValues[j] / expSum
		}

		// Cross-entropy loss: -sum(y_true * log(y_pred))
		// For one-hot encoded labels, this simplifies to -log(y_pred[true_label])
		trueLabel := labels[batchIndices[i]]
		loss += -math.Log(softmaxValues[i][trueLabel])
	}
	loss /= float64(batchSize)

	// Step 4: Compute Gradients (Backward)
	// Start with output layer gradients
	dZ3 := make([][]float64, batchSize)
	for i := range dZ3 {
		dZ3[i] = make([]float64, OutputDim)
		for j := 0; j < OutputDim; j++ {
			// Derivative of softmax cross-entropy is (softmax - one_hot_true)
			if j == labels[batchIndices[i]] {
				// Softmax - 1.0 for true class
				dZ3[i][j] = softmaxValues[i][j] - 1.0
			} else {
				// Softmax for other classes
				dZ3[i][j] = softmaxValues[i][j]
			}
		}
	}

	// Compute gradients for output layer weights
	dW3 := make([][]float64, HiddenDim2)
	for i := range dW3 {
		dW3[i] = make([]float64, OutputDim)
		for j := 0; j < OutputDim; j++ {
			for k := 0; k < batchSize; k++ {
				dW3[i][j] += a2[k][i] * dZ3[k][j]
			}
			dW3[i][j] /= float64(batchSize)
		}
	}

	// Compute gradients for output layer biases
	db3 := make([]float64, OutputDim)
	for i := 0; i < OutputDim; i++ {
		for j := 0; j < batchSize; j++ {
			db3[i] += dZ3[j][i]
		}
		db3[i] /= float64(batchSize)
	}

	// Compute gradients for hidden layer
	dZ2 := make([][]float64, batchSize)
	for i := range dZ2 {
		dZ2[i] = make([]float64, HiddenDim2)
		for j := 0; j < HiddenDim2; j++ {
			for k := 0; k < OutputDim; k++ {
				dZ2[i][j] += dZ3[i][k] * clientModel.W3[j][k]
			}
			// ReLU derivative: 0 if input <= 0, 1 otherwise
			if a2[i][j] <= 0 {
				dZ2[i][j] = 0
			}
		}
	}

	// Compute gradients for hidden layer weights
	dW2 := make([][]float64, HiddenDim1)
	for i := range dW2 {
		dW2[i] = make([]float64, HiddenDim2)
		for j := 0; j < HiddenDim2; j++ {
			for k := 0; k < batchSize; k++ {
				dW2[i][j] += a1Transposed[k][i] * dZ2[k][j]
			}
			dW2[i][j] /= float64(batchSize)
		}
	}

	db2 := make([]float64, HiddenDim2)
	for i := 0; i < HiddenDim2; i++ {
		for j := 0; j < batchSize; j++ {
			db2[i] += dZ2[j][i]
		}
		db2[i] /= float64(batchSize)
	}

	// Compute gradients for first hidden layer
	dA1 := make([][]float64, batchSize)
	for i := range dA1 {
		dA1[i] = make([]float64, HiddenDim1)
		for j := 0; j < HiddenDim1; j++ {
			for k := 0; k < HiddenDim2; k++ {
				dA1[i][j] += dZ2[i][k] * clientModel.W2[j][k]
			}
		}
	}

	// Transpose dA1 to have shape [hiddenDim1 x batchSize]
	dA1Transposed := make([][]float64, HiddenDim1)
	for i := range dA1Transposed {
		dA1Transposed[i] = make([]float64, batchSize)
		for j := 0; j < batchSize; j++ {
			dA1Transposed[i][j] = dA1[j][i]
		}
	}

	// Step 5: Update Client's Weights
	// Update W3 and b3
	for i := 0; i < HiddenDim2; i++ {
		for j := 0; j < OutputDim; j++ {
			clientModel.W3[i][j] -= LearningRate * dW3[i][j]
		}
	}

	for i := 0; i < OutputDim; i++ {
		clientModel.b3[i] -= LearningRate * db3[i]
	}

	// Update W2 and b2
	for i := 0; i < HiddenDim1; i++ {
		for j := 0; j < HiddenDim2; j++ {
			clientModel.W2[i][j] -= LearningRate * dW2[i][j]
		}
	}

	for i := 0; i < HiddenDim2; i++ {
		clientModel.b2[i] -= LearningRate * db2[i]
	}

	// ---- New Step 6: pack dA1 into 2 ciphertexts ----      ★ SIMD-weights
	neurons_per_block = HiddenDim1 / NeuronsPerCT // =2
	encGradBlk := make([]*rlwe.Ciphertext, neurons_per_block)

	// Reuse this buffer across iterations - memory optimization
	scratch := make([]float64, heContext.params.N()/2)

	for b := 0; b < neurons_per_block; b++ {
		// *** clear the buffer – otherwise data from the previous
		//     block would survive in the tail ***
		for i := range scratch {
			scratch[i] = 0
		}

		for n := 0; n < NeuronsPerCT; n++ {
			// We pack the gradient for each neuron across all examples
			copy(scratch[n*batchSize:(n+1)*batchSize],
				dA1Transposed[b*NeuronsPerCT+n])
		}

		pt := ckks.NewPlaintext(heContext.params, heContext.params.MaxLevel())
		heContext.encoder.Encode(scratch, pt)

		ct, err := heContext.encryptor.EncryptNew(pt)
		if err != nil {
			return nil, fmt.Errorf("encrypt packed grad: %v", err)
		}
		encGradBlk[b] = ct
	}
	return encGradBlk, nil
}

// Server performs backward pass and updates weights
func serverBackwardAndUpdate(heContext *HEContext, serverModel *ServerModel, encGradients []*rlwe.Ciphertext,
	cachedInputs []*rlwe.Ciphertext, learningRate float64) error {
	// Convert the standard server model to a packed homomorphic one
	heServer, err := convertToPacked(serverModel, heContext)
	if err != nil {
		return fmt.Errorf("failed to convert to packed homomorphic model: %v", err)
	}

	fmt.Println("Performing weight update...")

	// Use the cached inputs from the forward pass instead of creating placeholders
	// This is a critical fix - using real inputs for accurate weight updates

	// Use packedUpdate for SIMD optimization
	err = packedUpdate(heContext, heServer, cachedInputs, encGradients, learningRate, BatchSize)
	if err != nil {
		return fmt.Errorf("homomorphic backward error: %v", err)
	}

	// For demonstration, decrypt the updated weights
	blocks_count := HiddenDim1 / NeuronsPerCT
	for i := 0; i < InputDim; i++ {
		for b := 0; b < blocks_count; b++ {
			// Decrypt packed weight
			ptWeight := heContext.decryptor.DecryptNew(heServer.W[i][b])

			// Decode all slots
			values := make([]float64, heContext.params.N()/2)
			heContext.encoder.Decode(ptWeight, values)

			// Extract the weight values (first slot of each batch)
			for n := 0; n < NeuronsPerCT; n++ {
				neuronIdx := b*NeuronsPerCT + n
				if neuronIdx < HiddenDim1 {
					serverModel.W1[i][neuronIdx] = values[n*BatchSize]
				}
			}
		}
	}

	// Update biases
	for b := 0; b < blocks_count; b++ {
		// Decrypt packed bias
		ptBias := heContext.decryptor.DecryptNew(heServer.b[b])

		// Decode all slots
		values := make([]float64, heContext.params.N()/2)
		heContext.encoder.Decode(ptBias, values)

		// Extract the bias values (first slot of each batch)
		for n := 0; n < NeuronsPerCT; n++ {
			neuronIdx := b*NeuronsPerCT + n
			if neuronIdx < HiddenDim1 {
				serverModel.b1[neuronIdx] = values[n*BatchSize]
			}
		}
	}

	return nil
}

// serverBackwardAndUpdateHEParallel updates W1,b1 in-ciphertext using 'workers' goroutines
func serverBackwardAndUpdateHEParallel(
	ctx *HEContext,
	svr *HEServerModel,
	encX []*rlwe.Ciphertext, // len=inputDim
	encGradZ []*rlwe.Ciphertext, // len=hiddenDim1
	lr float64,
	B int,
	workers int,
) error {
	// Helper to capture first error
	var firstErr error
	var errMu sync.Mutex
	recordErr := func(e error) {
		errMu.Lock()
		if firstErr == nil {
			firstErr = e
		}
		errMu.Unlock()
	}

	if workers < 1 {
		workers = 1
	}
	params := ctx.params
	etaPt := scalarPlain(-lr/float64(B), params, ctx.encoder)

	rowsPerWkr := (HiddenDim1 + workers - 1) / workers
	var wg sync.WaitGroup

	for w := 0; w < workers; w++ {
		jStart := w * rowsPerWkr
		jEnd := min(HiddenDim1, jStart+rowsPerWkr)
		if jStart >= jEnd {
			continue
		}

		// each goroutine gets its own evaluator clone (thread-safe)
		eval := ctx.evaluator.ShallowCopy()

		wg.Add(1)
		go func(js, je int) {
			defer wg.Done()
			for j := js; j < je; j++ {
				// ------- bias update -------
				sumSlots, e := innerSumSlots(encGradZ[j], B, eval)
				if e != nil {
					recordErr(e)
					return
				}

				if e = eval.Mul(sumSlots, etaPt, sumSlots); e != nil {
					recordErr(e)
					return
				}
				if e = eval.Add(svr.b1[j], sumSlots, svr.b1[j]); e != nil {
					recordErr(e)
					return
				}

				// ------- weight updates over inputDim -------
				for i := 0; i < InputDim; i++ {
					prod, e := eval.MulNew(encX[i], encGradZ[j])
					if e != nil {
						recordErr(e)
						return
					}
					if e = eval.Relinearize(prod, prod); e != nil {
						recordErr(e)
						return
					}
					if e = eval.Mul(prod, etaPt, prod); e != nil {
						recordErr(e)
						return
					}
					if e = eval.Add(svr.W1[i][j], prod, svr.W1[i][j]); e != nil {
						recordErr(e)
						return
					}
				}
			}
		}(jStart, jEnd)
	}
	wg.Wait()
	return firstErr
}

// packedUpdate applies homomorphic weight updates using the efficient packed SIMD approach
func packedUpdate(heContext *HEContext, heServer *HEServerPacked, encInputs []*rlwe.Ciphertext, encGradients []*rlwe.Ciphertext, learningRate float64, batchSize int) error {
	eval := heContext.evaluator
	// Cache the learning rate constant plain
	ptLR := scalarPlain(-learningRate/float64(batchSize), heContext.params, heContext.encoder)

	// Cache reusable constants and operations
	// These don't need to be recalculated for each weight
	slots := heContext.params.N() / 2

	// Process weights first - use the single packed input to update all weights
	for i := 0; i < InputDim; i++ {
		// Get the single packed input
		encInput := encInputs[0]

		// For each output neuron group (NeuronsPerCT neurons per group)
		output_blocks := HiddenDim1 / NeuronsPerCT
		for b := 0; b < output_blocks; b++ {
			// Create a temporary ciphertext for the gradients
			tempGrad := encGradients[b].CopyNew()

			// Scale the gradient by -learningRate/batchSize
			if err := eval.Mul(tempGrad, ptLR, tempGrad); err != nil {
				return fmt.Errorf("error scaling gradient: %v", err)
			}

			// Create a temporary ciphertext for the input
			tempInput := encInput.CopyNew()

			// Multiply the input by the scaled gradient to get the weight update
			if err := eval.Mul(tempInput, tempGrad, tempInput); err != nil {
				return fmt.Errorf("error multiplying for weight update: %v", err)
			}

			// Relinearize the result
			if err := eval.Relinearize(tempInput, tempInput); err != nil {
				return fmt.Errorf("error relinearizing weight update: %v", err)
			}

			// Compute the inner sum to get the weight update
			weightUpdate, err := innerSumSlots(tempInput, slots, eval)
			if err != nil {
				return fmt.Errorf("error computing inner sum: %v", err)
			}

			// Apply the update to the weights
			if err := eval.Add(heServer.W[i][b], weightUpdate, heServer.W[i][b]); err != nil {
				return fmt.Errorf("error applying weight update: %v", err)
			}
		}
	}

	// Update biases using the gradient directly
	bias_blocks := HiddenDim1 / NeuronsPerCT
	for b := 0; b < bias_blocks; b++ {
		// Create a temporary ciphertext for the bias update
		tempBias := encGradients[b].CopyNew()

		// Scale the gradient by -learningRate/batchSize
		if err := eval.Mul(tempBias, ptLR, tempBias); err != nil {
			return fmt.Errorf("error scaling bias gradient: %v", err)
		}

		// Compute the inner sum to get the bias update
		biasUpdate, err := innerSumSlots(tempBias, slots, eval)
		if err != nil {
			return fmt.Errorf("error computing inner sum for bias: %v", err)
		}

		// Apply the update to the biases
		if err := eval.Add(heServer.b[b], biasUpdate, heServer.b[b]); err != nil {
			return fmt.Errorf("error applying bias update: %v", err)
		}
	}

	return nil
}
