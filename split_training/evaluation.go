package split

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// Evaluates the model on test data and returns accuracy
func evaluateModel(heContext *HEContext, clientModel *ClientModel, serverModel *ServerModel,
	images [][]float64, labels []int) float64 {

	fmt.Println("Evaluating model on test data...")

	// Parameters
	evalBatchSize := 8 // Use smaller batch size for evaluation
	numBatches := 10   // Only evaluate on a small subset for testing
	correct := 0

	// Process batches
	for batch := 0; batch < numBatches; batch++ {
		fmt.Printf("  Evaluating batch %d/%d\n", batch+1, numBatches)

		// Get batch indices
		startIdx := batch * evalBatchSize
		batchIndices := make([]int, evalBatchSize)
		for i := range batchIndices {
			batchIndices[i] = startIdx + i
		}

		// Client prepares and encrypts inputs
		encInputs, err := clientPrepareAndEncryptBatch(heContext, images, batchIndices)
		if err != nil {
			fmt.Printf("Error in client preparation: %v\n", err)
			continue
		}

		// Server performs forward pass
		encActivations, err := serverForwardPass(heContext, serverModel, encInputs)
		if err != nil {
			fmt.Printf("Error in server forward pass: %v\n", err)
			continue
		}

		// Client performs forward pass to get predictions
		predictions, err := clientEvaluateForwardPass(heContext, clientModel, encActivations, evalBatchSize)
		if err != nil {
			fmt.Printf("Error in client evaluation: %v\n", err)
			continue
		}

		// Count correct predictions
		for i := 0; i < evalBatchSize; i++ {
			if predictions[i] == labels[batchIndices[i]] {
				correct++
			}
		}
	}

	// Calculate accuracy
	accuracy := float64(correct) / float64(numBatches*evalBatchSize)
	return accuracy
}

// Client forward pass for evaluation (no backpropagation)
func clientEvaluateForwardPass(heContext *HEContext, clientModel *ClientModel, encActivations []*rlwe.Ciphertext, batchSize int) ([]int, error) {
	// Step 1: Receive & Decrypt
	a1 := make([][]float64, HiddenDim1)
	for i := 0; i < HiddenDim1; i++ {
		// Decrypt the ciphertext
		pt := heContext.decryptor.DecryptNew(encActivations[i])

		// Decode the plaintext
		decoded := make([]float64, batchSize)
		heContext.encoder.Decode(pt, decoded)

		a1[i] = decoded
	}

	// Transpose a1 to have shape [batchSize x hiddenDim1]
	a1Transposed := make([][]float64, batchSize)
	for i := range a1Transposed {
		a1Transposed[i] = make([]float64, HiddenDim1)
		for j := 0; j < HiddenDim1; j++ {
			a1Transposed[i][j] = a1[j][i]
		}
	}

	// Step 2: Forward Pass through Client's Second Layer
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

	// Step 3: Forward Pass through Client's Output Layer
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

	// Step 4: Get Predictions
	predictions := make([]int, batchSize)
	for i := 0; i < batchSize; i++ {
		// Find the index of the maximum value
		maxIdx := 0
		maxVal := a3[i][0]
		for j := 1; j < OutputDim; j++ {
			if a3[i][j] > maxVal {
				maxVal = a3[i][j]
				maxIdx = j
			}
		}
		predictions[i] = maxIdx
	}

	return predictions, nil
}
