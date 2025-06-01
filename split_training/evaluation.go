package split

import (
	"fmt"
)

// Evaluates the model on test data and returns accuracy
func evaluateModel(heContext *HEContext, clientModel *ClientModel, serverModel *ServerModel,
	images [][]float64, labels []int) float64 {

	fmt.Println("Evaluating model on test data...")

	// Parameters for evaluation
	numExamples := 20 // Only evaluate on a small subset for testing
	correct := 0

	// Process individual examples
	for i := 0; i < numExamples; i++ {
		fmt.Printf("  Evaluating example %d/%d\n", i+1, numExamples)

		// Create indices array with just this example
		indices := []int{i}

		// Client prepares and encrypts the single image
		encInputs, err := clientPrepareAndEncryptBatch(heContext, images, indices)
		if err != nil {
			fmt.Printf("Error in client preparation: %v\n", err)
			continue
		}

		// Server performs forward pass
		_, encActivations, err := ServerForwardPassWithLayerInputs(heContext, serverModel, encInputs)
		if err != nil {
			fmt.Printf("Error in server forward pass: %v\n", err)
			continue
		}

		// Client performs forward pass to get predictions
		predictions, _, err := PerformEvaluation(heContext, clientModel, encActivations, 1)
		if err != nil {
			fmt.Printf("Error in client evaluation: %v\n", err)
			continue
		}

		// Check if prediction is correct
		if predictions[0] == labels[i] {
			correct++
		}
	}

	// Calculate accuracy
	accuracy := float64(correct) / float64(numExamples)
	return accuracy
}

// EvaluateModelOnBatch evaluates the model on a batch of test examples
func EvaluateModelOnBatch(heContext *HEContext, clientModel *ClientModel, serverModel *ServerModel,
	images [][]float64, labels []int, batchIndices []int) (int, error) {

	batchSize := len(batchIndices)
	if batchSize == 0 {
		return 0, fmt.Errorf("empty batch")
	}

	// Client prepares and encrypts the batch of images
	encInputs, err := clientPrepareAndEncryptBatch(heContext, images, batchIndices)
	if err != nil {
		return 0, fmt.Errorf("error in client preparation: %v", err)
	}

	// Server performs forward pass
	_, encActivations, err := ServerForwardPassWithLayerInputs(heContext, serverModel, encInputs)
	if err != nil {
		return 0, fmt.Errorf("error in server forward pass: %v", err)
	}

	// Client performs forward pass to get predictions
	predictions, _, err := PerformEvaluation(heContext, clientModel, encActivations, batchSize)
	if err != nil {
		return 0, fmt.Errorf("error in client evaluation: %v", err)
	}

	// Count correct predictions
	correct := 0
	for i := 0; i < batchSize; i++ {
		if predictions[i] == labels[batchIndices[i]] {
			correct++
		}
	}

	return correct, nil
}
