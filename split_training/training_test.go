package split

import (
	"testing"
)

func TestSimpleTraining(t *testing.T) {
	// Initialize HE context
	heContext, err := initHE()
	if err != nil {
		t.Fatalf("Failed to initialize HE context: %v", err)
	}

	// Create dummy data - 2 small images (3x3 instead of 28x28)
	// We'll just use 9 pixels for testing
	smallInputDim := 9
	images := make([][]float64, 2)
	for i := range images {
		images[i] = make([]float64, smallInputDim)
		for j := range images[i] {
			images[i][j] = float64(i+j) / 10.0 // Simple pattern
		}
	}
	labels := []int{0, 1} // Simple labels

	// Create smaller models for testing
	clientModel := &ClientModel{
		W2: make([][]float64, HiddenDim1),
		b2: make([]float64, HiddenDim2),
		W3: make([][]float64, HiddenDim2),
		b3: make([]float64, OutputDim),
	}

	// Initialize with simple values
	for i := range clientModel.W2 {
		clientModel.W2[i] = make([]float64, HiddenDim2)
		for j := range clientModel.W2[i] {
			clientModel.W2[i][j] = 0.1
		}
	}
	for i := range clientModel.b2 {
		clientModel.b2[i] = 0.01
	}
	for i := range clientModel.W3 {
		clientModel.W3[i] = make([]float64, OutputDim)
		for j := range clientModel.W3[i] {
			clientModel.W3[i][j] = 0.1
		}
	}
	for i := range clientModel.b3 {
		clientModel.b3[i] = 0.01
	}

	// Server model
	serverModel := &ServerModel{
		W1: make([][]float64, InputDim),
		b1: make([]float64, HiddenDim1),
	}

	// Initialize with simple values
	for i := range serverModel.W1 {
		serverModel.W1[i] = make([]float64, HiddenDim1)
		for j := range serverModel.W1[i] {
			serverModel.W1[i][j] = 0.1
		}
	}
	for i := range serverModel.b1 {
		serverModel.b1[i] = 0.01
	}

	// Test batch indices
	batchIndices := []int{0}

	// Run both training methods
	t.Log("Testing standard training...")
	trainBatchWithTiming(heContext, clientModel, serverModel, images, labels, batchIndices, 0.01, nil)

	t.Log("Testing fully homomorphic training...")
	err = trainBatchFullHomomorphic(heContext, clientModel, serverModel, images, labels, batchIndices, 0.01)
	if err != nil {
		t.Fatalf("Error in fully homomorphic training: %v", err)
	}

	t.Log("Both training methods completed successfully")
}
