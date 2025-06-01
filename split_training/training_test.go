package split

import (
	"testing"
)

// setTestBatchSize sets the global BatchSize for a test and returns a function to restore the original value
func setTestBatchSize(size int) func() {
	original := BatchSize
	BatchSize = size
	return func() {
		BatchSize = original
	}
}

func TestSimpleTraining(t *testing.T) {
	// Set batch size for this test
	restore := setTestBatchSize(2)
	defer restore()

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

	// Create configuration with consistent hidden layer dimensions
	config := &ModelConfig{
		Arch:     []int{smallInputDim, 32, 32, OutputDim},
		SplitIdx: 1, // Split after the first hidden layer
	}

	// Create client and server models using the initialization functions
	// which will allocate the correct dimensions based on the configuration
	clientModel := initClientModel(config)
	serverModel := initServerModel(config)

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

func TestServerWithFirstLayerBatchPerCiphertext(t *testing.T) {
	// Set batch size for this test
	restore := setTestBatchSize(2)
	defer restore()

	// Initialize HE context
	heContext, err := initHE()
	if err != nil {
		t.Fatalf("Failed to initialize HE context: %v", err)
	}

	// Use real MNIST data but only the first few images
	images, labels, _, _, err := readMNISTData()
	if err != nil {
		t.Fatalf("Failed to load MNIST data: %v", err)
	}

	// Create configuration - server has the first layer and activation
	// Use HiddenDim2=32 for both hidden layers to avoid dimension mismatch warnings
	config := &ModelConfig{
		Arch:     []int{MnistPixels, 32, 32, OutputDim},
		SplitIdx: 1, // Server has 1 layer (input -> hidden1)
	}

	// Create client and server models
	clientModel := initClientModel(config)
	serverModel := initServerModel(config)

	// Test with very small batch size to ensure each image fits in a ciphertext
	batchSize := BatchSize
	batchIndices := []int{0, 1}

	// Test the forward pass first to isolate issues
	t.Log("Testing forward pass with batch per ciphertext...")

	// Prepare and encrypt the batch
	encInputs, err := clientPrepareAndEncryptBatch(heContext, images, batchIndices)
	if err != nil {
		t.Fatalf("Failed to prepare and encrypt batch: %v", err)
	}

	// Verify we have the right number of ciphertexts (one per image)
	if len(encInputs) != batchSize {
		t.Fatalf("Expected %d ciphertexts, got %d", batchSize, len(encInputs))
	}

	// Run server forward pass
	_, encActivations, err := ServerForwardPassWithLayerInputs(heContext, serverModel, encInputs)
	if err != nil {
		t.Fatalf("Failed in server forward pass: %v", err)
	}

	// Run client forward and backward pass
	_, err = clientForwardAndBackward(heContext, clientModel, encActivations, labels, batchIndices)
	if err != nil {
		t.Fatalf("Failed in client forward and backward: %v", err)
	}

	// Run full training batch to ensure everything works together
	t.Log("Testing full training with batch per ciphertext...")
	trainBatchWithTiming(heContext, clientModel, serverModel, images, labels, batchIndices, 0.01, nil)

	t.Log("Training completed successfully with batch per ciphertext")
}

func TestSimpleWeightStructure(t *testing.T) {
	// Set batch size for this test
	restore := setTestBatchSize(2)
	defer restore()

	// Initialize HE context
	heContext, err := initHE()
	if err != nil {
		t.Fatalf("Failed to initialize HE context: %v", err)
	}

	// Create a very small configuration with just a few neurons
	config := &ModelConfig{
		Arch:     []int{4, 2, 2}, // Tiny architecture for testing
		SplitIdx: 1,              // Server has 1 layer
	}

	// Create tiny client model
	clientModel := &ClientModel{
		Weights: make([][][]float64, 1), // Only one layer
		Biases:  make([][]float64, 1),
		Config:  config,
	}

	// Initialize client layer weights
	clientModel.Weights[0] = make([][]float64, 2) // 2 inputs from the server
	clientModel.Weights[0][0] = make([]float64, 2)
	clientModel.Weights[0][1] = make([]float64, 2)
	// Set some values
	clientModel.Weights[0][0][0] = 0.1
	clientModel.Weights[0][0][1] = 0.2
	clientModel.Weights[0][1][0] = 0.3
	clientModel.Weights[0][1][1] = 0.4

	// Initialize client biases
	clientModel.Biases[0] = make([]float64, 2)
	clientModel.Biases[0][0] = 0.01
	clientModel.Biases[0][1] = 0.02

	// Create tiny server model
	serverModel := &ServerModel{
		Weights: make([][][]float64, 1), // Only one layer
		Biases:  make([][]float64, 1),
		Config:  config,
	}

	// Initialize server layer weights
	serverModel.Weights[0] = make([][]float64, 4) // 4 inputs
	for i := range serverModel.Weights[0] {
		serverModel.Weights[0][i] = make([]float64, 2)
		for j := range serverModel.Weights[0][i] {
			serverModel.Weights[0][i][j] = 0.1 * float64(i+j+1)
		}
	}

	// Initialize server biases
	serverModel.Biases[0] = make([]float64, 2)
	serverModel.Biases[0][0] = 0.01
	serverModel.Biases[0][1] = 0.02

	// Create very simple 4-pixel "images"
	images := make([][]float64, 2)
	for i := range images {
		images[i] = make([]float64, 4)
		for j := range images[i] {
			images[i][j] = float64(i+j) / 10.0
		}
	}
	labels := []int{0, 1}

	batchIndices := []int{0, 1}

	// Test the forward pass
	t.Log("Testing weight structure with simple network...")

	// Prepare and encrypt the batch
	encInputs, err := clientPrepareAndEncryptBatch(heContext, images, batchIndices)
	if err != nil {
		t.Fatalf("Failed to prepare and encrypt batch: %v", err)
	}

	// Run server forward pass
	_, encActivations, err := ServerForwardPassWithLayerInputs(heContext, serverModel, encInputs)
	if err != nil {
		t.Fatalf("Failed in server forward pass: %v", err)
	}

	// Run client forward and backward pass
	_, err = clientForwardAndBackward(heContext, clientModel, encActivations, labels, batchIndices)
	if err != nil {
		t.Fatalf("Failed in client forward and backward: %v", err)
	}

	t.Log("Simple weight structure test passed")
}

func TestModelDimensions(t *testing.T) {
	// Create a test configuration
	config := &ModelConfig{
		Arch:     []int{784, 128, 64, 10}, // MNIST dimensions
		SplitIdx: 1,                       // Server has first layer
	}

	// Initialize client and server models
	clientModel := initClientModel(config)
	serverModel := initServerModel(config)

	// Check server model dimensions
	if len(serverModel.Weights) != 1 {
		t.Errorf("Expected server to have 1 layer, got %d", len(serverModel.Weights))
	}

	if len(serverModel.Weights[0]) != 784 {
		t.Errorf("Expected server input dimension to be 784, got %d", len(serverModel.Weights[0]))
	}

	if len(serverModel.Weights[0][0]) != 128 {
		t.Errorf("Expected server output dimension to be 128, got %d", len(serverModel.Weights[0][0]))
	}

	// Check client model dimensions
	if len(clientModel.Weights) != 2 {
		t.Errorf("Expected client to have 2 layers, got %d", len(clientModel.Weights))
	}

	if len(clientModel.Weights[0]) != 128 {
		t.Errorf("Expected client first layer input dimension to be 128, got %d", len(clientModel.Weights[0]))
	}

	if len(clientModel.Weights[0][0]) != 64 {
		t.Errorf("Expected client first layer output dimension to be 64, got %d", len(clientModel.Weights[0][0]))
	}

	if len(clientModel.Weights[1]) != 64 {
		t.Errorf("Expected client second layer input dimension to be 64, got %d", len(clientModel.Weights[1]))
	}

	if len(clientModel.Weights[1][0]) != 10 {
		t.Errorf("Expected client second layer output dimension to be 10, got %d", len(clientModel.Weights[1][0]))
	}

	// Verify GetLayerInputDim and GetLayerOutputDim match actual dimensions
	for l := 0; l < len(serverModel.Weights); l++ {
		inputDim := serverModel.GetLayerInputDim(l)
		if inputDim != len(serverModel.Weights[l]) {
			t.Errorf("Server layer %d: GetLayerInputDim returned %d but weights dimension is %d",
				l, inputDim, len(serverModel.Weights[l]))
		}

		outputDim := serverModel.GetLayerOutputDim(l)
		if outputDim != len(serverModel.Weights[l][0]) {
			t.Errorf("Server layer %d: GetLayerOutputDim returned %d but weights dimension is %d",
				l, outputDim, len(serverModel.Weights[l][0]))
		}
	}

	for l := 0; l < len(clientModel.Weights); l++ {
		inputDim := clientModel.GetLayerInputDim(l)
		if inputDim != len(clientModel.Weights[l]) {
			t.Errorf("Client layer %d: GetLayerInputDim returned %d but weights dimension is %d",
				l, inputDim, len(clientModel.Weights[l]))
		}

		outputDim := clientModel.GetLayerOutputDim(l)
		if outputDim != len(clientModel.Weights[l][0]) {
			t.Errorf("Client layer %d: GetLayerOutputDim returned %d but weights dimension is %d",
				l, outputDim, len(clientModel.Weights[l][0]))
		}
	}

	t.Log("Model dimensions match configuration as expected")
}

// TestConfigurableNetwork tests a server model with multiple layers
func TestConfigurableNetwork(t *testing.T) {
	// Set batch size for this test
	restore := setTestBatchSize(2)
	defer restore()

	// Initialize HE context
	heContext, err := initHE()
	if err != nil {
		t.Fatalf("Failed to initialize HE context: %v", err)
	}

	// Create a configuration with multiple server layers
	config := &ModelConfig{
		Arch:     []int{784, 64, 32, 16, 10}, // Deeper architecture for testing
		SplitIdx: 3,                          // Server has 3 layers (784->64->32->16), client has 1 (16->10)
	}

	// Initialize client and server models
	clientModel := initClientModel(config)
	serverModel := initServerModel(config)

	// Verify server model has the correct number of layers
	if len(serverModel.Weights) != 3 {
		t.Errorf("Expected server to have 3 layers, got %d", len(serverModel.Weights))
	}

	// Verify client model has the correct number of layers
	if len(clientModel.Weights) != 1 {
		t.Errorf("Expected client to have 1 layer, got %d", len(clientModel.Weights))
	}

	// Create sample data
	numSamples := 4
	images := make([][]float64, numSamples)
	for i := range images {
		images[i] = make([]float64, 784)
		for j := range images[i] {
			images[i][j] = float64(i+j) / 1000.0
		}
	}
	labels := []int{0, 1, 2, 3}
	batchIndices := []int{0, 1}

	// Test forward pass with layered inputs
	t.Log("Testing forward pass with multiple server layers...")

	// Prepare and encrypt the batch
	encInputs, err := clientPrepareAndEncryptBatch(heContext, images, batchIndices)
	if err != nil {
		t.Fatalf("Failed to prepare and encrypt batch: %v", err)
	}

	// Run server forward pass and verify we get cached layer inputs
	layerInputs, encActivations, err := ServerForwardPassWithLayerInputs(heContext, serverModel, encInputs)
	if err != nil {
		t.Fatalf("Failed in server forward pass: %v", err)
	}

	// Verify we have the correct number of layer inputs (numLayers + 1)
	expectedLayers := len(serverModel.Weights) + 1
	if len(layerInputs) != expectedLayers {
		t.Fatalf("Expected %d layer inputs, got %d", expectedLayers, len(layerInputs))
	}

	// Verify first layer input is the original input
	if len(layerInputs[0]) != len(encInputs) {
		t.Fatalf("First layer input should match original input: expected %d, got %d",
			len(encInputs), len(layerInputs[0]))
	}

	// Run client forward and backward pass
	encGradients, err := clientForwardAndBackward(heContext, clientModel, encActivations, labels, batchIndices)
	if err != nil {
		t.Fatalf("Failed in client forward and backward: %v", err)
	}

	// Make a copy of the original weights for comparison
	originalWeights := make([][][]float64, len(serverModel.Weights))
	for l := range serverModel.Weights {
		originalWeights[l] = make([][]float64, len(serverModel.Weights[l]))
		for i := range serverModel.Weights[l] {
			originalWeights[l][i] = make([]float64, len(serverModel.Weights[l][i]))
			copy(originalWeights[l][i], serverModel.Weights[l][i])
		}
	}

	// Run server backward pass with layered inputs
	err = serverBackwardAndUpdate(heContext, serverModel, encGradients, layerInputs, 0.01)
	if err != nil {
		t.Fatalf("Failed in server backward and update: %v", err)
	}

	// Verify weights have been updated for all layers
	weightsChanged := false
	for l := range serverModel.Weights {
		for i := range serverModel.Weights[l] {
			for j := range serverModel.Weights[l][i] {
				if serverModel.Weights[l][i][j] != originalWeights[l][i][j] {
					weightsChanged = true
					break
				}
			}
		}
	}

	if !weightsChanged {
		t.Fatalf("Server weights were not updated after backward pass")
	}

	t.Log("Configurable network test passed - multiple server layers successfully updated")
}
