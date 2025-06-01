package main

import (
	"fmt"
	"sync"
	"time"

	split "github.com/halilibrahimkanpak/cure_test/split_training"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Helper function for parallel execution
func parallelFor(start, end int, numWorkers int, fn func(int)) {
	if end <= start {
		return
	}

	if numWorkers <= 1 || end-start <= 1 {
		// Run sequentially for small ranges or single worker
		for i := start; i < end; i++ {
			fn(i)
		}
		return
	}

	// Adjust worker count if range is small
	if end-start < numWorkers {
		numWorkers = end - start
	}

	var wg sync.WaitGroup
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

func main() {
	fmt.Println("Loading MNIST data...")
	// Instead of loading real MNIST data, we'll create synthetic data for demonstration

	// Create synthetic data (784 pixels per image)
	totalSamples := 64
	trainImages := make([][]float64, totalSamples)
	trainLabels := make([]int, totalSamples)

	for i := 0; i < totalSamples; i++ {
		trainImages[i] = make([]float64, 784)
		// Fill with some pattern based on the label - ensure we stay in range [0,9]
		trainLabels[i] = i % 10

		for j := 0; j < 784; j++ {
			trainImages[i][j] = float64((i+j)%255) / 255.0
		}
	}

	// Initialize HE context
	fmt.Println("Initializing homomorphic encryption context...")
	heContext, err := split.InitHE()
	if err != nil {
		fmt.Printf("Error initializing HE context: %v\n", err)
		return
	}

	// Create model configuration with configurable architecture
	// Use a deeper server model with multiple layers
	config := &split.ModelConfig{
		Arch:     []int{784, 128, 64, 32, 10}, // Architecture with 3 server layers
		SplitIdx: 3,                           // Server has 3 layers (784->128->64->32), client has 1 (32->10)
	}

	// Initialize client and server models
	fmt.Println("Initializing models...")
	clientModel := split.InitClientModel(config)
	serverModel := split.InitServerModel(config)

	// Print model architecture
	fmt.Println("Server model architecture:")
	for l, layer := range serverModel.Weights {
		fmt.Printf("  Layer %d: %d inputs, %d outputs\n",
			l, len(layer), len(layer[0]))
	}

	fmt.Println("Client model architecture:")
	for l, layer := range clientModel.Weights {
		fmt.Printf("  Layer %d: %d inputs, %d outputs\n",
			l, len(layer), len(layer[0]))
	}

	// Prepare a small batch of data
	batchSize := 8 // Use 8 examples to better demonstrate batch processing
	numBatches := 1
	epochs := 1

	// Process only a subset of the data
	images := make([][]float64, batchSize*numBatches)
	labels := make([]int, batchSize*numBatches)

	for i := 0; i < batchSize*numBatches; i++ {
		images[i] = trainImages[i]
		labels[i] = trainLabels[i]
	}

	// Create a custom implementation for pixel-wise processing
	fmt.Println("Training model with configurable network architecture...")
	batchIndices := make([]int, batchSize)

	totalTrainingTime := time.Duration(0)
	totalEncryptTime := time.Duration(0)
	totalServerForwardTime := time.Duration(0)
	totalClientComputeTime := time.Duration(0)
	totalServerBackwardTime := time.Duration(0)

	// Learning rate
	learningRate := 0.01

	for epoch := 0; epoch < epochs; epoch++ {
		fmt.Printf("Epoch %d/%d\n", epoch+1, epochs)

		for batch := 0; batch < numBatches; batch++ {
			fmt.Printf("  Batch %d/%d\n", batch+1, numBatches)

			// Get indices for this batch
			startIdx := batch * batchSize
			for i := 0; i < batchSize; i++ {
				batchIndices[i] = startIdx + i
			}

			startTime := time.Now()

			// Phase 1: Client prepares batch
			encStart := time.Now()

			// Custom pixel-wise encryption for configurable networks
			// Create one ciphertext per pixel (784 ciphertexts)
			pixelCount := len(images[0]) // 784 for MNIST
			encPixels := make([]*rlwe.Ciphertext, pixelCount)

			// Mutex to protect access to error variable
			var mu sync.Mutex
			var encryptionError error

			// Parallel encryption of pixels
			numWorkers := 4 // Use 4 workers for parallel processing
			parallelFor(0, pixelCount, numWorkers, func(pixel int) {
				// Check if any errors occurred already
				mu.Lock()
				if encryptionError != nil {
					mu.Unlock()
					return
				}
				mu.Unlock()

				// Create a vector with this pixel's value for each image in the batch
				pixelValues := make([]float64, heContext.GetSlots())

				// Fill with zeros
				for i := range pixelValues {
					pixelValues[i] = 0
				}

				// Set the actual values for images in this batch
				for i, idx := range batchIndices {
					pixelValues[i] = images[idx][pixel]
				}

				// Encode and encrypt
				pt := ckks.NewPlaintext(heContext.GetParams(), heContext.GetParams().MaxLevel())
				heContext.GetEncoder().Encode(pixelValues, pt)

				ct, err := heContext.GetEncryptor().EncryptNew(pt)
				if err != nil {
					mu.Lock()
					encryptionError = err
					mu.Unlock()
					return
				}

				// Store the encrypted result
				mu.Lock()
				encPixels[pixel] = ct
				mu.Unlock()
			})

			// Check if any errors occurred during encryption
			if encryptionError != nil {
				fmt.Printf("Error during parallel encryption: %v\n", encryptionError)
				continue
			}

			encTime := time.Since(encStart)
			totalEncryptTime += encTime

			fmt.Printf("    Encrypted %d pixels in parallel in %v\n", pixelCount, encTime)

			// Phase 2: Server forward pass with multi-layer processing
			serverStart := time.Now()

			layerInputs, encActivations, err := split.ServerForwardPassWithLayerInputs(heContext, serverModel, encPixels)
			if err != nil {
				fmt.Printf("Error in server forward pass: %v\n", err)
				continue
			}

			serverTime := time.Since(serverStart)
			totalServerForwardTime += serverTime

			// Print some information about the layer inputs
			fmt.Printf("    Server forward pass completed in %v\n", serverTime)
			fmt.Printf("    Processed %d layers, received %d layer inputs\n",
				len(serverModel.Weights), len(layerInputs))

			// Phase 3: Client forward and backward pass
			clientStart := time.Now()

			encGradients, err := split.ClientForwardAndBackward(heContext, clientModel, encActivations, labels, batchIndices)
			if err != nil {
				fmt.Printf("Error in client computation: %v\n", err)
				continue
			}

			clientTime := time.Since(clientStart)
			totalClientComputeTime += clientTime

			// Phase 4: Server backward pass with multi-layer processing
			serverBackStart := time.Now()

			err = split.ServerBackwardAndUpdate(heContext, serverModel, encGradients, layerInputs, learningRate)
			if err != nil {
				fmt.Printf("Error in server backward pass: %v\n", err)
				continue
			}

			serverBackTime := time.Since(serverBackStart)
			totalServerBackwardTime += serverBackTime

			batchTime := time.Since(startTime)
			totalTrainingTime += batchTime

			fmt.Printf("    Batch completed in %v\n", batchTime)
			fmt.Printf("      Encryption: %v\n", encTime)
			fmt.Printf("      Server forward: %v\n", serverTime)
			fmt.Printf("      Client compute: %v\n", clientTime)
			fmt.Printf("      Server backward: %v\n", serverBackTime)
		}
	}

	// Print timing summary
	fmt.Println("\nTraining Summary:")
	fmt.Printf("  Total training time: %v\n", totalTrainingTime)
	fmt.Printf("  Average times per batch:\n")
	fmt.Printf("    Encryption: %v\n", totalEncryptTime/time.Duration(numBatches*epochs))
	fmt.Printf("    Server forward: %v\n", totalServerForwardTime/time.Duration(numBatches*epochs))
	fmt.Printf("    Client compute: %v\n", totalClientComputeTime/time.Duration(numBatches*epochs))
	fmt.Printf("    Server backward: %v\n", totalServerBackwardTime/time.Duration(numBatches*epochs))
}
