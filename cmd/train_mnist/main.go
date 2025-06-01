package main

import (
	"fmt"
	"math/rand"
	"runtime"
	"time"

	split "github.com/halilibrahimkanpak/cure_test/split_training"
)

func main() {
	// Set random seed for reproducibility
	rand.Seed(42)

	// Display CPU info
	fmt.Printf("Running on %d CPU cores\n", runtime.NumCPU())

	// Set parameters for training
	const (
		numSamples   = 64      // Total number of samples
		batchSize    = 4       // Size of each batch (64/16 = 4)
		numBatches   = 16      // Number of batches
		imageSize    = 28 * 28 // MNIST image size
		numClasses   = 10      // Number of output classes
		learningRate = 0.01    // Learning rate for training
		numEpochs    = 2       // Number of training epochs
	)

	// Create a minimal but realistic architecture
	testArch := []int{imageSize, 128, 32, numClasses} // Classic MNIST architecture
	splitIdx := 1                                     // Split after first hidden layer

	config := &split.ModelConfig{
		Arch:     testArch,
		SplitIdx: splitIdx,
	}

	// Initialize HE context
	fmt.Println("Initializing HE context...")
	startHE := time.Now()
	heContext, err := split.InitHE()
	elapsedHE := time.Since(startHE)
	if err != nil {
		fmt.Printf("Failed to initialize HE context: %v\n", err)
		return
	}
	fmt.Printf("HE context initialized in %v\n", elapsedHE)

	// Initialize models
	fmt.Println("Initializing models...")
	startClientModel := time.Now()
	clientModel := split.InitClientModel(config)
	elapsedClientModel := time.Since(startClientModel)
	fmt.Printf("Client model initialized in %v\n", elapsedClientModel)

	startServerModel := time.Now()
	serverModel := split.InitServerModel(config)
	elapsedServerModel := time.Since(startServerModel)
	fmt.Printf("Server model initialized in %v\n", elapsedServerModel)

	// Generate random sample data
	fmt.Println("Generating sample data...")
	images := make([][]float64, numSamples)
	labels := make([]int, numSamples)
	for i := range images {
		images[i] = make([]float64, imageSize)
		for j := range images[i] {
			images[i][j] = rand.Float64() // Random pixel values between 0 and 1
		}
		labels[i] = rand.Intn(numClasses) // Random label
	}

	// Run the training with the newly parallelized backpropagation
	fmt.Println("Starting training with parallel backpropagation...")

	// Training metrics
	totalClientTime := time.Duration(0)
	totalServerForwardTime := time.Duration(0)
	totalServerBackwardTime := time.Duration(0)
	totalEncryptionTime := time.Duration(0)

	// Run training for multiple epochs
	fmt.Printf("\nStarting training for %d epochs with %d samples in %d batches...\n",
		numEpochs, numSamples, numBatches)

	startTraining := time.Now()

	for epoch := 0; epoch < numEpochs; epoch++ {
		fmt.Printf("Epoch %d/%d\n", epoch+1, numEpochs)

		// Shuffle the data for each epoch
		indices := rand.Perm(numSamples)

		// Process each batch
		for batch := 0; batch < numBatches; batch++ {
			batchStart := batch * batchSize
			batchEnd := batchStart + batchSize
			if batchEnd > numSamples {
				batchEnd = numSamples
			}

			// Create batch indices
			batchIndices := make([]int, batchEnd-batchStart)
			for i := range batchIndices {
				batchIndices[i] = indices[batchStart+i]
			}

			fmt.Printf("  Batch %d/%d (samples %d-%d)\n",
				batch+1, numBatches, batchStart+1, batchEnd)

			// 1. Client prepare and encrypt batch
			startEncrypt := time.Now()
			encInputs, err := split.ClientPrepareAndEncryptBatch(heContext, images, batchIndices)
			encryptTime := time.Since(startEncrypt)
			totalEncryptionTime += encryptTime

			if err != nil {
				fmt.Printf("Error in batch preparation: %v\n", err)
				continue
			}

			// 2. Server forward pass
			startServerForward := time.Now()
			encActivations, err := split.ServerForwardPass(heContext, serverModel, encInputs)
			serverForwardTime := time.Since(startServerForward)
			totalServerForwardTime += serverForwardTime

			if err != nil {
				fmt.Printf("Error in server forward pass: %v\n", err)
				continue
			}

			// 3. Client forward and backward
			startClient := time.Now()
			encGradients, err := split.ClientForwardAndBackward(
				heContext, clientModel, encActivations, labels, batchIndices)
			clientTime := time.Since(startClient)
			totalClientTime += clientTime

			if err != nil {
				fmt.Printf("Error in client computation: %v\n", err)
				continue
			}

			// 4. Server backward and update
			startServerBackward := time.Now()
			err = split.ServerBackwardAndUpdate(
				heContext, serverModel, encGradients, encInputs, learningRate)
			serverBackwardTime := time.Since(startServerBackward)
			totalServerBackwardTime += serverBackwardTime

			if err != nil {
				fmt.Printf("Error in server update: %v\n", err)
				continue
			}

			// Print timing for this batch
			fmt.Printf("    Encryption: %v, Server forward: %v, Client: %v, Server backward: %v\n",
				encryptTime, serverForwardTime, clientTime, serverBackwardTime)
		}
	}

	totalTrainingTime := time.Since(startTraining)

	// Print overall training statistics
	fmt.Printf("\nTraining complete!\n")
	fmt.Printf("Total training time: %v\n", totalTrainingTime)
	fmt.Printf("Average times per batch:\n")
	batchCount := numBatches * numEpochs
	fmt.Printf("  Encryption: %v\n", totalEncryptionTime/time.Duration(batchCount))
	fmt.Printf("  Server forward: %v\n", totalServerForwardTime/time.Duration(batchCount))
	fmt.Printf("  Client computation: %v\n", totalClientTime/time.Duration(batchCount))
	fmt.Printf("  Server backward: %v\n", totalServerBackwardTime/time.Duration(batchCount))

	// Print relative speedup compared to your previous run
	fmt.Printf("\nParallel vs Previous Run Comparison:\n")
	previousServerBackward := float64(5.626601811 * float64(time.Second)) // From the previous output
	currentServerBackward := float64(totalServerBackwardTime / time.Duration(batchCount))
	speedup := previousServerBackward / currentServerBackward
	fmt.Printf("  Server backward speedup: %.2fx\n", speedup)
}
