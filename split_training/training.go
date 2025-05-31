package split

import (
	"fmt"
	"math/rand"
	"time"
)

// Struct to store performance timing metrics
type TimingMetrics struct {
	totalEncryptionTime     time.Duration
	totalServerForwardTime  time.Duration
	totalClientBackwardTime time.Duration
	totalServerBackwardTime time.Duration
	batchesProcessed        int
}

// Run is the main entry point for training and evaluation
func Run(cfg RunConfig) error {
	// Set batch size from config
	if cfg.BatchSize > 0 {
		BatchSize = cfg.BatchSize
	}

	// Set random seed
	rand.Seed(time.Now().UnixNano())

	// Initialize HE context
	fmt.Println("Initializing HE context...")
	heContext, err := initHE()
	if err != nil {
		return fmt.Errorf("failed to initialize HE context: %v", err)
	}

	// Load MNIST data
	fmt.Println("Loading MNIST data...")
	trainImages, trainLabels, testImages, testLabels, err := readMNISTData()
	if err != nil {
		return fmt.Errorf("failed to load MNIST data: %v", err)
	}

	var clientModelObj *ClientModel
	var serverModelObj *ServerModel

	// Initialize or load models based on mode
	if cfg.Mode == "eval" {
		fmt.Printf("Loading models from %s and %s...\n", cfg.ClientPath, cfg.ServerPath)
		clientModelObj, serverModelObj, err = loadModel(cfg.ClientPath, cfg.ServerPath)
		if err != nil {
			return fmt.Errorf("failed to load models: %v", err)
		}
	} else {
		// Initialize new models
		fmt.Println("Initializing models...")
		clientModelObj = initClientModel()
		serverModelObj = initServerModel()
	}

	// Training or evaluation based on mode
	if cfg.Mode != "eval" {
		fmt.Printf("Starting training with %d batches...\n", cfg.NumBatches)
		trainModelWithBatches(heContext, clientModelObj, serverModelObj,
			trainImages, trainLabels, Epochs, BatchSize, LearningRate, cfg.NumBatches, cfg.FullyHE)

		if cfg.SaveModels {
			fmt.Printf("Saving models to %s and %s...\n", cfg.ClientPath, cfg.ServerPath)
			err = saveModel(clientModelObj, serverModelObj, cfg.ClientPath, cfg.ServerPath)
			if err != nil {
				fmt.Printf("Failed to save models: %v\n", err)
			}
		}
	}

	// Evaluate the model
	if cfg.Mode != "train" {
		fmt.Println("Evaluating model...")
		accuracy := evaluateModel(heContext, clientModelObj, serverModelObj, testImages, testLabels)
		fmt.Printf("Test accuracy: %.2f%%\n", accuracy*100)
	}

	fmt.Println("Done.")
	return nil
}

// Trains the split learning model with a limited number of batches
func trainModelWithBatches(heContext *HEContext, clientModel *ClientModel, serverModel *ServerModel,
	images [][]float64, labels []int, epochs int, batchSize int, learningRate float64, maxBatches int, fullyHomomorphic bool) {

	numSamples := len(images)
	numBatches := numSamples / batchSize

	// Limit the number of batches
	if maxBatches > 0 && maxBatches < numBatches {
		numBatches = maxBatches
	}

	metrics := &TimingMetrics{}

	// Log the training mode
	if fullyHomomorphic {
		fmt.Println("Using fully homomorphic backpropagation mode...")
	} else {
		fmt.Println("Using standard homomorphic mode...")
	}

	for epoch := 0; epoch < epochs; epoch++ {
		fmt.Printf("Epoch %d/%d\n", epoch+1, epochs)

		// Shuffle data
		indices := rand.Perm(numSamples)

		epochStart := time.Now()

		// Process each batch
		for batch := 0; batch < numBatches; batch++ {
			if batch%10 == 0 {
				fmt.Printf("  Batch %d/%d\n", batch+1, numBatches)
			}

			// Get batch indices
			startIdx := batch * batchSize
			endIdx := startIdx + batchSize
			batchIndices := indices[startIdx:endIdx]

			// Choose which training method to use
			if fullyHomomorphic {
				// Train using fully homomorphic backpropagation
				err := trainBatchFullHomomorphic(heContext, clientModel, serverModel, images, labels, batchIndices, learningRate)
				if err != nil {
					fmt.Printf("Error in fully homomorphic training: %v\n", err)
					continue
				}
			} else {
				// Train using the standard method with timing
				trainBatchWithTiming(heContext, clientModel, serverModel, images, labels, batchIndices, learningRate, metrics)
			}
		}

		epochDuration := time.Since(epochStart)
		fmt.Printf("Epoch completed in %v\n", epochDuration)
	}

	// Print timing metrics only for standard mode
	if !fullyHomomorphic && metrics.batchesProcessed > 0 {
		fmt.Println("\nPerformance Metrics:")
		fmt.Printf("  Average encryption time: %v/batch\n",
			metrics.totalEncryptionTime/time.Duration(metrics.batchesProcessed))
		fmt.Printf("  Average server forward time: %v/batch\n",
			metrics.totalServerForwardTime/time.Duration(metrics.batchesProcessed))
		fmt.Printf("  Average client computation time: %v/batch\n",
			metrics.totalClientBackwardTime/time.Duration(metrics.batchesProcessed))
		fmt.Printf("  Average server backward time: %v/batch\n",
			metrics.totalServerBackwardTime/time.Duration(metrics.batchesProcessed))

		totalTime := metrics.totalEncryptionTime + metrics.totalServerForwardTime +
			metrics.totalClientBackwardTime + metrics.totalServerBackwardTime
		fmt.Printf("  Total processing time: %v\n", totalTime)
		fmt.Printf("  Average batch processing time: %v/batch\n",
			totalTime/time.Duration(metrics.batchesProcessed))
	}
}

// Trains the model on a single batch with timing measurements
func trainBatchWithTiming(heContext *HEContext, clientModel *ClientModel, serverModel *ServerModel,
	images [][]float64, labels []int, batchIndices []int, learningRate float64, metrics *TimingMetrics) {

	// Check if metrics is nil
	trackMetrics := metrics != nil

	// Phase 1: Client-Side Prep and Forward to Server
	encStart := time.Now()
	encInputs, err := clientPrepareAndEncryptBatch(heContext, images, batchIndices)
	encTime := time.Since(encStart)
	if trackMetrics {
		metrics.totalEncryptionTime += encTime
	}

	if err != nil {
		fmt.Printf("Error in client preparation: %v\n", err)
		return
	}

	// Phase 2: Server-Side Homomorphic Forward Pass
	serverStart := time.Now()
	encActivations, err := serverForwardPass(heContext, serverModel, encInputs)
	serverTime := time.Since(serverStart)
	if trackMetrics {
		metrics.totalServerForwardTime += serverTime
	}

	if err != nil {
		fmt.Printf("Error in server forward pass: %v\n", err)
		return
	}

	// Phase 3: Client-Side Plaintext Computation (Forward & Backward)
	clientStart := time.Now()
	encGradients, err := clientForwardAndBackward(heContext, clientModel, encActivations, labels, batchIndices)
	clientTime := time.Since(clientStart)
	if trackMetrics {
		metrics.totalClientBackwardTime += clientTime
	}

	if err != nil {
		fmt.Printf("Error in client forward and backward: %v\n", err)
		return
	}

	// Phase 4: Server-Side Homomorphic Backward Pass & Update
	serverBackStart := time.Now()
	// Pass the cached encInputs to serverBackwardAndUpdate for accurate weight updates
	err = serverBackwardAndUpdate(heContext, serverModel, encGradients, encInputs, learningRate)
	serverBackTime := time.Since(serverBackStart)
	if trackMetrics {
		metrics.totalServerBackwardTime += serverBackTime
		metrics.batchesProcessed++
	}

	if err != nil {
		fmt.Printf("Error in server backward and update: %v\n", err)
		return
	}
}

// trainBatchFullHomomorphic performs fully homomorphic training on a single batch
func trainBatchFullHomomorphic(heContext *HEContext, clientModel *ClientModel, serverModel *ServerModel,
	images [][]float64, labels []int, batchIndices []int, learningRate float64) error {

	// Convert the standard server model to a packed homomorphic one
	heServer, err := convertToPacked(serverModel, heContext)
	if err != nil {
		return fmt.Errorf("failed to convert to packed homomorphic model: %v", err)
	}

	// 1. Client: Prepare and encrypt the batch
	encInputs, err := clientPrepareAndEncryptBatch(heContext, images, batchIndices)
	if err != nil {
		return fmt.Errorf("client encryption error: %v", err)
	}

	// 2. Server: Forward pass
	encActivations, err := serverForwardPass(heContext, serverModel, encInputs)
	if err != nil {
		return fmt.Errorf("server forward pass error: %v", err)
	}

	// 3. Client: Forward and backward pass (now returns packed gradients)
	encGradBlk, err := clientForwardAndBackward(heContext, clientModel, encActivations, labels, batchIndices)
	if err != nil {
		return fmt.Errorf("client forward/backward error: %v", err)
	}

	// 4. Server: Fully homomorphic backward pass and weight update using packed SIMD
	// We'll use packedUpdate directly instead of going through serverBackwardAndUpdate
	err = packedUpdate(heContext, heServer, encInputs, encGradBlk,
		learningRate, len(batchIndices))
	if err != nil {
		return fmt.Errorf("homomorphic backward error: %v", err)
	}

	// 5. Convert updated homomorphic model back to standard model
	// For demonstration, decrypt the weights and update the standard model
	// In a real fully homomorphic system, we would keep the weights encrypted
	blk := HiddenDim1 / NeuronsPerCT
	for i := 0; i < InputDim; i++ {
		for b := 0; b < blk; b++ {
			// Decrypt packed weight
			ptWeight := heContext.decryptor.DecryptNew(heServer.W[i][b])

			// Decode all slots
			values := make([]float64, heContext.params.N()/2)
			heContext.encoder.Decode(ptWeight, values)

			// Extract the weight values (first slot of each batch)
			for n := 0; n < NeuronsPerCT; n++ {
				serverModel.W1[i][b*NeuronsPerCT+n] = values[n*BatchSize]
			}
		}
	}

	// Update biases
	for b := 0; b < blk; b++ {
		// Decrypt packed bias
		ptBias := heContext.decryptor.DecryptNew(heServer.b[b])

		// Decode all slots
		values := make([]float64, heContext.params.N()/2)
		heContext.encoder.Decode(ptBias, values)

		// Extract the bias values (first slot of each batch)
		for n := 0; n < NeuronsPerCT; n++ {
			serverModel.b1[b*NeuronsPerCT+n] = values[n*BatchSize]
		}
	}

	return nil
}

// Trains the model on a single batch (wrapper for backward compatibility)
func trainBatch(heContext *HEContext, clientModel *ClientModel, serverModel *ServerModel,
	images [][]float64, labels []int, batchIndices []int, learningRate float64) {

	// Call the timing version with nil metrics to ignore timing
	trainBatchWithTiming(heContext, clientModel, serverModel, images, labels, batchIndices, learningRate, nil)
}

// Trains the split learning model with timing metrics
func trainModel(heContext *HEContext, clientModel *ClientModel, serverModel *ServerModel,
	images [][]float64, labels []int, epochs int, batchSize int, learningRate float64) {

	// Call the version with unlimited batches and standard mode
	trainModelWithBatches(heContext, clientModel, serverModel, images, labels, epochs, batchSize, learningRate, 0, false)
}
