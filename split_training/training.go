package split

import (
	"fmt"
	"math/rand"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
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
	// Apply configuration to global settings
	ApplyConfig(&cfg)

	// Print configuration details
	PrintConfig(&cfg)

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
		clientModelObj = initClientModel(cfg.ModelCfg)
		serverModelObj = initServerModel(cfg.ModelCfg)
	}

	// Training or evaluation based on mode
	if cfg.Mode != "eval" {
		fmt.Printf("Starting training with %d batches...\n", cfg.NumBatches)

		// Choose which training method to use
		if cfg.FullyHE {
			// Use fully homomorphic training
			fmt.Println("Using fully homomorphic training...")
			trainModelWithBatches(heContext, clientModelObj, serverModelObj,
				trainImages, trainLabels, Epochs, BatchSize, LearningRate, cfg.NumBatches, true)
		} else if cfg.FullySIMD {
			// Use fully optimized SIMD training
			fmt.Println("Using fully optimized SIMD training...")
			trainModelFullSIMD(heContext, clientModelObj, serverModelObj,
				trainImages, trainLabels, Epochs, BatchSize, LearningRate, cfg.NumBatches)
		} else {
			// Use standard training
			fmt.Println("Using standard training...")
			trainModelWithBatches(heContext, clientModelObj, serverModelObj,
				trainImages, trainLabels, Epochs, BatchSize, LearningRate, cfg.NumBatches, false)
		}

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

	// Flag to track metrics (if metrics is not nil)
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
	layerInputs, encActivations, err := ServerForwardPassWithLayerInputs(heContext, serverModel, encInputs)
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
	// Pass the cached layer inputs to serverBackwardAndUpdate for accurate weight updates
	err = serverBackwardAndUpdate(heContext, serverModel, encGradients, layerInputs, learningRate)
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

	// 1. Client: Prepare and encrypt the batch
	encInputs, err := clientPrepareAndEncryptBatch(heContext, images, batchIndices)
	if err != nil {
		return fmt.Errorf("client encryption error: %v", err)
	}

	// 2. Server: Forward pass
	layerInputs, encActivations, err := ServerForwardPassWithLayerInputs(heContext, serverModel, encInputs)
	if err != nil {
		return fmt.Errorf("server forward pass error: %v", err)
	}

	// 3. Client: Forward and backward pass (now returns packed gradients)
	encGradBlk, err := clientForwardAndBackward(heContext, clientModel, encActivations, labels, batchIndices)
	if err != nil {
		return fmt.Errorf("client forward/backward error: %v", err)
	}

	// 4. Server: Fully homomorphic backward pass and weight update
	err = serverBackwardAndUpdate(heContext, serverModel, encGradBlk, layerInputs, learningRate)
	if err != nil {
		return fmt.Errorf("homomorphic backward error: %v", err)
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

// Trains the split learning model with a limited number of batches
// This version fully utilizes SIMD by keeping the server model in encrypted form
func trainModelFullSIMD(heContext *HEContext, clientModel *ClientModel, serverModel *ServerModel,
	images [][]float64, labels []int, epochs int, batchSize int, learningRate float64, maxBatches int) {

	numSamples := len(images)
	numBatches := numSamples / batchSize

	// Limit the number of batches
	if maxBatches > 0 && maxBatches < numBatches {
		numBatches = maxBatches
	}

	// Create a packed server model for SIMD operations - will keep this throughout training
	fmt.Println("Converting server model to packed form...")
	heServer, err := convertToPacked(serverModel, heContext)
	if err != nil {
		fmt.Printf("Failed to convert to packed model: %v\n", err)
		return
	}

	fmt.Println("Using fully optimized SIMD mode for training...")

	metrics := &TimingMetrics{}

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

			// Phase 1: Client-Side Prep and Forward to Server
			encStart := time.Now()
			encInputs, err := clientPrepareAndEncryptBatch(heContext, images, batchIndices)
			encTime := time.Since(encStart)
			metrics.totalEncryptionTime += encTime

			if err != nil {
				fmt.Printf("Error in client preparation: %v\n", err)
				continue
			}

			// Phase 2: Server-Side Homomorphic Forward Pass with packed model
			serverStart := time.Now()
			encActivations, err := serverForwardPassPacked(heContext, heServer, encInputs)
			serverTime := time.Since(serverStart)
			metrics.totalServerForwardTime += serverTime

			if err != nil {
				fmt.Printf("Error in server forward pass: %v\n", err)
				continue
			}

			// Phase 3: Client-Side Plaintext Computation (Forward & Backward)
			clientStart := time.Now()
			encGradients, err := clientForwardAndBackward(heContext, clientModel, encActivations, labels, batchIndices)
			clientTime := time.Since(clientStart)
			metrics.totalClientBackwardTime += clientTime

			if err != nil {
				fmt.Printf("Error in client forward and backward: %v\n", err)
				continue
			}

			// Phase 4: Server-Side Homomorphic Backward Pass & Update packed model directly
			serverBackStart := time.Now()
			err = packedUpdateDirect(heContext, heServer, encInputs, encGradients, learningRate, batchSize)
			serverBackTime := time.Since(serverBackStart)
			metrics.totalServerBackwardTime += serverBackTime
			metrics.batchesProcessed++

			if err != nil {
				fmt.Printf("Error in server backward and update: %v\n", err)
				continue
			}
		}

		epochDuration := time.Since(epochStart)
		fmt.Printf("Epoch completed in %v\n", epochDuration)
	}

	// Only decrypt and update the model at the end of training
	fmt.Println("Training complete. Converting model back to plaintext...")
	updateCompleteModelFromHE(heContext, serverModel, heServer, batchSize)

	// Print timing metrics
	if metrics.batchesProcessed > 0 {
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

// Forward pass using directly the packed server model (avoids repeated packing)
func serverForwardPassPacked(he *HEContext, heServer *HEServerPacked, encInputs []*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error) {
	// Check if input is valid
	if len(encInputs) == 0 || encInputs[0] == nil {
		return nil, fmt.Errorf("invalid input: empty or nil encInputs")
	}

	// Number of images in the batch
	batchSize := len(encInputs)

	// Process each image separately
	resultOutputs := make([]*rlwe.Ciphertext, batchSize)

	// For each image in the batch
	for batchIdx := 0; batchIdx < batchSize; batchIdx++ {
		// Process each layer in the server model
		currentLayerOutput := encInputs[batchIdx : batchIdx+1] // Slice with single ciphertext

		for l := 0; l < len(heServer.W); l++ {
			serverArch := heServer.Config.Arch
			inputDim := serverArch[l]
			outputDim := serverArch[l+1]

			// Calculate the number of blocks needed for the output neurons
			neuronsPerCT := heServer.NeuronsPerCT
			numBlocks := (outputDim + neuronsPerCT - 1) / neuronsPerCT

			// Prepare the next layer activations
			nextLayer := make([]*rlwe.Ciphertext, numBlocks)

			// Create a mutex for thread-safe access to shared resources
			var mutex sync.Mutex

			// Process each block of neurons in parallel
			parallelFor(0, numBlocks, func(b int) {
				// Initialize with bias for this neuron block
				blockResult := heServer.b[l][b].CopyNew()

				// Get the input for this image
				encInput := currentLayerOutput[0]

				// For each input dimension, multiply by weight and add to accumulator
				for i := 0; i < inputDim; i++ {
					// Create a temporary ciphertext for this multiplication
					temp := encInput.CopyNew()

					// Multiply the input by the weight for this neuron block
					he.evaluator.Mul(temp, heServer.W[l][i][b], temp)

					// Relinearize
					he.evaluator.Relinearize(temp, temp)

					// Add to the accumulator
					he.evaluator.Add(blockResult, temp, blockResult)
				}

				// Apply ReLU approximation to this block's activation
				activatedBlock, err := applyReLU(he, blockResult)
				if err != nil {
					fmt.Printf("Error in ReLU for block %d: %v\n", b, err)
					return
				}

				// Safely store the result
				mutex.Lock()
				nextLayer[b] = activatedBlock
				mutex.Unlock()
			})

			// Update current layer for the next iteration
			currentLayerOutput = nextLayer
		}

		// Store the final output for this image
		resultOutputs[batchIdx] = currentLayerOutput[0]
	}

	return resultOutputs, nil
}

// Update packed model directly without decrypting after each batch
func packedUpdateDirect(heContext *HEContext, heServer *HEServerPacked, encInputs []*rlwe.Ciphertext,
	encGradients []*rlwe.Ciphertext, learningRate float64, batchSize int) error {

	// Process only the first (last) layer of the server model
	// In future versions, this could be extended to all server layers
	l := len(heServer.W) - 1
	serverArch := heServer.Config.Arch
	inputDim := serverArch[l]
	outputDim := serverArch[l+1]

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

	return nil
}

// Update the complete model from packed HE model - only call at end of training
func updateCompleteModelFromHE(heContext *HEContext, serverModel *ServerModel, heServer *HEServerPacked, batchSize int) {
	// For each layer
	for l := 0; l < len(serverModel.Weights); l++ {
		// Call the layer-specific update function
		updateModelFromHE(heContext, serverModel, heServer, l, batchSize)
	}
}
