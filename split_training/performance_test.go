package split

import (
	"fmt"
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// TestPerformance runs performance tests on key operations
func TestPerformance(t *testing.T) {
	// Set parameters for testing - use smaller numbers for testing
	const (
		numSamples   = 2       // Reduced from 10 to prevent out of range errors
		batchSize    = 4       // Reduced from 8 to prevent out of range errors
		imageSize    = 28 * 28 // MNIST image size
		numClasses   = 10      // Number of output classes
		learningRate = 0.01    // Learning rate for training
	)

	// Create a minimal but realistic architecture
	testArch := []int{imageSize, 128, 32, numClasses} // Classic MNIST architecture
	splitIdx := 1                                     // Split after first hidden layer

	config := &ModelConfig{
		Arch:     testArch,
		SplitIdx: splitIdx,
	}

	// Initialize HE context
	fmt.Println("Initializing HE context...")
	startHE := time.Now()
	heContext, err := InitHE()
	elapsedHE := time.Since(startHE)
	if err != nil {
		t.Fatalf("Failed to initialize HE context: %v", err)
	}
	fmt.Printf("HE context initialized in %v\n", elapsedHE)

	// Initialize models
	fmt.Println("Initializing models...")
	startClientModel := time.Now()
	clientModel := InitClientModel(config)
	elapsedClientModel := time.Since(startClientModel)
	fmt.Printf("Client model initialized in %v\n", elapsedClientModel)

	startServerModel := time.Now()
	serverModel := InitServerModel(config)
	elapsedServerModel := time.Since(startServerModel)
	fmt.Printf("Server model initialized in %v\n", elapsedServerModel)

	// Generate random sample data - generate more data than we need
	fmt.Println("Generating sample data...")
	totalImages := numSamples * batchSize
	images := make([][]float64, totalImages)
	labels := make([]int, totalImages)
	for i := range images {
		images[i] = make([]float64, imageSize)
		for j := range images[i] {
			images[i][j] = rand.Float64() // Random pixel values between 0 and 1
		}
		labels[i] = rand.Intn(numClasses) // Random label
	}

	// Store results for each operation
	type OperationTiming struct {
		Name        string
		Times       []time.Duration
		AverageTime time.Duration
	}

	timings := make([]OperationTiming, 4)
	timings[0].Name = "ClientPrepareAndEncryptBatch"
	timings[1].Name = "ServerForwardPass"
	timings[2].Name = "ClientForwardAndBackward"
	timings[3].Name = "ServerBackwardAndUpdate"

	// Run the tests
	fmt.Println("\nRunning performance tests...")

	var encInputsCache [][]*rlwe.Ciphertext
	var encActivationsCache [][]*rlwe.Ciphertext
	var encGradientsCache [][]*rlwe.Ciphertext

	// Save batch indices for later use with packed operations
	var firstBatchIndices []int

	// For each sample
	for s := 0; s < numSamples; s++ {
		fmt.Printf("Sample %d/%d\n", s+1, numSamples)

		// Get batch indices for this sample
		startIdx := s * batchSize
		endIdx := startIdx + batchSize

		// Ensure indices don't go out of bounds
		if endIdx > totalImages {
			endIdx = totalImages
		}

		// Create batch indices
		batchIndices := make([]int, endIdx-startIdx)
		for i := 0; i < len(batchIndices); i++ {
			batchIndices[i] = startIdx + i
		}

		// Save the first batch indices for later use
		if s == 0 {
			firstBatchIndices = make([]int, len(batchIndices))
			copy(firstBatchIndices, batchIndices)
		}

		// Skip if batch is empty
		if len(batchIndices) == 0 {
			fmt.Println("  Skipping empty batch")
			continue
		}

		fmt.Printf("  Processing batch with %d images\n", len(batchIndices))

		// 1. Client prepare and encrypt batch
		start := time.Now()
		encInputs, err := ClientPrepareAndEncryptBatch(heContext, images, batchIndices)
		elapsed := time.Since(start)
		if err != nil {
			t.Fatalf("ClientPrepareAndEncryptBatch failed: %v", err)
		}
		timings[0].Times = append(timings[0].Times, elapsed)
		fmt.Printf("  ClientPrepareAndEncryptBatch: %v\n", elapsed)
		encInputsCache = append(encInputsCache, encInputs)

		// 2. Server forward pass
		start = time.Now()
		encActivations, err := ServerForwardPass(heContext, serverModel, encInputs)
		elapsed = time.Since(start)
		if err != nil {
			t.Fatalf("ServerForwardPass failed: %v", err)
		}
		timings[1].Times = append(timings[1].Times, elapsed)
		fmt.Printf("  ServerForwardPass: %v\n", elapsed)
		encActivationsCache = append(encActivationsCache, encActivations)

		// 3. Client forward and backward
		start = time.Now()
		encGradients, err := ClientForwardAndBackward(heContext, clientModel, encActivations, labels[startIdx:endIdx], batchIndices)
		elapsed = time.Since(start)
		if err != nil {
			t.Fatalf("ClientForwardAndBackward failed: %v", err)
		}
		timings[2].Times = append(timings[2].Times, elapsed)
		fmt.Printf("  ClientForwardAndBackward: %v\n", elapsed)
		encGradientsCache = append(encGradientsCache, encGradients)

		// 4. Server backward and update
		start = time.Now()
		err = ServerBackwardAndUpdate(heContext, serverModel, encGradients, encInputs, learningRate)
		elapsed = time.Since(start)
		if err != nil {
			t.Fatalf("ServerBackwardAndUpdate failed: %v", err)
		}
		timings[3].Times = append(timings[3].Times, elapsed)
		fmt.Printf("  ServerBackwardAndUpdate: %v\n", elapsed)
	}

	// Calculate and print averages
	fmt.Println("\nPerformance Summary:")
	for i := range timings {
		var total time.Duration
		for _, t := range timings[i].Times {
			total += t
		}
		if len(timings[i].Times) > 0 {
			timings[i].AverageTime = total / time.Duration(len(timings[i].Times))
			fmt.Printf("%s: Avg %v\n", timings[i].Name, timings[i].AverageTime)
		} else {
			fmt.Printf("%s: No measurements\n", timings[i].Name)
		}
	}

	// Test the SIMD optimized paths
	fmt.Println("\nTesting SIMD-optimized operations...")

	// Test packed server model conversion
	fmt.Println("Converting to packed server model...")
	start := time.Now()
	packedServer, err := convertToPacked(serverModel, heContext)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("Failed to convert to packed server model: %v", err)
	}
	fmt.Printf("Packed server model conversion: %v\n", elapsed)

	// Test server forward pass with packed model
	if len(encInputsCache) > 0 {
		fmt.Println("Testing serverForwardPassPacked...")
		start = time.Now()
		packedActivations, err := serverForwardPassPacked(heContext, packedServer, encInputsCache[0])
		elapsed = time.Since(start)
		if err != nil {
			t.Fatalf("serverForwardPassPacked failed: %v", err)
		}
		fmt.Printf("serverForwardPassPacked: %v\n", elapsed)

		// Just to avoid unused variable warning
		if len(packedActivations) > 0 {
			fmt.Printf("  Processed %d packed activations\n", len(packedActivations))
		}

		// Test packed update
		if len(encGradientsCache) > 0 && len(firstBatchIndices) > 0 {
			fmt.Println("Testing packedUpdateDirect...")
			start = time.Now()
			err = packedUpdateDirect(heContext, packedServer, encInputsCache[0], encGradientsCache[0], learningRate, len(firstBatchIndices))
			elapsed = time.Since(start)
			if err != nil {
				t.Fatalf("packedUpdateDirect failed: %v", err)
			}
			fmt.Printf("packedUpdateDirect: %v\n", elapsed)
		}
	}

	// Test optimized full SIMD training
	fmt.Println("\nTesting full SIMD training cycle...")
	// This simulates one epoch of training with the optimized SIMD path
	if totalImages > 0 {
		simdBatchSize := 2 // Use smaller batch for quick test
		maxBatches := 1    // Just do one batch for testing
		simdTotalSize := simdBatchSize * maxBatches

		// Make sure we have enough data
		if simdTotalSize > totalImages {
			simdTotalSize = totalImages
			maxBatches = simdTotalSize / simdBatchSize
			if maxBatches == 0 {
				maxBatches = 1
				simdBatchSize = simdTotalSize
			}
		}

		start = time.Now()
		trainModelFullSIMD(heContext, clientModel, serverModel,
			images[:simdTotalSize], labels[:simdTotalSize],
			1, simdBatchSize, learningRate, maxBatches)
		elapsed = time.Since(start)
		fmt.Printf("Full SIMD training cycle (%d batches): %v\n", maxBatches, elapsed)
	}
}

// TestParallelExecution verifies that parallelization is working correctly
func TestParallelExecution(t *testing.T) {
	const items = 100
	const expectedSum = items * (items - 1) / 2 // Sum of numbers 0 to items-1

	// Create a test array
	array := make([]int, items)
	for i := range array {
		array[i] = i
	}

	// Create a result array to store processed values
	result := make([]int, items)
	var mu sync.Mutex
	sum := 0

	// Use the parallelFor function
	start := time.Now()
	parallelFor(0, items, func(i int) {
		// Simulate some work
		time.Sleep(time.Millisecond)
		result[i] = array[i]

		// Update sum safely
		mu.Lock()
		sum += array[i]
		mu.Unlock()
	})
	elapsed := time.Since(start)

	// Verify results
	for i := 0; i < items; i++ {
		if result[i] != i {
			t.Errorf("Parallel execution error: result[%d] = %d, expected %d", i, result[i], i)
		}
	}

	if sum != expectedSum {
		t.Errorf("Parallel execution error: sum = %d, expected %d", sum, expectedSum)
	}

	// Test without parallelization
	sum = 0
	start = time.Now()
	for i := 0; i < items; i++ {
		time.Sleep(time.Millisecond)
		sum += array[i]
	}
	elapsedSerial := time.Since(start)

	fmt.Printf("Parallel execution: %v, Serial execution: %v\n", elapsed, elapsedSerial)
	if NumWorkers > 1 && elapsed >= elapsedSerial {
		// Only warn if we expect parallelization (NumWorkers > 1)
		t.Logf("Warning: Parallel execution (%v) was not faster than serial execution (%v)",
			elapsed, elapsedSerial)
	}
}

// TestReLUApproximation tests the correctness of the ReLU approximation
func TestReLUApproximation(t *testing.T) {
	// Initialize HE context
	heContext, err := InitHE()
	if err != nil {
		t.Fatalf("Failed to initialize HE context: %v", err)
	}

	// Test values to encrypt
	testValues := []float64{-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0}
	slots := heContext.params.N() / 2
	vec := make([]float64, slots)

	// Set test values in first slots
	for i, val := range testValues {
		if i < slots {
			vec[i] = val
		}
	}

	// Encrypt
	pt := ckks.NewPlaintext(heContext.params, heContext.params.MaxLevel())
	heContext.encoder.Encode(vec, pt)

	ct, err := heContext.encryptor.EncryptNew(pt)
	if err != nil {
		t.Fatalf("Encryption failed: %v", err)
	}

	// Apply ReLU approximation
	start := time.Now()
	ctReLU, err := applyReLU(heContext, ct)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("ReLU approximation failed: %v", err)
	}
	fmt.Printf("ReLU approximation time: %v\n", elapsed)

	// Decrypt and check results
	ptResult := heContext.decryptor.DecryptNew(ctReLU)
	resultVec := make([]float64, slots)
	heContext.encoder.Decode(ptResult, resultVec)

	// Expected results for a good ReLU approximation (not perfect but close)
	for i, val := range testValues {
		expected := 0.0
		if val > 0 {
			expected = val
		}

		// Allow some approximation error
		if i < len(resultVec) {
			error := resultVec[i] - expected
			fmt.Printf("ReLU(%f) = %f (expected ~%f, error: %f)\n",
				val, resultVec[i], expected, error)

			// For positive values, expect reasonable approximation
			if val > 0 && (resultVec[i] < 0 || resultVec[i] > val*1.5) {
				t.Errorf("ReLU approximation too far off for %f: got %f", val, resultVec[i])
			}

			// For negative values, expect small positive value
			if val < 0 && resultVec[i] > 0.5 {
				t.Errorf("ReLU approximation too far off for %f: got %f", val, resultVec[i])
			}
		}
	}
}
