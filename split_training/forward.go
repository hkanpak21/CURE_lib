package split

import (
	"fmt"
	"sync"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Client prepares and encrypts a batch of images with parallel processing
func clientPrepareAndEncryptBatch(he *HEContext, imgs [][]float64, idx []int) ([]*rlwe.Ciphertext, error) {
	B := len(idx)
	encInputs := make([]*rlwe.Ciphertext, InputDim)

	var wg sync.WaitGroup
	rowsPerWkr := (InputDim + NumWorkers - 1) / NumWorkers

	for w := 0; w < NumWorkers; w++ {
		start := w * rowsPerWkr
		end := min(InputDim, start+rowsPerWkr)
		if start >= end {
			continue
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			localEnc := he.encryptor // immutable
			localEncdr := he.encoder

			for i := s; i < e; i++ {
				vec := make([]float64, B)
				for j := 0; j < B; j++ {
					vec[j] = imgs[idx[j]][i]
				}
				pt := ckks.NewPlaintext(he.params, he.params.MaxLevel())
				localEncdr.Encode(vec, pt)
				ct, err := localEnc.EncryptNew(pt)
				if err != nil {
					fmt.Printf("Error encrypting input row %d: %v\n", i, err)
					return
				}
				encInputs[i] = ct
			}
		}(start, end)
	}
	wg.Wait()
	return encInputs, nil
}

// Server performs forward pass on encrypted inputs
func serverForwardPass(heContext *HEContext, serverModel *ServerModel, encInputs []*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error) {
	// Step 1: Linear Layer (Forward)
	// Perform homomorphic matrix-vector multiplication between the plaintext weight matrix W_1
	// and the encrypted input vectors

	// Initialize encrypted activations vector (before ReLU)
	encZ1 := make([]*rlwe.Ciphertext, HiddenDim1)

	// For each output neuron, compute the dot product with inputs
	for i := 0; i < HiddenDim1; i++ {
		// Start with bias
		ptBias := ckks.NewPlaintext(heContext.params, heContext.params.MaxLevel())
		biasVec := make([]float64, BatchSize) // Using batchSize from global constant
		for j := range biasVec {
			biasVec[j] = serverModel.b1[i]
		}
		heContext.encoder.Encode(biasVec, ptBias)

		// Initialize a ciphertext for the result
		encZ1[i] = heContext.encryptor.EncryptZeroNew(heContext.params.MaxLevel())

		// Add bias to the result
		heContext.evaluator.Add(encZ1[i], ptBias, encZ1[i])

		// Compute dot product with weights
		for j := 0; j < InputDim; j++ {
			// Create plaintext for the weight
			ptWeight := ckks.NewPlaintext(heContext.params, heContext.params.MaxLevel())
			weightVec := make([]float64, BatchSize) // Using batchSize from global constant
			for k := range weightVec {
				weightVec[k] = serverModel.W1[j][i]
			}
			heContext.encoder.Encode(weightVec, ptWeight)

			// Multiply input with weight
			encTemp, err := heContext.evaluator.MulNew(encInputs[j], ptWeight)
			if err != nil {
				return nil, fmt.Errorf("error in matrix multiplication: %v", err)
			}

			// No need to relinearize after multiplying ciphertext with plaintext
			// as the degree doesn't increase

			// Add to result
			heContext.evaluator.Add(encZ1[i], encTemp, encZ1[i])
		}
	}

	// Step 2: ReLU Activation (Forward)
	// Apply a simpler ReLU approximation based on Chebyshev polynomials in parallel
	fmt.Println("Applying ReLU activation with parallel processing...")
	encA1 := make([]*rlwe.Ciphertext, HiddenDim1)

	// Create a wait group to synchronize goroutines
	var wg sync.WaitGroup

	// Create a mutex for synchronizing error handling
	errMutex := &sync.Mutex{}
	var firstErr error

	// Calculate how many neurons each worker should process
	neuronsPerWorker := HiddenDim1 / NumWorkers
	if neuronsPerWorker == 0 {
		neuronsPerWorker = 1
	}

	// Start worker goroutines
	for w := 0; w < NumWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			// Calculate the range of neurons this worker will process
			startNeuron := workerID * neuronsPerWorker
			endNeuron := (workerID + 1) * neuronsPerWorker
			if workerID == NumWorkers-1 {
				endNeuron = HiddenDim1 // Last worker takes any remaining neurons
			}

			// Process each neuron in the assigned range
			for i := startNeuron; i < endNeuron; i++ {
				// Check if an error has already occurred
				errMutex.Lock()
				if firstErr != nil {
					errMutex.Unlock()
					return
				}
				errMutex.Unlock()

				// ReLU(x) ≈ 0.32 + 0.5*x + 0.23*x² (degree-2 Chebyshev approximation)

				// 1. Compute x²
				ctSquared, err := heContext.evaluator.MulNew(encZ1[i], encZ1[i])
				if err != nil {
					errMutex.Lock()
					if firstErr == nil {
						firstErr = fmt.Errorf("error computing square in ReLU: %v", err)
					}
					errMutex.Unlock()
					return
				}

				// Here we need to relinearize because we multiplied two ciphertexts
				err = heContext.evaluator.Relinearize(ctSquared, ctSquared)
				if err != nil {
					errMutex.Lock()
					if firstErr == nil {
						firstErr = fmt.Errorf("error relinearizing in ReLU: %v", err)
					}
					errMutex.Unlock()
					return
				}

				// 2. Compute 0.5*x
				ctScaledX, err := heContext.evaluator.MulNew(encZ1[i], 0.5)
				if err != nil {
					errMutex.Lock()
					if firstErr == nil {
						firstErr = fmt.Errorf("error scaling x in ReLU: %v", err)
					}
					errMutex.Unlock()
					return
				}

				// 3. Compute 0.23*x²
				ctScaledX2, err := heContext.evaluator.MulNew(ctSquared, 0.23)
				if err != nil {
					errMutex.Lock()
					if firstErr == nil {
						firstErr = fmt.Errorf("error scaling x² in ReLU: %v", err)
					}
					errMutex.Unlock()
					return
				}

				// 4. First add 0.5*x + 0.23*x²
				temp, err := heContext.evaluator.AddNew(ctScaledX, ctScaledX2)
				if err != nil {
					errMutex.Lock()
					if firstErr == nil {
						firstErr = fmt.Errorf("error adding terms in ReLU: %v", err)
					}
					errMutex.Unlock()
					return
				}

				// 5. Add the constant term 0.32 (create a constant ciphertext)
				// Initialize a ciphertext for the constant
				constantCt := heContext.encryptor.EncryptZeroNew(heContext.params.MaxLevel())

				// Add the constant value to it
				constant := ckks.NewPlaintext(heContext.params, heContext.params.MaxLevel())
				constantVec := make([]float64, BatchSize)
				for j := range constantVec {
					constantVec[j] = 0.32
				}
				heContext.encoder.Encode(constantVec, constant)
				heContext.evaluator.Add(constantCt, constant, constantCt)

				// Now add the constant ciphertext to our result
				var result *rlwe.Ciphertext
				result, err = heContext.evaluator.AddNew(temp, constantCt)
				if err != nil {
					errMutex.Lock()
					if firstErr == nil {
						firstErr = fmt.Errorf("error adding terms in ReLU: %v", err)
					}
					errMutex.Unlock()
					return
				}

				// Store the result in the shared array
				encA1[i] = result
			}
		}(w)
	}

	// Wait for all workers to finish
	wg.Wait()

	// Check if any errors occurred
	if firstErr != nil {
		return nil, firstErr
	}

	return encA1, nil
}
