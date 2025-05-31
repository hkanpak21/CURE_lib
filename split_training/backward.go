package split

import (
	"fmt"
	"math"
	"sync"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Client performs forward and backward pass after receiving encrypted activations
func clientForwardAndBackward(heContext *HEContext, clientModel *ClientModel, encActivations []*rlwe.Ciphertext,
	labels []int, batchIndices []int) ([]*rlwe.Ciphertext, error) {

	batchSize := len(batchIndices)

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

	// Step 2: Forward Pass through Client's Layers
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

	// Step 3: Compute Loss (Cross-Entropy Loss)
	// Apply softmax and compute cross-entropy loss
	loss := 0.0
	for i := 0; i < batchSize; i++ {
		// Apply softmax
		maxVal := a3[i][0]
		for j := 1; j < OutputDim; j++ {
			if a3[i][j] > maxVal {
				maxVal = a3[i][j]
			}
		}

		expSum := 0.0
		expValues := make([]float64, OutputDim)
		for j := 0; j < OutputDim; j++ {
			expValues[j] = math.Exp(a3[i][j] - maxVal)
			expSum += expValues[j]
		}

		softmax := make([]float64, OutputDim)
		for j := 0; j < OutputDim; j++ {
			softmax[j] = expValues[j] / expSum
		}

		// Compute cross-entropy loss
		loss -= math.Log(softmax[labels[batchIndices[i]]])
	}
	loss /= float64(batchSize)

	// Step 4: Backward Pass
	// Compute gradients for output layer
	dZ3 := make([][]float64, batchSize)
	for i := range dZ3 {
		dZ3[i] = make([]float64, OutputDim)
		for j := 0; j < OutputDim; j++ {
			// Derivative of softmax with cross-entropy loss
			if j == labels[batchIndices[i]] {
				dZ3[i][j] = a3[i][j] - 1
			} else {
				dZ3[i][j] = a3[i][j]
			}
		}
	}

	// Compute gradients for W3 and b3
	dW3 := make([][]float64, HiddenDim2)
	for i := range dW3 {
		dW3[i] = make([]float64, OutputDim)
		for j := 0; j < OutputDim; j++ {
			for k := 0; k < batchSize; k++ {
				dW3[i][j] += a2[k][i] * dZ3[k][j]
			}
			dW3[i][j] /= float64(batchSize)
		}
	}

	db3 := make([]float64, OutputDim)
	for i := 0; i < OutputDim; i++ {
		for j := 0; j < batchSize; j++ {
			db3[i] += dZ3[j][i]
		}
		db3[i] /= float64(batchSize)
	}

	// Compute gradients for second hidden layer
	dA2 := make([][]float64, batchSize)
	for i := range dA2 {
		dA2[i] = make([]float64, HiddenDim2)
		for j := 0; j < HiddenDim2; j++ {
			for k := 0; k < OutputDim; k++ {
				dA2[i][j] += dZ3[i][k] * clientModel.W3[j][k]
			}
		}
	}

	// Compute gradients for ReLU in second hidden layer
	dZ2 := make([][]float64, batchSize)
	for i := range dZ2 {
		dZ2[i] = make([]float64, HiddenDim2)
		for j := 0; j < HiddenDim2; j++ {
			if a2[i][j] > 0 {
				dZ2[i][j] = dA2[i][j]
			}
		}
	}

	// Compute gradients for W2 and b2
	dW2 := make([][]float64, HiddenDim1)
	for i := range dW2 {
		dW2[i] = make([]float64, HiddenDim2)
		for j := 0; j < HiddenDim2; j++ {
			for k := 0; k < batchSize; k++ {
				dW2[i][j] += a1Transposed[k][i] * dZ2[k][j]
			}
			dW2[i][j] /= float64(batchSize)
		}
	}

	db2 := make([]float64, HiddenDim2)
	for i := 0; i < HiddenDim2; i++ {
		for j := 0; j < batchSize; j++ {
			db2[i] += dZ2[j][i]
		}
		db2[i] /= float64(batchSize)
	}

	// Compute gradients for first hidden layer
	dA1 := make([][]float64, batchSize)
	for i := range dA1 {
		dA1[i] = make([]float64, HiddenDim1)
		for j := 0; j < HiddenDim1; j++ {
			for k := 0; k < HiddenDim2; k++ {
				dA1[i][j] += dZ2[i][k] * clientModel.W2[j][k]
			}
		}
	}

	// Transpose dA1 to have shape [hiddenDim1 x batchSize]
	dA1Transposed := make([][]float64, HiddenDim1)
	for i := range dA1Transposed {
		dA1Transposed[i] = make([]float64, batchSize)
		for j := 0; j < batchSize; j++ {
			dA1Transposed[i][j] = dA1[j][i]
		}
	}

	// Step 5: Update Client's Weights
	// Update W3 and b3
	for i := 0; i < HiddenDim2; i++ {
		for j := 0; j < OutputDim; j++ {
			clientModel.W3[i][j] -= LearningRate * dW3[i][j]
		}
	}

	for i := 0; i < OutputDim; i++ {
		clientModel.b3[i] -= LearningRate * db3[i]
	}

	// Update W2 and b2
	for i := 0; i < HiddenDim1; i++ {
		for j := 0; j < HiddenDim2; j++ {
			clientModel.W2[i][j] -= LearningRate * dW2[i][j]
		}
	}

	for i := 0; i < HiddenDim2; i++ {
		clientModel.b2[i] -= LearningRate * db2[i]
	}

	// ---- New Step 6: pack dA1 into 2 ciphertexts ----      ★ SIMD-weights
	blk := HiddenDim1 / NeuronsPerCT // =2
	encGradBlk := make([]*rlwe.Ciphertext, blk)

	scratch := make([]float64, heContext.params.N()/2)

	for b := 0; b < blk; b++ {
		// *** clear the buffer – otherwise data from the previous
		//     block would survive in the tail ***
		for i := range scratch {
			scratch[i] = 0
		}

		for n := 0; n < NeuronsPerCT; n++ {
			copy(scratch[n*batchSize:(n+1)*batchSize],
				dA1Transposed[b*NeuronsPerCT+n])
		}

		pt := ckks.NewPlaintext(heContext.params, heContext.params.MaxLevel())
		heContext.encoder.Encode(scratch, pt)

		ct, err := heContext.encryptor.EncryptNew(pt)
		if err != nil {
			return nil, fmt.Errorf("encrypt packed grad: %v", err)
		}
		encGradBlk[b] = ct
	}
	return encGradBlk, nil
}

// Server performs backward pass and updates weights
func serverBackwardAndUpdate(heContext *HEContext, serverModel *ServerModel, encGradients []*rlwe.Ciphertext, learningRate float64) error {
	// Convert the standard server model to a packed homomorphic one
	heServer, err := convertToPacked(serverModel, heContext)
	if err != nil {
		return fmt.Errorf("failed to convert to packed homomorphic model: %v", err)
	}

	fmt.Println("Performing weight update...")

	// Get the input placeholders (we don't have the actual inputs at this stage)
	// In a real implementation, we would cache these from the forward pass
	encInputs := make([]*rlwe.Ciphertext, InputDim)
	for i := 0; i < InputDim; i++ {
		// Create all-ones inputs for weight update
		// In the actual implementation we would use the real inputs that were cached
		values := make([]float64, heContext.params.N()/2)
		for j := range values {
			values[j] = 1.0 // Use 1.0 as a placeholder, in real implementation use actual inputs
		}

		pt := ckks.NewPlaintext(heContext.params, heContext.params.MaxLevel())
		heContext.encoder.Encode(values, pt)

		var err error
		encInputs[i], err = heContext.encryptor.EncryptNew(pt)
		if err != nil {
			return fmt.Errorf("error encrypting input placeholder: %v", err)
		}
	}

	// Use packedUpdate to update weights homomorphically
	err = packedUpdate(heContext, heServer, encInputs, encGradients, learningRate, BatchSize)
	if err != nil {
		return fmt.Errorf("homomorphic backward error: %v", err)
	}

	// For demonstration, decrypt the updated weights
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
				neuronIdx := b*NeuronsPerCT + n
				if neuronIdx < HiddenDim1 {
					serverModel.W1[i][neuronIdx] = values[n*BatchSize]
				}
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
			neuronIdx := b*NeuronsPerCT + n
			if neuronIdx < HiddenDim1 {
				serverModel.b1[neuronIdx] = values[n*BatchSize]
			}
		}
	}

	return nil
}

// serverBackwardAndUpdateHEParallel updates W1,b1 in-ciphertext using 'workers' goroutines
func serverBackwardAndUpdateHEParallel(
	ctx *HEContext,
	svr *HEServerModel,
	encX []*rlwe.Ciphertext, // len=inputDim
	encGradZ []*rlwe.Ciphertext, // len=hiddenDim1
	lr float64,
	B int,
	workers int,
) error {
	// Helper to capture first error
	var firstErr error
	var errMu sync.Mutex
	recordErr := func(e error) {
		errMu.Lock()
		if firstErr == nil {
			firstErr = e
		}
		errMu.Unlock()
	}

	if workers < 1 {
		workers = 1
	}
	params := ctx.params
	etaPt := scalarPlain(-lr/float64(B), params, ctx.encoder)

	rowsPerWkr := (HiddenDim1 + workers - 1) / workers
	var wg sync.WaitGroup

	for w := 0; w < workers; w++ {
		jStart := w * rowsPerWkr
		jEnd := min(HiddenDim1, jStart+rowsPerWkr)
		if jStart >= jEnd {
			continue
		}

		// each goroutine gets its own evaluator clone (thread-safe)
		eval := ctx.evaluator.ShallowCopy()

		wg.Add(1)
		go func(js, je int) {
			defer wg.Done()
			for j := js; j < je; j++ {
				// ------- bias update -------
				sumSlots, e := innerSumSlots(encGradZ[j], B, eval)
				if e != nil {
					recordErr(e)
					return
				}

				if e = eval.Mul(sumSlots, etaPt, sumSlots); e != nil {
					recordErr(e)
					return
				}
				if e = eval.Add(svr.b1[j], sumSlots, svr.b1[j]); e != nil {
					recordErr(e)
					return
				}

				// ------- weight updates over inputDim -------
				for i := 0; i < InputDim; i++ {
					prod, e := eval.MulNew(encX[i], encGradZ[j])
					if e != nil {
						recordErr(e)
						return
					}
					if e = eval.Relinearize(prod, prod); e != nil {
						recordErr(e)
						return
					}
					if e = eval.Mul(prod, etaPt, prod); e != nil {
						recordErr(e)
						return
					}
					if e = eval.Add(svr.W1[i][j], prod, svr.W1[i][j]); e != nil {
						recordErr(e)
						return
					}
				}
			}
		}(jStart, jEnd)
	}
	wg.Wait()
	return firstErr
}

// packedUpdate performs weight updates using packed ciphertexts for SIMD operations
func packedUpdate(
	ctx *HEContext,
	svr *HEServerPacked,
	encX []*rlwe.Ciphertext, // len=inputDim
	gradB []*rlwe.Ciphertext, // len=blk (=2)
	lr float64,
	B int,
) error {

	eta := scalarPlain(-lr/float64(B), ctx.params, ctx.encoder)
	mask := maskFirst(ctx.params, ctx.encoder, B) // keeps slot-0

	var wg sync.WaitGroup
	var firstErr error
	lock := sync.Mutex{}
	rec := func(e error) {
		lock.Lock()
		if firstErr == nil {
			firstErr = e
		}
		lock.Unlock()
	}

	rows := (InputDim + NumWorkers - 1) / NumWorkers
	for w := 0; w < NumWorkers; w++ {
		s, e := w*rows, min(InputDim, (w+1)*rows)
		if s >= e {
			continue
		}

		ev := ctx.evaluator.ShallowCopy()
		wg.Add(1)

		go func(start, end int) {
			defer wg.Done()

			for i := start; i < end; i++ {
				for blk := 0; blk < len(gradB); blk++ {

					// X[i] ⊙ gradB[blk]
					prod, err := ev.MulNew(encX[i], gradB[blk])
					if err != nil {
						rec(err)
						return
					}

					if err = ev.Relinearize(prod, prod); err != nil {
						rec(err)
						return
					}

					// Σ over batch
					sum, err := chunkSum(prod, B, ev)
					if err != nil {
						rec(err)
						return
					}

					// average × (−lr)
					if err = ev.Mul(sum, eta, sum); err != nil {
						rec(err)
						return
					}

					// keep single slot
					if err = ev.Mul(sum, mask, sum); err != nil {
						rec(err)
						return
					}

					// W ← W + Δ
					if err = ev.Add(svr.W[i][blk], sum, svr.W[i][blk]); err != nil {
						rec(err)
						return
					}
				}
			}
		}(s, e)
	}
	wg.Wait()
	if firstErr != nil {
		return firstErr
	}

	// --- bias update (much cheaper) ---------------------------------
	ev := ctx.evaluator
	for blk := 0; blk < len(gradB); blk++ {

		sum, err := chunkSum(gradB[blk], B, ev) // Σ_b dZ₁
		if err != nil {
			return err
		}
		if err = ev.Mul(sum, eta, sum); err != nil {
			return err
		}
		if err = ev.Mul(sum, mask, sum); err != nil {
			return err
		}
		if err = ev.Add(svr.b[blk], sum, svr.b[blk]); err != nil {
			return err
		}
	}
	return nil
}
