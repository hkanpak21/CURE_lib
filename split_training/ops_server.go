package split

import (
	"fmt"
	"log"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func (hePackedModel *HEServerPacked) UpdateModelFromHE(heContext *HEContext, serverModel *ServerModel, actualBatchSize int) error {
	// Update Biases
	if hePackedModel.b == nil {
		return fmt.Errorf("cannot update biases, hePackedModel.b is nil")
	}
	if serverModel.Biases == nil {
		return fmt.Errorf("cannot update biases, serverModel.Biases is nil")
	}

	for layer := 0; layer < len(serverModel.Biases) && layer < len(hePackedModel.b); layer++ {
		if serverModel.Biases[layer] == nil && hePackedModel.b[layer] != nil {
			// Initialize serverModel.Biases[layer] if needed, based on expected output dim
			// This situation suggests a mismatch in initialization or prior state.
			// For now, we'll error if serverModel.Biases[layer] is nil but we have packed biases.
			return fmt.Errorf("serverModel.Biases[%d] is nil but hePackedModel.b[%d] is not, cannot update", layer, layer)
		}
		if hePackedModel.b[layer] == nil {
			log.Printf("Warning: Packed biases for layer %d are nil. Skipping bias update for this layer.", layer)
			continue
		}

		outputDim := serverModel.GetLayerOutputDim(layer)
		numBiasBlocksCalculated := (outputDim + hePackedModel.NeuronsPerCT - 1) / hePackedModel.NeuronsPerCT

		// Ensure numBiasBlocksCalculated does not exceed the actual number of blocks in hePackedModel.b[layer]
		numBiasBlocksToProcess := min(numBiasBlocksCalculated, len(hePackedModel.b[layer]))

		for blk := 0; blk < numBiasBlocksToProcess; blk++ {
			if hePackedModel.b[layer][blk] == nil {
				log.Printf("Warning: Bias ciphertext is nil at layer %d, block %d. Skipping update for this block.", layer, blk)
				continue
			}

			pt := heContext.GetDecryptor().DecryptNew(hePackedModel.b[layer][blk])
			// Get a fresh buffer for each decryption
			plainVector := GetFloat64Buffer()
			// Ensure the buffer is cleared (or rely on overwriting by Decode)
			// GetFloat64Buffer should return a slice of size params.N()/2

			if err := heContext.GetEncoder().Decode(pt, plainVector); err != nil {
				PutFloat64Buffer(plainVector) // Return buffer on error
				return fmt.Errorf("decode error for bias block %d layer %d: %w", blk, layer, err)
			}

			// Determine how many neurons are actually in this block for this layer
			// This should cap at outputDim for the layer.
			startNeuronInLayer := blk * hePackedModel.NeuronsPerCT
			neuronsInBlock := 0
			if startNeuronInLayer < outputDim {
				neuronsInBlock = min(hePackedModel.NeuronsPerCT, outputDim-startNeuronInLayer)
			}

			for j := 0; j < neuronsInBlock; j++ {
				outputIdx := startNeuronInLayer + j
				// serverModel.Biases[layer] should be pre-allocated correctly
				if outputIdx < len(serverModel.Biases[layer]) {
					biasSlotIndex := j * actualBatchSize // Value for j-th neuron in block is at start of its batch segment
					if biasSlotIndex < len(plainVector) {
						serverModel.Biases[layer][outputIdx] = plainVector[biasSlotIndex]
					} else {
						PutFloat64Buffer(plainVector) // Return buffer
						return fmt.Errorf("biasSlotIndex %d out of bounds for plainVector (len %d) in updateModelFromHE layer %d, block %d, neuron_in_block %d",
							biasSlotIndex, len(plainVector), layer, blk, j)
					}
				} else {
					// This means outputIdx is >= outputDim of the layer, which should not happen if neuronsInBlock is correct.
					// Or serverModel.Biases[layer] is too small.
					log.Printf("Warning: outputIdx %d for bias is out of bounds for serverModel.Biases[%d] (len %d). Skipping.", outputIdx, layer, len(serverModel.Biases[layer]))
				}
			}
			// Clear and return buffer after use
			for k := range plainVector {
				plainVector[k] = 0
			}
			PutFloat64Buffer(plainVector)
		}
	}

	// Update Weights
	if hePackedModel.W == nil {
		return fmt.Errorf("cannot update weights, hePackedModel.W is nil")
	}
	if serverModel.Weights == nil {
		return fmt.Errorf("cannot update weights, serverModel.Weights is nil")
	}

	for l := 0; l < len(serverModel.Weights) && l < len(hePackedModel.W); l++ {
		if serverModel.Weights[l] == nil && hePackedModel.W[l] != nil {
			return fmt.Errorf("serverModel.Weights[%d] is nil but hePackedModel.W[%d] is not, cannot update", l, l)
		}
		if hePackedModel.W[l] == nil {
			log.Printf("Warning: Packed weights for layer %d are nil. Skipping weight update for this layer.", l)
			continue
		}

		inputDim := serverModel.GetLayerInputDim(l)
		outputDim := serverModel.GetLayerOutputDim(l)
		numWeightBlocksCalculated := (outputDim + hePackedModel.NeuronsPerCT - 1) / hePackedModel.NeuronsPerCT

		for i := 0; i < inputDim && i < len(hePackedModel.W[l]); i++ {
			if serverModel.Weights[l][i] == nil && hePackedModel.W[l][i] != nil {
				return fmt.Errorf("serverModel.Weights[%d][%d] is nil but hePackedModel.W[%d][%d] is not, cannot update", l, i, l, i)
			}
			if hePackedModel.W[l][i] == nil {
				log.Printf("Warning: Packed weights for layer %d, input %d are nil. Skipping.", l, i)
				continue
			}

			numWeightBlocksToProcess := min(numWeightBlocksCalculated, len(hePackedModel.W[l][i]))

			for blk := 0; blk < numWeightBlocksToProcess; blk++ {
				if hePackedModel.W[l][i][blk] == nil {
					log.Printf("Warning: Weight ciphertext is nil at layer %d, input %d, block %d. Skipping update for this block.", l, i, blk)
					continue
				}
				pt := heContext.GetDecryptor().DecryptNew(hePackedModel.W[l][i][blk])
				plainVector := GetFloat64Buffer() // Get fresh buffer

				if err := heContext.GetEncoder().Decode(pt, plainVector); err != nil {
					PutFloat64Buffer(plainVector) // Return buffer on error
					return fmt.Errorf("decode error for weight layer %d input %d block %d: %w", l, i, blk, err)
				}

				startNeuronInLayer := blk * hePackedModel.NeuronsPerCT
				neuronsInBlock := 0
				if startNeuronInLayer < outputDim {
					neuronsInBlock = min(hePackedModel.NeuronsPerCT, outputDim-startNeuronInLayer)
				}

				for j := 0; j < neuronsInBlock; j++ {
					outputIdx := startNeuronInLayer + j
					// serverModel.Weights[l][i] should be pre-allocated.
					if outputIdx < len(serverModel.Weights[l][i]) {
						weightSlotIndex := j * actualBatchSize // Value for j-th neuron in block
						if weightSlotIndex < len(plainVector) {
							serverModel.SetWeight(l, i, outputIdx, plainVector[weightSlotIndex])
						} else {
							PutFloat64Buffer(plainVector) // Return buffer
							return fmt.Errorf("weightSlotIndex %d out of bounds for plainVector (len %d) in updateModelFromHE layer %d, input %d, block %d, neuron_in_block %d",
								weightSlotIndex, len(plainVector), l, i, blk, j)
						}
					} else {
						log.Printf("Warning: outputIdx %d for weight is out of bounds for serverModel.Weights[%d][%d] (len %d). Skipping.", outputIdx, l, i, len(serverModel.Weights[l][i]))
					}
				}
				// Clear and return buffer
				for k := range plainVector {
					plainVector[k] = 0
				}
				PutFloat64Buffer(plainVector)
			}
		}
	}
	return nil
}

// cachedLayerInputs[0] = feature-packed A_C (input to L0)
// cachedLayerInputs[l] for l>0 = feature-packed output of server layer l-1 (input to server layer l)
func ServerBackwardAndUpdatePacked(
	heContext *HEContext,
	serverModel *ServerModel, // Plaintext model, mainly for topology/dimensions
	hePackedModel *HEServerPacked, // Encrypted model to be updated
	encGradientsFromClient []*rlwe.Ciphertext,
	cachedLayerInputs [][]*rlwe.Ciphertext,
	learningRate float64,
	actualBatchSize int,
) error {
	// hePackedModel is now passed in and updated directly.

	numServerLayers := len(serverModel.Weights) // Use len of plaintext model for layer count
	if numServerLayers == 0 {
		return nil // No server layers to update
	}
	if numServerLayers != len(hePackedModel.W) || numServerLayers != len(hePackedModel.b) {
		return fmt.Errorf("mismatch between serverModel layer count (%d) and hePackedModel layers (W: %d, b: %d)",
			numServerLayers, len(hePackedModel.W), len(hePackedModel.b))
	}

	// gradOutputCTs[l] means dL/da_l (gradient of loss wrt output of server layer l)
	gradOutputCTs := make([][]*rlwe.Ciphertext, numServerLayers)

	// Initialize gradOutputCTs for the layer that received gradients from the client
	// This corresponds to dL/da for the output of the last server layer.
	lastServerLayerIdx := numServerLayers - 1
	outputDimLastLayer := serverModel.GetLayerOutputDim(lastServerLayerIdx)
	numBlocksLastLayer := (outputDimLastLayer + hePackedModel.NeuronsPerCT - 1) / hePackedModel.NeuronsPerCT

	if len(encGradientsFromClient) != numBlocksLastLayer {
		return fmt.Errorf("mismatch in number of gradient blocks from client (%d) and expected for last server layer output (%d blocks for %d neurons)",
			len(encGradientsFromClient), numBlocksLastLayer, outputDimLastLayer)
	}
	gradOutputCTs[lastServerLayerIdx] = make([]*rlwe.Ciphertext, numBlocksLastLayer)
	for blk := 0; blk < numBlocksLastLayer; blk++ {
		if encGradientsFromClient[blk] == nil {
			return fmt.Errorf("encGradientsFromClient[%d] is nil", blk)
		}
		gradOutputCTs[lastServerLayerIdx][blk] = encGradientsFromClient[blk].CopyNew() // Clone to avoid modifying input
		log.Printf("[L%d B%d Grads] Received from Client Scale: %.5f, Level: %d", lastServerLayerIdx, blk, gradOutputCTs[lastServerLayerIdx][blk].Scale.Float64(), gradOutputCTs[lastServerLayerIdx][blk].Level())
	}

	// Iterate backwards through server layers
	for l := numServerLayers - 1; l >= 0; l-- {
		log.Printf("Starting backward update for server layer %d", l)

		// --- Gradient Propagation to Previous Layer (dL/da_{l-1}) ---
		if l > 0 {
			// This section calculates dL/da_{l-1} using dL/da_l and W_l.
			// dL/da_{l-1,k} = sum_j (dL/da_{l,j} * W_{l,kj}) * ReLU'(a_{l-1,k})
			// This is a complex operation involving matrix multiplication with packed ciphertexts.
			// For each input neuron k to layer l (output of layer l-1):
			//   contribution_k = 0
			//   For each output neuron j of layer l:
			//     contribution_k += gradOutputCTs[l][block_for_j] (extract slot for j) * hePackedModel.W[l][k_as_input_idx_to_l][block_for_j] (extract W_kj)
			// This part remains a significant TODO for multi-layer server backprop.
			// The current structure implies gradOutputCTs[l-1] would be populated here.
			// For now, we'll assume for single-layer server tests, this isn't strictly needed *if* only the last layer updates itself.
			// But for multi-layer, this is essential.

			// Placeholder for gradOutputCTs[l-1]
			prevLayerInputDim := serverModel.GetLayerInputDim(l) // = output dim of layer l-1
			numPrevBlocks := (prevLayerInputDim + hePackedModel.NeuronsPerCT - 1) / hePackedModel.NeuronsPerCT
			gradOutputCTs[l-1] = make([]*rlwe.Ciphertext, numPrevBlocks)
			// Fill with dummy/zero ciphertexts if not implemented
			for pb := 0; pb < numPrevBlocks; pb++ {
				zeroPt := ckks.NewPlaintext(heContext.params, heContext.params.MaxLevel())
				// encoder.Encode zeros... // TODO: encode actual zeros
				// For now, just encrypting an empty plaintext. This should be properly initialized.
				// Example: heContext.encoder.Encode(make([]float64, heContext.params.N()/2), zeroPt)
				gradOutputCTs[l-1][pb], _ = heContext.encryptor.EncryptNew(zeroPt)

			}
			log.Printf("[ServerBackwardUpdate L%d] Gradient propagation to previous server layer (L%d) is NOT YET FULLY IMPLEMENTED.", l, l-1)

		}

		// --- Update Biases for current layer l ---
		outputDim := serverModel.GetLayerOutputDim(l)
		numBiasBlocks := (outputDim + hePackedModel.NeuronsPerCT - 1) / hePackedModel.NeuronsPerCT

		for blk := 0; blk < numBiasBlocks; blk++ {
			if gradOutputCTs[l] == nil || blk >= len(gradOutputCTs[l]) || gradOutputCTs[l][blk] == nil {
				return fmt.Errorf("gradOutputCTs[%d][%d] is nil before cloning for bias update", l, blk)
			}
			deltaBiasCandidateCt := gradOutputCTs[l][blk].CopyNew()
			log.Printf("[L%d B%d BiasUpd] Initial grad dL/da_l Scale: %.2f, Lvl: %d", l, blk, deltaBiasCandidateCt.Scale.Float64(), deltaBiasCandidateCt.Level())

			// Sum gradients across the batch for each bias: sum_samples (dL/da_l)
			if err := heContext.evaluator.InnerSum(deltaBiasCandidateCt, actualBatchSize, hePackedModel.NeuronsPerCT, deltaBiasCandidateCt); err != nil {
				return fmt.Errorf("InnerSum error for BiasUpdate L%d B%d: %w", l, blk, err)
			}
			log.Printf("[L%d B%d BiasUpd] Post-InnerSum (sum_batch(dL/da_l)) Scale: %.2f, Lvl: %d", l, blk, deltaBiasCandidateCt.Scale.Float64(), deltaBiasCandidateCt.Level())

			// Multiply by -learningRate / actualBatchSize
			lrScalar := -learningRate / float64(actualBatchSize)
			lrBsPt := ckks.NewPlaintext(heContext.params, heContext.params.MaxLevel())
			if deltaBiasCandidateCt.Level() < lrBsPt.Level() {
				lrBsPt.Resize(deltaBiasCandidateCt.Degree(), deltaBiasCandidateCt.Level())
			}
			// Create a slice of the scalar repeated for all slots
			scalarVector := make([]float64, heContext.params.N()>>1)
			for i := range scalarVector {
				scalarVector[i] = lrScalar
			}
			heContext.encoder.Encode(scalarVector, lrBsPt)
			// For Mul(ct, pt), Lattigo recommends pt.Scale = ct.Scale for scalar mult or pt.Scale = params.DefaultScale()
			// If pt.Scale matches ct.Scale, then ct = ct * pt[0] (effectively).
			// If pt.Scale is DefaultScale, then ct = ct * (pt[0] * ct.Scale / DefaultScale)
			// To achieve simple scalar multiplication (ct_out = scalar * ct_in), and keep ct_in's scale,
			// we'd typically use MulConst, or ensure pt has the scalar and its scale is DefaultScale.
			// The Mul operation will result in scale_ct * scale_pt. Then Rescale divides by DefaultScale.
			// So if pt has scalar s and scale DefaultScale, then (ct * s) has scale OldScale*DefaultScale.
			// After rescale: (ct*s) has scale OldScale. This is usually desired for scalar mult.
			lrBsPt.Scale = heContext.params.DefaultScale()

			log.Printf("[L%d B%d BiasUpd] lrBsPt Scale: %.2f, Lvl: %d. deltaBias Scale: %.2f, Lvl: %d", l, blk, lrBsPt.Scale.Float64(), lrBsPt.Level(), deltaBiasCandidateCt.Scale.Float64(), deltaBiasCandidateCt.Level())

			if err := heContext.evaluator.Mul(deltaBiasCandidateCt, lrBsPt, deltaBiasCandidateCt); err != nil {
				return fmt.Errorf("error multiplying bias delta by (lr/bs) for L%d B%d: %w", l, blk, err)
			}
			log.Printf("[L%d B%d BiasUpd] Post-LR-Mul Scale: %.2f, Lvl: %d", l, blk, deltaBiasCandidateCt.Scale.Float64(), deltaBiasCandidateCt.Level())

			if err := heContext.evaluator.Rescale(deltaBiasCandidateCt, deltaBiasCandidateCt); err != nil {
				return fmt.Errorf("error rescaling bias delta for L%d B%d: %w", l, blk, err)
			}
			log.Printf("[L%d B%d BiasUpd] Post-LR-Rescale (final delta_B) Scale: %.2f, Lvl: %d", l, blk, deltaBiasCandidateCt.Scale.Float64(), deltaBiasCandidateCt.Level())

			if hePackedModel.b[l][blk] == nil {
				return fmt.Errorf("nil bias ciphertext b[%d][%d] before adding delta", l, blk)
			}
			log.Printf("[L%d B%d BiasUpd] B_old Scale: %.2f, Lvl: %d", l, blk, hePackedModel.b[l][blk].Scale.Float64(), hePackedModel.b[l][blk].Level())

			if err := heContext.evaluator.Add(hePackedModel.b[l][blk], deltaBiasCandidateCt, hePackedModel.b[l][blk]); err != nil {
				return fmt.Errorf("error adding delta to b[%d][%d]: %w", l, blk, err)
			}
			log.Printf("[L%d B%d BiasUpd] B_new Scale: %.2f, Lvl: %d", l, blk, hePackedModel.b[l][blk].Scale.Float64(), hePackedModel.b[l][blk].Level())
		}

		// --- Update Weights for current layer l ---
		inputDimLayerL := serverModel.GetLayerInputDim(l)
		numWeightBlocks := (outputDim + hePackedModel.NeuronsPerCT - 1) / hePackedModel.NeuronsPerCT

		for i_in := 0; i_in < inputDimLayerL; i_in++ {
			if l >= len(cachedLayerInputs) || i_in >= len(cachedLayerInputs[l]) || cachedLayerInputs[l][i_in] == nil {
				return fmt.Errorf("nil cached input activation: cachedLayerInputs[L%d][idx%d]", l, i_in)
			}
			actCipher := cachedLayerInputs[l][i_in]
			log.Printf("[L%d W_in%d WU] actCipher (a_{l-1,i_in}) Scale: %.2f, Lvl: %d", l, i_in, actCipher.Scale.Float64(), actCipher.Level())

			for blk := 0; blk < numWeightBlocks; blk++ {
				if gradOutputCTs[l] == nil || blk >= len(gradOutputCTs[l]) || gradOutputCTs[l][blk] == nil {
					return fmt.Errorf("gradOutputCTs[L%d][B%d] is nil for weight update", l, blk)
				}
				gradBlockCt := gradOutputCTs[l][blk]
				log.Printf("[L%d W_in%d B%d WU] gradBlock (dL/da_l) Scale: %.2f, Lvl: %d", l, i_in, blk, gradBlockCt.Scale.Float64(), gradBlockCt.Level())

				prodCt, err := heContext.evaluator.MulNew(gradBlockCt, actCipher)
				if err != nil {
					return fmt.Errorf("Mul error (grad*act) for W[L%d][in%d][B%d]: %w", l, i_in, blk, err)
				}
				log.Printf("[L%d W_in%d B%d WU] prodCt (grad*act) Initial Scale: %.2f, Lvl: %d", l, i_in, blk, prodCt.Scale.Float64(), prodCt.Level())

				if err := heContext.evaluator.Rescale(prodCt, prodCt); err != nil {
					return fmt.Errorf("Rescale error (grad*act) for W[L%d][in%d][B%d]: %w", l, i_in, blk, err)
				}
				log.Printf("[L%d W_in%d B%d WU] prodCt (grad*act) Post-Rescale Scale: %.2f, Lvl: %d", l, i_in, blk, prodCt.Scale.Float64(), prodCt.Level())

				deltaWCandidateCt := prodCt
				if err = heContext.evaluator.InnerSum(deltaWCandidateCt, actualBatchSize, hePackedModel.NeuronsPerCT, deltaWCandidateCt); err != nil {
					return fmt.Errorf("InnerSum error for dW for W[L%d][in%d][B%d]: %w", l, i_in, blk, err)
				}
				log.Printf("[L%d W_in%d B%d WU] deltaW Post-InnerSum Scale: %.2f, Lvl: %d", l, i_in, blk, deltaWCandidateCt.Scale.Float64(), deltaWCandidateCt.Level())

				lrScalarW := -learningRate / float64(actualBatchSize)
				lrWeightPt := ckks.NewPlaintext(heContext.params, heContext.params.MaxLevel())
				if deltaWCandidateCt.Level() < lrWeightPt.Level() {
					lrWeightPt.Resize(deltaWCandidateCt.Degree(), deltaWCandidateCt.Level())
				}
				// Create a slice of the scalar repeated for all slots
				scalarWVector := make([]float64, heContext.params.N()>>1)
				for i_slot := range scalarWVector {
					scalarWVector[i_slot] = lrScalarW
				}
				heContext.encoder.Encode(scalarWVector, lrWeightPt)
				lrWeightPt.Scale = heContext.params.DefaultScale()

				log.Printf("[L%d W_in%d B%d WU] lrWeightPt Scale: %.2f, Lvl: %d. deltaW Scale: %.2f, Lvl: %d", l, i_in, blk, lrWeightPt.Scale.Float64(), lrWeightPt.Level(), deltaWCandidateCt.Scale.Float64(), deltaWCandidateCt.Level())

				if err := heContext.evaluator.Mul(deltaWCandidateCt, lrWeightPt, deltaWCandidateCt); err != nil {
					return fmt.Errorf("Mul error (deltaW * lr/bs) for W[L%d][in%d][B%d]: %w", l, i_in, blk, err)
				}
				log.Printf("[L%d W_in%d B%d WU] deltaW Post-LR-Mul Scale: %.2f, Lvl: %d", l, i_in, blk, deltaWCandidateCt.Scale.Float64(), deltaWCandidateCt.Level())

				if err := heContext.evaluator.Rescale(deltaWCandidateCt, deltaWCandidateCt); err != nil {
					return fmt.Errorf("Rescale error (deltaW * lr/bs) for W[L%d][in%d][B%d]: %w", l, i_in, blk, err)
				}
				log.Printf("[L%d W_in%d B%d WU] deltaW Post-LR-Rescale (final delta_W) Scale: %.2f, Lvl: %d", l, i_in, blk, deltaWCandidateCt.Scale.Float64(), deltaWCandidateCt.Level())

				if hePackedModel.W[l][i_in][blk] == nil {
					return fmt.Errorf("nil weight W[L%d][in%d][B%d] before adding delta", l, i_in, blk)
				}
				log.Printf("[L%d W_in%d B%d WU] W_old Scale: %.2f, Lvl: %d", l, i_in, blk, hePackedModel.W[l][i_in][blk].Scale.Float64(), hePackedModel.W[l][i_in][blk].Level())

				if err := heContext.evaluator.Add(hePackedModel.W[l][i_in][blk], deltaWCandidateCt, hePackedModel.W[l][i_in][blk]); err != nil {
					return fmt.Errorf("Add error (W_old + delta_W) for W[L%d][in%d][B%d]: %w", l, i_in, blk, err)
				}
				log.Printf("[L%d W_in%d B%d WU] W_new Scale: %.2f, Lvl: %d", l, i_in, blk, hePackedModel.W[l][i_in][blk].Scale.Float64(), hePackedModel.W[l][i_in][blk].Level())
			}
		}
	}
	log.Println("ServerBackwardAndUpdate finished all layer updates.")
	return nil
}
