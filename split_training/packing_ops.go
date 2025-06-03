package split

import (
	"fmt"
	"sync"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// serverPackFeaturesFromSamples converts sample-wise packed ciphertexts to feature-wise packed ciphertexts.
// samplePackedCTs: [CT_sample0, CT_sample1, ...], where CT_sample_s = Enc(f_s0, f_s1, ..., f_s(N_FEATURES-1), 0...)
// Each feature f_si is assumed to occupy one slot, contiguously, starting from slot 0 in CT_sample_s.
// Output: [CT_feature0, CT_feature1, ...], where CT_feature_j = Enc(f_0j, f_1j, ..., f_(BATCH_SIZE-1)j, 0...)
// Feature f_sj from sample s will be placed in slot s of CT_feature_j.
// Assumes slotsPerFeatureElement is 1.
func serverPackFeaturesFromSamples(
	heCtx *HEContext,
	samplePackedCTs []*rlwe.Ciphertext,
	batchSize int,
	numFeatures int,
) ([]*rlwe.Ciphertext, error) {

	if len(samplePackedCTs) == 0 && numFeatures > 0 && batchSize > 0 { // Check if input implies work but is empty
		return nil, fmt.Errorf("samplePackedCTs is empty but batchSize and numFeatures are non-zero")
	}
	if numFeatures == 0 {
		return []*rlwe.Ciphertext{}, nil // No features to pack
	}
	if batchSize == 0 {
		// If batch size is 0, create numFeatures ciphertexts, each an encryption of zeros.
		// This ensures the output structure is consistent.
		featurePackedCTs := make([]*rlwe.Ciphertext, numFeatures)
		localEncryptor := rlwe.NewEncryptor(heCtx.params, heCtx.pk)
		zeroPt := ckks.NewPlaintext(heCtx.params, heCtx.params.MaxLevel())
		for j := 0; j < numFeatures; j++ {
			ct, err := localEncryptor.EncryptNew(zeroPt)
			if err != nil {
				return nil, fmt.Errorf("error encrypting zero plaintext for feature %d (empty batch): %v", j, err)
			}
			featurePackedCTs[j] = ct
		}
		return featurePackedCTs, nil
	}

	if len(samplePackedCTs) != batchSize {
		return nil, fmt.Errorf("len(samplePackedCTs) %d does not match batchSize %d", len(samplePackedCTs), batchSize)
	}

	featurePackedCTs := make([]*rlwe.Ciphertext, numFeatures)
	var mu sync.Mutex
	var firstError error

	maskSlot0Pt, err := heCtx.GetSlotMaskPT(0)
	if err != nil {
		return nil, fmt.Errorf("failed to get slot 0 mask plaintext: %v", err)
	}

	// Assuming parallelFor and NumWorkers are defined in the package (e.g., from forward.go or params.go)
	parallelFor(0, numFeatures, func(featureIdx int) {
		mu.Lock()
		if firstError != nil {
			mu.Unlock()
			return
		}
		mu.Unlock()

		localEncryptor := rlwe.NewEncryptor(heCtx.params, heCtx.pk)
		zeroPt := ckks.NewPlaintext(heCtx.params, heCtx.params.MaxLevel())
		targetCtJ, err := localEncryptor.EncryptNew(zeroPt)
		if err != nil {
			mu.Lock()
			if firstError == nil {
				firstError = fmt.Errorf("error encrypting zero plaintext for feature %d: %v", featureIdx, err)
			}
			mu.Unlock()
			return
		}

		for s := 0; s < batchSize; s++ {
			if samplePackedCTs[s] == nil {
				mu.Lock()
				if firstError == nil {
					firstError = fmt.Errorf("samplePackedCTs[%d] is nil for feature %d processing", s, featureIdx)
				}
				mu.Unlock()
				return
			}

			// Workaround: Use CopyNew() due to persistent linter issues with ShallowCopy variants.
			var tempCt *rlwe.Ciphertext
			if samplePackedCTs[s] != nil { // Guard against nil pointer dereference
				tempCt = samplePackedCTs[s].CopyNew()
			} else {
				// This case should be caught by the nil check for samplePackedCTs[s] earlier in the loop.
				// If somehow reached, propogate error or handle appropriately.
				mu.Lock()
				if firstError == nil {
					firstError = fmt.Errorf("critical: samplePackedCTs[%d] became nil before CopyNew for feature %d", s, featureIdx)
				}
				mu.Unlock()
				return
			}

			// Rotate to bring f_sj (feature `featureIdx` of sample `s`) to slot 0
			if err := heCtx.evaluator.Rotate(tempCt, -featureIdx, tempCt); err != nil {
				mu.Lock()
				if firstError == nil {
					firstError = fmt.Errorf("L0 pack: rotate sample %d for feature %d to slot 0: %v", s, featureIdx, err)
				}
				mu.Unlock()
				return
			}

			// Isolate f_sj at slot 0
			if err := heCtx.evaluator.Mul(tempCt, maskSlot0Pt, tempCt); err != nil {
				mu.Lock()
				if firstError == nil {
					firstError = fmt.Errorf("L0 pack: mul mask sample %d feature %d: %v", s, featureIdx, err)
				}
				mu.Unlock()
				return
			}
			if err := heCtx.evaluator.Rescale(tempCt, tempCt); err != nil {
				mu.Lock()
				if firstError == nil {
					firstError = fmt.Errorf("L0 pack: rescale mask sample %d feature %d: %v", s, featureIdx, err)
				}
				mu.Unlock()
				return
			}

			// Rotate f_sj (now isolated at slot 0 in tempCt) to slot `s`
			if err := heCtx.evaluator.Rotate(tempCt, s, tempCt); err != nil {
				mu.Lock()
				if firstError == nil {
					firstError = fmt.Errorf("L0 pack: rotate sample %d feature %d to slot %d: %v", s, featureIdx, s, err)
				}
				mu.Unlock()
				return
			}

			// Add to accumulator for feature `featureIdx`
			if err := heCtx.evaluator.Add(targetCtJ, tempCt, targetCtJ); err != nil {
				mu.Lock()
				if firstError == nil {
					firstError = fmt.Errorf("L0 pack: add sample %d to feature %d accumulator: %v", s, featureIdx, err)
				}
				mu.Unlock()
				return
			}
		}

		mu.Lock()
		if firstError == nil {
			featurePackedCTs[featureIdx] = targetCtJ
		}
		mu.Unlock()
	})

	if firstError != nil {
		return nil, firstError
	}
	return featurePackedCTs, nil
}

// Remove local parallelFor and min, assuming they are accessible package-wide.
