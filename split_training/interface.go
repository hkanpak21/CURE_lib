package split

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// InitHE initializes and returns a new homomorphic encryption context
func InitHE() (*HEContext, error) {
	return initHE()
}

// InitClientModel initializes a new client model with the given configuration
func InitClientModel(config *ModelConfig) *ClientModel {
	return initClientModel(config)
}

// InitServerModel initializes a new server model with the given configuration
func InitServerModel(config *ModelConfig) *ServerModel {
	return initServerModel(config)
}

// ClientPrepareAndEncryptBatch prepares and encrypts a batch of images for processing
func ClientPrepareAndEncryptBatch(he *HEContext, imgs [][]float64, idx []int) ([]*rlwe.Ciphertext, error) {
	return clientPrepareAndEncryptBatch(he, imgs, idx)
}

// ServerForwardPass performs the forward pass on the server side
func ServerForwardPass(he *HEContext, serverModel *ServerModel, encInputs []*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error) {
	return serverForwardPass(he, serverModel, encInputs)
}

// ClientForwardAndBackward performs forward and backward passes on the client side
func ClientForwardAndBackward(heContext *HEContext, clientModel *ClientModel, encActivations []*rlwe.Ciphertext,
	labels []int, batchIndices []int) ([]*rlwe.Ciphertext, error) {
	return clientForwardAndBackward(heContext, clientModel, encActivations, labels, batchIndices)
}

// ServerBackwardAndUpdate performs the backward pass and updates the server model weights
func ServerBackwardAndUpdate(heContext *HEContext, serverModel *ServerModel, encGradients []*rlwe.Ciphertext,
	cachedInputs []*rlwe.Ciphertext, learningRate float64) error {
	return serverBackwardAndUpdate(heContext, serverModel, encGradients, cachedInputs, learningRate)
}
