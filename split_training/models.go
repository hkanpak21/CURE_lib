package split

import (
	"fmt"
	"math/rand"
	"os"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// ClientModel represents the client-side part of the split learning model
type ClientModel struct {
	// Second layer (client's first layer)
	W2 [][]float64 // [hiddenDim1 x hiddenDim2] = [128 x 32]
	b2 []float64   // [hiddenDim2] = [32]

	// Output layer
	W3 [][]float64 // [hiddenDim2 x outputDim] = [32 x 10]
	b3 []float64   // [outputDim] = [10]
}

// ServerModel represents the server-side part of the split learning model
type ServerModel struct {
	// First layer
	W1 [][]float64 // [inputDim x hiddenDim1] = [784 x 128]
	b1 []float64   // [hiddenDim1] = [128]
}

// HEServerModel represents the fully homomorphic version of the server model
type HEServerModel struct {
	// Encrypted weights and biases
	W1 [][]*rlwe.Ciphertext // [inputDim][hiddenDim1]
	b1 []*rlwe.Ciphertext   // [hiddenDim1]
}

// HEServerPacked represents a packed version of the server model for SIMD operations
type HEServerPacked struct {
	// Packed as [inputDim][hiddenDim1/neuronsPerCT]
	W [][]*rlwe.Ciphertext
	// Packed biases: [hiddenDim1/neuronsPerCT]
	b []*rlwe.Ciphertext
}

// Initialize a client model with random weights
func initClientModel() *ClientModel {
	// Initialize with random weights and biases
	W2 := make([][]float64, HiddenDim1)
	for i := range W2 {
		W2[i] = make([]float64, HiddenDim2)
		for j := range W2[i] {
			W2[i][j] = rand.NormFloat64() * 0.1
		}
	}

	b2 := make([]float64, HiddenDim2)
	for i := range b2 {
		b2[i] = rand.NormFloat64() * 0.1
	}

	W3 := make([][]float64, HiddenDim2)
	for i := range W3 {
		W3[i] = make([]float64, OutputDim)
		for j := range W3[i] {
			W3[i][j] = rand.NormFloat64() * 0.1
		}
	}

	b3 := make([]float64, OutputDim)
	for i := range b3 {
		b3[i] = rand.NormFloat64() * 0.1
	}

	return &ClientModel{
		W2: W2,
		b2: b2,
		W3: W3,
		b3: b3,
	}
}

// Initialize a server model with random weights
func initServerModel() *ServerModel {
	// Initialize with random weights and biases
	W1 := make([][]float64, InputDim)
	for i := range W1 {
		W1[i] = make([]float64, HiddenDim1)
		for j := range W1[i] {
			W1[i][j] = rand.NormFloat64() * 0.1
		}
	}

	b1 := make([]float64, HiddenDim1)
	for i := range b1 {
		b1[i] = rand.NormFloat64() * 0.1
	}

	return &ServerModel{
		W1: W1,
		b1: b1,
	}
}

// convertToHomomorphicModel converts a standard ServerModel to a fully homomorphic HEServerModel
func convertToHomomorphicModel(serverModel *ServerModel, heContext *HEContext) (*HEServerModel, error) {
	heModel := &HEServerModel{
		W1: make([][]*rlwe.Ciphertext, InputDim),
		b1: make([]*rlwe.Ciphertext, HiddenDim1),
	}

	// Encrypt W1 (weights)
	for i := 0; i < InputDim; i++ {
		heModel.W1[i] = make([]*rlwe.Ciphertext, HiddenDim1)
		for j := 0; j < HiddenDim1; j++ {
			// Create plaintext
			pt := ckks.NewPlaintext(heContext.params, heContext.params.MaxLevel())

			// Create vector where all slots have the same weight value
			values := make([]float64, heContext.params.N()/2)
			for k := range values {
				values[k] = serverModel.W1[i][j]
			}

			// Encode values
			heContext.encoder.Encode(values, pt)

			// Encrypt
			var err error
			heModel.W1[i][j], err = heContext.encryptor.EncryptNew(pt)
			if err != nil {
				return nil, fmt.Errorf("error encrypting weight: %v", err)
			}
		}
	}

	// Encrypt b1 (biases)
	for i := 0; i < HiddenDim1; i++ {
		// Create plaintext
		pt := ckks.NewPlaintext(heContext.params, heContext.params.MaxLevel())

		// Create vector where all slots have the same bias value
		values := make([]float64, heContext.params.N()/2)
		for k := range values {
			values[k] = serverModel.b1[i]
		}

		// Encode values
		heContext.encoder.Encode(values, pt)

		// Encrypt
		var err error
		heModel.b1[i], err = heContext.encryptor.EncryptNew(pt)
		if err != nil {
			return nil, fmt.Errorf("error encrypting bias: %v", err)
		}
	}

	return heModel, nil
}

// convertToPacked converts a ServerModel to a packed HEServerPacked for SIMD operations
func convertToPacked(server *ServerModel, he *HEContext) (*HEServerPacked, error) {
	blk := HiddenDim1 / NeuronsPerCT // =2 for 128/64
	packed := &HEServerPacked{
		W: make([][]*rlwe.Ciphertext, InputDim),
		b: make([]*rlwe.Ciphertext, blk),
	}

	// helper alloc
	line := make([]float64, he.params.N()/2)

	// --- weights ---
	for i := 0; i < InputDim; i++ {
		packed.W[i] = make([]*rlwe.Ciphertext, blk)
		for b := 0; b < blk; b++ {
			for n := 0; n < NeuronsPerCT; n++ {
				copy(line[n*BatchSize:(n+1)*BatchSize],
					repeat(server.W1[i][b*NeuronsPerCT+n], BatchSize))
			}
			pt := ckks.NewPlaintext(he.params, he.params.MaxLevel())
			he.encoder.Encode(line, pt)
			ct, err := he.encryptor.EncryptNew(pt)
			if err != nil {
				return nil, err
			}
			packed.W[i][b] = ct
		}
	}

	// --- biases ---
	for b := 0; b < blk; b++ {
		for n := 0; n < NeuronsPerCT; n++ {
			copy(line[n*BatchSize:(n+1)*BatchSize],
				repeat(server.b1[b*NeuronsPerCT+n], BatchSize))
		}
		pt := ckks.NewPlaintext(he.params, he.params.MaxLevel())
		he.encoder.Encode(line, pt)
		ct, err := he.encryptor.EncryptNew(pt)
		if err != nil {
			return nil, err
		}
		packed.b[b] = ct
	}
	return packed, nil
}

// SaveModel saves the client and server models to files
func saveModel(clientModel *ClientModel, serverModel *ServerModel, clientPath, serverPath string) error {
	// Save client model
	clientFile, err := os.Create(clientPath)
	if err != nil {
		return fmt.Errorf("failed to create client model file: %v", err)
	}
	defer clientFile.Close()

	// Write client model dimensions
	fmt.Fprintf(clientFile, "%d %d %d\n", HiddenDim1, HiddenDim2, OutputDim)

	// Write W2
	for i := 0; i < HiddenDim1; i++ {
		for j := 0; j < HiddenDim2; j++ {
			fmt.Fprintf(clientFile, "%f ", clientModel.W2[i][j])
		}
		fmt.Fprintln(clientFile)
	}

	// Write b2
	for i := 0; i < HiddenDim2; i++ {
		fmt.Fprintf(clientFile, "%f ", clientModel.b2[i])
	}
	fmt.Fprintln(clientFile)

	// Write W3
	for i := 0; i < HiddenDim2; i++ {
		for j := 0; j < OutputDim; j++ {
			fmt.Fprintf(clientFile, "%f ", clientModel.W3[i][j])
		}
		fmt.Fprintln(clientFile)
	}

	// Write b3
	for i := 0; i < OutputDim; i++ {
		fmt.Fprintf(clientFile, "%f ", clientModel.b3[i])
	}
	fmt.Fprintln(clientFile)

	// Save server model
	serverFile, err := os.Create(serverPath)
	if err != nil {
		return fmt.Errorf("failed to create server model file: %v", err)
	}
	defer serverFile.Close()

	// Write server model dimensions
	fmt.Fprintf(serverFile, "%d %d\n", InputDim, HiddenDim1)

	// Write W1
	for i := 0; i < InputDim; i++ {
		for j := 0; j < HiddenDim1; j++ {
			fmt.Fprintf(serverFile, "%f ", serverModel.W1[i][j])
		}
		fmt.Fprintln(serverFile)
	}

	// Write b1
	for i := 0; i < HiddenDim1; i++ {
		fmt.Fprintf(serverFile, "%f ", serverModel.b1[i])
	}
	fmt.Fprintln(serverFile)

	return nil
}

// LoadModel loads client and server models from files
func loadModel(clientPath, serverPath string) (*ClientModel, *ServerModel, error) {
	// Load client model
	clientFile, err := os.Open(clientPath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open client model file: %v", err)
	}
	defer clientFile.Close()

	// Read client model dimensions
	var h1, h2, out int
	_, err = fmt.Fscanf(clientFile, "%d %d %d\n", &h1, &h2, &out)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read client model dimensions: %v", err)
	}

	// Check dimensions match expected values
	if h1 != HiddenDim1 || h2 != HiddenDim2 || out != OutputDim {
		return nil, nil, fmt.Errorf("model dimensions mismatch")
	}

	// Initialize client model
	clientModel := &ClientModel{
		W2: make([][]float64, HiddenDim1),
		b2: make([]float64, HiddenDim2),
		W3: make([][]float64, HiddenDim2),
		b3: make([]float64, OutputDim),
	}

	// Read W2
	for i := 0; i < HiddenDim1; i++ {
		clientModel.W2[i] = make([]float64, HiddenDim2)
		for j := 0; j < HiddenDim2; j++ {
			_, err = fmt.Fscanf(clientFile, "%f", &clientModel.W2[i][j])
			if err != nil {
				return nil, nil, fmt.Errorf("failed to read W2: %v", err)
			}
		}
		fmt.Fscanln(clientFile)
	}

	// Read b2
	for i := 0; i < HiddenDim2; i++ {
		_, err = fmt.Fscanf(clientFile, "%f", &clientModel.b2[i])
		if err != nil {
			return nil, nil, fmt.Errorf("failed to read b2: %v", err)
		}
	}
	fmt.Fscanln(clientFile)

	// Read W3
	for i := 0; i < HiddenDim2; i++ {
		clientModel.W3[i] = make([]float64, OutputDim)
		for j := 0; j < OutputDim; j++ {
			_, err = fmt.Fscanf(clientFile, "%f", &clientModel.W3[i][j])
			if err != nil {
				return nil, nil, fmt.Errorf("failed to read W3: %v", err)
			}
		}
		fmt.Fscanln(clientFile)
	}

	// Read b3
	for i := 0; i < OutputDim; i++ {
		_, err = fmt.Fscanf(clientFile, "%f", &clientModel.b3[i])
		if err != nil {
			return nil, nil, fmt.Errorf("failed to read b3: %v", err)
		}
	}

	// Load server model
	serverFile, err := os.Open(serverPath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open server model file: %v", err)
	}
	defer serverFile.Close()

	// Read server model dimensions
	var in, h int
	_, err = fmt.Fscanf(serverFile, "%d %d\n", &in, &h)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read server model dimensions: %v", err)
	}

	// Check dimensions match expected values
	if in != InputDim || h != HiddenDim1 {
		return nil, nil, fmt.Errorf("model dimensions mismatch")
	}

	// Initialize server model
	serverModel := &ServerModel{
		W1: make([][]float64, InputDim),
		b1: make([]float64, HiddenDim1),
	}

	// Read W1
	for i := 0; i < InputDim; i++ {
		serverModel.W1[i] = make([]float64, HiddenDim1)
		for j := 0; j < HiddenDim1; j++ {
			_, err = fmt.Fscanf(serverFile, "%f", &serverModel.W1[i][j])
			if err != nil {
				return nil, nil, fmt.Errorf("failed to read W1: %v", err)
			}
		}
		fmt.Fscanln(serverFile)
	}

	// Read b1
	for i := 0; i < HiddenDim1; i++ {
		_, err = fmt.Fscanf(serverFile, "%f", &serverModel.b1[i])
		if err != nil {
			return nil, nil, fmt.Errorf("failed to read b1: %v", err)
		}
	}
	fmt.Fscanln(serverFile)

	return clientModel, serverModel, nil
}
