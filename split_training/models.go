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
	// Weights and biases for client layers
	Weights [][][]float64 // 3D array [layer][input][output]
	Biases  [][]float64   // Array of bias vectors for each layer
	Config  *ModelConfig  // Architecture configuration
}

// ServerModel represents the server-side part of the split learning model
type ServerModel struct {
	// Weights and biases for server layers
	Weights [][][]float64 // 3D array [layer][input][output]
	Biases  [][]float64   // Array of bias vectors for each layer
	Config  *ModelConfig  // Architecture configuration
}

// HEServerModel represents the fully homomorphic version of the server model
type HEServerModel struct {
	// Encrypted weights and biases
	Weights [][][]*rlwe.Ciphertext // Array of weight matrices for each layer
	Biases  [][]*rlwe.Ciphertext   // Array of bias vectors for each layer
	Config  *ModelConfig           // Architecture configuration
}

// HEServerPacked represents a packed version of the server model for SIMD operations
type HEServerPacked struct {
	// Packed weights as [layer][input_dim][output_dim/neuronsPerCT]
	W [][][]*rlwe.Ciphertext
	// Packed biases: [layer][output_dim/neuronsPerCT]
	b [][]*rlwe.Ciphertext
	// Number of neurons per ciphertext
	NeuronsPerCT int
	// Architecture configuration
	Config *ModelConfig
}

// Initialize a client model with random weights
func initClientModel(config *ModelConfig) *ClientModel {
	if config == nil {
		config = &ModelConfig{
			Arch:     DefaultArch,
			SplitIdx: 0,
		}
	}

	clientLayers := len(config.Arch) - config.SplitIdx - 1
	weights := make([][][]float64, clientLayers)
	biases := make([][]float64, clientLayers)

	// Initialize weights and biases for each layer
	for l := 0; l < clientLayers; l++ {
		// Get the actual dimensions from the architecture configuration
		// This ensures the dimensions match between GetLayerInputDim and weight array sizes
		inputDim := config.Arch[l+config.SplitIdx]
		outputDim := config.Arch[l+config.SplitIdx+1]

		// Initialize with scaled random weights (Xavier initialization)
		scale := 1.0 / float64(inputDim)
		weights[l] = make([][]float64, inputDim)
		for i := range weights[l] {
			weights[l][i] = make([]float64, outputDim)
			for j := range weights[l][i] {
				weights[l][i][j] = rand.NormFloat64() * scale
			}
		}

		biases[l] = make([]float64, outputDim)
		for i := range biases[l] {
			biases[l][i] = rand.NormFloat64() * 0.1
		}
	}

	return &ClientModel{
		Weights: weights,
		Biases:  biases,
		Config:  config,
	}
}

// Initialize a server model with random weights
func initServerModel(config *ModelConfig) *ServerModel {
	if config == nil {
		config = &ModelConfig{
			Arch:     DefaultArch,
			SplitIdx: 0,
		}
	}

	serverLayers := config.SplitIdx + 1
	weights := make([][][]float64, serverLayers-1) // No weights for input layer
	biases := make([][]float64, serverLayers-1)    // No biases for input layer

	// Initialize weights and biases for each layer
	for l := 0; l < serverLayers-1; l++ {
		inputDim := config.Arch[l]
		outputDim := config.Arch[l+1]

		// Initialize with scaled random weights (Xavier initialization)
		scale := 1.0 / float64(inputDim)
		weights[l] = make([][]float64, inputDim)
		for i := range weights[l] {
			weights[l][i] = make([]float64, outputDim)
			for j := range weights[l][i] {
				weights[l][i][j] = rand.NormFloat64() * scale
			}
		}

		biases[l] = make([]float64, outputDim)
		for i := range biases[l] {
			biases[l][i] = rand.NormFloat64() * 0.1
		}
	}

	return &ServerModel{
		Weights: weights,
		Biases:  biases,
		Config:  config,
	}
}

// Get the weight at specific layer, input, output indices
func (s *ServerModel) GetWeight(layer, input, output int) float64 {
	if layer < 0 || layer >= len(s.Weights) ||
		input < 0 || input >= len(s.Weights[layer]) ||
		output < 0 || output >= len(s.Weights[layer][input]) {
		fmt.Printf("Warning: Invalid weight index: [%d][%d][%d], weights dims: %d x %d x %d\n",
			layer, input, output, len(s.Weights),
			len(s.Weights[layer]), len(s.Weights[layer][input]))
		return 0
	}
	return s.Weights[layer][input][output]
}

// Set the weight at specific layer, input, output indices
func (s *ServerModel) SetWeight(layer, input, output int, value float64) {
	if layer < 0 || layer >= len(s.Weights) ||
		input < 0 || input >= len(s.Weights[layer]) ||
		output < 0 || output >= len(s.Weights[layer][input]) {
		fmt.Printf("Warning: Invalid weight index: [%d][%d][%d], weights dims: %d x %d x %d\n",
			layer, input, output, len(s.Weights),
			len(s.Weights[layer]), len(s.Weights[layer][input]))
		return
	}
	s.Weights[layer][input][output] = value
}

// Get the weight at specific layer, input, output indices
func (c *ClientModel) GetWeight(layer, input, output int) float64 {
	if layer < 0 || layer >= len(c.Weights) ||
		input < 0 || input >= len(c.Weights[layer]) ||
		output < 0 || output >= len(c.Weights[layer][input]) {
		fmt.Printf("Warning: Invalid weight index: [%d][%d][%d], weights dims: %d x %d x %d\n",
			layer, input, output, len(c.Weights),
			len(c.Weights[layer]), len(c.Weights[layer][input]))
		return 0
	}
	return c.Weights[layer][input][output]
}

// Set the weight at specific layer, input, output indices
func (c *ClientModel) SetWeight(layer, input, output int, value float64) {
	if layer < 0 || layer >= len(c.Weights) ||
		input < 0 || input >= len(c.Weights[layer]) ||
		output < 0 || output >= len(c.Weights[layer][input]) {
		fmt.Printf("Warning: Invalid weight index: [%d][%d][%d], weights dims: %d x %d x %d\n",
			layer, input, output, len(c.Weights),
			len(c.Weights[layer]), len(c.Weights[layer][input]))
		return
	}
	c.Weights[layer][input][output] = value
}

// GetLayerInputDim returns the input dimension for a given layer
func (s *ServerModel) GetLayerInputDim(layer int) int {
	return s.Config.Arch[layer]
}

// GetLayerOutputDim returns the output dimension for a given layer
func (s *ServerModel) GetLayerOutputDim(layer int) int {
	return s.Config.Arch[layer+1]
}

// GetLayerInputDim returns the input dimension for a given layer
func (c *ClientModel) GetLayerInputDim(layer int) int {
	return c.Config.Arch[layer+c.Config.SplitIdx]
}

// GetLayerOutputDim returns the output dimension for a given layer
func (c *ClientModel) GetLayerOutputDim(layer int) int {
	return c.Config.Arch[layer+c.Config.SplitIdx+1]
}

// convertToHomomorphicModel converts a standard ServerModel to a fully homomorphic HEServerModel
func convertToHomomorphicModel(serverModel *ServerModel, heContext *HEContext) (*HEServerModel, error) {
	serverLayers := len(serverModel.Weights)
	heModel := &HEServerModel{
		Weights: make([][][]*rlwe.Ciphertext, serverLayers),
		Biases:  make([][]*rlwe.Ciphertext, serverLayers),
		Config:  serverModel.Config,
	}

	// For each layer in the server model
	for l := 0; l < serverLayers; l++ {
		inputDim := serverModel.GetLayerInputDim(l)
		outputDim := serverModel.GetLayerOutputDim(l)

		// Allocate arrays for this layer
		heModel.Weights[l] = make([][]*rlwe.Ciphertext, inputDim)
		heModel.Biases[l] = make([]*rlwe.Ciphertext, outputDim)

		// Encrypt weights
		for i := 0; i < inputDim; i++ {
			heModel.Weights[l][i] = make([]*rlwe.Ciphertext, outputDim)
			for j := 0; j < outputDim; j++ {
				// Create plaintext
				pt := ckks.NewPlaintext(heContext.params, heContext.params.MaxLevel())

				// Create vector where all slots have the same weight value
				values := make([]float64, heContext.params.N()/2)
				for k := range values {
					values[k] = serverModel.Weights[l][i][j]
				}

				// Encode values
				heContext.encoder.Encode(values, pt)

				// Encrypt
				var err error
				heModel.Weights[l][i][j], err = heContext.encryptor.EncryptNew(pt)
				if err != nil {
					return nil, fmt.Errorf("error encrypting weight: %v", err)
				}
			}
		}

		// Encrypt biases
		for j := 0; j < outputDim; j++ {
			// Create plaintext
			pt := ckks.NewPlaintext(heContext.params, heContext.params.MaxLevel())

			// Create vector where all slots have the same bias value
			values := make([]float64, heContext.params.N()/2)
			for k := range values {
				values[k] = serverModel.Biases[l][j]
			}

			// Encode values
			heContext.encoder.Encode(values, pt)

			// Encrypt
			var err error
			heModel.Biases[l][j], err = heContext.encryptor.EncryptNew(pt)
			if err != nil {
				return nil, fmt.Errorf("error encrypting bias: %v", err)
			}
		}
	}

	return heModel, nil
}

// convertToPacked converts a ServerModel to a packed HEServerPacked for SIMD operations
func convertToPacked(server *ServerModel, he *HEContext) (*HEServerPacked, error) {
	serverLayers := len(server.Weights)

	// Calculate how many neurons to pack per ciphertext
	slots := he.params.N() / 2
	var neuronsPerCT int
	neuronsPerCT = calculateNeuronsPerCT(slots, BatchSize, 64) // Default max is 64 neurons

	packed := &HEServerPacked{
		W:            make([][][]*rlwe.Ciphertext, serverLayers),
		b:            make([][]*rlwe.Ciphertext, serverLayers),
		NeuronsPerCT: neuronsPerCT,
		Config:       server.Config,
	}

	// For each layer in the server model
	for l := 0; l < serverLayers; l++ {
		inputDim := server.GetLayerInputDim(l)
		outputDim := server.GetLayerOutputDim(l)

		// Calculate how many blocks needed
		numBlocks := (outputDim + neuronsPerCT - 1) / neuronsPerCT

		// Allocate arrays for this layer
		packed.W[l] = make([][]*rlwe.Ciphertext, inputDim)
		packed.b[l] = make([]*rlwe.Ciphertext, numBlocks)

		// helper alloc for encoding
		line := make([]float64, he.params.N()/2)

		// --- weights ---
		for i := 0; i < inputDim; i++ {
			packed.W[l][i] = make([]*rlwe.Ciphertext, numBlocks)

			for b := 0; b < numBlocks; b++ {
				// Clear the line buffer
				for k := range line {
					line[k] = 0
				}

				// Pack neurons for this block
				startNeuron := b * neuronsPerCT
				endNeuron := min(startNeuron+neuronsPerCT, outputDim)

				for n := startNeuron; n < endNeuron; n++ {
					neuronOffset := (n - startNeuron) * BatchSize
					weight := server.Weights[l][i][n]

					// Replicate the weight value across batch slots
					for batch := 0; batch < BatchSize; batch++ {
						line[neuronOffset+batch] = weight
					}
				}

				// Encode and encrypt
				pt := ckks.NewPlaintext(he.params, he.params.MaxLevel())
				he.encoder.Encode(line, pt)
				ct, err := he.encryptor.EncryptNew(pt)
				if err != nil {
					return nil, err
				}
				packed.W[l][i][b] = ct
			}
		}

		// --- biases ---
		for b := 0; b < numBlocks; b++ {
			// Clear the line buffer
			for k := range line {
				line[k] = 0
			}

			// Pack biases for this block
			startNeuron := b * neuronsPerCT
			endNeuron := min(startNeuron+neuronsPerCT, outputDim)

			for n := startNeuron; n < endNeuron; n++ {
				neuronOffset := (n - startNeuron) * BatchSize
				bias := server.Biases[l][n]

				// Replicate the bias value across batch slots
				for batch := 0; batch < BatchSize; batch++ {
					line[neuronOffset+batch] = bias
				}
			}

			// Encode and encrypt
			pt := ckks.NewPlaintext(he.params, he.params.MaxLevel())
			he.encoder.Encode(line, pt)
			ct, err := he.encryptor.EncryptNew(pt)
			if err != nil {
				return nil, err
			}
			packed.b[l][b] = ct
		}
	}

	return packed, nil
}

// saveModel saves client and server models to files
func saveModel(clientModel *ClientModel, serverModel *ServerModel, clientPath, serverPath string) error {
	// Create client model file
	clientFile, err := os.Create(clientPath)
	if err != nil {
		return fmt.Errorf("failed to create client model file: %v", err)
	}
	defer clientFile.Close()

	// Create server model file
	serverFile, err := os.Create(serverPath)
	if err != nil {
		return fmt.Errorf("failed to create server model file: %v", err)
	}
	defer serverFile.Close()

	// Save client model
	// First, save architecture and split point
	fmt.Fprintf(clientFile, "%d %d\n", len(clientModel.Config.Arch), clientModel.Config.SplitIdx)
	for _, dim := range clientModel.Config.Arch {
		fmt.Fprintf(clientFile, "%d ", dim)
	}
	fmt.Fprintln(clientFile)

	// Then save weights and biases for each layer
	fmt.Fprintf(clientFile, "%d\n", len(clientModel.Weights)) // Number of layers
	for l := 0; l < len(clientModel.Weights); l++ {
		inputDim := clientModel.GetLayerInputDim(l)
		outputDim := clientModel.GetLayerOutputDim(l)

		// Save dimensions
		fmt.Fprintf(clientFile, "%d %d\n", inputDim, outputDim)

		// Save weights
		for i := 0; i < inputDim; i++ {
			for j := 0; j < outputDim; j++ {
				fmt.Fprintf(clientFile, "%f ", clientModel.Weights[l][i][j])
			}
			fmt.Fprintln(clientFile)
		}

		// Save biases
		for j := 0; j < outputDim; j++ {
			fmt.Fprintf(clientFile, "%f ", clientModel.Biases[l][j])
		}
		fmt.Fprintln(clientFile)
	}

	// Save server model
	// First, save architecture and split point
	fmt.Fprintf(serverFile, "%d %d\n", len(serverModel.Config.Arch), serverModel.Config.SplitIdx)
	for _, dim := range serverModel.Config.Arch {
		fmt.Fprintf(serverFile, "%d ", dim)
	}
	fmt.Fprintln(serverFile)

	// Then save weights and biases for each layer
	fmt.Fprintf(serverFile, "%d\n", len(serverModel.Weights)) // Number of layers
	for l := 0; l < len(serverModel.Weights); l++ {
		inputDim := serverModel.GetLayerInputDim(l)
		outputDim := serverModel.GetLayerOutputDim(l)

		// Save dimensions
		fmt.Fprintf(serverFile, "%d %d\n", inputDim, outputDim)

		// Save weights
		for i := 0; i < inputDim; i++ {
			for j := 0; j < outputDim; j++ {
				fmt.Fprintf(serverFile, "%f ", serverModel.Weights[l][i][j])
			}
			fmt.Fprintln(serverFile)
		}

		// Save biases
		for j := 0; j < outputDim; j++ {
			fmt.Fprintf(serverFile, "%f ", serverModel.Biases[l][j])
		}
		fmt.Fprintln(serverFile)
	}

	return nil
}

// loadModel loads client and server models from files
func loadModel(clientPath, serverPath string) (*ClientModel, *ServerModel, error) {
	// Open client model file
	clientFile, err := os.Open(clientPath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open client model file: %v", err)
	}
	defer clientFile.Close()

	// Open server model file
	serverFile, err := os.Open(serverPath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open server model file: %v", err)
	}
	defer serverFile.Close()

	// Load client model
	// First, load architecture and split point
	var archLen, splitIdx int
	if _, err := fmt.Fscanf(clientFile, "%d %d\n", &archLen, &splitIdx); err != nil {
		return nil, nil, fmt.Errorf("failed to read client model header: %v", err)
	}

	// Read architecture
	arch := make([]int, archLen)
	for i := 0; i < archLen; i++ {
		if _, err := fmt.Fscanf(clientFile, "%d ", &arch[i]); err != nil {
			return nil, nil, fmt.Errorf("failed to read architecture: %v", err)
		}
	}
	fmt.Fscanln(clientFile) // Consume newline

	// Create model config
	config := &ModelConfig{
		Arch:     arch,
		SplitIdx: splitIdx,
	}

	// Read number of layers
	var clientLayers int
	if _, err := fmt.Fscanf(clientFile, "%d\n", &clientLayers); err != nil {
		return nil, nil, fmt.Errorf("failed to read number of layers: %v", err)
	}

	// Create client model
	clientModel := &ClientModel{
		Weights: make([][][]float64, clientLayers),
		Biases:  make([][]float64, clientLayers),
		Config:  config,
	}

	// Read weights and biases for each layer
	for l := 0; l < clientLayers; l++ {
		var inputDim, outputDim int
		if _, err := fmt.Fscanf(clientFile, "%d %d\n", &inputDim, &outputDim); err != nil {
			return nil, nil, fmt.Errorf("failed to read layer dimensions: %v", err)
		}

		// Read weights
		clientModel.Weights[l] = make([][]float64, inputDim)
		for i := 0; i < inputDim; i++ {
			clientModel.Weights[l][i] = make([]float64, outputDim)
			for j := 0; j < outputDim; j++ {
				if _, err := fmt.Fscanf(clientFile, "%f ", &clientModel.Weights[l][i][j]); err != nil {
					return nil, nil, fmt.Errorf("failed to read weight: %v", err)
				}
			}
			fmt.Fscanln(clientFile) // Consume newline
		}

		// Read biases
		clientModel.Biases[l] = make([]float64, outputDim)
		for j := 0; j < outputDim; j++ {
			if _, err := fmt.Fscanf(clientFile, "%f ", &clientModel.Biases[l][j]); err != nil {
				return nil, nil, fmt.Errorf("failed to read bias: %v", err)
			}
		}
		fmt.Fscanln(clientFile) // Consume newline
	}

	// Load server model
	// First, load architecture and split point (but use the same config)
	if _, err := fmt.Fscanf(serverFile, "%d %d\n", &archLen, &splitIdx); err != nil {
		return nil, nil, fmt.Errorf("failed to read server model header: %v", err)
	}

	// Skip architecture
	for i := 0; i < archLen; i++ {
		var dummy int
		if _, err := fmt.Fscanf(serverFile, "%d ", &dummy); err != nil {
			return nil, nil, fmt.Errorf("failed to skip architecture: %v", err)
		}
	}
	fmt.Fscanln(serverFile) // Consume newline

	// Read number of layers
	var serverLayers int
	if _, err := fmt.Fscanf(serverFile, "%d\n", &serverLayers); err != nil {
		return nil, nil, fmt.Errorf("failed to read number of layers: %v", err)
	}

	// Create server model
	serverModel := &ServerModel{
		Weights: make([][][]float64, serverLayers),
		Biases:  make([][]float64, serverLayers),
		Config:  config,
	}

	// Read weights and biases for each layer
	for l := 0; l < serverLayers; l++ {
		var inputDim, outputDim int
		if _, err := fmt.Fscanf(serverFile, "%d %d\n", &inputDim, &outputDim); err != nil {
			return nil, nil, fmt.Errorf("failed to read layer dimensions: %v", err)
		}

		// Read weights
		serverModel.Weights[l] = make([][]float64, inputDim)
		for i := 0; i < inputDim; i++ {
			serverModel.Weights[l][i] = make([]float64, outputDim)
			for j := 0; j < outputDim; j++ {
				if _, err := fmt.Fscanf(serverFile, "%f ", &serverModel.Weights[l][i][j]); err != nil {
					return nil, nil, fmt.Errorf("failed to read weight: %v", err)
				}
			}
			fmt.Fscanln(serverFile) // Consume newline
		}

		// Read biases
		serverModel.Biases[l] = make([]float64, outputDim)
		for j := 0; j < outputDim; j++ {
			if _, err := fmt.Fscanf(serverFile, "%f ", &serverModel.Biases[l][j]); err != nil {
				return nil, nil, fmt.Errorf("failed to read bias: %v", err)
			}
		}
		fmt.Fscanln(serverFile) // Consume newline
	}

	return clientModel, serverModel, nil
}
