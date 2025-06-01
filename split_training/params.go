package split

import (
	"encoding/json"
	"fmt"
)

// MNIST dataset dimensions
const (
	MnistRows     = 28
	MnistCols     = 28
	MnistPixels   = MnistRows * MnistCols // 784
	MnistTrainNum = 60000
	MnistTestNum  = 10000
	NumClasses    = 10
)

// Network architecture dimensions
const (
	InputDim   = MnistPixels // 784
	HiddenDim1 = 128
	HiddenDim2 = 32
	OutputDim  = NumClasses // 10
)

// We pack 64 hidden neurons per ciphertext.
// Works for N=4096 and any batch B ≤ 32   (64×32 = 2048 slots)
const (
	NeuronsPerCT = 64 // ★ SIMD-weights
)

// Training parameters
const (
	Epochs       = 1
	LearningRate = 0.01
	NumWorkers   = 4 // For parallel operations
)

// Default batch size (can be overridden by command-line flag)
var BatchSize = 8 // Reduced from 64 for faster testing

// Default architecture when not specified
var DefaultArch = []int{MnistPixels, 128, 32, NumClasses}

// ModelConfig holds the model architecture and split point
type ModelConfig struct {
	Arch     []int `json:"arch"`     // Network layer dimensions
	SplitIdx int   `json:"splitIdx"` // Index of the split point
}

// ParseConfig parses a JSON string into a ModelConfig
func ParseConfig(jsonStr string) (*ModelConfig, error) {
	var config ModelConfig
	if err := json.Unmarshal([]byte(jsonStr), &config); err != nil {
		return nil, err
	}
	return &config, nil
}

// Validate checks if the model configuration is valid
func (mc *ModelConfig) Validate() bool {
	// Must have at least 2 layers (input and output)
	if mc == nil || len(mc.Arch) < 2 {
		return false
	}

	// Split index must be between 0 and len(Arch)-2
	// 0 means server has no layers (client does all work)
	// len(Arch)-2 means client has just the output layer
	if mc.SplitIdx < 0 || mc.SplitIdx >= len(mc.Arch)-1 {
		return false
	}

	return true
}

// RunConfig holds the runtime configuration for training/evaluation
type RunConfig struct {
	Mode       string       // 'train' or 'eval'
	NumBatches int          // Number of batches for training
	BatchSize  int          // Mini-batch size
	FullyHE    bool         // Use fully homomorphic backpropagation
	FullySIMD  bool         // Use fully optimized SIMD training (keeps model encrypted)
	SaveModels bool         // Save models after training
	ClientPath string       // Path to save/load client model
	ServerPath string       // Path to save/load server model
	ModelCfg   *ModelConfig // Network architecture configuration
}

// calculateNeuronsPerCT determines how many neurons can be packed into one ciphertext
func calculateNeuronsPerCT(slots, batchSize, maxPerCT int) int {
	// Maximum neurons that can fit given the batch size
	maxFit := slots / batchSize

	// Cap at a reasonable value to avoid numeric issues
	if maxFit > maxPerCT {
		return maxPerCT
	}

	// Ensure at least 1 neuron per ciphertext
	if maxFit < 1 {
		return 1
	}

	return maxFit
}

// ApplyConfig applies the configuration to the global variables
func ApplyConfig(config *RunConfig) {
	if config == nil {
		return
	}

	// Update BatchSize if specified
	if config.BatchSize > 0 {
		BatchSize = config.BatchSize
	}
}

// PrintConfig prints the current configuration details
func PrintConfig(config *RunConfig) {
	fmt.Println("\nConfiguration Details:")
	fmt.Printf("  Batch Size: %d\n", BatchSize)

	// Print training mode
	if config != nil {
		fmt.Printf("  Training Mode: ")
		if config.FullyHE {
			fmt.Println("Fully Homomorphic")
		} else if config.FullySIMD {
			fmt.Println("Fully Optimized SIMD")
		} else {
			fmt.Println("Standard")
		}
	}

	if config != nil && config.ModelCfg != nil {
		fmt.Println("  Architecture:")
		fmt.Printf("    Layers: %v\n", config.ModelCfg.Arch)
		fmt.Printf("    Split Index: %d\n", config.ModelCfg.SplitIdx)
		fmt.Printf("    Server Layers: %d\n", config.ModelCfg.SplitIdx+1)
		fmt.Printf("    Client Layers: %d\n", len(config.ModelCfg.Arch)-config.ModelCfg.SplitIdx-1)
	} else {
		fmt.Println("  Architecture: Using Default")
		fmt.Printf("    Layers: %v\n", DefaultArch)
	}
	fmt.Println("")
}
