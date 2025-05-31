package split

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

// RunConfig holds the configuration for a training or evaluation run
type RunConfig struct {
	Mode       string // "train" | "eval"
	NumBatches int
	BatchSize  int
	FullyHE    bool
	SaveModels bool
	ClientPath string
	ServerPath string
}
