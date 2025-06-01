package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	split "github.com/halilibrahimkanpak/cure_test/split_training"
)

func main() {
	// Define default configuration
	cfg := split.RunConfig{
		NumBatches: 50,
		BatchSize:  8,
		FullyHE:    false,
		FullySIMD:  false,
		SaveModels: false,
		ClientPath: "client_model.txt",
		ServerPath: "server_model.txt",
		ModelCfg:   nil, // Will use default if not specified
	}

	// Parse command-line arguments
	if len(os.Args) > 1 {
		cfg.Mode = os.Args[1]
		switch cfg.Mode {
		case "train":
			// Parse flags separately after the mode argument
			flagSet := flag.NewFlagSet("train", flag.ExitOnError)
			flagSet.IntVar(&cfg.NumBatches, "batches", 50, "Number of batches to use for training")
			flagSet.BoolVar(&cfg.FullyHE, "he", false, "Use fully homomorphic backpropagation")
			flagSet.BoolVar(&cfg.FullySIMD, "simd", false, "Use fully optimized SIMD training (keeps model encrypted)")
			flagSet.BoolVar(&cfg.SaveModels, "save", false, "Save trained models")
			flagSet.StringVar(&cfg.ClientPath, "client", "client_model.txt", "Client model filename")
			flagSet.StringVar(&cfg.ServerPath, "server", "server_model.txt", "Server model filename")
			flagSet.IntVar(&cfg.BatchSize, "batch", 8, "Mini-batch size (<= CKKS slots)")

			// Add new flags for architecture configuration
			var archStr string
			var splitIdx int
			flagSet.StringVar(&archStr, "arch", "784,128,32,10", "Network architecture as comma-separated integers")
			flagSet.IntVar(&splitIdx, "split-index", 0, "Split point index (0 ≤ split-index < len(arch)-1)")

			flagSet.Parse(os.Args[2:])

			// Parse architecture string
			if archStr != "" {
				parts := strings.Split(archStr, ",")
				arch := make([]int, 0, len(parts))

				for _, part := range parts {
					dim, err := strconv.Atoi(part)
					if err != nil {
						fmt.Printf("Error parsing architecture: %v\n", err)
						os.Exit(1)
					}
					arch = append(arch, dim)
				}

				// Create model config
				cfg.ModelCfg = &split.ModelConfig{
					Arch:     arch,
					SplitIdx: splitIdx,
				}

				// Validate configuration
				if !cfg.ModelCfg.Validate() {
					fmt.Println("Invalid model configuration. Please check architecture and split index.")
					os.Exit(1)
				}
			}

		case "eval":
			if len(os.Args) > 3 {
				cfg.ClientPath = os.Args[2]
				cfg.ServerPath = os.Args[3]
			} else {
				fmt.Println("Error: Missing model paths for evaluation")
				fmt.Println("Usage: go run . eval <client_model> <server_model>")
				os.Exit(1)
			}
		default:
			fmt.Println("Usage: go run . [train|eval] [options]")
			fmt.Println("  train                     - Train a new model")
			fmt.Println("  train --batches <num>     - Train with specified number of batches")
			fmt.Println("  train --he                - Use fully homomorphic backpropagation")
			fmt.Println("  train --simd              - Use fully optimized SIMD training (keeps model encrypted)")
			fmt.Println("  train --save [client_path] [server_path] - Train and save model")
			fmt.Println("  train --arch=\"784,128,32,10\" - Set network architecture")
			fmt.Println("  train --split-index=1     - Set split point (0=server only, 1=first hidden)")
			fmt.Println("  eval <client_path> <server_path> - Evaluate a saved model")
			os.Exit(1)
		}
	}

	// Run the main logic
	err := split.Run(cfg)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
}
