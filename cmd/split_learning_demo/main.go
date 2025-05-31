package main

import (
	"flag"
	"fmt"
	"os"

	split "github.com/halilibrahimkanpak/cure_test/split_training"
)

func main() {
	// Define default configuration
	cfg := split.RunConfig{
		NumBatches: 50,
		BatchSize:  8,
		FullyHE:    false,
		SaveModels: false,
		ClientPath: "client_model.txt",
		ServerPath: "server_model.txt",
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
			flagSet.BoolVar(&cfg.SaveModels, "save", false, "Save trained models")
			flagSet.StringVar(&cfg.ClientPath, "client", "client_model.txt", "Client model filename")
			flagSet.StringVar(&cfg.ServerPath, "server", "server_model.txt", "Server model filename")
			flagSet.IntVar(&cfg.BatchSize, "batch", 8, "Mini-batch size (<= CKKS slots)")
			flagSet.Parse(os.Args[2:])

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
			fmt.Println("  train --save [client_path] [server_path] - Train and save model")
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
