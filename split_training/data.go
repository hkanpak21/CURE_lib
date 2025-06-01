package split

import (
	"encoding/binary"
	"fmt"
	"os"
)

// DataPath represents the path to the MNIST data files
var DataPath = "/Users/halilibrahimkanpak/Documents/Coding/CURE_lib/data"

//go:generate bash -c "mkdir -p ./data && cd ./data && curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz && curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz && curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz && curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz && gzip -d *.gz || true"

// Function to read MNIST data
func readMNISTData() ([][]float64, []int, [][]float64, []int, error) {
	// Define file paths
	trainImagesPath := DataPath + "/train-images-idx3-ubyte"
	trainLabelsPath := DataPath + "/train-labels-idx1-ubyte"
	testImagesPath := DataPath + "/t10k-images-idx3-ubyte"
	testLabelsPath := DataPath + "/t10k-labels-idx1-ubyte"

	// Read training images
	trainImages, err := readMNISTImages(trainImagesPath)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("error reading training images: %v", err)
	}

	// Read training labels
	trainLabels, err := readMNISTLabels(trainLabelsPath)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("error reading training labels: %v", err)
	}

	// Read test images
	testImages, err := readMNISTImages(testImagesPath)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("error reading test images: %v", err)
	}

	// Read test labels
	testLabels, err := readMNISTLabels(testLabelsPath)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("error reading test labels: %v", err)
	}

	return trainImages, trainLabels, testImages, testLabels, nil
}

// Helper function to read MNIST images
func readMNISTImages(filename string) ([][]float64, error) {
	// Open the file
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Read the header
	header := make([]int32, 4)
	err = binary.Read(file, binary.BigEndian, &header)
	if err != nil {
		return nil, err
	}

	// Check the magic number
	if header[0] != 2051 {
		return nil, fmt.Errorf("invalid magic number: %d", header[0])
	}

	// Get number of images, rows, and columns
	numImages := int(header[1])
	numRows := int(header[2])
	numCols := int(header[3])
	numPixels := numRows * numCols

	// Read the image data
	imageData := make([]byte, numImages*numPixels)
	err = binary.Read(file, binary.BigEndian, imageData)
	if err != nil {
		return nil, err
	}

	// Convert to float64 array and normalize (0-1)
	images := make([][]float64, numImages)
	for i := 0; i < numImages; i++ {
		images[i] = make([]float64, numPixels)
		for j := 0; j < numPixels; j++ {
			images[i][j] = float64(imageData[i*numPixels+j]) / 255.0
		}
	}

	return images, nil
}

// Helper function to read MNIST labels
func readMNISTLabels(filename string) ([]int, error) {
	// Open the file
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Read the header
	header := make([]int32, 2)
	err = binary.Read(file, binary.BigEndian, &header)
	if err != nil {
		return nil, err
	}

	// Check the magic number
	if header[0] != 2049 {
		return nil, fmt.Errorf("invalid magic number: %d", header[0])
	}

	// Get number of labels
	numLabels := int(header[1])

	// Read the label data
	labelData := make([]byte, numLabels)
	err = binary.Read(file, binary.BigEndian, labelData)
	if err != nil {
		return nil, err
	}

	// Convert to int array
	labels := make([]int, numLabels)
	for i := 0; i < numLabels; i++ {
		labels[i] = int(labelData[i])
	}

	return labels, nil
}
