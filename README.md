# CURE\_lib: Homomorphic Split Learning for MLPs

**CURE\_lib** provides a Go implementation of split learning for MLPs using CKKS homomorphic encryption (via Lattigo v6). You define an arbitrary MLP, split it into “server” and “client” segments, and train securely: the client encrypts its cut-layer activations in SIMD-packed ciphertexts, the server continues forward/backward under HE, updates its encrypted weights, then returns encrypted gradients. At inference you can either run a small batch or pack one image per ciphertext.

---

## Key Features

* **Configurable Architecture** via JSON `ModelConfig` (e.g. `{"arch":[784,128,32,10],"splitIdx":1}`).

* **Three Training Modes**:

  1. **Standard Split-HE**: server decrypts intermediate weights each batch.
  2. **Fully HE**: server keeps weights encrypted throughout and only decrypts once at the end.
  3. **SIMD-Packed**: server maintains a packed-HE model and updates just the final layer under HE each batch.

* **Two Inference Options**:

  * **SIMD-Batch** (default): pack each image into one ciphertext.
  * **Single-Ciphertext** (requires minor additions): pack all 784 pixels of one image into one ciphertext.

* **MNIST Example** built-in: loading, training, saving/loading models, and evaluation.

---

## Directory Overview

```
split/
├── he_context.go       # CKKS parameters, keys, HEContext
├── models.go           # ClientModel, ServerModel, HEServerModel, HEServerPacked
├── client_forward.go   # clientPrepareAndEncryptBatch, PerformEvaluation
├── client_backward.go  # clientForwardAndBackward (plaintext forward/backward + gradient packing)
├── server_forward.go   # serverForwardPass, serverForwardPassPacked
├── server_backward.go  # serverBackwardAndUpdate, packedUpdateDirect
├── train.go            # Run, trainModelWithBatches, trainBatchWithTiming, trainModelFullSIMD
├── eval.go             # evaluateModel, EvaluateModelOnBatch
├── save_load.go        # saveModel, loadModel
├── config.go           # ModelConfig, RunConfig, flag parsing
├── helpers.go          # innerSumSlots, scalarPlain, parallelFor, etc.
├── data_io.go          # readMNISTData, readMNISTImages, readMNISTLabels
└── split_test.go       # unit tests
```

---

## Quick Start

1. **Prerequisites**

   * Go 1.18+
   * Internet (to fetch Lattigo v6 and MNIST)
   * \~100 MB free disk space

2. **Get MNIST**

   ```bash
   cd split
   go generate
   ```

3. **Build**

   ```bash
   cd ..
   go mod tidy
   go build ./split
   ```

4. **Train**

   ```bash
   ./split \
     -mode=train \
     -batchSize=8 \
     -numBatches=100 \
     -fullyHE=false \
     -fullySIMD=false \
     -clientModel=client.bin \
     -serverModel=server.bin \
     -modelConfig='{"arch":[784,128,32,10],"splitIdx":1}'
   ```

5. **Evaluate**

   ```bash
   ./split \
     -mode=eval \
     -clientModel=client.bin \
     -serverModel=server.bin \
     -modelConfig='{"arch":[784,128,32,10],"splitIdx":1}'
   ```

---

## Usage Summary

1. **HE Setup** (`he_context.go`):
   Builds CKKS with default parameters (N=8192, depth≈6).

2. **Model Init** (`models.go`):

   * `initClientModel(config)` and `initServerModel(config)` to create random weights.
   * `convertToPacked(...)` if running SIMD-packed or fully HE.

3. **Client Side**:

   * **Encrypt & Pack** (`clientPrepareAndEncryptBatch`): one ciphertext per image.
   * **Plaintext Forward/Backward** (`clientForwardAndBackward`): decrypt server activations, do ReLU/softmax, compute gradients, repack into HE ciphertexts.

4. **Server Side**:

   * **Forward** (`serverForwardPass`): multiply/add encrypted activations with plaintext weights, apply HE-ReLU.
   * **Backward & Update** (`serverBackwardAndUpdate`): decrypt gradients, update plaintext weights (or under HE if fullyHE), and optionally decrypt updated weights.
   * **SIMD-Packed Option** (`serverForwardPassPacked` + `packedUpdateDirect`): maintain a packed HE model and update only the final layer under HE per batch; decrypt everything at end.

5. **Saving/Loading** (`save_load.go`):
   Save plaintext weights/biases to files, and load them back.

6. **Evaluation** (`eval.go`):
   For each test sample (batchSize=1), encrypt → server forward → decrypt → client forward → softmax.

---

## Testing

Run all tests:

```bash
go test ./split -timeout 30s
```

---

## Configuration

* **Network**: specify `arch` (layer dims) and `splitIdx` (0 ≤ splitIdx ≤ len(arch)−2).
* **BatchSize** and **NumBatches**: control how many samples per batch and how many batches to train.
* **LearningRate** and **Epochs**: set in `config.go`.
* **HE Params**: edit `LogN`, `LogQ`, `LogP`, and `LogDefaultScale` in `he_context.go` if you need deeper networks or fewer slots.

---

## License

MIT License. Contributions welcome!
