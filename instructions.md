# PRD – Split‑Learning Library v0·4  (Configurable architectures + security hardening)

## 1. Objective

Upgrade the existing CKKS‑based split‑learning prototype so that it

1. **eliminates all adversarial attack vectors arising from weight disclosure**,
2. fixes the numeric issues spotted in the second‑pass review,
3. introduces a fully *configurable* network layout & split point, and
4. boosts runtime with safe parallelism.

## 2. Scope

* Go code under `split/…` and unit tests under `split_test/…`.
* No UX/CLI polish—only a minimal configuration interface (JSON + flags).

## 3. Stakeholders

| Role              | Name              | Interest                                                               |
| ----------------- | ----------------- | ---------------------------------------------------------------------- |
| Researcher / User | Halil (and peers) | Train arbitrary MLPs on MNIST‑like data without leaking server weights |
| Security reviewer | TBD               | Verify zero exposure of model parameters                               |
| Maintainer        | Future AI agent   | Clean, modular codebase                                                |

## 4. Functional Requirements

### F‑1  Configurable architecture

* **Input**: `--arch="784,128,32,10"` or JSON `[784,128,128,128,32,10]`.
* **Validation**: First element = input dim; last = output dim; all others >0.
* **Automatic split**: flag `--split-index=k` (0 ≤ k < len(arch)−1).

  * Layers `0..k` live on the **server**.
  * Layers `k+1..end` live on the **client**.
* **Weight inits** scale ∝ 1/√fan\_in.
* **Packing**: `NeuronsPerCT` is auto‑chosen s.t. `Slots ≥ batch×NeuronsPerCT`.

### F‑2  Server‑side forward (packed)

* Accepts **one** ciphertext block per 64 pixels (or auto block size).
* Dot‑product kernel uses *rotate + Mul(CT, PT) + add* pattern.
* No plaintext leaks at any stage.

### F‑3  ReLU approximation

* Degree‑2 Chebyshev with cached plaintext constants (`ptC0,C1,C2`).
* Rescale once per neuron; relinearise **after** the square.

### F‑4  Gradient packing & weight update

* Use `InnerSumLog(ct, B)` where `B = batchSize`.
* Update weights with CT×PT only; biases via slot‑0 mask.

### F‑5  Parallelism

* All large per‑neuron loops run in `NumWorkers` goroutines.
* Each goroutine obtains `eval := evaluator.ShallowCopy()`.

## 5. Security & Privacy Requirements

| ID  | Requirement                                                                                                                                                                           |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S‑1 | **No server weight decryption in production path**.  Any helper that calls `decryptor.DecryptNew(weight‑CT)` must be `//debug`‑gated and compile‑time excluded via `build tag debug`. |
| S‑2 | Remove or rewrite functions that create *all‑ones placeholder ciphertexts*.                                                                                                           |
| S‑3 | Unit test `TestWeightUndisclosed` ensures no exported method returns plaintext weights.                                                                                               |

## 6. Non‑functional Requirements

* **R‑1** Throughput target: ≤2 seconds for a single forward+backward batch (arch `784,128,32,10`, batch 32) on 8‑core CPU.
* **R‑2** CI: `go test ./...` runs < 90 s on GitHub runner.
* **R‑3** Code health: `golangci-lint` passes default rules.

## 7. Implementation Tasks

| #  | Title                                 | Detail                                                                                          |
| -- | ------------------------------------- | ----------------------------------------------------------------------------------------------- |
| 1  | **Param parsing**                     | New `model.Config` holds `Arch []int` and `SplitIdx`.  Add JSON+flag parser.                    |
| 2  | **Weight structs → slices-of-slices** | Generate `[][]float64` sized by `Arch`.  Update `convertToPacked`, `convertToHomomorphicModel`. |
| 3  | **CT×PT forward kernel**              | Replace CT×CT by CT×PT.  Cache plaintext weight rows.  Use power‑of‑two rotations.              |
| 4  | **Rotate‑accumulate helper**          | `func dotPacked(encImg *CT, ptWRow []*PT, slotsPerPixel int) *CT`.                              |
| 5  | **InnerSum replacement**              | Drop `innerSumSlots`; add `chunkSum(ct, B)` using `InnerSumLog`.                                |
| 6  | **Batch‑size propagation**            | All functions accept explicit `batch int`.  Remove hard‑coded globals.                          |
| 7  | **Delete dead code**                  | Remove `serverBackwardAndUpdate` and any function unreachable from `training.go`.               |
| 8  | **Parallel loops**                    | Introduce `parallel.For(start,end,fn)` util; refactor forward, backward, packing.               |
| 9  | **Security gate**                     | Move every decrypt‑for‑debug to file with `//go:build debug`.                                   |
| 10 | **Unit tests**                        | \* `TestDotProductCorrect` (plain vs HE)                                                        |

```
* `TestPackedUpdateChangesWeights`
* `TestNoPlainWeights` (reflection scan) |
```

## 8. Acceptance Criteria

1. `go test ./...` passes with `-race`.
2. `demo/main.go --arch="784,128,32,10" --split-index=0` completes one epoch and loss decreases.
3. Build with `-tags=prod` does **not** link the debug decrypt helpers.
4. Benchmarks show ≥2× speed‑up vs previous commit.

## 9. Out‑of‑scope

* Fancy activation functions > degree‑2.
* GPU offload.
* Advanced ciphertext compression.

---

**End of PRD**
