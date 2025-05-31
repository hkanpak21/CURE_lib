## Proposed repository layout

```
cure/
│
├── cmd/                        # CLI entry points
│   ├── cure-train/             # plaintext training
│   ├── cure-split/             # split-learning training
│   └── cure-infer/             # inference / benchmarking
│
├── internal/                   # helper code, not exported
│   ├── parallel/               # worker-pool, goroutine utils
│   └── math/                   # padding, index maps, misc numerics
│
├── pkg/
│   ├── he/                     # LOW-LEVEL homomorphic toolkit
│   │   ├── params/             # CURE-S / M / L parameter sets, keygen
│   │   ├── ops/                # atomic ops: dot, matmul, conv, pool
│   │   ├── rescale/            # scale manager & policies
│   │   └── boot/               # bootstrap wrapper
│   │
│   ├── layers/                 # DL layers (plain + HE variants)
│   │   ├── linear.go
│   │   ├── conv.go
│   │   ├── pool.go
│   │   ├── activation.go
│   │   └── residual.go
│   │
│   ├── model/                  # graph, builder DSL, splitter
│   │   ├── graph.go
│   │   ├── builder.go
│   │   └── onnx.go             # (future) import / export helpers
│   │
│   ├── train/                  # optimisers, schedulers, losses
│   └── data/                   # dataset loaders, augmentations
│
├── tests/                      # go test packages live here
│   ├── he/
│   ├── layers/
│   ├── integration/
│   └── benchmarks/
│
├── examples/                   # tiny runnable demos
│   ├── mlp_plain.go
│   ├── resnet_split.go
│   └── gan_generate.go
│
├── scripts/                    # helper bash / python scripts
│
├── docs/
│   ├── ARCHITECTURE.md         # packing layout, scale flow
│   ├── PARAMS.md               # security & modulus chains
│   └── CONTRIB.md              # coding style, PR checklist
│
└── go.mod
```

### Rationale

| Folder       | Responsibility                                                                                       | Why it matters |
| ------------ | ---------------------------------------------------------------------------------------------------- | -------------- |
| `pkg/he`     | *Pure CKKS primitives* only.<br>Easy to vendor into other projects without dragging ML code.         |                |
| `pkg/layers` | Wraps `he` ops **and** their plaintext twins, so you keep one API regardless of deployment back-end. |                |
| `pkg/model`  | Graph engine, builder DSL, split-learning utilities.  Keeps layer code free from graph concerns.     |                |
| `internal`   | Small helpers you don’t want in the public API; Go will enforce the boundary.                        |                |
| `cmd`        | CLI binaries live here; each directory is a `package main`.                                          |                |
| `tests`      | Test-only packages avoid import cycles and keep benchmarking tidy.                                   |                |
| `docs`       | Central place for design notes and parameter tables—stops PDF drift.                                 |                |

---

## Migration plan (high-level)

1. **Create new modules**

   ```bash
   mkdir -p pkg/he/ops pkg/he/params pkg/layers pkg/model internal/parallel
   ```
2. **Move files**

   * `pkg/he/ops.go`      → `pkg/he/ops/ops.go`
   * `pkg/he/conv.go`     → `pkg/layers/conv.go`
   * tests under `*_test.go` follow their code.  Update import paths.
3. **Split package names**

   * `package he` (for low-level)
   * `package layers` (depends on `he`)
   * `package model` (depends on `layers`)
4. **Introduce interfaces**

   ```go
   type Cipher interface { Add(Cipher) Cipher; Mul(Cipher) Cipher }
   ```

   so layer code can swap **plaintext** and **homomorphic** tensors.
5. **CI smoke test** — run `go test ./...` after each batch move.

---

## `REPO_LAYOUT.md` (sample)

```markdown
# Repository layout (v2)

``cure`` follows a “top–down” hierarchy:

```

cmd/            – CLI binaries (train / infer / split)
pkg/
he/           – low–level CKKS primitives, no DL logic
layers/       – Conv, Linear, Pool … implemented for
both plaintext and homomorphic tensors
model/        – graph builder, splitter, ONNX I/O
internal/       – non-exported helpers (parallelism, math)
tests/          – all unit, fuzz and benchmark packages
examples/       – small runnable demos
docs/           – design notes, parameter tables, tutorials

````

### Import rules

* `pkg/he/**` **MUST NOT** import anything outside `pkg/he`.
* `pkg/layers` may import `pkg/he` but never vice-versa.
* CLI (`cmd/*`) may import anything under `pkg/…` but nothing under `internal/`.

### How to add a new HE op

1. Put the core math in `pkg/he/ops`.
2. Wire it into a layer wrapper in `pkg/layers`.
3. Add unit test under `tests/he/` and (optionally) a benchmark.

### Build & test

```bash
go test ./...
go vet ./...
golangci-lint run
````

---

With this structure you gain:

* **Clear public API surface** (`pkg/...`)
* **Separation of concerns** (math vs. DL vs. CLI)
* **Easy vendoring** of `pkg/he` by outside users
* **Scalable testing** (benchmarks isolated in `tests/benchmarks`)

Start the refactor *before* adding ResNet code—you’ll thank yourself later.

[1]: https://github.com/hkanpak21/cure_test "GitHub - hkanpak21/cure_test"
