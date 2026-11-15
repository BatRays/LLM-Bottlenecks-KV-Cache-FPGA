---
title: "Complete LLM Inference with a Compressed KV Cache on AMD Alveo U55C"
subtitle: "Version 3 - UIUC HACC mlCluster edition | Labs 0-3 retained verbatim"
author: "Student engineering team - practical implementation laboratory"
date: "November 15, 2026"
---

# Project introduction and non-negotiable objectives

**What changed:** your team expects access to the **AMD Alveo U55C cards** in UIUC's AMD-Xilinx Center of Excellence / HACC mlCluster. That replaces the previously unspecified classroom FPGA with an HBM-equipped server accelerator. **Labs 0-3 are reproduced verbatim in Part I** and should not be repeated. Their statement that the host computes the transformer was the original teaching assumption; starting at Lab 4, the new architecture supersedes it.

**The actual finish line:** after the host sends prompt token IDs and preloaded model weights, **FPGA programmable logic performs every neural-network inference operation**: embeddings, RMSNorm, Q/K/V projections, RoPE, causal attention, reading/writing a compressed KV cache, output projection, SwiGLU MLP, residual additions, logits and next-token selection. The host performs tokenizer/text I/O, weights upload, XRT launches and output display. CPU/GPU PyTorch is the *golden reference*, **not** the inference engine in the hardware run. HBM is external memory on the accelerator card but is still FPGA-owned for computation.

**What this manual is:** a stepwise engineering specification, workstream division, software-golden procedure, SystemVerilog interface contract, RTL verification plan, HBM addressing and transfer strategy, Vitis/XRT bring-up playbook, measured-performance protocol and error guide. **What it is not:** a finished, synthesized 0.5B-parameter Qwen design or guaranteed cluster installation instructions. The exact platform files, scheduler commands and XRT version **must be recorded when access is granted**. Commands with `<...>` are **templates**, not commands to paste unchanged.

## Target U55C platform - physical facts versus available resources

| U55C physical specification | Nominal value | Design consequence |
|:--|:--|:--|
| FPGA | Virtex UltraScale+ XCU55C | Vitis-compatible RTL kernels, multi-SLR timing concerns |
| HBM2 | 16 GB, **32 x 512 MB pseudo-channels** | Place weight/cache buffers explicitly; never assume a single uniform RAM port |
| Aggregate HBM bandwidth | **Up to 460 GB/s** theoretical | Must benchmark actual kernel bandwidth; no single master reaches this automatically |
| DSP slices | 9,024 physical maximum | Matrix/vector pipelines can scale, but DSPs are also used by other logic |
| LUTs / flip-flops | 1,304K / 2,607K | Actual available logic depends on installed shell and placement |
| On-chip BRAM / URAM | 70.9 Mb BRAM / 270 Mb URAM | Use on-chip tiles, scratchpads, prefetch FIFOs and partial reduction buffers |
| Host interface | PCIe | XRT controls kernels and transfers weight/input/output buffers |

Values are **card-level specifications**, not a resource guarantee for the reconfigurable user partition. Manufacturer references: [DS978](https://docs.amd.com/r/en-US/ds978-u55c/Product-Details), [U55C platform](https://docs.amd.com/r/en-US/ug1120-alveo-platforms/U55C-Gen3x16-XDMA-base_3-Platform), [HBM configuration](https://docs.amd.com/r/en-US/ug1700-vitis-accelerated-data-center/HBM-Configuration-and-Use). Record actual `platforminfo` and build utilization before sizing your datapath.

## Two implementation tracks - both FPGA-compute, not FPGA-offload

**Track A, mandatory:** execute the entire supplied *TinyDecoder-2L-64D* architecture on **one U55C**, with INT8 weights, Q8.8 activations (initial reference), configurable **INT16 versus INT4 KV cache**, and greedy token decoding. Start with an HBM-backed memory layout even if an on-chip-only prototype would fit. This validates the architecture and full board/toolchain pipeline. A randomly initialized model is enough for numerical bring-up, but train the supplied tiny classifier if you want meaningful finance-post outputs.

**Track B, ambitious:** run pretrained **Qwen2.5-0.5B-Instruct** entirely in FPGA logic, with model weights resident in U55C HBM and tensor-by-tensor matching against a quantized software oracle. Memory *capacity* is much less restrictive than on Spartan-7, but **compute throughput, numerical fidelity, HBM access patterns, routing and long build times** remain hard. Treat this as a gated stretch goal, not an automatic result of having 16 GB HBM.

**Minimum defensible success** = tiny full-transformer in U55C logic with verified compressed cache. **Stronger success** = meaningful trained tiny model + accurately measured speed/memory tradeoffs. **Research-grade success** = pretrained Qwen-compatible compute + measured gain from compressed cache, both executed on FPGA.

## End-to-end dataflow - read this before dividing work

```text
                      HOST COMPUTER (Linux, XRT)
  text -> tokenizer -> token IDs      checkpoint -> weight quantization
                |                                    |
                +----------- PCIe / XRT ------------+
                                    |
                     Vitis platform / static shell
                                    |
               +--------------------v---------------------+
               |              U55C RTL KERNEL            |
               |                                         |
               | AXI4-Lite control + DMA/AXI4-M masters  |
               |                 |                       |
               |        sequencer / token FSM            |
               |                 |                       |
               |   embedding -> [transformer layer x N]  |
               |                   |                     |
               |       RMSNorm -> Q/K/V -> RoPE          |
               |                   |                     |
               |       attention <-> INT4 KV codec      |
               |                   |                     |
               |       output proj -> residual          |
               |                   |                     |
               |       RMSNorm -> SwiGLU -> residual     |
               |                   |                     |
               |   final norm -> LM head -> argmax       |
               |                                         |
               |  BRAM / URAM tiles <-> AXI HBM masters  |
               +--------------------+--------------------+
                                    |
            +-----------------------v---------------------+
            |     U55C HBM2 (32 pseudo-channels)         |
            | weights | packed K/V + scales | scratch    |
            +---------------------------------------------+
                      |
                generated token IDs -> XRT -> host text display
```

**No PCIe round-trip for K/V on each generated token.** Keep the cache in HBM and reconstruct it near the attention datapath. An XRT call per token can still introduce measurable *control overhead*; batch generation inside a persistent kernel later if needed.

## Recommended phases and acceptance gates

| Work period | Labs | Concrete demonstrable output | Gate |
|:--|:--|:--|:--|
| Completed | 0-3 | Transformer fundamentals, baseline Qwen, real KV traces | Preserve files |
| Before cluster access | 4-7 | Model and memory design, team handoffs, bit-exact goldens | Signed **spec-v1** |
| RTL development | 8-12 | Matvec, nonlinear arithmetic, GQA, HBM-addressed compressed KV, full token decode | RTL and oracle agree |
| U55C bring-up | 13-14 | First HBM kernel, XRT input/output, hardware emulation, hardware xclbin | Measured hardware output |
| Scientific comparison | 15 | A/B FPGA cache-mode benchmark and CPU/GPU reference | Reproducible measurements |
| Application and scale | 16-17 | Offline finance event labels; optional pretrained Qwen | Accurate scope claims |

**Work division warning:** an HBM-capable kernel is *not* the same thing as a working compressed-cache controller, and a correct compressor *alone* does not establish full FPGA inference. Each team must satisfy its own golden tests and a shared end-to-end test.


# Part I. Previous work — Labs 0-3 (UNMODIFIED)

# Part I. Concepts from first principles

## Lab 0 - Setup your environment

### 0.1 Understand what runs where

An LLM is a very large collection of numerical operations, especially matrix multiplications. A conventional FPGA board will **not** run a competitive modern generative model simply because it can execute multipliers. Model weights, internal activations and memory bandwidth are the obstacle. For this project, the **host** runs the transformer and the **FPGA** processes a well-defined stream of its numeric KV data. We can benchmark the FPGA block independently before attempting a live host-to-board connection.

### 0.2 Install the software

1. Install Python 3.10+ and, on Windows, create a virtual environment from PowerShell: `py -3.10 -m venv .venv` then `.\.venv\Scripts\Activate.ps1`.
2. From the project directory run `python -m pip install --upgrade pip` and `python -m pip install -r requirements.txt`.
3. Install the correct GPU-enabled PyTorch build from [PyTorch's installer](https://pytorch.org/get-started/locally/) if CUDA is available. CPU-only is fine for small tests. `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` confirms your backend.
4. Run `python -m pytest -q`; the project should show **10 passed**. Do this before installing ML models: these tests run on synthetic tensors and do not download anything.
5. Run `python -m src.fpga_vectors`. Expect the bytes `10 7f 78 1f` and two files under `results/`.
6. Install Vivado and verify that you can run an ordinary RTL behavioral simulation. **No board is needed yet.**

**Expected artifact:** screenshot or text log showing the Python version, PyTorch version, CUDA availability, test pass, and FPGA simulation tool version.

**Common errors:** `python not recognized` means the interpreter is not on PATH (use the full `python.exe` path); `No module named src` means you are not running commands from the root project directory; `CUDA unavailable` means a CPU-only torch wheel or incompatible driver was installed. The model download requires an internet connection the first time.

### 0.3 Project directory walkthrough

- `src/quant.py`: your Python INT4 numerical specification.
- `src/01_baseline.py`: loads Qwen and inspects its real KV cache.
- `src/02_extract_kv.py`: saves representative key/value data, compares sizes.
- `src/03_quality_loop.py`: inserts INT4 rounding/decompression between decoding steps.
- `src/04_hf_cache_benchmark.py`: benchmarks a **different**, genuine compressed-resident software cache.
- `rtl/int4_pack.sv`: fixed-scale, streaming INT4 quantizer/packer.
- `rtl/tb_int4_pack.sv`: self-checking behavioral testbench with output stalls.
- `rtl/int4_unpack.sv`: streaming fixed-scale INT4-to-Q8.8 decompressor.
- `rtl/tb_int4_roundtrip.sv`: self-checking pack/decompress pipeline test with backpressure.
- `src/06_codec_sweep.py`: tested 2/4/8-bit software sweep and plot generator.
- `src/05_signal_demo.py`: artificial financial events to classification labels.
- `src/07_evaluate_signals.py`: compute coverage, macro-F1 and latency from demo labels.
- `results/`: generated results; do not commit giant traces to Git.

**Exit criterion:** You can explain what each file is supposed to prove, and you have not mixed up the software compressed-cache benchmark with FPGA acceleration.

## Lab 1 - Learn how an LLM works

### 1.1 Tokens are numbers, not words

A language model cannot process a string directly. A **tokenizer** turns text into integer token IDs. Some tokens represent whole words, others subwords, symbols or spaces. For instance, a sentence might become a sequence of 12 integers. The vocabulary maps each token ID to a learned vector; this table is called an **embedding**. The model repeatedly transforms those vectors and assigns scores (logits) to possible next tokens. It converts the scores into probabilities and selects or samples one token, then repeats.

An LLM is **not** continuously reading the entire internet while generating. It runs a fixed numerical network using the prompt and any explicitly supplied context. In this project we run **inference**, not training: we use existing model weights, and optimize the temporary information needed to generate more tokens.

### 1.2 Why the transformer uses attention

Suppose the prompt says, “The company cut its forecast because **it** saw weaker demand.” The model needs a mechanism to connect the word “it” with relevant earlier words. A transformer uses **attention** to compute weighted combinations of token information. Each transformer layer creates three distinct vectors from its input:

- **Query (Q):** What information is the current token seeking?
- **Key (K):** What pattern or information does an earlier token offer?
- **Value (V):** What information should that earlier token contribute if selected?

Q, K, V are **numerical arrays, not English strings**. Each is calculated using a learned weight matrix: `Q = X Wq`, `K = X Wk`, `V = X Wv` (layout conventions vary). For one head the attention equation is:

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^{T}}{\sqrt{d}}+M\right)V$$

Here `d` is head dimension, `M` is the causal mask, and `softmax` turns raw scores into positive weights summing to 1. The causal mask prevents a token from using words that have not been generated yet.

### 1.3 Tiny attention calculation (do this by hand)

For a simple head with dimension `d=1`, choose current query `q=1`, two previous keys `k1=1` and `k2=0`, and values `v1=10`, `v2=20`. The score vector is `[1, 0]`. `softmax([1,0])` is approximately `[0.7311,0.2689]`. The output is `0.7311*10 + 0.2689*20 = 12.689`. The original two values were blended according to how relevant their **keys** were to the **query**.

**Your task:** Calculate the output for q=2, keys `[0,1]`, values `[5,25]`; you should get approximately 22.62 (softmax weights `[0.1192,0.8808]`). If you cannot explain why this is not simply a weighted average chosen beforehand, revisit Q and K: **they compute the weights dynamically**.

### 1.4 A model contains many attention heads and layers

A head is a separate way to attend to information. A **layer** includes attention plus other computations, notably feedforward networks and normalization. Many layers are stacked. New token generation involves passing through all layers; therefore **every layer** has its own K/V cache. The stored keys at layer 0 are **not** interchangeable with the keys at layer 23.

Modern models often use **grouped-query attention (GQA)**. The number of **query heads** can be larger than the number of **key/value heads**. Multiple query heads share one stored KV head. Consequently, the KV memory formula uses `num_key_value_heads`, **not** the number of query heads. A positional technique called **RoPE** changes key and query representations based on token positions; careless cache manipulation can break it. We will compress the **already produced** K and V tensors and will not attempt to modify RoPE logic in the basic labs.

### 1.5 Prefill vs decode

**Prefill:** process the prompt's T tokens and construct the initial KV cache. **Decode:** produce subsequent output tokens one at a time; at each new token, use its query against the prior keys, mix values, and append its own K/V vectors. During long generation, reading the large KV cache can cost significant memory traffic. Cache compression attempts to reduce that traffic and capacity.

**Check your understanding:** (1) Why cache K and V but not a permanent copy of every Q? (2) Why is there one KV cache per layer? (3) When would a 4-bit cached value change the next token? (4) Why could a 4-bit cache be slower than FP16 for a very short prompt? **Answer key** is in Appendix A.

## Lab 2 - Run the first real LLM

Use [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), a compact and accessible model with Apache 2.0 licensing. It has 24 layers, 14 query heads, 2 KV heads, hidden size 896, and head dimension `896/14=64`, according to its published configuration. This is a particularly useful model to **see GQA in action**. Do not confuse a small development model with a production-quality trading text model.

Run `python -m src.01_baseline`. The first run downloads model weights. Observe the prompt token count, 24 cache layers, layer-0 `K` and `V` shapes, generated answer, and total time. On a one-example prompt you should see shapes of the form `[1,2,T,64]`, with `T` determined by the tokenizer and chat-template tokens. If you use CPU, cached tensors may be FP32; if CUDA, this project requests FP16.

**Prediction before running:** Which dimension should grow as your prompt gets longer? How many bytes does the first layer need? Multiply all four shape dimensions by the number of bytes per element, then multiply by **two** because K and V are both stored. Finally multiply by 24 layers.

**Deliverable:** a short report containing model ID, package version, device, exact tensor shapes, dtype, prompt length, and full generated response. Do not use a GPU-memory reading as though it represented KV memory alone: model weights occupy much more memory for short prompts.

## Lab 3 - Understand the KV-cache memory bottleneck

### 3.1 The key formula

For a full-attention, uniform-layer model:

$$M_{KV}=2\,B\,L\,H_{KV}\,T\,D\,s$$

Here `B` = batch size, `L` = number of layers, `H_KV` = KV heads per layer, `T` = cached tokens, `D` = elements per KV head, `s` = bytes per element. The factor of 2 counts K **and** V.

For Qwen2.5-0.5B, batch 1, 24 layers, 2 KV heads, dimension 64, FP16 (`s=2`):

| Sequence length | Raw FP16 KV size |
|:--|--:|
| 512 tokens | 6 MiB |
| 2,048 tokens | 24 MiB |
| 4,096 tokens | 48 MiB |
| 8,192 tokens | 96 MiB |
| 16,384 tokens | 192 MiB |

For a batch size of 8, multiply those by 8. These are **analytical payload sizes**, not allocator readings or GPU total memory. When caching a sliding-window architecture, the growth can stop at its window, so inspect the actual model.

### 3.2 Why 4-bit should matter

FP16 = 16 bits per scalar. INT4 = 4 bits per scalar. If 100% of the KV payload were coded in INT4, then its ideal payload compression would be 16/4 = **4x**. But the model needs to know how to interpret those 4-bit numbers. Quantizers usually save a **scale** (and sometimes zero-point, outliers, and a high-precision residual). These are additional bytes. Consequently **actual 4-bit KV compression is less than 4x**, even though the ideal numerical-payload ratio is 4x.

### 3.3 Capture the real vectors

Run `python -m src.02_extract_kv`. The script requests an LLM forward pass, visits all 24 cache layers, and writes `results/kv_analysis.json` and a layer-0 tensor trace. Find `shape`, `max_abs_error`, `mean_abs_error`, and `compression_ratio`. The script reports the ratio of a *hypothetical packed representation* against the actual tensor dtype; it does not change the LLM's live memory allocation.

Use the report to compare K and V separately. Their distributions can differ. You are now looking at the actual numerical data your RTL will eventually process, rather than synthetic examples.

**Checkoff:** Explain why an FPGA implementing 4-bit quantization needs more than a 4-bit output wire: what will receive that compressed stream, where will scale metadata go, and how will attention recover the values?


# Part II. U55C platform readiness and design freeze

## Lab 4 - U55C feasibility, first contact, and model sizing (EVERYONE)

**Objective:** choose an implementable first model and capture a verified, cluster-specific platform contract. **Do not wait for your cluster account to complete Steps 4.1-4.3.**

### 4.1 Before cluster access: gather your completed Lab 0-3 artifacts

1. Save `results/` KV traces, software compressor quantization sweep and 10-test baseline; record Python/Torch/Transformers versions and the exact Qwen revision.
2. Extract the previous **Full-FPGA Extension Starter ZIP** into the existing repo, preserving completed `src/` and `rtl/`. Keep `full_fpga/spec/model_spec.json` as the configuration source.
3. Run from project root: `python -m pytest -q tests full_fpga/tests`; then `python -m full_fpga.python.feasibility`; `python -m full_fpga.python.tiny_model`.
4. Freeze **TinyDecoder-2L-64D** for Track A: 2 layers, E=64, Hq=4, Hkv=2, D=16, FFN=128, vocab=262, max context 64, INT8 weights, Q8.8 activations, INT4 KV payload + a 2-byte scale per vector. Do not silently change this while RTL is underway.
5. Make a `results/lab04_feasibility.csv` that lists each tensor's bytes, lifetime, memory region and owner.

### 4.2 Numerical capacity worksheet - calculate it yourselves

For every layer and token, **uncompressed** K+V storage is `2 * Hkv * D * 2` bytes at INT16. **Packed INT4 with one 2-byte scale per K vector and one per V vector** is `2 * Hkv * (D/2 + 2)` bytes when D even. For the tiny model with 2 layers, Hkv=2, D=16, T=64:

- INT16 cache: `2 layers * 64 tokens * 2 (K,V) * 2 heads * 16 dims * 2 bytes = 16,384 bytes`.
- INT4 cache: `2 * 64 * 2 * 2 * (8 payload + 2 scale) = 5,120 bytes`.
- Compression = `16,384 / 5,120 = 3.2x`, assuming no extra alignment/padding. Actual HBM allocation footprint **may** be much larger if you pad each vector to 32/64-byte boundaries; report both *logical* and *physical* bytes.

For Qwen2.5-0.5B: ~490M weights -> **~980 MB FP16**, **~490 MB INT8**, or theoretical **~245 MB packed INT4**, without metadata and duplicated buffers. Qwen's FP16 KV cache at 4096 tokens with 24 layers, Hkv=2, D=64 is `2*24*4096*2*64*2=50,331,648 bytes = 48 MiB` per sequence. This is *much smaller* than weight storage at low batch size, so cache compression may show bigger storage benefits than speedups. Estimate with actual tensor alignment and measured bandwidth, not headline peaks.

### 4.3 First cluster access - capture facts, do not guess commands

UIUC's public instructions describe a shared development node for compilation and compute access through a scheduler; the public status page currently identifies U55Cs on **HACC mlCluster Kubernetes workers**. The access mechanism, platform/shell versions and partitions **may change**. Use **the onboarding instructions given to your team** rather than inventing a Kubernetes resource name or assuming Slurm. Never run direct hardware work on a shared login node without authorization.

On the authorized **development node**, execute only read-only inventory commands:

```bash
hostname
uname -a
command -v vivado; command -v v++; command -v platforminfo
command -v xbutil; command -v xrt-smi
vivado -version | head -n 8
v++ --version
find "${XILINX_VITIS:-/opt}" -name '*.xpfm' 2>/dev/null | head -n 20
# When a specific platform path is assigned:
platforminfo -p "$PLATFORM"  # only after exporting PLATFORM=/absolute/path/file.xpfm
```

On an **allocated FPGA worker / approved container** (not necessarily the development node), inspect the runtime/device using the administrator-supported command (`xbutil examine`, or a version-compatible equivalent); save output. Some installations need explicit device passthrough into a container. Never attempt driver/shell flashing, device reset, or privileged XRT commands unless an admin requests them.

Create a **platform manifest** containing: cluster node and allocation procedure, exact `.xpfm` path, shell name, Vitis/Vivado/XRT versions, RTL kernel control protocol, kernel clock(s), available HBM bank connectivity, maximum kernel masters, XRT device index, Linux compiler, allowed emulator, build/output directories, quota and scheduler job template. Collect `platforminfo -p` output if available.

### 4.4 Cluster-safe operating rules

- UIUC publicly states **Vitis flow only**, **use installed shells**, **no custom OS/root/JTAG**, and shared development resources. A conventional Vivado standalone bitstream workflow does **not** deploy a custom U55C kernel to this cluster.
- Develop/simulate small RTL blocks on your local machine when possible; use the shared build node for time-consuming link/place/route only when ready.
- Submit hardware runs only using the **admin-provided** scheduler/container mechanism. Start with a 1-card/short-time job; do not reserve multiple U55Cs while idle.
- Keep access tokens and SSH credentials out of Git. Do not assume Kubernetes `kubectl` is available on your personal computer.

**Deliverables:** `results/platform_manifest.md`, `results/lab04_feasibility.csv`, copy of actual `platforminfo` output, and a one-page resource-risk assessment. **Checkoff:** a teammate can explain why 16 GB HBM makes weight capacity plausible but does not guarantee a fast transformer.


## Lab 5 - U55C architecture contract and team split

**Objective:** keep independent developers compatible. Freeze tensor formats, AXI/stream boundaries, HBM ownership and integration schedule **before** real RTL is written.

### 5.1 Four workstreams with early handoff dates

| Track | Work owned | First week's required artifact | Completion test |
|:--|:--|:--|:--|
| A - model / goldens | tiny model, optional training, weight quantization, bit-exact oracle, traces, Qwen map | `model_spec.json`, integer oracle, first-token trace | cached vs full-prefix match; exact exported operands |
| B - RTL compute | matrix-vector MAC array, RMSNorm, RoPE, SiLU/SwiGLU, softmax | standalone `matvec.sv` + self-checking TB | signed/overflow/latency tests |
| C - attention and KV | GQA, INT4 codec, cache HBM layout, burst packing, metadata, cache toggle | `cache_layout.md`, address calculator, INT4 TB | read-after-write and token reuse tests |
| D - platform / integration | AXI-Lite control, AXI4 HBM master, host XRT, `.xo` packaging, `.xclbin` linking, CI, benchmark | emulator hello-world and DMA test plan | one U55C kernel executes and returns known bytes |

For a 3-person team, track D is rotating but **one named integrator** signs off all merges. Do not defer the HBM interface until the matrix and attention blocks are finished. Review stubs with all four tracks by the end of Lab 5.

### 5.2 Recommended repository extension (do not modify Labs 0-3)

```text
kv_fpga_project/
  src/  rtl/  tests/                 # Labs 0-3: leave intact
  full_fpga/                       # prior extension: float/integer reference
    spec/model_spec.json
    python/ ...
    rtl/llm_pkg.sv
    golden/
  u55c/
    spec/protocol_v1.md            # shared interfaces and timing
    spec/hbm_layout.csv            # every buffer, byte offset, alignment
    spec/platform_manifest.md      # populated after cluster access
    rtl/kv_addr.sv                 # deterministic cache byte-address math
    rtl/axi_hbm_reader.sv          # burst requests, beat FIFO, error handling
    rtl/axi_hbm_writer.sv
    rtl/kv_cache_controller.sv     # codec + HBM I/O and metadata
    rtl/compute_kernel.sv          # Vitis-compatible top-level RTL wrapper
    rtl/inference_fsm.sv           # prefill and decode scheduling
    tb/tb_kv_addr.sv
    tb/tb_axi_bursts.sv
    tb/tb_inference.sv
    host/host.cpp                  # XRT, weight/input/output buffers
    build/connectivity.cfg         # actual port -> HBM placements
    build/pack_kernel.tcl          # Vivado IP packaging + package_xo
    scripts/                       # inventory / test scripts
    results/                       # isolated benchmark runs
```

The appended ZIP provides **platform inventory and cache-address test scaffolding**, **not** a completed transformer `.xo` or `.xclbin`. Every file marked as an RTL/host deliverable must be implemented by your team.

### 5.3 Hard numerical contract - copied into protocol_v1.md

- Matrix product: `y[row] = sum W[row, col]*x[col]`. Weights flattened row-major. Initial formats: weights signed Q1.7 INT8, activations signed Q8.8 INT16, accumulator signed INT32 or wider, arithmetic right shift by 7, signed saturation to INT16. Later quantization revisions must fork the protocol and regenerate all goldens.
- K/V data after RoPE: signed Q8.8 values; INT4 signed codes `-8..7`; **low nibble = even component** and **high nibble = odd**. Scale is an **INT16 Q8.8 integer** per K-head vector and per V-head vector (not IEEE FP16). Codec must define tie-breaking and negative division precisely.
- Tiny model: L=2, T=64, Hkv=2, D=16. Cache strides in bytes: `ENTRY_BYTES = D/2 + SCALE_BYTES = 8+2=10` logical bytes per vector. K/V layout must have separate base addresses and correct byte alignment. One token consumes 2 layers x 2 heads x (K+V) x 10 = **80 logical bytes**, not necessarily 80 physical HBM bytes.
- GQA: Query head `qh` maps to KV head `floor(qh/(Hq/Hkv))`, when Hq divisible by Hkv. K is cached **after** RoPE, V without RoPE. New token sees itself in causal attention.
- A block uses `start` when idle, raises `busy` until completion, pulses `done` for one cycle; reset behavior and error semantics fixed. Ready/valid stream payload remains stable during backpressure.
- Cache policy is chosen before benchmarking: either the **current token is also compressed then decompressed**, or current K/V are used at high precision while only old history is decompressed. For a fair A/B test, document the chosen policy and apply the same attention path where feasible.

### 5.4 Memory ownership and AXI interfaces

Design **one Vitis RTL kernel** for the first integrated version. It owns all neural operations and has AXI4-Lite command/status, one or more AXI4 master ports to HBM, and optional AXI streams *inside* the kernel. The host sees buffer objects in device memory; it does not copy KV back to CPU during token generation. The AXI adapter handles outstanding read/write transactions, bursts, `RVALID/RREADY` backpressure, `BRESP/RRESP`, byte strobes, alignment, ordering and 4-KiB boundary constraints. The math blocks should consume a small **ready/valid tensor stream**, not drive the AXI bus directly.

Start with ports conceptually named `m_axi_weights`, `m_axi_cache`, `m_axi_scratch`. **Actual port names and connectivity syntax come from your packaged IP and chosen platform**. Do not assume a kernel can read all 32 pseudo-channels through one AXI master at maximum speed. Use a single port for bring-up, then partition if utilization profiling justifies it.

### 5.5 Simple top-level control/status contract

Choose **either** standard XRT-managed `ap_ctrl_hs` (recommended first: `xrt::kernel` / `run.wait`) **or** user-managed custom AXI4-Lite registers (`xrt::ip`); do not mix their register maps. The lab's control register sketch is **an internal design template**, not a replacement for the official `ap_ctrl_hs` register layout. For XRT-managed flow, map pointer/scalar args using kernel metadata. For custom user-managed flow, do **not** name a custom register `CTRL` (reserved in the supported flow). Read your installed release's UG1700/UG1702 documentation before packaging.

**Checkoff:** signed shared `protocol_v1.md`, stable `model_spec.json`, HBM logical address calculator, one dummy/tie-off kernel compiled through the supported flow, CI unit tests, and a named owner for each module and memory region.


## Lab 6 - Build the tiny full-transformer software golden reference

**Goal:** create a tiny causal transformer with the *same architectural operations as your intended RTL*. Your prior Qwen experiment remains a separate scientific reference; it does not provide compatible weights for your smaller model.

### 6.1 The exact inference algorithm

Define one token's hidden vector as `x` of length `E`. Each decoder layer performs, in this order (inference only, no dropout):

```text
h      = RMSNorm(x, norm1)
q      = Wq @ h + bq              # shape Hq x D
k      = Wk @ h + bk              # shape Hkv x D
v      = Wv @ h + bv              # shape Hkv x D
q, k   = RoPE(q,k,absolute_position)
append (k,v) to THIS layer's KV cache
for each query head hq:
  kv_head = floor(hq / (Hq/Hkv))
  score[t] = dot(q[hq], cached_k[kv_head,t]) / sqrt(D)
  prob = softmax(score over t=0..current_position)
  a[hq] = sum_t prob[t] * cached_v[kv_head,t]
x      = x + Wo @ concat(a)       # residual connection
u      = RMSNorm(x, norm2)
g      = Wgate @ u
r      = Wup @ u
x      = x + Wdown @ (SiLU(g) * r)
```

After the last layer, apply `final_RMSNorm(x)`, then `logits = tied_embedding_weights @ x`, then choose `argmax(logits)`. For the first prompt token, all layers have empty caches; when computing the second token, each layer retrieves the key/value computed by that same layer on the first token. Do not share a single cache between layers.

**RoPE:** rotate pairs of Q and K elements based on the absolute token position. `RoPE` applies to Q/K, not V; cached K is stored **after** RoPE. Store position indexes consistently across software and RTL.

**RMSNorm:** `RMSNorm(x)=g * x / sqrt(mean(x^2)+epsilon)`. The learned gain vector `g` must be exported. Do not substitute layer normalization because LayerNorm subtracts the mean; RMSNorm does not.

**SwiGLU:** `SiLU(z)=z/(1+exp(-z))`, and MLP intermediate is `SiLU(Wgate*u) * (Wup*u)`. It is not simply a ReLU MLP.

### 6.2 Create and test the floating-point model

Implement a PyTorch model under `full_fpga/python/tiny_model.py` with `TinyDecoder(config)`, `forward(input_ids, use_cache=False)`, `decode_one(token_id, cache)`, and `generate(prompt_ids, max_new_tokens, greedy=True)`. Build the **uncached** forward pass first, with a causal mask; then add a cached incremental pass. Keep the option to capture intermediate tensors (hidden states, raw projected Q/K/V, rotated K, attention scores/probs, MLP gate/up, logits) for every layer/token.

A crucial unit test: for the same input tokens and weights, `full_forward(prompt).logits[:, -1, :]` and `incremental_decode(prompt).last_logits` must match within a small FP32 tolerance (e.g. `atol=1e-5`, `rtol=1e-4`). If they differ, stop: you likely have a RoPE position, head repeat, cache insertion, or mask error. Use CPU FP32 for this floating-point identity test. In PyTorch 2.x API versions, KV structures vary; use your own simple `{keys,values}` arrays in this educational model.

**Your test fixture:** byte-encode `b"NEWS: AAA beat earnings. SIGNAL:"` (shorten/truncate to 64 tokens if necessary), prepend BOS, run prefill, then greedily generate two output tokens. Save exact token IDs as well as the float logits. At this stage randomly initialized weights may emit meaningless tokens. That is fine for functionality; training comes later.

### 6.3 Optional: train it to generate meaningful signal tokens

A randomly initialized network is still an autoregressive transformer, but cannot demonstrate useful financial interpretation. Train a small dataset offline on CPU/GPU (training is *not* part of FPGA inference). Construct simple templated posts with randomized tickers and phrasing, paired with POS/NEG/NEU label tokens. Example: `NEWS: AAA raises guidance. SIGNAL:` -> `POS` -> `EOS`; `NEWS: BBB cuts forecast. SIGNAL:` -> `NEG` -> `EOS`. Use a training/test split **by template family and company**, not just a random row split, to reduce memorization/leakage. Make contradictory and neutral test cases.

Train using cross-entropy on the next-token label and EOS; optionally add general language-model next-token loss if you want generated words as well as label tokens. Save `tiny_model.pt`, `model_spec.json`, tokenizer mapping, training seed, train/test split, and sample outputs. When your weights are frozen, calculate baseline label accuracy and token/sequence agreement. This is a *toy financial-language demonstration*, not evidence of profitable trading.

### 6.4 Golden tracing specification

For each deterministic prompt, export the following as `*.npy` **plus** human-readable `*.hex` where applicable:

| Trace | Per-layer / per-token shape | Hardware consumer |
|:--|:--|:--|
| `input_hidden` | `[E]` | embedding/scratch |
| `norm1` | `[E]` | matrix projection |
| `q_pre, q_rope` | `[Hq,D]` | attention |
| `k_pre, k_rope, v` | `[Hkv,D]` | cache compressor |
| `packed_k,packed_v,scale` | `[Hkv,D/2]` plus scale | cache memory |
| `scores, probabilities` | `[Hq,t+1]` | softmax |
| `attn_out` | `[E]` | output projection |
| `mlp_gate/up/down` | `[F]` / `[E]` | MLP |
| `layer_output` | `[E]` | next layer |
| `final_logits` | `[V]` | argmax |

The **float golden** verifies mathematical intent. It is *not* the bit-exact RTL oracle. In Lab 7 you will generate **integer golden** values with the same saturating and shifting rules as the RTL. Preserve both.

**Checkoff:** two-token cached vs uncached equality; downloadable/check-in model weights; reproducible traces and train/evaluation separation; explain exactly which tensors live on FPGA.

## Lab 7 - Quantize weights and build the fixed-point oracle

**Goal:** remove any ambiguity between PyTorch floating-point arithmetic and FPGA integer arithmetic, while preserving the same model architecture.

### 7.1 The golden arithmetic contract

Start with `x` in signed Q8.8 (an integer value divided by 256 represents the real number) and `w` in signed Q1.7 (integer divided by 128). For a single matrix dot product, calculate:

```text
acc = 0  (signed 32 or wider)
for j in 0..N-1:
    acc = acc + int32(x_q[j]) * int32(w_q[j])
y_q = saturate_int16(arithmetic_shift_right(acc,7))
```

Example: `x_q=[256,-128]` represents `[1.0,-0.5]`; `w_q=[64,64]` represents `[0.5,0.5]`. Products `[16384,-8192]` sum to `8192`, right-shift 7 gives `64`, or **0.25** in Q8.8. This test must pass in Python and RTL before anything more complicated is attempted.

The Q1.7 range is approximately `[-1,0.9921875]`; large weights require a different scale, such as a per-matrix shift. **The starter format is a teaching contract, not a quality guarantee.** Keep the float model's reference untouched and measure quantization error. Better calibrated post-training quantization or quantization-aware retraining is an optional optimization, not a silent alteration to the spec.

### 7.2 Choose KV scales with a deterministic rule

For each K or V vector of length D in Q8.8, let `maxabs=max(abs(x_q[d]))`. A simple *power-of-two* scale in Q8.8 can avoid a general division on FPGA: choose the smallest positive power-of-two scale (in Q8.8 integer units) large enough that `maxabs <= 7*scale_q`, then clamp the INT4 code to `[-8,7]`. To avoid overflow with `maxabs=0`, use `scale_q=1` (one integer unit, representing 1/256). Define negative-code rounding carefully and test exact ties. This is only one allowed compressor policy; per-vector absmax with non-power-of-two scales is another, but must be specified end-to-end.

To quantize using **truncation toward zero** with a positive integer scale (easy to implement exactly):

```text
q[d] = clamp(trunc_toward_zero(x_q[d] / scale_q), -8, 7)
packed[d/2] = (q[2*d+1] & 15) << 4 | (q[2*d] & 15)
reconstructed_q[d] = q[d] * scale_q  # saturate to INT16 if needed
```

In RTL, dividing by a **power of two** can use an arithmetic shift, but note that arithmetic right shift on negatives **floors**, whereas `trunc_toward_zero()` rounds toward zero. To implement this exact codec, either add correction for negative remainders or define the first version's quantizer to use flooring everywhere and change the Python oracle accordingly. A correct final design must agree **bit-for-bit**, including negative inputs. Do not quietly mix the two rules.

***Important interface detail:*** older `rtl/int4_pack.sv` and `rtl/int4_unpack.sv` from the original starter use a teaching fixed scale. They are useful as standalone components but **not automatically interchangeable** with the dynamic-per-vector codec above. Put the scale generator, packed data and scale store in the new `kv_store.sv`; validate the whole interface together.

### 7.3 Export `.mem` images, not just `.pt` checkpoints

For every matrix, emit one file containing row-major signed INT8 values, encoded as two hexadecimal digits (`-1` -> `ff`, `-128` -> `80`). Use `$readmemh` or a BRAM initialization flow that works with your target Vivado setup. Export normalization gains, per-matrix scales/shift counts, biases, RoPE sine/cosine lookup tables, and activation-function lookup tables in **explicit fixed-point formats**, with an index table giving address, shape, datatype, and size. Store a SHA-256 hash of every image in the build manifest.

Create a weight manifest like this:

| Tensor name | Shape | Storage | Order | Address source |
|:--|:--|:--|:--|:--|
| `tok_embeddings` | `[V,E]` | INT8/Q1.7 or INT16 (decide once) | row-major | manifest |
| `layers.0.wq` | `[E,E]` | INT8/Q1.7 | row-major | manifest |
| `layers.0.wk` | `[Hkv*D,E]` | INT8/Q1.7 | row-major | manifest |
| `layers.0.wv` | `[Hkv*D,E]` | INT8/Q1.7 | row-major | manifest |
| `layers.0.wo` | `[E,E]` | INT8/Q1.7 | row-major | manifest |
| `layers.0.wgate/wup` | `[F,E]` | INT8/Q1.7 | row-major | manifest |
| `layers.0.wdown` | `[E,F]` | INT8/Q1.7 | row-major | manifest |
| `layers.0.norm1/norm2` | `[E]` | chosen fixed-point format | 1-D | manifest |

Duplicate for layer 1, plus final RMSNorm. If tied embeddings are INT8, output logits can reuse that array; still check scales and accumulation width.

### 7.4 How to validate quantization without fooling yourselves

Run in stages: **(a)** float-trained model, **(b)** software integer model with exactly RTL-compatible shifts/saturation/LUTs, **(c)** future RTL model. Measure per-op max absolute integer error (target zero for stage b vs c), and task label accuracy for stage a vs b. An accuracy drop between a and b is not a Verilog bug if b matches its oracle; it is a precision/model-calibration problem. Keep those failure categories separate.

**Checkoff:** Python integer tests cover zero, two's-complement extremes, rounding ties, saturated products, odd/even nibble order, random vectors, and at least one whole tiny-model token step. Export golden input and output bytes for *every primitive*.

# Part III. Compute blocks, HBM, and causal attention

## Lab 8 - Signed matrix engine with HBM-to-BRAM tiling

**Objective:** implement the core arithmetic that eventually consumes U55C HBM bursts. Start with a simple signed dot-product in RTL simulation; scale to a tiled architecture only after it is bit-exact.

### 8.1 First build - tiny deterministic product

Test `x=[256,-128]` (Q8.8) and `w=[64,64]` (Q1.7). `acc=8192`; `acc >>> 7 = 64` (Q8.8 => 0.25). Compare against `full_fpga/python/fixed_oracle.py`. Test signed extremes, zero, negative products, saturation, reset while busy, and start-during-busy. Save cycle counts as well as data values.

Implement a reusable `matvec_core.sv` with `start/busy/done`, matrix rows/cols, input and weight stream handshakes, output ready/valid, and an INT32+ accumulator. **Do not connect raw Python float arrays to RTL**: emit exactly signed INT8 and INT16 hex bytes with stable endianness.

### 8.2 Tile for the real memory system

The recommended architecture is `HBM read -> burst FIFO -> BRAM/URAM weight tile -> P parallel MAC lanes -> partial-sum bank -> output FIFO`. Keep the current activation vector on chip and reuse it across multiple weight rows. For a matrix of shape `[M,N]`, a P-lane row-parallel engine nominally computes `M*N/P` multiply-accumulates, plus startup/tail/loads. Its actual timing depends on whether you have **P independent operands per clock**. A naive one-port BRAM feeding P multipliers has no such parallelism.

Freeze a concrete initial tile **only for the tiny model**: e.g. 8 output rows x 16 input columns, 8 MAC lanes and 8 independent accumulators, with BRAM packed to supply all lanes. This is a **design target**, not a measured U55C performance. Increase P and tile dimensions only after routing and HBM burst profiling.

### 8.3 Standalone simulation and verification

1. Write a behavioral BRAM/FIFO latency model with a documented 1- or 2-cycle read latency.
2. Assert that each accepted AXI/read beat becomes exactly one ordered FIFO beat (no duplicate/lost beats under stall).
3. Compare 2x2, 8x16, 64x64, and one random quantized row against integer oracle; verify all rows with a stable matrix addressing convention.
4. Demonstrate correct `done` only after final output handshake; stall output halfway through a tile and ensure accumulator state does not corrupt.
5. Synthesize **only this engine** to report LUT/FF/DSP/BRAM/URAM utilization and initial Fmax. Do not infer whole-LLM performance from standalone Fmax.

**Checkoff:** self-checking `tb_matvec.sv`, hardware resource estimate, cycle accounting, tile diagram, and a plan to map HBM read bursts into the weight FIFO.


## Lab 9 - Nonlinear operations: normalization, RoPE, SiLU, softmax

**Goal:** implement the functions that prevent a transformer from being merely a chain of matrix multiplies. Work with **the same fixed-point formats and oracle** in every block.

### 9.1 RMSNorm

For E=64 input elements `x_q`, compute `mean_square = (sum(x_i*x_i))/E` in an adequately wide accumulator. Use `epsilon` in the proper **squared-value** units. Then compute the reciprocal square root and multiply each input by `gain[i]`. The basic hardware solution is a ROM/table approximation to `1/sqrt(z)` plus shifts for range normalization; a higher-fidelity alternative is a small iterative approximation.

**First test in floats:** inputs `[3,4]`, gain `[1,1]`, zero epsilon. Mean-square is 12.5; normalized outputs are approximately `[0.848528,1.131371]`. Your FPGA fixed-point result will approximate these; compare against the fixed-point oracle, not blindly against infinite-precision values. Test all-zero x, large x, a one-hot input, and signs. Record absolute and relative error. A hardware LUT needs an explicit input domain, saturation policy, and treatment of zero; undefined table addresses are not acceptable.

### 9.2 RoPE: rotate position-dependent pairs

For each **pair** `(x_even,x_odd)`, apply:

```text
x_even_new = x_even*cos(theta) - x_odd*sin(theta)
x_odd_new  = x_even*sin(theta) + x_odd*cos(theta)
```

`theta` depends on absolute sequence position and feature-pair index; use the exact same formula as your tiny PyTorch model. Store cos/sin values in a BRAM table, e.g. Q1.15, and use a wide intermediate product followed by a documented shift and saturation. Position zero should be the identity rotation. Q and K are rotated separately; V is not. Golden test positions 0, 1, 2 and 63, and assert approximate preservation of squared norm (within quantization error). If Q and K use *different* position schemes, incremental decode will break.

### 9.3 SiLU and gated MLP

The MLP requires `SiLU(g)*u`; do not apply SiLU to the `up` branch. Implement `SiLU` using a lookup table plus interpolation or a small piecewise approximation over a declared clamped input interval, e.g. [-8,+8], in Q8.8. Outside the table range specify a deterministic approximation (`SiLU(z)` tends to 0 for strongly negative z, and to z for strongly positive z). Run fixed-point tests at -8, -2, -1, 0, +1, +2, +8 and at LUT boundaries. Multiplying the SiLU output and `u` requires a correctly shifted, saturated result.

### 9.4 Stable causal softmax

For a query head and `t+1` valid keys:

```
m = max(score[0..t])
u[i] = exp(score[i] - m)     # prevents positive exponent overflow
p[i] = u[i] / sum_j u[j]
```

A *first working FPGA design* can compute three sequential passes: find max; fetch exponent values from a LUT; accumulate the sum and normalize. Exponent inputs are `<=0` after maximum subtraction; clamp values below the LUT range to an appropriately small/nonnegative value. Use wider accumulators for the sum (64 probabilities each near one can overflow a 16-bit sum). Division can start as a simple sequential integer divider; optimize later with reciprocal lookup or Newton iteration. A reusable softmax over at most `Tmax=64` elements is sufficient.

**Numerical example:** scores `[1,0]` have softmax weights about `[0.731059,0.268941]`; scores `[1001,1000]` produce **the same** weights after subtracting max. If the latter overflows your hardware, the implementation is incorrect. Test identical scores (uniform distribution), large negative differences, one-token attention (exactly weight 1), and a non-power-of-two live length.

### 9.5 Verification integration gate

Every math block must have two separate correctness reports: a float-vs-integer *approximation-error* report, and an integer-oracle-vs-RTL *bit-exactness* report. RTL should match the integer oracle exactly (unless the protocol permits a narrow tolerance for an explicitly approved alternative math approximation). Do not combine all four operations into attention until these tests pass independently.

**Checkoff:** `tb_rmsnorm`, `tb_rope`, `tb_silu`, `tb_softmax` report PASS; each block has deterministic latency bounds and its input/output numeric format documented.


## Lab 10 - Real HBM-resident INT4 KV cache and GQA attention

**Objective:** implement a compressed cache **that the FPGA actually stores and reuses**. This is the primary research subsystem. Do not accept a demonstration that merely compresses and immediately decompresses a vector with no retained history.

### 10.1 Data lifetime and exact cache address

For each transformer layer `l`, token `t`, KV head `h`, vector kind K or V, the core produces `D` signed Q8.8 elements. K must be rotated with RoPE **before** caching. Quantize each vector to signed INT4 with a documented scale, pack two nibbles/byte, store packed payload and scale. On attention for token t, read all relevant prior K/V vectors for positions 0..t, reconstruct their Q8.8 elements and calculate causal attention.

Initial **logical** layout: separate base addresses for K and V. Define:

```text
entry_bytes = (D/2) + SCALE_BYTES         # D even
index(l,t,h) = ((l*Tmax + t)*Hkv + h)
K_addr = K_base + index * entry_bytes
V_addr = V_base + index * entry_bytes
payload occupies first D/2 bytes; scale occupies final 2 bytes.
```

For the tiny model `entry_bytes=10`; a 64-byte HBM memory beat may span several logical entries. **Never** assume writing 10 bytes via an arbitrary 64-byte aligned burst is always supported by the platform without byte strobe/masking/rmw logic. Easiest correct first implementation: arrange each logical entry in its own **32- or 64-byte physical slot** and use aligned transfers, clearly recording the physical overhead. Then improve by packing multiple logical entries per aligned HBM line with byte enables or a full-line read/modify/write controller. Compare *logical payload bytes*, *physical allocated bytes*, and *actual AXI traffic bytes* separately.

Suggested first physical layout: `SLOT_BYTES=64` for every vector, `physical_addr = base + index*64`; payload+scale at offsets 0..9, remainder reserved zero. INT16 baseline similarly uses `D*2=32` payload bytes in a 64-byte slot. Under this first layout **both modes allocate the same 64 bytes/entry**, so **no claim of actual HBM footprint reduction** is allowed. Your optimization assignment is to implement a packed physical layout where INT4 really uses fewer 64-byte lines. This distinction prevents a common false positive.

### 10.2 Exact INT4 codec + edge cases

1. Choose a deterministic positive per-vector scale (e.g. smallest power of two in Q8.8 sufficient to fit most/all values; avoid `scale=0`).
2. Quantize with an explicitly specified signed rule. For truncation-toward-zero division by `2^s`, correct the negative arithmetic-shift behavior or update **both** oracle and RTL to use floor.
3. Pack even feature into low nibble, odd into high. Example `[1,-1,-8,7]` becomes bytes `F1 78`.
4. Sign-extend each 4-bit code before multiplying by the scale. Saturate to Q8.8 signed INT16.
5. Test all signed nibble values `-8..7`, zero vector, max/min Q8.8, odd address alignment, backpressure, repeat reads, overlapping writes and reset during bursts.

The original `rtl/int4_pack.sv` is a **fixed-scale teaching codec** and not a drop-in implementation of this dynamic per-vector policy. Use it as a starting primitive only.

### 10.3 Make memory traffic measurable

Maintain separate 64-bit counters for logical payload bytes, memory issued bytes, accepted AXI read beats, write beats, bursts, stalls and cache occupancy. To fairly attribute speedups to INT4, a first **same-bitstream** design may switch between INT16 and INT4 modes; however, record which architecture is physically active. A dual-mode circuit's resources are not the area of a stand-alone compressor design.

### 10.4 GQA and attention unit

Query heads may be more numerous than KV heads. `kv_head = q_head / (Hq/Hkv)`. For t, compute scores `dot(Q[qhead], K[kvhead,0..t]) / sqrt(D)`, apply **causal** masking, stable softmax and a weighted sum of V for 0..t. The current token's K/V must have been inserted or provided through a forwarded path before attention consumes them. Choose one path and freeze it in tests.

Make **two cache modes**: `baseline = 16-bit K/V` and `compressed = packed INT4 + scale`; run the **same** layer/MLP, input token IDs and weights. The compressed case may differ numerically and produce different generated tokens; quality metrics therefore complement throughput metrics.

### 10.5 Verification ladder

| Simulation | What must be checked |
|:--|:--|
| `tb_kv_addr` | start/middle/end of l/t/h ranges; K/V do not overlap; no 64-bit wrap |
| `tb_codec` | all sign/round/saturation cases; pack/unpack under stalls |
| `tb_cache_store` | 64-byte aligned writes, no corruption of neighboring entries, full clear/reset |
| `tb_cache_history` | write 3 tokens, read token 0 again during token 2; no PC/host KV movement |
| `tb_attention` | 1 token, 2 tokens, causal 3-token case; GQA head map; scales used |
| `tb_axi_stalls` | randomized ARREADY/RVALID/AWREADY/WREADY/BVALID; correct data ordering and responses |

**Checkoff:** simulation and, later, hardware trace that attention's subsequent token reads **the exact bytes it wrote for earlier tokens**. Report a packed-slot optimization (or explicitly explain if only logical compression, not physical HBM savings, is achieved).


## Lab 11 - MLP, one complete transformer block, and residual paths

**Goal:** combine the verified modules to compute one complete decoder block from input `x` to output `x_next`. The previous stage proves attention; this stage proves the learned computation surrounding it.

### 11.1 The MLP datapath

Compute `u = RMSNorm(x_after_attention, norm2)`, `gate = Wgate @ u`, `up = Wup @ u`, `mixed = SiLU(gate) * up`, `mlp_out = Wdown @ mixed`, then add the residual. Gate/up arrays are length F=128; input/output are length E=64. With a single reused matvec engine, execute gate, up, and down sequentially; keep intermediate arrays in scratch BRAM. No second physical matrix engine is required to make a true FPGA inference path.

### 11.2 Residual addition must be explicit

If the attention branch outputs `a`, the first residual is `x1 = saturate_int16(x + a)`. The second is `x2 = saturate_int16(x1 + mlp_out)`. Omitting a residual changes the neural model, not merely its speed. For testing, configure projection weights to all zeros: the block should reduce to the input (subject to exact zero-bias assumptions). This is an excellent integration smoke test.

### 11.3 Layer transaction and golden fixture

Create `transformer_block.sv` that accepts `layer_id`, `token_position`, `input_hidden_base`, `output_hidden_base` and `start`. It calls norm1, Q/K/V, RoPE, cache/attention, Wo, residual, norm2, MLP, residual, then pulses `done`. When each subblock completes, record a monotonically increasing trace event with **module name, cycle counter and layer/token**. The Python fixed oracle should export the expected output **after each** sub-operation, not just the final token. This is essential for debugging numerical mismatches in a 2-layer network.

**Test order:** zero weights; identity-like weights; short random weights; trained/quantized weights; prefill length 2; decode token 3. For a test that contains two layers, verify that the second layer receives the output of the first—not the original embedding—and uses its **own** K/V storage. Assert correct reset/clean-cache state on a new prompt.

**Checkoff:** bit-exact intermediate comparisons for a whole layer on at least three deterministic prompts; no old cache state survives a reset between prompts.


## Lab 12 - Complete FPGA autoregressive decoder (one kernel)

**Objective:** run **embedding -> every transformer layer -> logits -> next-token selection** without Python/Torch running any layer. This is the key distinction between the full-FPGA project and a KV-cache coprocessor.

### 12.1 Kernel execution schedule

For prompt IDs `[t0,t1,...,tn-1]`, run **prefill**: embed each position and execute all layers in correct order, writing each layer's own K/V history. After processing `tn-1`, read the final logits and greedy-argmax for the first new token. Then **decode** each generated token using the existing per-layer histories; append new K/V; repeat until maximum output length or EOS. Save complete token IDs in an FPGA-owned output buffer and return them to host *only after* the requested generation finishes (preferred final mode). First bring-up may use one host launch/token for simplicity, but disclose dispatch overhead in benchmarks.

Canonical per-layer processing order for a decoder-only Qwen-like block:

```text
hidden -> RMSNorm -> Q/K/V projections -> RoPE(Q,K) -> cache K,V
       -> masked GQA attention -> output projection -> residual add
       -> RMSNorm -> SiLU(gate)*up -> down projection -> residual add
```

Then final RMSNorm -> LM head -> argmax. **No layer executes on CPU/GPU during a hardware generation.** Host loads token IDs/weights ahead of launch, checks done/status after and detokenizes afterwards.

### 12.2 Correctness vs numerical quality

There are three independent references: trained floating point tiny model, integer bit-exact software oracle, and SystemVerilog RTL/hardware. Accept **exact byte agreement** between integer oracle and RTL, and measure quality loss versus float. When implementing exp/reciprocal/sqrt via LUT, freeze rounding, range clipping and interpolation in the numerical oracle before hardware.

**Failure signature:** first generated token matches but second diverges -> inspect cached K after RoPE, per-layer cache indexing, attention score masking, kv scale, and token-position updates. If the prefill last-logit result already diverges, debug layer 0 intermediate traces first.

### 12.3 Checkoffs

- Behavioral SV simulation processes at least 4 prompt tokens + 3 generated tokens.
- RTL traces agree with integer oracle for every stage of one token; compressed and uncompressed modes can have different oracle output, so generate both reference traces.
- First FPGA execution returns expected bytes for an identity or simpler test model **before** full transformer deployment.
- Final board run returns a complete generated token sequence using **one XRT launch for multiple tokens** when feasible.


# Part IV. U55C HBM pipeline, runtime integration, and measurement

## Lab 13 - HBM bandwidth laboratory and complete prefill/decode schedule

**Objective:** identify the actual memory bottleneck *before* blaming KV compression performance. HBM is not interchangeable with on-chip BRAM, and bandwidth is not just a device data-sheet number.

### 13.1 HBM calibration experiment - do this before the LLM

Implement a Vitis-compatible **memory copy/read-reduce** kernel whose only job is to read a known device buffer, apply a simple checksum/XOR, and write a result. First test it using XRT on one chosen bank, then over buffers bound to multiple bank groups if the platform permits. Independently measure:

- Device-to-kernel **read** bandwidth for 4 KiB, 64 KiB, 1 MiB and 64 MiB buffers.
- Device write bandwidth and read+write contention.
- At least two patterns: contiguous aligned burst and strided/random-like access.
- Throughput with one master, then two masters if supported; note outstanding depth, AXI width, clock, burst count, average stall time.
- Host-to-device and device-to-host PCIe time **separately**; those do not count as internal HBM bandwidth.

Use `bytes actually transferred / measured kernel seconds` for sustained GB/s. `fclk * bus_bytes` is only an interface upper bound. Do not report a device aggregate 460 GB/s as an experimental result.

### 13.2 Split weight and cache traffic logically

Weights generally dwarf KV storage for small-batch, short-context decoding. A 490M-weight INT4 model requires roughly 245 MB of payload **per streamed full-weight pass** if weights are not persistently buffered. Lower bound `T_weight >= model_weight_bytes / measured_weight_BW`. Attention cache traffic adds `T_cache >= actual_cache_read_bytes / measured_cache_BW`, but **sum of the independent times is not a guaranteed latency** when operations overlap or share bandwidth. Measure stall cycles rather than assuming all accesses overlap perfectly.

Start with a **single HBM read master and serialized compute**, not multiple independent `m_axi` ports that create unroutable congestion. Use weights in larger sequential bursts and cache in separate allocations. Once correctness is established, increase burst length, FIFO depths and port count; record HBM crossbar contention and changes to routing/Fmax.

### 13.3 Prefill and autoregressive decode instrumentation

Counters in FPGA must distinguish: `prefill_cycles`, `decode_cycles`, `weight_read_bytes`, `cache_read_bytes`, `cache_write_bytes`, `cache_stall_cycles`, `MAC_busy_cycles`, `softmax_cycles`, and `generated_tokens`. Compute **time-to-first-token**, average decode token latency, tokens/s for 32+ tokens, and total end-to-end host+FPGA wall time. Reset counters and cache for each benchmark run. Prefill can first process prompt tokens sequentially (correct but slow), then be improved with parallel prompt attention.

**Checkoff:** a bandwidth roofline worksheet using *measured* HBM BW, an HBM smoke-test log, and a token-by-token traffic trace that distinguishes weight and cache access.


## Lab 14 - UIUC U55C deployment: RTL kernel -> XO -> XCLBIN -> XRT

**Objective:** execute a genuine bitstream on an allocated HACC mlCluster U55C. **This replaces the prior generic board/UART/JTAG Lab 14.**

### 14.1 What the installed tools actually do

- **Vivado**: synthesize and package your SystemVerilog/AXI IP for Vitis.
- **`package_xo` in Vivado Tcl**: turn a packaged RTL IP + interface metadata into a `.xo` kernel object.
- **`v++ --link`**: link your `.xo` with the administrator-installed U55C **`.xpfm` platform** and connectivity into an `.xclbin` containing configured logic and platform metadata.
- **XRT C++ host**: open the device, load `.xclbin`, allocate/migrate HBM buffers, launch kernel, wait, read output.
- **Simulation and hardware emulation**: validate functionality before expending full implementation cycles; support varies by installed shell/tool release.

Source: [AMD UG1700 RTL packaging](https://docs.amd.com/r/en-US/ug1700-vitis-accelerated-data-center/Creating-and-Packaging-RTL-Kernels), [AMD package_xo](https://docs.amd.com/r/2026.1-English/ug1702-vitis-accelerated-reference/package_xo-Command), [XRT API](https://xilinx.github.io/XRT/master/html/xrt_native_apis.html). Match examples to the installed tool version. **Do not attempt custom platform flashing.**

### 14.2 Recommended step order

1. **Local simulation:** `xvlog -sv ...; xelab ...; xsim ...` (or your chosen simulator). Prove RTL handshake and numerical tests. The exact file list must be created by your team.
2. **Platform discovery on development node:** record `v++ --version`, `vivado -version`, `$XILINX_XRT`, `platforminfo -p "$PLATFORM"`. Save valid clocks, memory groups and shell ID.
3. **First kernel = simple known-output kernel**, not the full transformer. Package a kernel with one control interface and memory read/write ports; prove correct operation in hardware emulation or on an allocated U55C.
4. **Package RTL IP:** use Vivado IP Packager, associate *every top-level port* with its appropriate clock/reset/AXI interface, then run the Tcl packaging command below.
5. **Link Vitis xclbin:** use `v++` and a deliberately small platform-specific connectivity configuration; inspect warnings and `link_summary` for HBM mapping.
6. **Run through the cluster-approved scheduler/Kubernetes allocation:** copy host executable and xclbin into the approved workspace; never assume direct device access from the development VM.
7. **First full transformer:** run tiny model for one token, compare exact bytes with oracle; extend to 4 prompt and 3 generated tokens, then 32-token benchmark.

### 14.3 Command skeletons - replace verified paths, names, flags

**Vivado packaging (inside Vivado Tcl after IP packaged):**

```tcl
# Only after Vivado IP packaging has created the actual IP directory:
package_xo -force \
  -xo_path /absolute/path/build/kv_llm.xo \
  -kernel_name kv_llm \
  -ip_directory /absolute/path/to/packaged_ip \
  -ctrl_protocol ap_ctrl_hs
```

**Vitis link (development node; the platform may require version-specific options):**

```bash
export PLATFORM=/absolute/path/from-admin/installed_platform.xpfm
v++ --link -t hw --platform "$PLATFORM" \
  --config u55c/build/connectivity.cfg \
  -o u55c/build/kv_llm.xclbin u55c/build/kv_llm.xo
```

The connectivity config's port names are **not guessed**; derive them from the XO's kernel metadata, then use the platform's valid memory names. A common *illustrative* form is:

```ini
[connectivity]
nk=kv_llm:1:kv_llm_1
sp=kv_llm_1.m_axi_weights:HBM[0:3]
sp=kv_llm_1.m_axi_cache:HBM[4:5]
```

**This example is not a ready-to-run configuration.** The actual shell may expose different bank-group capabilities, the RTL port names must match and a single port's connectivity range is not a promise of parallel bandwidth. Use `platforminfo` and the cluster's approved U55C examples to validate, and separate K and V ports only after correct baseline.

### 14.4 Host-code contract

For an XRT-managed kernel using `ap_ctrl_hs`, a host program conceptually does the following:

```cpp
xrt::device device(device_index);
auto uuid = device.load_xclbin("kv_llm.xclbin");
xrt::kernel kernel(device, uuid, "kv_llm");
auto weights = xrt::bo(device, weight_bytes, kernel.group_id(0));
// Other buffers allocated with group_id(their actual argument indices).
// weights.write(...); weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);
// auto run = kernel(weights, cache, input, output, ...);
// run.wait(); output.sync(XCL_BO_SYNC_BO_FROM_DEVICE); output.read(...);
```

**This is explanatory pseudocode**, not a compilable host file: argument positions, HBM groups, scalar registers and compatible API are defined only after packaging. Write a real compiled XRT host program after checking the installed XRT API. If instead using `user_managed` control, replace `xrt::kernel`/`run` with `xrt::ip` and manually honor the custom command registers; do not mix them.

### 14.5 Hardware bring-up checklist (one gate at a time)

1. Kernel reads 4 test integers from HBM, returns expected checksum.
2. Kernel writes a full cache line, host reads it back **after kernel completion**.
3. Kernel reads/writes multiple cache lines and survives backpressure.
4. Run FPGA matvec on fixed oracle test vectors.
5. Run one transformer block with integer oracle agreement.
6. Run all layers and one output token with identical weights.
7. Run multiple tokens with correct history; compare compressed vs uncompressed modes.
8. Only then enable parallel masters, large tiles, prefetch and burst optimization.

**Checkoff:** `.xo`, `.xclbin`, build logs, `platforminfo`, host executable, approved job submission record, hardware output and checksum. If hardware access has not yet arrived, report *simulation/hw_emu* status honestly, not 'verified on U55C'.


## Lab 15 - Scientific benchmarking: baseline, FPGA INT16, FPGA INT4

**Objective:** isolate changes caused by the KV codec instead of confusing different models or compute platforms.

### 15.1 Required runs

| Experiment | Transformer executes where? | Cache | Purpose |
|:--|:--|:--|:--|
| A: PyTorch reference | CPU or GPU | normal high-precision | semantic/quality reference; **not** direct accelerator A/B |
| B: U55C baseline | Entire model on U55C logic | uncompressed INT16 | within-platform control |
| C: U55C compressed | Entire identical model on U55C logic | packed INT4 + scales | compressor impact |
| D (optional) | U55C compute with *software-equivalent* codec | same format | separate compute-vs-codec overhead |

For B vs C, hold the model architecture/weights/quantization, prompts, host/XRT version, device, clock target, batch, prefill length, generated token count, and HBM allocation policy as constant as possible. If two bitstreams are needed, report DSP/LUT/Fmax differences, placement variability and actual attained clocks; prefer a controlled runtime mode if feasible. **Generated output tokens may differ because quantization is lossy**; compare both the *same forced token sequence* (fair hardware timing) and autonomous generation (quality/behavior).

### 15.2 Measurements and definitions

| Metric | Exact calculation | Why |
|:--|:--|:--|
| Logical ratio | 16-bit KV payload bytes / (INT4 payload + scale bytes) | Codec efficiency without memory allocation artifacts |
| Physical HBM ratio | actual reserved/padded baseline bytes / actual compressed bytes | Real storage capacity benefit |
| AXI traffic ratio | counted actual baseline cache bytes / counted compressed cache bytes | Real bandwidth reduction |
| Prefill time | first launch/start -> final prompt token complete | Prompt processing |
| First-token latency | host input ready -> first emitted new token | Live signal response latency |
| Decode rate | generated tokens / elapsed decode seconds | Model-level throughput |
| FPGA-only cycles | `counter_end - counter_start`; convert using **actual clock** | Excludes XRT launch/PCIe |
| End-to-end time | tokenization + H2D + dispatch + FPGA + D2H + display | Honest user-visible performance |
| Quality | token agreement, forced-prefix perplexity or logit error, classification macro-F1 | Quantization impact |
| Hardware cost | LUT/FF/DSP/BRAM/URAM, Fmax, build time | Design tradeoffs |

Repeat each run at least 5 times with a warm-up. Record median and spread (e.g. p10-p90), prompt IDs, model hash, platform shell, bitstream hash, clock, memory allocation metadata, measured HBM BW, host CPU and any throttling. In a shared cluster, record whether the card was exclusive during benchmarking.

### 15.3 Controlled context sweep

Use prompt lengths 8, 32 and 64 for tiny model; decode 8, 16 and 32 tokens while obeying its max-context limit (total context <=64). For Qwen only if genuinely implemented, add context lengths 128, 512, 2048, 4096 (subject to configured RoPE/max length). Batch size 1 initially. Report prefill and decode separately. **The tiny model's KV is so small that its performance may be dominated by weights/control rather than cache traffic; that's useful insight, not a failure.**

### 15.4 Required figures and failure analysis

1. Logical, physical and traffic compression ratio vs context length.
2. FPGA prefill latency and decode tokens/s, B vs C, with uncertainty bars.
3. Weight bytes vs cache bytes per generated token and HBM utilization.
4. INT4 absolute/relative reconstruction error and output quality.
5. LUT/FF/DSP/BRAM/URAM/Fmax and achieved frequency.
6. End-to-end host+FPGA latency, including XRT launch and PCIe upload/download.

Do not claim HFT performance from a synthetic tiny-model signal classifier or from PCIe microbenchmarks alone. An offline paper-signal demonstration is appropriate; sub-millisecond exchange execution and real alpha are entirely different claims.

**Checkoff:** a one-command benchmark harness or fully documented submission/run script, saved raw CSV, plots, repeated results and a limitations table. Distinguish *predicted* bandwidth from *measured* bandwidth.


# Part V. Application and scale-up

## Lab 16 - Finance demonstration: posts to reproducible paper signals

**Goal:** run a real end-to-end financial-language demonstration on the FPGA. Keep this educational application distinct from production high-frequency trading.

### 16.1 Start with a tiny classified vocabulary

The default TinyDecoder has special output tokens `POS`, `NEG`, `NEU`. Present an input of the form `NEWS: [text]. SIGNAL:` and generate a label token. The host maps this token to `+1`, `-1`, or `0` (positive, negative, neutral), but the **label decision is made by FPGA transformer arithmetic**, not by a host rule. The host can package the resulting classification into a CSV record with event timestamp, label, FPGA inference completion time, and model/bitstream hash.

### 16.2 Use an explicitly labeled, offline dataset

Use synthetic earnings and guidance headlines first: e.g. "raises outlook", "misses revenue", "maintains guidance". Keep mutually contradictory cases and neutral cases. Hold out unseen templates and compare accuracy/F1 for A, B, and C. Add actual public financial posts later only when you have time-aligned labels and usage rights. Do not treat a retrospective classifier on random historical posts as evidence of a feasible HFT strategy.

### 16.3 Signal latency and paper outcome

The relevant latency is not just the FPGA MAC latency. Timestamp a data event's arrival, tokenize it, transfer token IDs, run prefill/next-token inference, transfer output, and register a paper action. Measure this **total reaction latency** as well as internal compute cycles. You may evaluate paper signals against later price changes in an offline dataset, but control look-ahead leakage, spread, fees, stale quotes and execution delay. Do **not** place live orders as a project requirement.

### 16.4 Baselines and claims

A dictionary keyword classifier and a small software financial-sentiment model can serve as additional application baselines. Do not confuse model quality with hardware novelty: your FPGA achievement is **full transformer inference plus compressed cache**. Classification F1 is useful to show the cache's approximation effects, not to assert a profitable strategy.

**Checkoff:** physical board accepts an encoded post and returns a label with measured latency; benchmark table compares labels from the same checkpoint under uncompressed and compressed FPGA cache.


## Lab 17 - Gated scale-up: pretrained Qwen2.5-0.5B on ONE U55C

**Objective:** use U55C HBM to make a pretrained Qwen attempt plausible while retaining rigorous feasibility gates. Qwen is **not** automatically feasible at useful throughput simply because weights fit in 16 GB.

### 17.1 Frozen Qwen model contract

Reference Qwen2.5-0.5B-Instruct architecture: ~490M parameters, 24 decoder layers, E=896, FFN=4864, Hq=14, Hkv=2, head dimension=64, vocab=151,936. Confirm exact `config.json`, tokenizer revision, tied embedding setting, bias conventions and checkpoint tensor names; **do not reuse tiny model token IDs or training weights**. Pretrained inference is sensitive to the exact RMSNorm epsilon, RoPE positions and quantization scales.

A theoretical INT8 weight payload is ~490 MB and INT4 is ~245 MB, plus scales/biases/rounding metadata. Even FP16 model weights at ~980 MB fit by capacity. But a **single output token may require streaming many of those bytes** from HBM, making decode throughput memory bound. Also output LM-head logits over ~152k vocabulary entries are nontrivial and require a reduction/argmax plan.

### 17.2 Mandatory go/no-go before any 24-layer hardware build

1. U55C HBM read bandwidth is measured, with relevant contiguous/strided access patterns.
2. Track A whole model runs on a real U55C (not just behavioral simulation).
3. One Qwen RMSNorm/QKV layer matches a **quantized software oracle** using exported pretrained parameters.
4. The integer model's quality degradation is tolerable on a fixed test corpus. If not, calibrate per-group/per-channel scales or use wider arithmetic before hardware expansion.
5. Synthesis/implementation passes timing with a small reusable matrix engine, and estimated weight-bytes/token divided by measured bandwidth satisfies your project's throughput goal.
6. Team can load a full checkpoint into assigned HBM banks and verify hashes without using host neural compute during inference.

### 17.3 Scaling architecture

Use one reusable **tiled weight-streamed** matrix engine rather than instantiate 24 complete layers. Load layer-specific Q/K/V/O/MLP weights in HBM and schedule the same arithmetic resources for each layer. Use double buffering for HBM weight tiles and activation scratchpads if timing allows. Cache K/V by layer, token and KV head, with enough address space and alignment to hold the desired context.

Add quantization in stages: pretrained tensor export -> quantized Python reference -> matrix engine agreement -> one whole block -> two blocks -> all 24 blocks -> full prompt prefill -> decoded tokens. Do **not** skip the software quantized reference and expect bitwise agreement with full-precision Qwen. Train a custom checkpoint only if you clearly label it as **not pretrained Qwen**.

### 17.4 A simple lower bound, not a throughput claim

If a pass reads 245 MB of INT4 weight payload and measured sustainable weight bandwidth is `B` GB/s, a simplistic lower bound is `T_weight >= 0.245/B seconds per generated token` (using decimal units, ignoring metadata/overlap). At *hypothetical* 100 GB/s this is **2.45 ms/token just for streaming raw weights**, not total latency. Actual latency includes memory scheduling, matrix operations, attention, nonlinear operations, stalls and output control. The **460 GB/s card peak is not a measured application result**.

**Checkoff:** either a fully measured Qwen hardware run on U55C, or a clearly labeled technical feasibility report with passing one-layer/quantized software results. Never label a tiny-model demo as running pretrained Qwen.


# Part VI. Team execution, integration gates and reporting

## Work plan tailored to the period before and after U55C access

The week numbers are *relative project stages*, not promises of delivery time. Divide ownership **now**, and treat the platform team as an active contributor rather than a final-stage helper.

| Stage | Golden-model lead | Compute RTL lead | Cache / attention lead | Platform / verification lead | Shared gate |
|:--|:--|:--|:--|:--|:--|
| A (before access) | Freeze tiny spec; golden tensors | Matvec signed TB | Cache layout and address TB | Repository/CI, kernel interface design | `spec-v1` and common test |
| B (before access) | Fixed-point oracle, export images | MAC/FIFO and tile TB | INT4 pack/scale + history tests | XRT host skeleton, emulated AXI BFMs | RTL primitives pass |
| C (access arrives) | Tiny checkpoint/hash | Matvec resource test | Burst cache controller | `platforminfo`, shell, scheduler, HBM smoke test | one allocated U55C returns checksum |
| D | oracle tracing of one layer | RMSNorm/RoPE/SiLU/softmax | GQA attention integration | XO packaging, xclbin, trace assertions | one-layer oracle agreement |
| E | trained tiny model and label set | Pipelining/resource repairs | Full cache mode toggle | tiny FPGA token execution | full tiny FPGA decoder passes |
| F | autonomous vs forced-prefix quality | MAC throughput tuning | physical HBM packing optimization | repeated B/C experiment | raw CSV and plots |
| G (optional) | Qwen export + quantized goldens | Qwen dims/tile schedule | longer-context cache | 24-layer integration plan | Qwen go/no-go |

**Daily handoff rule:** every module has a documented input format, timing, one self-checking testcase, and reviewer other than author. A failing integration test blocks merge even if standalone synthesis reports a high Fmax. Keep the interface fixed unless everyone approves a versioned update.

## Minimal live demonstration script

1. Show the actual U55C `.xpfm`/shell, XRT/Vitis versions and the built `.xclbin` fingerprint.
2. Load model weights and token IDs from host, then run a **complete transformer** on one U55C. Explain clearly that all neural math is FPGA logic.
3. Generate multiple tokens with **uncompressed INT16 cache**, show reference comparison and FPGA cycle counter.
4. Repeat with **packed INT4 cache**, identical checkpoint/prompt, show prior tokens read from HBM and the attention unit using reconstructed K/V.
5. Display actual **HBM allocated bytes and AXI traffic**, not only the ideal 4x code-width change; show whether packing eliminated padding.
6. Show achieved clocks, LUTs/DSPs/URAM/BRAM, prefill/decode/host latency and run-to-run spread.
7. If trained: run an offline finance post and show label with honest caveats; if Qwen runs, show its exact checkpoint and prompt.

## Suggested self-assessment rubric (100 points)

| Dimension | Points | Required evidence |
|:--|--:|:--|
| Model and platform specifications | 10 | Labs 0-3 retained, numerical contract, actual platform manifest |
| Quantized software oracle and reproducibility | 15 | exported operands, bit-exact model fixtures |
| SystemVerilog neural arithmetic | 20 | signed matvec + norm/RoPE/softmax/MLP testbenches |
| FPGA-resident compressed cache and GQA | 20 | actual HBM read-after-write and scale/payload verification |
| Complete FPGA decoder | 20 | end-to-end multi-token generated IDs in RTL / on card |
| Real U55C XRT deployment and benchmarking | 10 | installed-shell xclbin, repeated measured results |
| Report quality and risk accounting | 5 | reproducible logs, actual-vs-predicted distinctions |

**Naming accuracy:** a valid model simulation is not an on-card FPGA result; a standalone RTL compressor is not an FPGA LLM; an FPGA tiny transformer is not pretrained Qwen; and a synthetic financial classifier is not a live HFT trading strategy.


## Appendix A - Worked calculations and numerical details

### A.1 Why GQA cache is smaller than query count suggests

If `Hq=4`, `Hkv=2`, and `D=16`, there are 64 query features but only `2*16=32` K features and 32 V features per layer/token. Cached history stores K/V *per KV head*, not per Q head. Q head 0 and 1 access the same K/V head 0; Q head 2 and 3 access K/V head 1. A naive cache sized using `Hq` would waste memory and misrepresent your compression ratio.

### A.2 Exact INT4 storage example

Assume `q = [1,-1,-8,7]` (four signed 4-bit codes). Two's-complement nibble values are `[1,15,8,7]`, and low-nibble-first bytes are `0xF1` and `0x78`. With `scale_q = 32` (Q8.8 units = real 0.125), the reconstructed signed Q8.8 integers are `[32,-32,-256,224]`, representing `[0.125,-0.125,-1.0,0.875]`. Every implementation must generate the **same** bytes and values. Notice that `0xF1` is not two ordinary unsigned samples; the high nibble requires sign extension.

### A.3 A simple data-flow trace for three input tokens

```
Prompt tokens: [BOS, 'N', 'E'] (example only)
After BOS: each layer has K/V at position 0
After 'N': each layer has K/V at positions 0,1
After 'E': each layer has K/V at positions 0,1,2;
           last logits predict token G0
Host/FPGA sees G0; FPGA feeds G0 back as next input
After G0: caches grow to position 3; new logits predict G1
After G1: caches grow to position 4; new logits predict G2
```

This is why the first generated token is obtained *from the last prompt token's logits*. The cache stores the generated token's KV only when that token is next processed.

### A.4 Fixed-point pitfalls

- A 16x8 signed multiply may fit in 24 bits, but summing up to 128 products generally needs more width. Use a carefully derived accumulator width and intentionally sized intermediate signals.
- `>>>` in SystemVerilog is an arithmetic shift **only when the left operand is signed**. A concatenation or cast may remove signedness unexpectedly.
- Q8.8 32767 represents about 127.996; -32768 represents -128. A saturation checker must clamp *before* truncating to 16 bits.
- `logic [3:0]` is unsigned by default; for a signed INT4 nibble use `$signed(nibble)` or explicit sign extension, and test -8, -1, 0 and 7.
- Pretrained attention computations require more precision than random test vectors may reveal. Passing bit-exact integer tests proves RTL correctness against the chosen format but not that the quantized network retains useful language quality.
- A `scale_q` in Q8.8 is **not** IEEE-754 half-precision and must not be called FP16 in a storage report.

### A.5 FPGA weight streaming bound

If a large model requires `Bweight` bytes of weights per token and real **sustained** external-memory bandwidth is `BW`, a rough memory lower bound is `Bweight/BW` seconds/token. Even if DSP compute were instantaneous, an external memory interface cannot deliver those bytes faster than that idealized limit. Weight compression, coalesced reads, buffering, and fusing multiple operators become crucial. Report whether your FPGA ever meets the bandwidth-bound condition rather than assigning that limit automatically.

## Appendix B - Common failures and targeted fixes

| Symptom | Likely cause | Concrete fix |
|:--|:--|:--|
| Cached vs uncached float PyTorch logits differ | RoPE position offset, cache append order or wrong causal mask | Compare one-layer two-token Q/K/V/score dumps and positions |
| RTL result has correct magnitude but wrong sign | unsigned nibble or multiply/shift | `$signed` for input and product; compare -8, -1 tests |
| INT4 byte order differs from oracle | high/low nibbles swapped | Test `q=[1,-1,-8,7] -> F1 78` |
| Layer 1 gives nonsensical values, layer 0 is correct | wrong layer weight/cache base or residual register | Check layer offset and intermediate golden per sub-op |
| First generated token correct; second differs | off-by-one decode or stale cache | Trace exact processed token vs predicted token timestamps |
| `done` drops or produces duplicates | start/busy/done FSM protocol mismatch | Assert `done` one cycle and only after accepted start |
| All scales are zero | no maxabs minimum, Q conversion or writes missing | Set zero vector scale=1; inspect BRAM location |
| LLM works only without cache | cached K storage before/after RoPE inconsistent | Store *post-RoPE* K and track absolute position |
| Query heads mismatch | GQA mapping using Hq instead of Hkv | Assert pairs (0,1) and (2,3) map to correct KV head |
| Softmax produces overflow/NaN-like results | no max-subtraction or too-narrow exp sum | Stable softmax plus wide accumulator and bounded LUT |
| Negative Q8.8 values consistently biased | floor vs truncation rounding mismatch | Freeze one exact rule, update RTL and oracle in lockstep |
| Tiny model output labels all same | training/quantization degraded classifier or data imbalance | Inspect float baseline, per-label confusion, quantized oracle |
| BRAM synthesizes as large FF array | unsupported inferred RAM pattern / asynchronous read | Use synchronous read inferred BRAM or vendor IP |
| Vivado says timing pass with missing real clock | XDC not applied, unconstrained paths | Verify `report_clocks`, `report_exceptions`, path coverage |
| Board cannot fit tiny model despite enough aggregate bits | BRAM port/width fragmentation, LUT cost, parallel PEs | Serialize MAC, split memories into legal primitives, inspect utilization |
| FPGA slower after INT4 compression | decompressor overhead, cache already on chip | Measure scale reads, stalls, cycles/token, long-context behavior |
| FPGA faster than GPU on tiny model but claim is grandiose | mismatched model, batch, transfer or clocks | Provide configuration table and separate kernel/end-to-end timing |
| External DDR data appears corrupted | controller ready/valid, burst alignment/endian | Test deterministic streaming memory fill/read before neural code |
| "Full FPGA" trace still includes CPU matrix operations | hidden fallback in Python or hard processor firmware | Audit inference path and timestamp host/FPGA commands |

## Appendix C - Legacy internal module APIs (not the XRT / ap_ctrl_hs register map)

You may adapt widths if a spec revision is agreed by all contributors. The aim is to avoid ambiguous handoffs. These are **interfaces to implement**, not complete synthesizable modules.

```systemverilog
// A starting point: define in one package, not independently per module.
package llm_pkg;
    parameter int L       = 2;
    parameter int E       = 64;
    parameter int HQ      = 4;
    parameter int HKV     = 2;
    parameter int D       = 16;
    parameter int FF      = 128;
    parameter int TMAX    = 64;
    parameter int VOCAB   = 262;
    typedef logic signed [15:0] act_t;
    typedef logic signed [7:0]  wt_t;
    typedef logic signed [31:0] acc_t;
endpackage
```

Suggested control registers (addresses relative to the host-control peripheral; do **not** write literal addresses into all modules):

| Offset | Register | Direction | Definition |
|:--|:--|:--|:--|
| `0x00` | `CONTROL` | W | bit0 START, bit1 RESET_CACHE, bit2 CACHE_INT4 |
| `0x04` | `STATUS` | R | bit0 BUSY, bit1 DONE, bit2 ERROR |
| `0x08` | `TOKEN_IN` | W | next prompt token ID (0..261) |
| `0x0C` | `TOKEN_OUT` | R | last generated token ID |
| `0x10` | `PROMPT_LEN` | W/R | number of valid input tokens |
| `0x14` | `LAYER_IDX` | R | active layer debug counter |
| `0x18` | `POS_IDX` | R | current absolute position |
| `0x1C` | `CYCLE_LO` | R | performance cycles low word |
| `0x20` | `CYCLE_HI` | R | performance cycles high word |
| `0x24` | `KV_BYTES_LO` | R | physical cache traffic low word |
| `0x28` | `ERROR_CODE` | R | format/length/protocol diagnostics |

AXI4-Lite, UART packet memory map, or simple on-chip register files are all possible. Choose **one transport** and make host tests independent of it. Use `DONE` semantics that work when the host reads slowly: a one-cycle RTL `done` pulse can set a sticky `STATUS.DONE` bit cleared on command acknowledgment.

## Appendix D - Suggested evidence checklist

- [ ] Labs 0-3 unchanged; preserved Python and KV data (historical reference)
- [ ] Board inventory and model memory/DSP budget are complete
- [ ] `model_spec.json` and exact numeric/address protocol signed off
- [ ] Tiny PyTorch model: incremental decode equals full causal forward
- [ ] Trained or deliberately fixed tiny model checkpoint with documented provenance
- [ ] INT8 weight export + integer oracle + golden traces with hashes
- [ ] Self-checking SystemVerilog matrix engine + fixed-point primitives
- [ ] Cache payload/scale BRAM and verified low/high signed nibble handling
- [ ] Real GQA attention using **compressed-resident** cache
- [ ] Entire transformer block then all layers + logits + argmax on RTL
- [ ] Generated token fed back internally for subsequent decode
- [ ] Physical bitstream and board output trace, or simulation-only limitation disclosed
- [ ] FPGA uncompressed vs FPGA INT4 cache benchmark (same weights/model)
- [ ] CPU/GPU reference shown separately with fair configuration labels
- [ ] Financial classification demo and end-to-end latency broken down
- [ ] Final report explains why Qwen scaling requires additional RAM/bandwidth


# Appendix F. U55C troubleshooting and onboarding checklist

| Symptom | Most likely explanation | First diagnostic |
|:--|:--|:--|
| `v++` cannot find platform | Wrong shell version or `$PLATFORM` path | Admin-provided `.xpfm`; `platforminfo -p` |
| XO links but memory args wrong | XO metadata / connectivity `sp` mismatch | Kernel port/argument list; link summary |
| Host sees no U55C | Running on dev VM rather than allocated GPU/FPGA worker, or device not exposed | Approved scheduler job and XRT `xbutil examine` |
| HBM read works but cache corrupts | Beat alignment, byte strobes, endian or burst boundary error | 64-byte line write/read + one-token RTL trace |
| Memory allocation fails | Wrong HBM group id / bank not connected / capacity | `kernel.group_id(arg)` and platform map |
| Hardware emulation hangs | AXI response never completes, start/done handshake, outstanding count | transaction trace, add timeout and outstanding assertion |
| Synthesis huge DSP/LUT use | accidental replicated matrices/layers, wide unrolled loops | resource report at hierarchy level |
| Fmax drops sharply at integration | SLR crossings, high-fanout control, HBM port routing | timing critical paths and placer SLR utilization |
| Tiny model runs but quality is poor | INT8/Q8.8 accuracy / LUT approximations rather than memory architecture | float vs integer oracle, one-layer error histogram |
| Cache codec ratio looks 3.2x but HBM allocation equal | every entry is padded to 64 bytes | compute *physical* and *logical* ratios separately |
| First token matches, later ones do not | RoPE position or cache head/layer address | trace K/V for tokens 0..2 at every layer |
| Inference takes longer with INT4 | extra unpack/scale, poor access pattern or weight traffic dominates | stall and weight/cache traffic counters |

**When access arrives, request from the cluster staff** (in one concise message): exact user guide/job submission example, supported U55C `.xpfm` / `platforminfo`, Vitis/Vivado/XRT versions, compiler setup script, XRT device diagnostics permissions, example RTL-kernel XO+XRT project, approved emulation target, resource quota, log/scratch location and available monitoring options. Do not ask for root or try to reprogram the shell.

# Appendix G. Worked physical vs logical cache sizing

With TinyDecoder L=2,T=64,Hkv=2,D=16, INT4 has 8 packed bytes + 2 scale bytes = 10 bytes for each K or V vector. There are `2*64*2*2=512` vector entries, so **logical compressed = 512*10=5,120 bytes** and **logical uncompressed INT16 = 512*32=16,384 bytes**. Ratio is **3.2x**, *not* 6.4x. (The 6.4x arises from accidentally counting half as many vectors; this appendix intentionally makes you cross-check every dimension.) At 64-byte padded slots the actual allocated bytes become `512*64 = 32,768` for either mode, so physical storage ratio is **1.0x** until packed layout is implemented. At a packed-64-byte layout and fully utilized lines the INT4 allocation rounds `5,120` to `5,120` bytes (80 full lines) while INT16 is `16,384` bytes (256 lines), yielding 3.2x physical storage, excluding allocator/inter-bank metadata. Actual AXI traffic depends on memory bursts, cache read pattern and any read-modify-write.

**This is why memory layout belongs in the benchmark and not just in a diagram.** Every cache size reported must state whether it is logical tensor data, allocated HBM storage or traffic transferred.

# Appendix H. References and reproducibility protocol

- AMD [Alveo U55C data sheet DS978](https://docs.amd.com/r/en-US/ds978-u55c/Product-Details) - physical card resources, capacity, bandwidth.
- AMD [U55C platform UG1120](https://docs.amd.com/r/en-US/ug1120-alveo-platforms/U55C-Gen3x16-XDMA-base_3-Platform) - HBM and `platforminfo`.
- AMD [Data Center Acceleration Using Vitis UG1700](https://docs.amd.com/r/en-US/ug1700-vitis-accelerated-data-center/Creating-and-Packaging-RTL-Kernels) - RTL kernel conventions.
- AMD [`package_xo` UG1702](https://docs.amd.com/r/2026.1-English/ug1702-vitis-accelerated-reference/package_xo-Command) - Tcl syntax.
- AMD [XRT native C++ APIs](https://xilinx.github.io/XRT/master/html/xrt_native_apis.html) - buffer allocation and kernel launch.
- UIUC [HACC Getting Started](https://xilinx-center.csl.illinois.edu/xacc-cluster/xacc-user-guide/xacc-getting-started/) and [Cluster Status](https://xilinx-center.csl.illinois.edu/xacc-cluster/hacc-cluster-status/) - access constraints and current public node summary; contact staff for the actual enabled workflow.
- AMD [HBM Configuration and Use](https://docs.amd.com/r/en-US/ug1700-vitis-accelerated-data-center/HBM-Configuration-and-Use) - HBM memory system considerations.
- Prior completed Labs 0-3 and prior Full-FPGA Extension Starter remain the source for basic concepts and tiny Python model.

For each significant result archive: Git commit, checkpoint SHA-256, weight manifest hash, model config hash, Vitis shell/platform version, exact binary hash, kernel clock, HBM placement config, host executable revision, test IDs, raw measurements and emulator/simulation/hardware designation.

