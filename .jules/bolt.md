## 2024-05-14 - Optimize DataFrame array extraction in Pandas
**Learning:** For DataFrames with few columns (like OHLC), constructing intermediate multi-column slices (`df[['A', 'B']].iloc[:N]`) to extract a 2D numpy array causes a significant copy and object-creation overhead. Iterating over the columns and extracting the 1D underlying arrays directly via `.values` before slicing (`df['A'].values[:N]`) is about ~3x faster.
**Action:** When calculating metrics or iterating over a few fixed columns, extract 1D numpy arrays via `df[col].values` individually rather than performing 2D multi-column vectorization, and use `np.divide` with a `where` mask for safe division instead of masking boolean index extraction.

## 2024-05-15 - Caching Autoregressive Transformer Causal Masks
**Learning:** Generating the $O(N^2)$ causal mask in `generate_square_subsequent_mask` using `torch.triu(torch.ones(sz, sz))` inside every `forward()` pass causes unnecessary memory reallocation and moving the resulting tensor from CPU to GPU via `.to(device)` adds cross-device synchronization latency, which is a noticeable bottleneck during tight training loops.
**Action:** Use the highly optimized `torch.nn.Transformer.generate_square_subsequent_mask(sz, device=...)` method that creates the tensor directly on the target device, and cache the resulting mask instances at the class level (keyed by `(size, device)`) to completely avoid recomputation and migration overhead on subsequent calls.

## 2024-05-16 - Avoid eager file existence checks in PyTorch Datasets
**Learning:** Checking `.exists()` on thousands of files during `Dataset.__init__` adds $O(N)$ system calls, causing massive and completely unnecessary startup latency when running training loops, specifically because Datasets should load data lazily.
**Action:** When initializing PyTorch Datasets, do string interpolation/path building to construct lists of file paths, but DO NOT verify they exist upfront. Rely on EAFP (Easier to Ask for Forgiveness than Permission) and allow missing files to throw standard `FileNotFoundError` during `__getitem__`.

## 2024-05-17 - Optimize tensor normalization using in-place operations
**Learning:** Performing tensor normalization like `image = image / 255.0` creates a temporary tensor and forces a memory allocation. In tight loops like `Dataset.__getitem__`, this allocation overhead is surprisingly costly. Using the in-place counterpart `image.div_(255.0)` eliminates the temporary allocation and speeds up the division step by over 2.5x per sample.
**Action:** Use in-place tensor operations like `.div_()`, `.mul_()`, `.add_()`, etc., instead of out-of-place arithmetic operators for large array manipulation in data-processing loops to save memory allocations and CPU overhead.

## 2024-05-18 - Avoiding costly tensor transpositions in Transformers
**Learning:** PyTorch's `TransformerDecoderLayer` defaults to `batch_first=False`, which expects `(SeqLen, Batch, Dim)` inputs. In a model where earlier stages (like `VisionEncoder`) naturally output `(Batch, SeqLen, Dim)`, adapting the inputs by calling `.transpose(0, 1)` creates non-contiguous tensors. This causes implicit copies during subsequent operations or cache misses, which adds up to noticeable overhead when generating tokens autoregressively.
**Action:** Always instantiate `TransformerDecoderLayer` and custom `PositionalEncoding` modules with `batch_first=True` when dealing with `(Batch, SeqLen, Dim)` data, avoiding unnecessary tensor dimension permutations during the forward pass.
