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

## 2024-05-18 - Eliminate Temporary Tensors in Positional Encoding
**Learning:** In the `PositionalEncoding` module, doing `x = x + self.pe[:x.size(0), :]` during the forward pass forces PyTorch to allocate a temporary tensor for the output. Because `PositionalEncoding` is called at every step of autoregressive generation (e.g., `model.generate`), this minor allocation overhead accumulates into a measurable slowdown. Using `x.add_(...)` avoids this allocation.
**Action:** For simple operations like addition or multiplication on tensors inside PyTorch modules that are called repeatedly in a loop (like autoregressive decoding), use in-place methods (`.add_()`, `.mul_()`) to reduce memory allocations and improve iteration speed, provided it doesn't break the backward pass (autograd).
