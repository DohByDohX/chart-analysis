## 2024-05-14 - Optimize DataFrame array extraction in Pandas
**Learning:** For DataFrames with few columns (like OHLC), constructing intermediate multi-column slices (`df[['A', 'B']].iloc[:N]`) to extract a 2D numpy array causes a significant copy and object-creation overhead. Iterating over the columns and extracting the 1D underlying arrays directly via `.values` before slicing (`df['A'].values[:N]`) is about ~3x faster.
**Action:** When calculating metrics or iterating over a few fixed columns, extract 1D numpy arrays via `df[col].values` individually rather than performing 2D multi-column vectorization, and use `np.divide` with a `where` mask for safe division instead of masking boolean index extraction.

## 2024-05-15 - Caching Autoregressive Transformer Causal Masks
**Learning:** Generating the $O(N^2)$ causal mask in `generate_square_subsequent_mask` using `torch.triu(torch.ones(sz, sz))` inside every `forward()` pass causes unnecessary memory reallocation and moving the resulting tensor from CPU to GPU via `.to(device)` adds cross-device synchronization latency, which is a noticeable bottleneck during tight training loops.
**Action:** Use the highly optimized `torch.nn.Transformer.generate_square_subsequent_mask(sz, device=...)` method that creates the tensor directly on the target device, and cache the resulting mask instances at the class level (keyed by `(size, device)`) to completely avoid recomputation and migration overhead on subsequent calls.

## 2024-05-16 - Safe vs Unsafe In-Place PyTorch Operations
**Learning:** In PyTorch modules, using in-place operations (e.g., `.div_()`) on newly created temporary tensors (like loaded image arrays) eliminates unnecessary allocations and improves performance safely. However, mutating input tensors passed to `forward` methods (e.g., `x.add_()` on an argument `x`) is an unsafe anti-pattern that can break the autograd graph if the unmodified tensor is needed for backpropagation elsewhere.
**Action:** Always favor in-place operations (`.div_()`, `.mul_()`, etc) for newly created, intermediate tensors within a data pipeline or forward pass, but strictly use out-of-place operations (`x = x + y`) for any tensors passed as inputs to a module or function.

## 2024-05-18 - Dictionary Array Indexing Over DataFrame Initialization
**Learning:** Extracting a single element (e.g., the last element of a time series) directly from a dictionary of lists (e.g. `dict_data['Close'][-1]`) is significantly faster than wrapping the entire dictionary in a `pandas.DataFrame` just to use `.iloc[-1]`, due to DataFrame instantiation overhead.
**Action:** When accessing a single row or index from array-like or list-based dictionaries, use direct dictionary and list indexing instead of full DataFrame conversion.
