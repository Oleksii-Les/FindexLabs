# Lab 05 — why threads did not help, why processes did, and what that cost


## 1. What kind of "slow" is indexing?

Indexing is **CPU-bound**: tokenizing (a regex, `casefold`, Unicode normalization) and counting
(`Counter`, `dict`, `array`) are pure-Python bytecode. The serial build takes 19.12 s (min–max 18.59–23.94 s) and its CPU time is 19.02 s, i.e. 99.5 % of the wall clock was the processor working, not waiting. Reading the files is a small part:
23 s of the 30 s serial build (warm file cache). *My prediction before measuring:* ~30 s.

## 2. What the GIL is

CPython frees an object when its reference count reaches zero. Updating that count from two
threads at once would corrupt it, so the interpreter has one lock, the **GIL**: a thread must hold it
to execute Python bytecode. Only one thread does at any moment; the holder is asked to give it up
every `sys.getswitchinterval()` = 5 ms, and a thread gives it up voluntarily while it waits for
I/O (`read`, `recv`, `sleep`) or inside a C function that releases it.

It protects the **interpreter's internals**, not **your invariants**: `counter += 1` is several
bytecodes (`dis` shows load, add, store), and a switch between them loses an update (section 6).

## 3. Why threads did not scale

| Executor | Workers | Wall, s | CPU, s | CPU / wall | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| serial |	1 |	19.12 |	19.02 |	0.99 ≈ 1.0 |	1.00x |
| threads |	2 |	18.99 |	18.88 |	0.99 |	1.01x |
| threads	| 4 |	19.16 |	19.00	| 0.99 |	1.00x |
| threads |	8 |	19.55 |	19.70 |	1.01 ≈ 1.0 |	0.98x |

Wall time did not drop and CPU time did not rise above wall time: at every instant **one** thread
ran bytecode, the other seven waited for the GIL. Eight threads add only switching and
contention cost (2.2 % slower than serial). `experiments/gil_demos.py` experiment 1 shows
the same in six lines.

## 4. Why processes scaled, and the bill

Each worker is a separate interpreter with its **own GIL**, so tokenizing really runs in parallel:

| Executor | Workers | Wall, s | CPU, s | Peak RSS of parent, MiB | Merge, s | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| serial | 1 | 19.12 | 19.02 | 255 | 0.17 | 1.00x | – |
| processes | 1 | 24.53 | 22.38 | 497 | 0.26 | 0.78x | 78 % |
| processes | 2 | 14.43 | 25.00 | 438 | 0.36 | 1.32x | 66 % |
| processes | 4 | 10.14 | 32.73 | 421 | 0.45 | 1.88x | 47 % |
| processes | 8 | 9.07 | 44.83 | 460 | 0.56 | **2.11x** | 26 % |

It is paid for in three currencies:

* **CPU time vs wall time.** 44.83 s of CPU for 9.07 s of wall clock at 8 workers (CPU / wall = 4.9, so on average
  about five logical CPUs were busy). Serial needed 19.02 s of CPU: the parallel run burns **25.8 s of CPU more**
  (2.4x the total work) to finish 2.1x sooner. The extra seconds are start-up of 8 interpreters
  (`spawn` re-imports the modules in every worker, about 0.28 s each), pickling arguments and results,
  the parent's unpickling and merge, and hyper-threads running slower than a full core.
  With **one** worker the process pool is even 28 % *slower* than serial (24.53 s vs 19.12 s): all the costs
  and no parallelism.
* **Memory.** The parent peaked at 460 MiB (serial: 255 MiB) because it holds every partial
  index before merging; each worker process also has its own interpreter and its own chunk
  (visible in Task Manager, not in the parent's number).
* **Pickle tax.** Only `Path`s travel in (2.9 KiB for the whole corpus). Partial indexes come back as `array('I')`
  buffers. `experiments/pickle_tax.py` (150 documents, 91.5 MiB of text, 8 chunks):

  | | build | pickle | unpickle | merge | payload | tax |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | no positions | 27.48 s | 6.06 s | 2.39 s | 0.51 s | 66.9 MiB (0.73x the text) | 30.7 % |
  | with positions | 31.73 s | 7.32 s | 2.82 s | 0.77 s | 136.6 MiB (1.49x the text) | 31.9 % |

  Roughly **a third of the build time** is spent turning results into bytes and back. Sending the *texts* in
  would be much cheaper (89.4 MiB: dump 0.19 s, load 0.08 s, about 30 times less than pickling the partial indexes), but then the parent has to
  read and hold the whole corpus first, and the tokenizing still happens in the workers. I send paths in and
  compact partial indexes out; the return path is where the tax is paid.

## 5. Where Amdahl's law shows up

Amdahl: `speedup(N) = 1 / (s + (1 - s) / N)`, where `s` is the part that cannot be parallelized.

* `parbench report` counts only planning and merge as sequential: 0.58 s of 19.12 s, **s = 3.0 %**, ceiling
  `1/s` = 33x. That predicts 1.94x / 3.67x / 6.61x for 2 / 4 / 8 workers; I measured **1.32x / 1.88x / 2.11x**.
  Far too optimistic, so `s` must be bigger than that.
* The real sequential part also includes unpickling the results in the parent (≈ 1.66 s when scaled
  from `pickle_tax.py` to the 19 s build) and pool start-up. With it `s ≈ 11.7 %`, ceiling ≈ 8.5x, predictions
  1.79x / 2.96x / 4.39x. Closer, still above the measurement.
* The remaining gap: **4 physical cores, 8 logical CPUs**. Two hyper-threads share one core's execution
  units, so going from 4 to 8 workers gave only 1.88x → 2.11x. Also memory bandwidth, uneven chunks
  (books differ a lot in size) and worker-side pickling.

The merge time itself grows with the number of workers (0.17 s serial → 0.56 s with 8 process workers): more partial indexes
to concatenate, and it is exactly the part that no extra worker can speed up.
It could be shrunk (fewer, bigger chunks; a tree merge), which moves the ceiling, not the law.

## 6. What changed on the free-threaded build (3.13t)

`sys._is_gil_enabled()` → `False` (Python 3.13.16t, run with `PYTHON_GIL=0`). 
3.13.16 experimental free-threading build (main, Oct  9 2026, 15:23:08) [MSC v.1944 64 bit (AMD64)] False

| Interpreter | Executor | Workers | Wall, s | CPU, s | Peak RSS, MiB | Merge, s | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3.13t [GIL OFF] | serial | 1 | 19.10 | 19.02 | 274 | 0.24 | 1.00x |
| 3.13t [GIL OFF] | threads | 1 | 19.54 | 19.53 | 323 | 0.42 | 0.98x |
| 3.13t [GIL OFF] | threads | 2 | 11.86 | 21.44 | 339 | 0.52 | 1.61x |
| 3.13t [GIL OFF] | threads | 4 | 8.96 | 27.23 | 439 | 0.66 | **2.13x** |
| 3.13t [GIL OFF] | threads | 8 | 9.04 | 36.88 | 488 | 0.80 | 2.11x |

The same threads that gave nothing under the GIL now scale: 1.61x with 2 threads, 2.13x with 4, and then the same
wall as the 8-process run (9.04 s vs 9.07 s) while needing less CPU (36.9 s vs 44.8 s) and no pickling.
At 2 and 4 workers the threads are 18 % and 12 % faster than processes. The single-thread overhead of the free-threaded
build is invisible here (serial 19.10 s vs 19.12 s on 3.12), but the merge is 41 % slower (0.24 s vs 0.17 s),
and it grows to 0.80 s with 8 threads. The plateau at ~2.1x is the same as for processes: same 4 physical cores,
same sequential part.

**The race that was hiding.** Eight threads counting tokens of 40 different documents into one shared
structure (`experiments/race_demo.py`, correct total **4,774,789** tokens):

| Variant | 3.12 (GIL) | 3.13t (no GIL) |
| --- | --- | --- |
| A shared `Counter.update(tokens)` | OK | OK |
| B shared dict, `d[t] = d.get(t, 0) + 1` | **LOST UPDATES** (up to 15,644 missing) | **LOST UPDATES** (up to 8,804 missing) |
| C shared dict + one `Lock` per document | OK | OK |
| D one `Counter` per thread, merged after | OK | OK |

Variant A is safe under the GIL only because `Counter.update` runs its inner loop in C without
releasing the lock; without the GIL its read-then-write can lose updates (fill in the 3.13t column). Variant B is a genuine
read-modify-write in Python and is racy even on the GIL build, as the table shows. The production code has **no shared
mutable state**: `build_partial` writes only to its own arrays and `merge` runs in one thread. That is
variant D, and it is correct on every build. A `Lock` (C) is also correct but serializes the work.

Note: in the small `gil_demos.py` test a plain `x += 1` lost **0** updates on my machine, and so did
*read; function call; write* and the `Lock` variant. Absence of a visible loss proves nothing: the window is tiny and
depends on the interpreter version and timing; the race in variant B is the real, reproducible one.

## 7. The opposite experiment: I/O-bound work

`experiments/io_bound_demo.py --mode urls`, 30 downloads:

| Threads | 1 | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Time, s | 38.2 | 20.7 | 15.3 | 11.3 | 10.5 |
| Speedup | 1.0x | 1.8x | 2.5x | 3.4x | 3.6x |

While a thread waits for the network it releases the GIL, so the waits overlap and threads scale, under the GIL, with no
processes. The curve flattens at 3.6x because bandwidth and the server become the limit (with simulated
`sleep` the speedup is close to the number of threads). At 32 threads the server cut a connection
(`IncompleteRead`) and the demo, which does not retry, stopped. With a warm file cache `--mode files` shows no gain:
there is nothing to wait for.

## 8. Decision

`--executor` defaults to **auto**: processes for corpora of at least 2 MB, serial below that
(`MIN_PARALLEL_BYTES`), because on my machine processes win by **2.11x** at 8 workers (1.88x at 4) and lose on tiny inputs, where
start-up dominates (with 1 worker they are 28 % slower than serial). Eight workers are barely better than four
(hyper-threads), so 4 would be a reasonable choice too. Free-threaded threads are as fast and use less CPU, but
3.13t is not the interpreter most people have, so processes stay the default.
