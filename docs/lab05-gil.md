# Lab 05 — why threads did not help, why processes did, and what that cost

> **Draft.** The argument is laid out and every mechanism below was checked against the code.
> Replace each `TODO` with *your* measurements (from `bench/*.json`, `experiments/*`) and
> rewrite the prose in your own words before you submit: this page is what you will be asked
> to explain at the whiteboard.

## 1. What kind of "slow" is indexing?

Indexing is **CPU-bound**: tokenizing (a regex, `casefold`, Unicode normalization) and counting
(`Counter`, `dict`, `array`) are pure-Python bytecode. Reading the files is a small part:
`TODO` s of the `TODO` s serial build (warm file cache). *My prediction before measuring:* `TODO`.

## 2. What the GIL is

CPython frees an object when its reference count reaches zero. Updating that count from two
threads at once would corrupt it, so the interpreter has one lock, the **GIL**: a thread must hold it
to execute Python bytecode. Only one thread does at any moment; the holder is asked to give it up
every `sys.getswitchinterval()` = 5 ms, and a thread gives it up voluntarily while it waits for
I/O (`read`, `recv`, `sleep`) or inside a C function that releases it.

It protects the **interpreter's internals**, not **your invariants**: `counter += 1` is several
bytecodes (`dis` shows load, add, store), and a switch between them loses an update (section 6).

## 3. Why threads did not scale

| Executor | Workers | Wall, s | CPU, s | CPU / wall |
| --- | ---: | ---: | ---: | ---: |
| serial | 1 | `TODO` | `TODO` | `TODO` ≈ 1.0 |
| threads | 8 | `TODO` | `TODO` | `TODO` ≈ 1.0 |

Wall time did not drop and CPU time did not rise above wall time: at every instant **one** thread
ran bytecode, the other seven waited for the GIL. Eight threads add only switching and
contention cost (`TODO` % slower than serial). `experiments/gil_demos.py` experiment 1 shows
the same in six lines.

## 4. Why processes scaled, and the bill

Each worker is a separate interpreter with its **own GIL**, so tokenizing really runs in parallel:
speedup `TODO`x with `TODO` workers. It is paid for in three currencies:

* **CPU time vs wall time.** `TODO` s of CPU for `TODO` s of wall clock. The extra CPU seconds are
  start-up of `TODO` interpreters (`spawn` re-imports the modules in every worker), pickling
  arguments and results, and the parent's merge.
* **Memory.** The parent peaked at `TODO` MiB (serial: `TODO` MiB) because it holds every partial
  index before merging; each worker process also has its own interpreter and its own chunk
  (visible in Task Manager, not in the parent's number).
* **Pickle tax.** Only `Path`s travel in (a few KiB). Partial indexes come back as `array('I')`
  buffers: `TODO`x the size of the text without positions, `TODO`x with them, costing
  `TODO` % of build time to pickle and unpickle (`experiments/pickle_tax.py`). Sending
  documents instead of paths would also make the parent read and hold the whole corpus first.

## 5. Where Amdahl's law shows up

The planning and the merge run in the parent, sequentially: `TODO` s of a `TODO` s serial build
(`s` = `TODO`). Amdahl: `speedup(N) = 1 / (s + (1 - s) / N)`, so with infinite workers the ceiling
is `1/s` = `TODO`x. The measured speedup at `TODO` workers is `TODO`x, below the ceiling because
pool start-up, pickling and load imbalance are sequential or wasted time too, and because
`TODO` logical CPUs are `TODO` physical cores (hyper-threads share an execution unit).
The merge could be shrunk (fewer, bigger chunks; a tree merge), which moves the ceiling, not the law.

## 6. What changed on the free-threaded build (3.13t)

`sys._is_gil_enabled()` → `False`. Threads rows: wall `TODO` s at `TODO` workers, speedup `TODO`x
(single-thread overhead of the free-threaded build: `TODO` %).

**The race that was hiding.** Eight threads counting tokens of different documents into one shared
structure (`experiments/race_demo.py`, correct total `TODO`):

| Variant | 3.12 (GIL) | 3.13t (no GIL) |
| --- | --- | --- |
| A shared `Counter.update(tokens)` | `TODO` | `TODO` |
| B shared dict, `d[t] = d.get(t, 0) + 1` | `TODO` | `TODO` |
| C shared dict + one `Lock` per document | `TODO` | `TODO` |
| D one `Counter` per thread, merged after | `TODO` | `TODO` |

Variant A is safe under the GIL only because `Counter.update` runs its inner loop in C without
releasing the lock; without the GIL its read-then-write loses updates. Variant B is a genuine
read-modify-write in Python and is racy on both builds. The production code has **no shared
mutable state**: `build_partial` writes only to its own arrays and `merge` runs in one thread. That is
variant D, and it is correct on every build. A `Lock` (C) is also correct but serializes the work.

## 7. The opposite experiment: I/O-bound work

`experiments/io_bound_demo.py --mode urls` (or `sleep`): `TODO` tasks, 1 thread `TODO` s, 16 threads
`TODO` s (`TODO`x). While a thread waits, it releases the GIL, so the waits overlap. With a warm file
cache `--mode files` shows no gain: there is nothing to wait for.

## 8. Decision

`--executor` defaults to **auto**: processes for corpora of at least 2 MB, serial below that
(`MIN_PARALLEL_BYTES`), because processes win on my machine by `TODO`x and lose on tiny inputs where
start-up dominates. *(If processes did not win on your machine, say so here and set
`DEFAULT_PARALLEL_EXECUTOR` in `parallel.py` accordingly.)*
