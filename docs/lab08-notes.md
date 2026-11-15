# Lab 08 — notes for the defence

## The measurement I want to show
Scoring one query of three common words, 17 006 documents: **28.7 ms with the loop of lab 3, 0.49 ms with NumPy** (58x);
the same change under load: **28 → 210 requests/s, p95 1 015 → 133 ms**. Command: `uv run python experiments/numpy_table.py index.bin`.

## The bug that taught me something
*"The rare term is slower with NumPy."* My first table had two rows below 1x (a term in 3 documents: 0.5x, in 1 document:
0.4x). It was not a bug in the arithmetic (the 69 equivalence tests passed), it was the **fixed cost of a vectorised call**:
three arrays the size of the corpus (`np.zeros(17 006)` ×3) and an `np.fromiter` cost ~30 µs, while the old loop did the whole
job for 3 postings in 17 µs. Fix: `VectorIndex.worth_it` — below 128 postings per query the engine keeps the old loop. What
it explained: NumPy moves the cost from *per element* to *per call*, so the benefit is proportional to the length of the
list, and I should have predicted the table before measuring it.

A second one, in the semantic part. I wrote a test "a text that says *ship* (never *vessel*) is closer to the query
*vessel* than any text about music" and it **failed** with 6 LSA dimensions: some cosines were **negative** (−0.15). That is
not a defect: two words that never occur together but occur in the same kind of text get one shared component ("sea") and
one *opposing* component ("ship, not vessel"). Keeping more dimensions than there are topics keeps that contrast. With 3
dimensions on a 2-topic toy corpus the cosine is 1.0. Lesson: the number of dimensions is a modelling choice with a visible
cost, and a cosine below zero is not an error.

## Hypotheses against what happened (docs/profiling.md)
| Hypothesis written before | Result |
| --- | --- |
| per-posting `score()` + `idf()` is interpreter overhead, a vector expression removes it | yes: 51 % of the query; scoring went 58x faster |
| `Posting` objects are an allocation cost, arrays need none | yes (the NumPy path never builds one) |
| the heap over `SearchResult` is O(n log k) but constant-heavy; `argpartition` is O(n) | yes; the subtle part was **ties**: `heapq.nlargest` orders by (score, doc id) both descending, so the boundary needs an explicit rule (tested with Hypothesis against a full sort) |
| in `index`, per-token Python loop is the cost; `findall` removes half of the tokenizer | partly: 5.5 s → 4.9 s only (the per-token dict/list work stays) |
| after the fix, something else is the bottleneck | yes: Boolean `merge_or` is 77 % of the new query |

## Reflection, short answers
* **Why is a loop over Python objects expensive and a NumPy loop is not?** Each element of a Python loop is a bytecode
  round trip: look up names, call a function, box a float, allocate an object. A ufunc runs one C loop over contiguous
  `int32`/`float64` memory, with no per-element object. The arithmetic is the same; the overhead per element differs ~50x.
* **Why `argpartition` and not `sort`?** For the top 10 of 15 000 scores a full sort is O(n log n); `argpartition` finds
  the k-th value in O(n) and only the k winners are sorted. Ties at the boundary have to be cut by an explicit rule.
* **What does broadcasting do in BM25?** `norm[docs]` is an array of the same length as `tf`; the scalars (`idf`, `k1`)
  are stretched across it without being copied. The expression has no loop in Python.
* **Why unit vectors?** Cosine similarity = dot product when both vectors have length 1, so the similarity of the query
  to every fragment is one matrix–vector product (BLAS), and the matrix can be a plain memory-mapped file.
* **Why RRF and not a weighted sum of scores?** BM25 scores are unbounded and depend on the query, cosines live in [−1, 1];
  to add them I would need a calibration that changes with every query. RRF uses ranks only: `Σ 1/(60 + rank)`.
* **Why a document's score is its best fragment?** A book is thousands of topics; its average vector describes none of
  them. The best fragment keeps the one passage that matches, and gives the snippet to show.
* **Python or native time?** Scalene says the scorer lines are almost all Python time; the tokenizer regex and
  `array.frombytes` are native. That is why NumPy helps the first and cannot help the second.
* **Why is the semantic result so weak here?** LSA from 17 000 mixed technical documents is a coarse model, the labels are
  strict, and 5 paraphrases are very few. I did not run the neural model (no network in the build environment); I would
  not claim anything about it before running it.
