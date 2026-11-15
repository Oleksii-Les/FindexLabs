# Video, 2–3 minutes (script)

Record the screen (OBS, or Win+G on Windows) with a microphone; upload as an unlisted YouTube video or drop an `.mp4` into the
GitHub release; put the link at the top of the README. About 330 spoken words.

| Time | On screen | Say |
| --- | --- | --- |
| 0:00 | the live page | "findex is a search engine I built over eight labs. This is the deployed service; it searches N documents." |
| 0:15 | type `logging handler`, Enter | "Keyword search: BM25 ranking, snippets with the matches highlighted, scores, pages." |
| 0:35 | add `AND NOT` and a quoted phrase | "Boolean operators and exact phrases come from the inverted index with token positions." |
| 0:55 | switch *Search by* to Hybrid, type a question | "Hybrid: BM25 and a meaning-based ranking merged with Reciprocal Rank Fusion. It finds documents that use other words." |
| 1:20 | `docs/profiling.md` flame graph, then the speed-up table | "The slow part was one Python call per posting. I measured first, then replaced the loop with NumPy: 58 times faster for scoring, the service handles 7.5 times more requests." |
| 1:55 | the precision table | "Honest result: on paraphrases the offline model does not help yet; on keyword queries it is on par. The neural model is wired in but I could not run it in the sandbox." |
| 2:20 | `/api/docs` and the GitHub Actions page | "Everything is behind a JSON API with typed validation, tested, 490 tests, with a benchmark job in CI." |
| 2:40 | README top | "Code, measurements and limits are in the repository." |
