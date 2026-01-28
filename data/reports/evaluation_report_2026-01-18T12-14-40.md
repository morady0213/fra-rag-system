# FRA RAG System Evaluation Report

## Overview
- **Dataset**: fra_starter
- **Size**: 15 items
- **Date**: 2026-01-18T12:14:40.368877
- **Overall Score**: 50.44%

---

## Retrieval Metrics

| Metric | Score |
|--------|-------|
| hit_rate | 0.4667 |
| mrr | 0.3889 |
| precision | 0.1867 |
| recall | 0.8810 |
| ndcg | 0.3947 |
| map | 1.1286 |

---

## Generation Metrics

| Metric | Score |
|--------|-------|
| faithfulness | 0.3865 |
| answer_relevancy | 0.9115 |
| context_utilization | 1.0000 |
| answer_correctness | 0.3207 |

---

## Arabic-Specific Metrics

| Metric | Score |
|--------|-------|
| citation_accuracy | 0.5000 |
| anti_hallucination | 0.7067 |
| number_accuracy | 0.9000 |
| article_accuracy | 0.0000 |

---

## Performance

| Metric | Value |
|--------|-------|
| Average Latency | 7415 ms |
| P95 Latency | 12590 ms |


---

## Breakdown by Question Type

| Type | Avg Score |
|------|-----------|
| factual | 2.1417 |
| comparison | 1.1354 |
| conditional | 1.4071 |
| edge_case | 0.9125 |
| procedural | 1.1628 |
| aggregation | 0.8056 |
