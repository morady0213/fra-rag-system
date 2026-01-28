# FRA RAG System Evaluation Report

## Overview
- **Dataset**: fra_starter
- **Size**: 15 items
- **Date**: 2026-01-18T12:23:02.121520
- **Overall Score**: 52.93%

---

## Retrieval Metrics

| Metric | Score |
|--------|-------|
| hit_rate | 0.6667 |
| mrr | 0.5444 |
| precision | 0.2667 |
| recall | 0.8667 |
| ndcg | 0.5585 |
| map | 1.0842 |

---

## Generation Metrics

| Metric | Score |
|--------|-------|
| faithfulness | 0.4033 |
| answer_relevancy | 0.9289 |
| context_utilization | 1.0000 |
| answer_correctness | 0.3226 |

---

## Arabic-Specific Metrics

| Metric | Score |
|--------|-------|
| citation_accuracy | 0.5000 |
| anti_hallucination | 0.6933 |
| number_accuracy | 0.9000 |
| article_accuracy | 0.0000 |

---

## Performance

| Metric | Value |
|--------|-------|
| Average Latency | 6648 ms |
| P95 Latency | 15843 ms |


---

## Breakdown by Question Type

| Type | Avg Score |
|------|-----------|
| factual | 2.1379 |
| comparison | 1.1673 |
| conditional | 1.4216 |
| edge_case | 1.0194 |
| procedural | 1.1212 |
| aggregation | 0.7576 |
