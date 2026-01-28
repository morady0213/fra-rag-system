# FRA RAG System Evaluation Report

## Overview
- **Dataset**: fra_starter
- **Size**: 15 items
- **Date**: 2026-01-25T11:01:22.823361
- **Overall Score**: 61.25%

---

## Retrieval Metrics

| Metric | Score |
|--------|-------|
| hit_rate | 0.6667 |
| mrr | 0.5444 |
| precision | 0.2667 |
| recall | 0.8667 |
| ndcg | 0.5585 |
| map | 0.8333 |

---

## Generation Metrics

| Metric | Score |
|--------|-------|
| faithfulness | 0.7667 |
| answer_relevancy | 0.8600 |
| context_utilization | 0.9067 |
| answer_correctness | 0.3252 |

---

## Arabic-Specific Metrics

| Metric | Score |
|--------|-------|
| citation_accuracy | 0.5000 |
| anti_hallucination | 0.8467 |
| number_accuracy | 0.9000 |
| article_accuracy | 0.0000 |

---

## Performance

| Metric | Value |
|--------|-------|
| Average Latency | 2792 ms |
| P95 Latency | 5014 ms |


---

## Breakdown by Question Type

| Type | Avg Score |
|------|-----------|
| factual | 2.2103 |
| comparison | 1.1561 |
| conditional | 1.5574 |
| edge_case | 1.2250 |
| procedural | 1.3048 |
| aggregation | 0.7048 |
