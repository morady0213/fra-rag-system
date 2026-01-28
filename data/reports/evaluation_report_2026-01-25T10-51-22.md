# FRA RAG System Evaluation Report

## Overview
- **Dataset**: fra_starter
- **Size**: 15 items
- **Date**: 2026-01-25T10:51:22.234307
- **Overall Score**: 57.55%

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
| faithfulness | 0.6351 |
| answer_relevancy | 0.7978 |
| context_utilization | 0.8933 |
| answer_correctness | 0.3325 |

---

## Arabic-Specific Metrics

| Metric | Score |
|--------|-------|
| citation_accuracy | 0.5000 |
| anti_hallucination | 0.8667 |
| number_accuracy | 0.9000 |
| article_accuracy | 0.0000 |

---

## Performance

| Metric | Value |
|--------|-------|
| Average Latency | 3483 ms |
| P95 Latency | 5301 ms |


---

## Breakdown by Question Type

| Type | Avg Score |
|------|-----------|
| factual | 2.1920 |
| comparison | 1.1680 |
| conditional | 1.4947 |
| edge_case | 0.8667 |
| procedural | 1.3063 |
| aggregation | 0.7048 |
