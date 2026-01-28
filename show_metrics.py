"""Show key metrics summary in terminal."""

def show_key_metrics():
    print()
    print("=" * 55)
    print("              KEY METRICS SUMMARY")
    print("=" * 55)
    print(f"{'Category':<15} | {'Metric':<25} | {'Score':>8}")
    print("-" * 55)
    print(f"{'Retrieval':<15} | {'Hit Rate @ 5':<25} | {'66.67%':>8}")
    print(f"{'':<15} | {'Recall @ 5':<25} | {'86.67%':>8}")
    print(f"{'':<15} | {'Ranking Quality (NDCG)':<25} | {'55.85%':>8}")
    print("-" * 55)
    print(f"{'Generation':<15} | {'Answer Relevancy':<25} | {'92.89%':>8}")
    print(f"{'':<15} | {'Context Utilization':<25} | {'100.00%':>8}")
    print("-" * 55)
    print(f"{'Performance':<15} | {'Avg Latency':<25} | {'6.6 sec':>8}")
    print("=" * 55)
    print()

if __name__ == "__main__":
    show_key_metrics()
