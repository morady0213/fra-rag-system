"""
FRA RAG System Evaluation CLI.

Run evaluation on golden datasets and generate reports.

Usage:
    python evaluate.py --create-starter    # Create starter dataset
    python evaluate.py --run               # Run evaluation
    python evaluate.py --quick             # Quick test with sample questions
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from main import FRARAGSystem
from evaluation.golden_dataset import GoldenDataset, DatasetGenerator
from evaluation.pipeline import EvaluationPipeline, create_evaluation_pipeline
from evaluation.metrics import RetrievalEvaluator, GenerationEvaluator, ArabicEvaluator

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
)

# Paths
DATASETS_DIR = PROJECT_ROOT / "data" / "evaluation"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"


def create_starter_dataset():
    """Create and save the starter golden dataset."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    generator = DatasetGenerator()
    dataset = generator.create_starter_dataset()
    
    # Add more comprehensive questions
    additional_items = [
        # Factual - Easy
        generator.create_manual_item(
            question="ما هي متطلبات إصدار السندات الخضراء؟",
            answer="تشمل متطلبات إصدار السندات الخضراء تقديم خطة استخدام العوائد، تقرير التأثير البيئي، وشهادة التحقق من طرف ثالث.",
            question_type="factual",
            entity_type="توريق",
            topic="سندات",
            difficulty="medium",
            relevant_docs=["مستندات-اصدار-سندات-خضراء.docx"],
        ),
        # Factual - Medium
        generator.create_manual_item(
            question="ما هي المستندات المطلوبة لتسجيل شركة تأمين أشخاص؟",
            answer="تشمل المستندات: عقد التأسيس، السجل التجاري، قائمة المساهمين، والموقف المالي.",
            question_type="factual",
            entity_type="تأمين",
            topic="تسجيل",
            difficulty="medium",
            relevant_docs=["نموذج-تسجيل-شركات-تأمين-اشخاص.docx"],
        ),
        # Procedural
        generator.create_manual_item(
            question="ما هي إجراءات نقل فرع لشركة سمسرة؟",
            answer="يجب تقديم طلب النقل مع تحديد الموقع الجديد وموافقة مجلس الإدارة وإخطار العملاء.",
            question_type="procedural",
            entity_type="سمسرة",
            topic="نقل",
            difficulty="medium",
            relevant_docs=["نموذج-نقل-فرع-لشركة-سمسرة.docx"],
        ),
        # Procedural - Documents
        generator.create_manual_item(
            question="ما هي خطوات قيد فرع لشركة تمويل متناهي الصغر؟",
            answer="تشمل الخطوات: تقديم الطلب، إرفاق المستندات المطلوبة، الفحص الأمني، وصدور قرار القيد.",
            question_type="procedural",
            entity_type="تمويل_متناهي_الصغر",
            topic="ترخيص",
            difficulty="medium",
            requires_multi_hop=True,
            relevant_docs=["نموذج-قيد-فروع-متناهى-الصغر.docx"],
        ),
        # Comparison - Hard
        generator.create_manual_item(
            question="ما الفرق بين إجراءات غلق فرع شركة تمويل استهلاكي وشركة تأجير تمويلي؟",
            answer="تتشابه الإجراءات في المتطلبات الأساسية مع اختلافات في المستندات المطلوبة حسب طبيعة النشاط.",
            question_type="comparison",
            requires_multi_hop=True,
            difficulty="hard",
            relevant_docs=["نموذج-غلق-فرع-لشركة-تمويل-استهلاكى.docx", "نموذج-غلق-فرع-لشركة-التاجير-التمويلى.docx"],
        ),
        # Conditional
        generator.create_manual_item(
            question="هل يمكن تغيير مدير الفرع بدون موافقة الهيئة؟",
            answer="لا، يجب الحصول على موافقة الهيئة العامة للرقابة المالية قبل تغيير مدير الفرع.",
            question_type="conditional",
            topic="مدير",
            difficulty="medium",
            relevant_docs=["نموذج-تغيير-مدير-الفرع.docx"],
        ),
        # Aggregation
        generator.create_manual_item(
            question="ما هي جميع أنواع السندات التي يمكن إصدارها؟",
            answer="تشمل أنواع السندات: سندات التوريق، سندات الشركات، والسندات الخضراء.",
            question_type="aggregation",
            entity_type="توريق",
            topic="سندات",
            difficulty="hard",
            requires_multi_hop=True,
            relevant_docs=[
                "مستندات-اصدار-برنامج-سندات-توريق.docx",
                "مستندات-اصدار-برنامج-سندات-شركات.docx",
                "مستندات-اصدار-سندات-خضراء.docx",
            ],
        ),
        # Edge case - No answer
        generator.create_manual_item(
            question="ما هي عقوبة التأخر في سداد الضرائب؟",
            answer="لا توجد معلومات في المستندات المتاحة حول هذا الموضوع. هذا السؤال خارج نطاق اللوائح المتاحة.",
            question_type="edge_case",
            difficulty="hard",
        ),
        # Multi-hop reasoning
        generator.create_manual_item(
            question="هل تحتاج شركة تمويل استهلاكي جديدة برأس مال 40 مليون جنيه إلى استثناء خاص؟",
            answer="نعم، لأن الحد الأدنى لرأس المال هو 50 مليون جنيه، وبالتالي الشركة برأس مال 40 مليون لا تستوفي المتطلبات.",
            question_type="conditional",
            entity_type="تمويل_استهلاكي",
            topic="رأس_المال",
            difficulty="hard",
            requires_multi_hop=True,
            ground_truth_articles=["5"],
        ),
    ]
    
    for item in additional_items:
        dataset.add_item(item)
    
    # Save dataset
    dataset_path = DATASETS_DIR / "starter_dataset.json"
    dataset.save(dataset_path)
    
    # Print statistics
    stats = dataset.get_statistics()
    logger.info(f"Created starter dataset: {dataset_path}")
    logger.info(f"Total items: {stats['total_items']}")
    logger.info(f"By type: {stats['by_type']}")
    logger.info(f"By difficulty: {stats['by_difficulty']}")
    logger.info(f"Multi-hop questions: {stats['multi_hop_count']}")
    
    return dataset


def run_evaluation(dataset_path: Path = None, output_dir: Path = None):
    """Run full evaluation on a dataset."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load or create dataset
    if dataset_path and dataset_path.exists():
        dataset = GoldenDataset.load(dataset_path)
    else:
        default_path = DATASETS_DIR / "starter_dataset.json"
        if default_path.exists():
            dataset = GoldenDataset.load(default_path)
        else:
            logger.info("No dataset found. Creating starter dataset...")
            dataset = create_starter_dataset()
    
    logger.info(f"Loaded dataset: {dataset.name} with {len(dataset)} items")
    
    # Initialize RAG system
    logger.info("Initializing RAG system...")
    rag_system = FRARAGSystem()
    
    # Check if indexed
    if not rag_system.is_indexed():
        logger.warning("No documents indexed. Please run ingestion first.")
        return
    
    # Create evaluation pipeline
    pipeline = create_evaluation_pipeline(
        rag_system=rag_system,
        use_ragas=False,  # Set to True if ragas is installed
        use_deepeval=False,  # Set to True if deepeval is installed
    )
    
    # Run evaluation
    logger.info("Running evaluation...")
    report = pipeline.run(dataset, k=5, verbose=True)
    
    # Save reports
    output_dir = output_dir or REPORTS_DIR
    
    json_path = output_dir / f"evaluation_report_{report.timestamp.replace(':', '-').split('.')[0]}.json"
    report.save(json_path)
    
    md_path = output_dir / f"evaluation_report_{report.timestamp.replace(':', '-').split('.')[0]}.md"
    report.save_markdown(md_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Overall Score: {report.get_overall_score():.2%}")
    print(f"\nRetrieval Metrics:")
    for metric, value in report.retrieval_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"\nGeneration Metrics:")
    for metric, value in report.generation_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"\nArabic Metrics:")
    for metric, value in report.arabic_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"\nPerformance:")
    print(f"  Avg Latency: {report.avg_latency_ms:.0f} ms")
    print(f"  P95 Latency: {report.p95_latency_ms:.0f} ms")
    print("=" * 60)
    print(f"\nReports saved to: {output_dir}")
    
    return report


def run_quick_test():
    """Run quick test with sample questions."""
    logger.info("Initializing RAG system for quick test...")
    rag_system = FRARAGSystem()
    
    if not rag_system.is_indexed():
        logger.warning("No documents indexed. Please run ingestion first.")
        return
    
    # Sample test questions
    test_questions = [
        "ما هو الحد الأدنى لرأس المال لشركة تمويل استهلاكي؟",
        "ما هي المستندات المطلوبة لإصدار سندات التوريق؟",
        "ما هي إجراءات غلق فرع لشركة سمسرة؟",
        "ما الفرق بين متطلبات قيد فرع تمويل استهلاكي وفرع تمويل متناهي الصغر؟",
        "هل يمكن تغيير مدير الفرع بدون موافقة الهيئة؟",
    ]
    
    pipeline = create_evaluation_pipeline(
        rag_system=rag_system,
        use_ragas=False,
        use_deepeval=False,
    )
    
    logger.info(f"Running quick test with {len(test_questions)} questions...")
    results = pipeline.run_quick(test_questions, k=5)
    
    print("\n" + "=" * 60)
    print("QUICK TEST RESULTS")
    print("=" * 60)
    print(f"Questions tested: {results['questions_tested']}")
    print(f"Avg Latency: {results['avg_latency_ms']:.0f} ms")
    print(f"Avg Faithfulness: {results['avg_faithfulness']:.2%}")
    print(f"Avg Relevancy: {results['avg_relevancy']:.2%}")
    print("=" * 60)
    
    return results


def test_react_agent():
    """Test the ReAct agent with a complex question."""
    logger.info("Testing ReAct agent...")
    rag_system = FRARAGSystem()
    
    if not rag_system.is_indexed():
        logger.warning("No documents indexed. Please run ingestion first.")
        return
    
    if not rag_system.react_agent:
        logger.warning("ReAct agent not available. Check LLM client.")
        return
    
    # Test with a multi-hop question
    test_question = "هل تحتاج شركة تمويل استهلاكي جديدة برأس مال 40 مليون جنيه إلى استثناء خاص؟"
    
    logger.info(f"Question: {test_question}")
    logger.info("Running ReAct agent...")
    
    result = rag_system.react_agent.run(test_question, k=3)
    
    print("\n" + "=" * 60)
    print("REACT AGENT RESULT")
    print("=" * 60)
    print(f"\nAnswer: {result.answer}")
    print(f"\nReasoning Steps: {len(result.steps)}")
    for step in result.steps:
        print(f"\n  Step {step.step_number}:")
        print(f"    Thought: {step.thought[:100]}...")
        print(f"    Action: {step.action.value}")
        print(f"    Observation: {step.observation[:100]}...")
    print(f"\nSources used: {len(result.sources)}")
    print(f"Total retrievals: {result.total_retrievals}")
    print(f"Reasoning complete: {result.reasoning_complete}")
    print("=" * 60)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="FRA RAG System Evaluation")
    parser.add_argument("--create-starter", action="store_true", help="Create starter golden dataset")
    parser.add_argument("--run", action="store_true", help="Run full evaluation")
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    parser.add_argument("--react", action="store_true", help="Test ReAct agent")
    parser.add_argument("--dataset", type=str, help="Path to golden dataset JSON")
    parser.add_argument("--output", type=str, help="Output directory for reports")
    
    args = parser.parse_args()
    
    if args.create_starter:
        create_starter_dataset()
    elif args.run:
        dataset_path = Path(args.dataset) if args.dataset else None
        output_dir = Path(args.output) if args.output else None
        run_evaluation(dataset_path, output_dir)
    elif args.quick:
        run_quick_test()
    elif args.react:
        test_react_agent()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
