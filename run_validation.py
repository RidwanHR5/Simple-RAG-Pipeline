import os
import json
import datetime
from datasets import load_dataset, Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from app import process_input, get_rag_chain


def setup_evaluator_llm():
    """Initialize and return the LLM for RAGAS evaluation."""
    judge_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )
    return LangchainLLMWrapper(judge_llm)


def setup_evaluator_embeddings():
    """Initialize and return embeddings for RAGAS evaluation."""
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )
    return LangchainEmbeddingsWrapper(hf_embeddings)


def load_test_dataset():
    """Load the Amnesty QA evaluation dataset."""
    print("📥 Loading 'Amnesty QA' Dataset...")
    dataset = load_dataset("explodinggradients/amnesty_qa", "english_v2", split="eval")
    return dataset


def prepare_knowledge_base(dataset):
    """Prepare the knowledge base by combining all contexts from the dataset."""
    print("⚙️  Simulating User Upload & Processing...")
    all_contexts = []
    for row in dataset:
        all_contexts.extend(row['contexts'])
    
    # Remove duplicates while preserving order
    unique_contexts = list(dict.fromkeys(all_contexts))
    full_document = "\n\n".join(unique_contexts)
    
    vectorstore = process_input("Text", full_document)
    return vectorstore


def run_rag_inference(rag_chain, dataset):
    """Run RAG inference on all test questions and collect responses."""
    print(f"🚀 Running Validation on {len(dataset)} test cases...")
    
    answers = []
    retrieved_contexts = []
    
    for i, row in enumerate(dataset):
        question = row['question']
        response = rag_chain.invoke({
            "input": question,
            "chat_history": []
        })
        
        answers.append(response["answer"])
        ctx_strings = [doc.page_content for doc in response["context"]]
        retrieved_contexts.append(ctx_strings)
        
        print(f"   [{i+1}/{len(dataset)}] Processed...")
    
    return answers, retrieved_contexts


def create_evaluation_dataset(dataset, answers, retrieved_contexts):
    """Create RAGAS evaluation dataset from results."""
    return Dataset.from_dict({
        "question": dataset['question'],
        "answer": answers,
        "contexts": retrieved_contexts,
        "ground_truth": dataset['ground_truth']
    })


def run_ragas_evaluation(eval_dataset, evaluator_llm, evaluator_embeddings):
    """Run RAGAS evaluation metrics."""
    print("\n📈 Calculating Ragas Metrics...")
    
    results = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )
    
    return results


def save_results(results):
    """Save evaluation results to timestamped directory."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"validation_runs/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    df = results.to_pandas()
    
    # Save detailed results
    csv_path = os.path.join(output_dir, "detailed_results.csv")
    df.to_csv(csv_path, index=False)
    
    # Calculate and save summary scores
    numeric_cols = df.select_dtypes(include=['number'])
    summary_scores = numeric_cols.mean().to_dict()
    
    summary_path = os.path.join(output_dir, "summary_scores.json")
    with open(summary_path, "w") as f:
        json.dump(summary_scores, f, indent=4)
    
    return output_dir, summary_path, csv_path, summary_scores


def print_summary(summary_scores, output_dir, summary_path, csv_path):
    """Print validation summary to console."""
    print("\n" + "="*40)
    print("✅ VALIDATION COMPLETE")
    print("="*40)
    print(f"Summary Scores: {summary_scores}")
    print(f"\n📂 Files saved to: {output_dir}/")
    print(f"   1. {summary_path} (Averages)")
    print(f"   2. {csv_path} (Row-by-row details)")


def main():
    """Main validation pipeline."""
    try:
        # Setup evaluators
        evaluator_llm = setup_evaluator_llm()
        evaluator_embeddings = setup_evaluator_embeddings()
        
        # Load test dataset
        dataset = load_test_dataset()
        
        # Prepare knowledge base
        vectorstore = prepare_knowledge_base(dataset)
        rag_chain = get_rag_chain(vectorstore)
        
        # Run inference
        answers, retrieved_contexts = run_rag_inference(rag_chain, dataset)
        
        # Create evaluation dataset
        eval_dataset = create_evaluation_dataset(dataset, answers, retrieved_contexts)
        
        # Run evaluation
        results = run_ragas_evaluation(eval_dataset, evaluator_llm, evaluator_embeddings)
        
        # Save results
        output_dir, summary_path, csv_path, summary_scores = save_results(results)
        
        # Print summary
        print_summary(summary_scores, output_dir, summary_path, csv_path)
        
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        raise


if __name__ == "__main__":
    main()
