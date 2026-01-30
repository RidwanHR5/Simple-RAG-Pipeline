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

judge_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0
)
evaluator_llm = LangchainLLMWrapper(judge_llm)

hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'}
)
evaluator_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

print("📥 Loading 'Amnesty QA' Dataset...")
dataset_source = load_dataset("explodinggradients/amnesty_qa", "english_v2", split="eval")

print("⚙️  Simulating User Upload & Processing...")
all_contexts = []
for row in dataset_source:
    all_contexts.extend(row['contexts'])
full_simulated_doc = "\n\n".join(set(all_contexts)) 

vectorstore = process_input("Text", full_simulated_doc)
rag_chain = get_rag_chain(vectorstore)

print(f"🚀 Running Validation on {len(dataset_source)} test cases...")
answers = []
retrieved_contexts = []

for i, row in enumerate(dataset_source):
    question = row['question']
    response = rag_chain.invoke({"input": question, "chat_history": []})
    answers.append(response["answer"])
    ctx_strings = [doc.page_content for doc in response["context"]]
    retrieved_contexts.append(ctx_strings)
    print(f"   [{i+1}/{len(dataset_source)}] Processed...")

print("\n📈 Calculating Ragas Metrics...")
eval_dataset = Dataset.from_dict({
    "question": dataset_source['question'],
    "answer": answers,
    "contexts": retrieved_contexts,
    "ground_truth": dataset_source['ground_truth']
})

results = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"validation_runs/{timestamp}"
os.makedirs(output_dir, exist_ok=True)

df = results.to_pandas()

csv_path = os.path.join(output_dir, "detailed_results.csv")
df.to_csv(csv_path, index=False)

numeric_cols = df.select_dtypes(include=['number']) 
summary_scores = numeric_cols.mean().to_dict()

summary_path = os.path.join(output_dir, "summary_scores.json")
with open(summary_path, "w") as f:
    json.dump(summary_scores, f, indent=4)

print("\n" + "="*40)
print("✅ VALIDATION COMPLETE")
print("="*40)
print(f"Summary Scores: {summary_scores}")
print(f"\n📂 Files saved to: {output_dir}/")
print(f"   1. {summary_path} (Averages)")
print(f"   2. {csv_path} (Row-by-row details)")