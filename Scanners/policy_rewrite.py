import json
import torch
from transformers import pipeline
from Scanners.Complete_Scanner import CompleteCipherCloudScanner

device = 0 if torch.cuda.is_available() else -1  # GPU if available, else CPU

print(f"⚡ Using device: {'GPU' if device >= 0 else 'CPU'}")

llm = pipeline(
    "text-generation",
    model="./model_cache/deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B",
    cache_dir="./model_cache",
    device=device
)

def rewrite_policy(policy_json: dict) -> str:
    """
    Rewrite risky IAM policies using HuggingFace DeepSeek model.
    """
    prompt = (
        "This IAM policy is risky. Rewrite it to be secure while preserving necessary permissions:\n"
        f"{json.dumps(policy_json, indent=2)}\n"
        "Rewritten Secure Policy:"
    )

    result = llm(prompt, max_new_tokens=512, temperature=0.3)

    result_text = result[0].get('generated_text', "")
    if "Rewritten Secure Policy:" in result_text:
        rewritten = result_text.split("Rewritten Secure Policy:")[-1].strip()
    else:
        rewritten = result_text.strip()

    return rewritten

def main():
    scanner = CompleteCipherCloudScanner()

    # Example: Load a policy from file
    with open("example_policy.json") as f:
        policy = json.load(f)

    # Step 1: Run binary + family scan
    scan_result = scanner.complete_scan(policy)

    # Step 2: Rewrite risky policies
    if scan_result['binary_result']['is_risky']:
        print("🚨 Risky policy detected! Sending to LLM for rewrite...")
        rewritten_policy = rewrite_policy(policy)
        print("\n🔒 Rewritten Secure Policy:\n")
        print(rewritten_policy)
    else:
        print("✅ Policy is not risky. No rewrite needed.")

if __name__ == "__main__":
    main()