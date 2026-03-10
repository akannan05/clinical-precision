import json
import requests
import os
import time
from dotenv import load_dotenv

load_dotenv()
OR_KEY = os.getenv("OPEN_ROUTER_KEY")
MODEL_NAME = "stepfun/step-3.5-flash:free"

def get_judge_verdict(question, truth, gold_res, raw_res):
    prompt = f"""You are a Senior Medical Educator. Compare two AI responses to a USMLE-style question.

Question: {question}
Correct Answer Letter: {truth}

Model A (Gold): {gold_res}
Model B (Raw): {raw_res}

Evaluation Criteria:
1. Accuracy: Did the model reach the correct final answer ({truth})?
2. Logical Depth: Did it provide a clinical reasoning path (pathophysiology, differential diagnosis)?
3. Professionalism: Is the tone appropriate for a medical assistant?

Instructions:
- If Model A provides correct reasoning and Model B only provides a short answer, Model A wins.
- If both are wrong, mark it as a 'Tie'.
- Your output must be in JSON format: {{"winner": "A" or "B" or "Tie", "reason": "one sentence explanation"}}"""

    headers = {
        "Authorization": f"Bearer {OR_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": { "type": "json_object" }
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=30
        )
        res_json = response.json()
        return json.loads(res_json['choices'][0]['message']['content'])
    except Exception as e:
        print(f"Error calling judge: {e}")
        return {"winner": "Tie", "reason": "API Error"}

def main():
    with open("results/eval_comparison.json", "r") as f:
        data = json.load(f)

    stats = {"A": 0, "B": 0, "Tie": 0}
    final_report = []

    print(f"⚖️ Judging {len(data)} clinical cases with Step 3.5 Flash...")

    for i, item in enumerate(data):
        verdict = get_judge_verdict(
            item['question'],
            item['ground_truth'],
            item['gold_response'],
            item['raw_response']
        )

        winner = verdict.get("winner", "Tie")
        stats[winner] += 1

        item["judge_winner"] = winner
        item["judge_reason"] = verdict.get("reason")
        final_report.append(item)

        print(f"[{i+1}/{len(data)}] Winner: {winner} | Reason: {item['judge_reason']}")
        time.sleep(1) 

    # final analysis
    with open("results/final_report.json", "w") as f:
        json.dump({"stats": stats, "results": final_report}, f, indent=2)

    print("\n" + "="*30)
    print("FINAL EVALUATION SUMMARY")
    print("="*30)
    print(f"Gold Model Wins: {stats['A']}")
    print(f"Raw Model Wins:  {stats['B']}")
    print(f"Ties:            {stats['Tie']}")
    print("="*30)

if __name__ == "__main__":
    main()
