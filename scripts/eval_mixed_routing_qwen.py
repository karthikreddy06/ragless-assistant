import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.executor import ToolExecutor
from app.agent.orchestrator import AgentOrchestrator
from app.agent.web_search import FakeWebSearchClient
from app.agent.llm import OllamaClient


def eval_mixed_question():
    canned_results = [
        {
            "title": "UK Statutory Leave Guidelines",
            "url": "https://gov.uk/annual-leave-entitlement",
            "snippet": "Statutory annual leave entitlement in the UK is 28 days for full-time workers.",
        },
    ]

    q = "How does our annual leave compare to UK statutory minimum?"
    modes = ["true", "false"]
    results = {}
    sample_verification_raw = None

    llm = OllamaClient(model="qwen2.5:3b")

    for mode in modes:
        os.environ["FAST_PATH_ENABLED"] = mode
        results[f"fast_path_{mode}"] = []

        for run_idx in range(1, 4):
            fake_web = FakeWebSearchClient(canned_results=canned_results)
            executor = ToolExecutor(web_client=fake_web)
            orchestrator = AgentOrchestrator(llm_client=llm, executor=executor)

            initial_web_calls = len(fake_web.calls)
            resp = orchestrator.run(q, user_id="eval-user")
            final_web_calls = len(fake_web.calls)
            web_called = final_web_calls > initial_web_calls

            tools_chosen = [s.tool for s in resp.plan.steps] if (resp.plan and resp.plan.steps) else []
            plan_valid = resp.plan is not None

            verif_json = json.loads(resp.verification.model_dump_json()) if resp.verification else None
            if sample_verification_raw is None and verif_json:
                sample_verification_raw = verif_json

            record = {
                "run": run_idx,
                "question": q,
                "plan_valid": plan_valid,
                "tools_chosen": tools_chosen,
                "web_called": web_called,
                "verified": resp.verified,
                "unresolved": resp.verification.unresolved if resp.verification else [],
                "answer": resp.answer,
            }
            results[f"fast_path_{mode}"].append(record)
            print(f"[MIXED] Mode: fast_path={mode} Run {run_idx}: tools={tools_chosen}, web_called={web_called}, valid={plan_valid}, verified={resp.verified}")

    out_file = Path("docs/results/mixed_routing_eval.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"results": results, "sample_verification": sample_verification_raw}, f, indent=2)

    print("\nSample Verification Raw JSON:")
    print(json.dumps(sample_verification_raw, indent=2))


if __name__ == "__main__":
    eval_mixed_question()
