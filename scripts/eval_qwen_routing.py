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


def run_routing_evaluation():
    canned_results = [
        {
            "title": "NVIDIA Nemotron 3 Architecture & Hardware Requirements",
            "url": "https://developer.nvidia.com/nemotron-3",
            "snippet": "NVIDIA Nemotron-3 models require 8x H100 SXM GPUs for FP8 inference and Triton Server 24.08.",
        },
        {
            "title": "UK Statutory Leave Guidelines",
            "url": "https://gov.uk/annual-leave-entitlement",
            "snippet": "Statutory annual leave entitlement in the UK is 28 days for full-time workers.",
        },
    ]

    questions = [
        ("company", "Who is the head of Engineering?"),
        ("external", "What are the current NVIDIA Nemotron deployment requirements?"),
        ("mixed", "How does our annual leave compare to UK statutory minimum?"),
    ]

    modes = ["true", "false"]
    results = {}

    llm = OllamaClient(model="qwen2.5:3b")

    for mode in modes:
        os.environ["FAST_PATH_ENABLED"] = mode
        results[f"fast_path_{mode}"] = {}

        for category, q in questions:
            run_records = []
            for run_idx in range(1, 4):
                fake_web = FakeWebSearchClient(canned_results=canned_results)
                executor = ToolExecutor(web_client=fake_web)
                orchestrator = AgentOrchestrator(llm_client=llm, executor=executor)

                initial_web_calls = len(fake_web.calls)
                resp = orchestrator.run(q, user_id="eval-user")
                final_web_calls = len(fake_web.calls)
                web_called = final_web_calls > initial_web_calls

                plan_steps = []
                tools_chosen = []
                if resp.plan and resp.plan.steps:
                    for s in resp.plan.steps:
                        tools_chosen.append(s.tool)
                        plan_steps.append({
                            "tool": s.tool,
                            "input": s.input,
                            "operation": s.operation,
                            "args": s.args,
                        })

                # Determine stopping layer if any error/refusal
                stopping_layer = "None (Executed)"
                if resp.plan is None:
                    stopping_layer = "Validator (Plan validation failed)"
                elif any(not tr.success for tr in resp.tool_results):
                    stopping_layer = "Executor (Tool execution rejected)"
                elif not resp.verified:
                    stopping_layer = "Verifier (Grounded verification failed)"

                rec = {
                    "run": run_idx,
                    "question": q,
                    "plan_valid": resp.plan is not None,
                    "tools_chosen": tools_chosen,
                    "web_called": web_called,
                    "stopping_layer": stopping_layer,
                    "verified": resp.verified,
                    "answer_preview": resp.answer[:150] if resp.answer else "",
                }
                run_records.append(rec)
                print(f"[{category.upper()}] Mode: fast_path={mode} Run {run_idx}: tools={tools_chosen}, web_called={web_called}, valid={rec['plan_valid']}, verified={rec['verified']}")

            results[f"fast_path_{mode}"][category] = run_records

    out_file = Path("docs/results/phase4_routing_eval.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation complete. Saved to {out_file}")


if __name__ == "__main__":
    run_routing_evaluation()
