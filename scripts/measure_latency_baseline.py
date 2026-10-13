import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Isolate database
temp_dir = tempfile.TemporaryDirectory()
db_path = Path(temp_dir.name) / "latency_baseline.db"
os.environ["DATABASE_PATH"] = str(db_path)

from app.database import initialize_database, seed_database
initialize_database()
seed_database()

from app.agent.executor import ToolExecutor
from app.agent.orchestrator import AgentOrchestrator
from app.agent.web_search import FakeWebSearchClient
from app.agent.llm import OllamaClient


def measure_baseline(output_file: str = "docs/results/latency_before.json"):
    questions = [
        "How many employees are in AI?",
        "Where is the company located?",
        "How does our annual leave compare to the UK statutory minimum?",
    ]

    canned_results = [
        {
            "title": "UK Statutory Leave Guidelines",
            "url": "https://gov.uk/annual-leave-entitlement",
            "snippet": "Statutory annual leave entitlement in the UK is 28 days for full-time workers.",
        },
    ]

    modes = [True, False]
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "qwen2.5:3b",
        "provider": "ollama",
        "runs": [],
        "summary": {},
    }

    llm = OllamaClient(model="qwen2.5:3b")

    for fast_path in modes:
        mode_str = "fast_path_true" if fast_path else "fast_path_false"
        os.environ["FAST_PATH_ENABLED"] = "true" if fast_path else "false"
        print(f"\n==========================================")
        print(f"Testing FAST_PATH_ENABLED={fast_path}")
        print(f"==========================================")

        for q in questions:
            print(f"\nQuestion: {q}")
            for run_idx in range(1, 4):
                fake_web = FakeWebSearchClient(canned_results=canned_results)
                executor = ToolExecutor(web_client=fake_web)
                orchestrator = AgentOrchestrator(llm_client=llm, executor=executor)
                user_id = f"baseline-user-{mode_str}-{run_idx}"

                t0 = time.perf_counter()
                resp = orchestrator.run(q, user_id=user_id)
                proc_time_ms = round((time.perf_counter() - t0) * 1000, 2)

                record = {
                    "question": q,
                    "fast_path_enabled": fast_path,
                    "run": run_idx,
                    "llm_calls_count": resp.llm_calls_count,
                    "llm_latency_ms": resp.llm_latency_ms,
                    "processing_time_ms": resp.processing_time_ms or proc_time_ms,
                    "total_measured_time_ms": proc_time_ms,
                    "verified": resp.verified,
                    "answer": resp.answer,
                }
                results["runs"].append(record)
                print(
                    f"  Run {run_idx}: calls={resp.llm_calls_count}, "
                    f"llm_lat={resp.llm_latency_ms}, "
                    f"proc_ms={record['processing_time_ms']}, "
                    f"verified={resp.verified}"
                )

    # Compute summary
    for fast_path in modes:
        mode_key = "fast_path_true" if fast_path else "fast_path_false"
        results["summary"][mode_key] = {}
        for q in questions:
            matching = [
                r for r in results["runs"]
                if r["fast_path_enabled"] == fast_path and r["question"] == q
            ]
            if matching:
                avg_proc = round(sum(r["processing_time_ms"] for r in matching) / len(matching), 2)
                avg_calls = round(sum(r["llm_calls_count"] for r in matching) / len(matching), 2)
                results["summary"][mode_key][q] = {
                    "avg_processing_time_ms": avg_proc,
                    "avg_llm_calls_count": avg_calls,
                    "runs_count": len(matching),
                }

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved baseline latency measurements to {output_file}")
    temp_dir.cleanup()
    return results


if __name__ == "__main__":
    measure_baseline()
