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
db_path = Path(temp_dir.name) / "latency_measure_after.db"
os.environ["DATABASE_PATH"] = str(db_path)

from app.database import initialize_database, seed_database
initialize_database()
seed_database()

from app.agent.executor import ToolExecutor
from app.agent.orchestrator import AgentOrchestrator
from app.agent.web_search import FakeWebSearchClient
from app.agent.llm import OllamaClient


def measure_after(output_file: str = "docs/results/latency_after.json"):
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

    configs = [
        {
            "config_name": "default_flags (fast_path=true, deterministic=true)",
            "fast_path": "true",
            "deterministic_answers": "true",
        },
        {
            "config_name": "fast_path_off (fast_path=false, deterministic=true)",
            "fast_path": "false",
            "deterministic_answers": "true",
        },
        {
            "config_name": "deterministic_answers_off (fast_path=true, deterministic=false)",
            "fast_path": "true",
            "deterministic_answers": "false",
        },
        {
            "config_name": "all_flags_off (fast_path=false, deterministic=false)",
            "fast_path": "false",
            "deterministic_answers": "false",
        },
    ]

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "qwen2.5:3b",
        "provider": "ollama",
        "runs": [],
        "summary": {},
    }

    llm = OllamaClient(model="qwen2.5:3b")

    for cfg in configs:
        cfg_name = cfg["config_name"]
        os.environ["FAST_PATH_ENABLED"] = cfg["fast_path"]
        os.environ["DETERMINISTIC_ANSWERS_ENABLED"] = cfg["deterministic_answers"]
        print(f"\n==========================================")
        print(f"Testing Config: {cfg_name}")
        print(f"==========================================")

        for q in questions:
            print(f"\nQuestion: {q}")
            for run_idx in range(1, 4):
                fake_web = FakeWebSearchClient(canned_results=canned_results)
                executor = ToolExecutor(web_client=fake_web)
                orchestrator = AgentOrchestrator(llm_client=llm, executor=executor)
                user_id = f"bench-user-{run_idx}"

                t0 = time.perf_counter()
                resp = orchestrator.run(q, user_id=user_id)
                proc_time_ms = round((time.perf_counter() - t0) * 1000, 2)

                record = {
                    "config": cfg_name,
                    "fast_path_enabled": cfg["fast_path"] == "true",
                    "deterministic_answers_enabled": cfg["deterministic_answers"] == "true",
                    "question": q,
                    "run": run_idx,
                    "answer_mode": resp.answer_mode,
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
                    f"mode={resp.answer_mode}, "
                    f"proc_ms={record['processing_time_ms']}, "
                    f"verified={resp.verified}"
                )

    # Compute summary
    for cfg in configs:
        cfg_name = cfg["config_name"]
        results["summary"][cfg_name] = {}
        for q in questions:
            matching = [
                r for r in results["runs"]
                if r["config"] == cfg_name and r["question"] == q
            ]
            if matching:
                avg_proc = round(sum(r["processing_time_ms"] for r in matching) / len(matching), 2)
                avg_calls = round(sum(r["llm_calls_count"] for r in matching) / len(matching), 1)
                all_lats = [lat for r in matching if r["llm_latency_ms"] for lat in r["llm_latency_ms"]]
                avg_llm_lat = round(sum(all_lats) / len(all_lats), 2) if all_lats else 0.0
                modes = list(set(r["answer_mode"] for r in matching))
                results["summary"][cfg_name][q] = {
                    "avg_processing_time_ms": avg_proc,
                    "avg_llm_calls_count": avg_calls,
                    "avg_llm_call_latency_ms": avg_llm_lat,
                    "answer_modes": modes,
                    "runs_count": len(matching),
                    "all_verified": all(r["verified"] for r in matching),
                }

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Latency measurements saved to {output_file}")


if __name__ == "__main__":
    measure_after()
