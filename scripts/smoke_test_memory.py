import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.llm import OllamaClient
from app.agent.memory import get_memory_store
from app.agent.orchestrator import AgentOrchestrator
from app.database import initialize_database, seed_database


def run_demo_scenario(repetition: int) -> Dict[str, Any]:
    print(f"\n==================================================")
    print(f"RUNNING DEMO SCENARIO: REPETITION {repetition}/3")
    print(f"==================================================")

    # Initialize and seed database
    initialize_database()
    seed_database()

    # Create fresh user ID per repetition to test clean lifecycle
    user_id = f"demo-user-rep-{repetition}"
    conn = get_memory_store().get_connection()
    try:
        with conn:
            conn.execute("DELETE FROM user_completed_items WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM user_profile WHERE user_id = ?", (user_id,))
    finally:
        conn.close()

    client = OllamaClient(timeout=120)
    orchestrator = AgentOrchestrator(llm_client=client)

    scenario_steps = [
        {"day": 1, "query": "Help me complete onboarding", "expected_op": "get_progress"},
        {"day": 2, "query": "I completed security training", "expected_op": "mark_item_complete"},
        {"day": 3, "query": "What do I still need to do?", "expected_op": "get_progress"},
    ]

    step_results = []
    latencies = []
    llm_calls = []
    fast_path_count = 0
    write_confirm_count = 0
    total_writes = 0

    initial_total_items = 0
    day1_remaining = 0
    day3_remaining = 0

    for step_info in scenario_steps:
        day = step_info["day"]
        q = step_info["query"]
        expected_op = step_info["expected_op"]

        t0 = time.perf_counter()
        resp = orchestrator.run(q, user_id=user_id)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        latencies.append(elapsed_ms)
        llm_calls.append(resp.llm_calls_count or 0)

        plan_valid = resp.plan is not None
        has_memory = bool(resp.memory_used)
        op_matched = False
        if resp.plan and resp.plan.steps:
            for s in resp.plan.steps:
                if s.tool == "memory" and s.operation == expected_op:
                    op_matched = True
                    break

        # Check if fast-path handled it (0 planner calls)
        fast_path_triggered = (resp.llm_calls_count is not None and resp.llm_calls_count <= 1)
        if fast_path_triggered:
            fast_path_count += 1

        # Check write confirmation
        if resp.memory_writes:
            total_writes += len(resp.memory_writes)
            for w in resp.memory_writes:
                if w.get("confirmed"):
                    write_confirm_count += 1

        # Track remaining counts from memory tool result
        memory_data = None
        for tr in resp.tool_results:
            if tr.tool == "memory" and isinstance(tr.data, dict):
                memory_data = tr.data
                break

        if day == 1 and memory_data:
            day1_remaining = memory_data.get("remaining_count", 0)
            initial_total_items = memory_data.get("total_items", 0)
        elif day == 3 and memory_data:
            day3_remaining = memory_data.get("remaining_count", 0)

        print(f"\n[Day {day}] User ({user_id}): \"{q}\"")
        print(f"  Plan: {resp.plan.goal if resp.plan else 'NO PLAN'} -> {[s.tool + '.' + (s.operation or '') for s in (resp.plan.steps if resp.plan else [])]}")
        print(f"  Latency: {elapsed_ms}ms | LLM Calls: {resp.llm_calls_count}")
        print(f"  Verification: {'PASS' if resp.verification.passed else 'FAIL'}")
        print(f"  Answer: {resp.answer}")

        step_results.append({
            "day": day,
            "query": q,
            "latency_ms": elapsed_ms,
            "llm_calls": resp.llm_calls_count,
            "plan_valid": plan_valid,
            "op_matched": op_matched,
            "plan": resp.plan.model_dump() if resp.plan else None,
            "tool_results": [tr.model_dump() for tr in resp.tool_results] if resp.tool_results else [],
            "verification": resp.verification.model_dump() if resp.verification else None,
            "verification_passed": resp.verification.passed if resp.verification else False,
            "memory_writes": resp.memory_writes,
            "answer": resp.answer,
        })

    # Correctness check: Day 3 remaining count must be exactly Day 1 remaining count minus 1
    count_correct = (day3_remaining == (day1_remaining - 1))

    return {
        "repetition": repetition,
        "user_id": user_id,
        "latencies": latencies,
        "llm_calls": llm_calls,
        "fast_path_count": fast_path_count,
        "write_confirm_count": write_confirm_count,
        "total_writes": total_writes,
        "count_correct": count_correct,
        "initial_total": initial_total_items,
        "day1_remaining": day1_remaining,
        "day3_remaining": day3_remaining,
        "step_results": step_results,
    }


def main():
    import os
    from app.agent.fast_path import is_fast_path_enabled

    fast_path_on = is_fast_path_enabled()
    mode_str = "FAST-PATH ENABLED" if fast_path_on else "FAST-PATH DISABLED (qwen2.5:3b Planner)"

    print("==================================================")
    print("PHASE 3 LIVE SMOKE SUITE: PERSISTENT MEMORY")
    print(f"Provider: Ollama (qwen2.5:3b) | Mode: {mode_str} | Repetitions: 3")
    print("==================================================")

    runs = []
    for rep in range(1, 4):
        run_res = run_demo_scenario(rep)
        runs.append(run_res)

    # Compute aggregate statistics
    all_latencies = [lat for r in runs for lat in r["latencies"]]
    all_llm_calls = [c for r in runs for c in r["llm_calls"]]
    total_steps = len(all_latencies)
    valid_plans = sum(1 for r in runs for s in r["step_results"] if s["plan_valid"])
    matched_ops = sum(1 for r in runs for s in r["step_results"] if s["op_matched"])
    total_fast_path = sum(r["fast_path_count"] for r in runs)
    total_writes = sum(r["total_writes"] for r in runs)
    confirmed_writes = sum(r["write_confirm_count"] for r in runs)
    all_counts_correct = all(r["count_correct"] for r in runs)

    plan_valid_rate = round((valid_plans / total_steps) * 100, 1)
    correct_op_rate = round((matched_ops / total_steps) * 100, 1)
    p50_lat = round(statistics.median(all_latencies), 2)
    p95_lat = round(sorted(all_latencies)[int(len(all_latencies) * 0.95)], 2)
    mean_lat = round(statistics.mean(all_latencies), 2)
    fast_path_rate = round((total_fast_path / total_steps) * 100, 1)
    write_confirm_rate = round((confirmed_writes / max(1, total_writes)) * 100, 1)

    print("\n" + "=" * 60)
    print(f"PHASE 3 LIVE SMOKE RESULTS ({mode_str})")
    print("=" * 60)
    print(f"Total Workflow Runs:        3 repetitions (9 turns total)")
    print(f"Plan Validity Rate:         {plan_valid_rate}% ({valid_plans}/{total_steps})")
    print(f"Correct Op Selection Rate:  {correct_op_rate}% ({matched_ops}/{total_steps})")
    print(f"Remaining Count Correct:    {'100.0% (PASSED)' if all_counts_correct else 'FAIL'}")
    print(f"Write Confirmation Rate:    {write_confirm_rate}% ({confirmed_writes}/{total_writes})")
    print(f"Fast-Path Trigger Rate:     {fast_path_rate}% ({total_fast_path}/{total_steps})")
    print(f"Latency p50:                {p50_lat} ms")
    print(f"Latency p95:                {p95_lat} ms")
    print(f"Latency Mean:               {mean_lat} ms")
    print(f"Avg LLM Calls per Request:  {round(statistics.mean(all_llm_calls), 2)}")

    # Save results to docs/results/
    out_dir = Path("docs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "fastpath" if fast_path_on else "planner"
    out_file = out_dir / f"phase3_smoke_{suffix}.json"
    summary_data = {
        "provider": "ollama",
        "model": "qwen2.5:3b",
        "mode": "fast_path" if fast_path_on else "llm_planner",
        "fast_path_enabled": fast_path_on,
        "repetitions": 3,
        "metrics": {
            "plan_validity_percent": plan_valid_rate,
            "correct_operation_rate_percent": correct_op_rate,
            "remaining_count_correct": all_counts_correct,
            "write_confirmation_rate_percent": write_confirm_rate,
            "fast_path_trigger_rate_percent": fast_path_rate,
            "latency_p50_ms": p50_lat,
            "latency_p95_ms": p95_lat,
            "latency_mean_ms": mean_lat,
            "avg_llm_calls_per_turn": round(statistics.mean(all_llm_calls), 2),
        },
        "runs": runs,
    }
    with open(out_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    # Also keep general phase3_smoke.json updated
    with open(out_dir / "phase3_smoke.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nDetailed metrics saved to: {out_file}")

    print(f"\nDetailed metrics saved to: {out_file}")


if __name__ == "__main__":
    main()
