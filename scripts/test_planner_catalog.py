import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.llm import OllamaClient
from app.database import get_onboarding_items
import json

items = get_onboarding_items()
catalog = "\n".join(f"- ID {it['id']}: {it['title']}" for it in items)

system_prompt = f"""You are a strict task planner for an enterprise knowledge assistant.
Your job is to break down a user question into sequential steps using ONLY allowed tools.

Available Onboarding Checklist Items:
{catalog}

Allowed Tools:
- 'company_knowledge': Queries verified SQLite corporate knowledge. Parameter: 'input'.
- 'memory': Persistent structured user profile and onboarding progress tracker.
  Parameter: 'operation' (one of: 'get_profile', 'get_progress', 'set_profile', 'mark_item_complete', 'set_onboarding_status')
  Parameter: 'args' (structured object, e.g. {{'item_id': <integer>}} for mark_item_complete)

When to use Memory operations:
- 'get_progress': User asks what onboarding tasks are remaining, left to do, or asks for help completing onboarding.
- 'mark_item_complete': User explicitly states they completed or finished a checklist item (args: {{'item_id': <integer ID from catalog>}}).

Output valid JSON ONLY matching:
{{
  "goal": "Description",
  "steps": [
    {{"tool": "memory", "operation": "mark_item_complete", "args": {{"item_id": 1}}}}
  ]
}}"""

client = OllamaClient(timeout=30)
for q in ["Help me complete onboarding", "I completed security training", "What do I still need to do?"]:
    res = client.generate_json(f'User Question: "{q}"\nGenerate JSON plan:', system=system_prompt)
    print("Q:", q)
    print("Plan:", res)
