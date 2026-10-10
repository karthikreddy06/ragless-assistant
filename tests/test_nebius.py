import os
import unittest
from unittest.mock import MagicMock, patch
import requests

from app.agent.llm import (
    NebiusAuthError,
    NebiusClient,
    NebiusError,
    NebiusRateLimitError,
    NebiusTimeoutError,
    OllamaClient,
    check_active_llm_health,
    get_llm_client,
    redact_secrets,
    strip_reasoning,
)
from app.agent.orchestrator import AgentOrchestrator
from app.agent.planner import AgentPlanner
from app.agent.schemas import AgentPlan, PlanStep
from app.agent.validator import validate_plan
try:
    import sys
    from pathlib import Path
    _proj_root = str(Path(__file__).resolve().parent.parent)
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)
    _tests_dir = str(Path(__file__).resolve().parent)
    if _tests_dir not in sys.path:
        sys.path.insert(0, _tests_dir)
    from test_agent import FakeLLMClient
except ImportError:
    from tests.test_agent import FakeLLMClient


class TestNebiusClientMocked(unittest.TestCase):
    """Safe unit tests for NebiusClient using mocked HTTP responses (no network/key needed)."""

    def setUp(self):
        self.secret_key = "nebius_secret_test_key_987654321"
        self.client = NebiusClient(
            api_key=self.secret_key,
            base_url="https://api.tokenfactory.nebius.com/v1/",
            model="nvidia/nemotron-3-super-120b-a12b",
            timeout=5,
            max_retries=2,
        )

    @patch("requests.post")
    def test_successful_text_generation(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Chennai is the headquarters."}}]
        }
        mock_post.return_value = mock_resp

        result = self.client.generate_text("Where is the company?")
        self.assertEqual(result, "Chennai is the headquarters.")
        self.assertNotIn(self.secret_key, result)

    @patch("requests.post")
    def test_successful_json_generation(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"goal": "find location", "steps": []}'}}]
        }
        mock_post.return_value = mock_resp

        result = self.client.generate_json("Where is the company?")
        self.assertIn("find location", result)
        # Verify response_format was sent in payload
        called_payload = mock_post.call_args[1]["json"]
        self.assertEqual(called_payload["response_format"], {"type": "json_object"})

    @patch("requests.post")
    def test_authentication_failure_raises_nebius_auth_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = f"Invalid API key: {self.secret_key}"
        mock_post.return_value = mock_resp

        with self.assertRaises(NebiusAuthError) as ctx:
            self.client.generate_text("Hello")
        # Ensure key was redacted from the exception
        err_msg = str(ctx.exception)
        self.assertNotIn(self.secret_key, err_msg)
        self.assertIn("[REDACTED]", err_msg)

    @patch("time.sleep", return_value=None)
    @patch("requests.post")
    def test_rate_limit_retry_then_success(self, mock_post, mock_sleep):
        # 1st attempt: 429; 2nd attempt: 200
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.text = "Rate limit exceeded"

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {
            "choices": [{"message": {"content": "Recovered response"}}]
        }

        mock_post.side_effect = [resp_429, resp_200]
        result = self.client.generate_text("Query after rate limit")
        self.assertEqual(result, "Recovered response")
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once()

    @patch("time.sleep", return_value=None)
    @patch("requests.post")
    def test_server_error_exhaustion_raises_nebius_error(self, mock_post, mock_sleep):
        resp_500 = MagicMock()
        resp_500.status_code = 500
        resp_500.text = "Internal Server Error"
        mock_post.return_value = resp_500

        with self.assertRaises(NebiusError):
            self.client.generate_text("Query")
        self.assertEqual(mock_post.call_count, 2)

    @patch("requests.post")
    def test_timeout_raises_nebius_timeout_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")
        with self.assertRaises(NebiusTimeoutError):
            self.client.generate_text("Query")

    @patch("requests.post")
    def test_malformed_empty_choices_raises_nebius_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": []}
        mock_post.return_value = mock_resp

        with self.assertRaises(NebiusError):
            self.client.generate_text("Query")


class TestReasoningStripping(unittest.TestCase):
    """Test chain-of-thought (<think>...</think>) stripping for reasoning models."""

    def test_strip_reasoning_closed_tags(self):
        raw = "<think>\nLet me analyze the company database.\nTarget is location.\n</think>Chennai"
        stripped = strip_reasoning(raw)
        self.assertEqual(stripped, "Chennai")

    def test_strip_reasoning_multiline_json(self):
        raw = (
            "<think>\nNeed to output valid plan.\nStep 1: company_knowledge.\n</think>\n"
            '{"goal": "Check location", "steps": [{"tool": "company_knowledge", "input": "location"}]}'
        )
        stripped = strip_reasoning(raw)
        self.assertTrue(stripped.startswith('{"goal": "Check location"'))
        self.assertNotIn("<think>", stripped)
        self.assertNotIn("Need to output valid plan", stripped)

    def test_strip_reasoning_case_insensitive(self):
        raw = "<THINK>internal thought</THINK>Output answer."
        self.assertEqual(strip_reasoning(raw), "Output answer.")

    def test_strip_reasoning_unclosed_trailing(self):
        raw = "Partial answer <think>thought that was cut off"
        self.assertEqual(strip_reasoning(raw), "Partial answer")


class TestSecretRedaction(unittest.TestCase):
    """Verify that sensitive API keys are strictly redacted from errors and logs."""

    def test_redact_secrets_replaces_key(self):
        secret = "nb-token-abc123xyz"
        text = f"Connection failed to Nebius using Bearer {secret}."
        redacted = redact_secrets(text, secret)
        self.assertNotIn(secret, redacted)
        self.assertIn("Bearer [REDACTED]", redacted)

    def test_redact_secrets_noop_when_empty(self):
        self.assertEqual(redact_secrets("Normal text", ""), "Normal text")
        self.assertEqual(redact_secrets("Normal text", None), "Normal text")


class TestProviderFactory(unittest.TestCase):
    """Test get_llm_client() switching between providers."""

    @patch.dict(os.environ, {"LLM_PROVIDER": "ollama"})
    def test_default_ollama_provider(self):
        client = get_llm_client()
        self.assertIsInstance(client, OllamaClient)
        self.assertEqual(client.provider_name, "ollama")

    @patch.dict(os.environ, {"LLM_PROVIDER": "nebius", "NEBIUS_API_KEY": "fake_test_key"})
    def test_nebius_provider_selected(self):
        client = get_llm_client()
        self.assertIsInstance(client, NebiusClient)
        self.assertEqual(client.provider_name, "nebius")
        self.assertEqual(client.model_name, "nvidia/nemotron-3-super-120b-a12b")

    @patch.dict(os.environ, {"LLM_PROVIDER": "nebius", "NEBIUS_API_KEY": ""})
    def test_nebius_missing_key_raises_auth_error(self):
        with self.assertRaises(NebiusAuthError) as ctx:
            get_llm_client()
        self.assertIn("missing", str(ctx.exception).lower())

    @patch.dict(os.environ, {"LLM_PROVIDER": "unsupported_provider"})
    def test_unsupported_provider_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            get_llm_client()
        self.assertIn("unsupported", str(ctx.exception).lower())


class TestIntentPreservationInPlanner(unittest.TestCase):
    """Verify that planner prompt rules and plans preserve original intent without drifting."""

    def test_planner_preserves_head_of_department_entity(self):
        # Scripted plan preserving the exact role and department
        preserves_json = (
            '{"goal": "Find head of Engineering department", "steps": ['
            '{"tool": "company_knowledge", "input": "Who is the head of the Engineering department?"}'
            ']}'
        )
        fake_llm = FakeLLMClient(plan_json=preserves_json, text_response="")
        planner = AgentPlanner(fake_llm)
        plan_dict = planner.plan("Who is the head of the Engineering department?")

        validated = validate_plan(plan_dict)
        step_input = validated.steps[0].input.lower()
        self.assertIn("head", step_input)
        self.assertIn("engineering", step_input)
        # Verify it did not drift to "headcount"
        self.assertNotIn("headcount", step_input)


class TestHealthAndResponseMetadata(unittest.TestCase):
    """Test health reporting and metadata fields on AgentResponse."""

    def test_agent_response_contains_provider_metadata(self):
        fake_llm = FakeLLMClient(
            plan_json='{"goal": "Find info", "steps": [{"tool": "company_knowledge", "input": "location"}]}',
            text_response="Location is Chennai."
        )
        # Set provider properties on FakeLLMClient
        fake_llm.provider_name = "nebius"
        fake_llm.model_name = "nvidia/nemotron-3-super-120b-a12b"

        orchestrator = AgentOrchestrator(llm_client=fake_llm)
        with patch("app.agent.executor.process_query", return_value={"retrieved_data": [{"category": "company", "key": "location", "value": "Chennai"}], "source": "sqlite"}):
            resp = orchestrator.run("Where is the company located?")

        self.assertEqual(resp.llm_provider, "nebius")
        self.assertEqual(resp.llm_model, "nvidia/nemotron-3-super-120b-a12b")
        self.assertGreaterEqual(resp.llm_calls_count, 1)


    @patch.dict(os.environ, {"LLM_PROVIDER": "nebius", "NEBIUS_API_KEY": ""})
    def test_health_reports_nebius_without_secrets(self):
        health = check_active_llm_health()
        self.assertEqual(health["provider"], "nebius")
        self.assertFalse(health["available"])
        self.assertNotIn("Authorization", str(health))
        self.assertNotIn("Bearer", str(health))


if __name__ == "__main__":
    unittest.main()
