from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ==========================================
# Memory Tool Operation Schemas
# ==========================================

ALLOWED_MEMORY_OPERATIONS = {
    "get_profile",
    "get_progress",
    "set_profile",
    "mark_item_complete",
    "set_onboarding_status",
}


class SetProfileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Optional[str] = Field(default=None, max_length=100)
    department: Optional[str] = Field(default=None, max_length=100)
    preferences: Optional[Dict[str, Any]] = None


class MarkItemCompleteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_id: int = Field(..., ge=1)


class SetOnboardingStatusArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["not_started", "in_progress", "completed"]


class GetProfileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GetProgressArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WebResearchStepArgs(BaseModel):
    """Schema for web_research tool arguments with strict bounds and extra='forbid'."""
    model_config = ConfigDict(extra="forbid")
    query: str = Field(..., min_length=3, max_length=200, description="Web search query string")
    max_results: int = Field(default=3, ge=1, le=5, description="Maximum number of results to fetch (1-5)")


class PlanStep(BaseModel):
    """Single step in an agent plan supporting company_knowledge, memory, or web_research."""

    model_config = ConfigDict(extra="forbid")

    tool: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Name of the tool to invoke ('company_knowledge', 'memory', or 'web_research')",
    )
    input: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Natural language query or input for company_knowledge",
    )
    operation: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Operation identifier for memory tool",
    )
    args: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured arguments for tool operation (extra='forbid', user_id forbidden)",
    )

    @model_validator(mode="after")
    def validate_step(self):
        t = self.tool.strip().lower()
        if t == "company_knowledge":
            if not self.input or not self.input.strip():
                raise ValueError("Step for tool 'company_knowledge' must have a non-empty 'input'.")
        elif t == "memory":
            if not self.operation or not self.operation.strip():
                raise ValueError("Step for tool 'memory' must specify an 'operation'.")
            op = self.operation.strip().lower()
            if op not in ALLOWED_MEMORY_OPERATIONS:
                raise ValueError(
                    f"Disallowed memory operation '{self.operation}'. Allowed: {sorted(ALLOWED_MEMORY_OPERATIONS)}"
                )
            args_dict = self.args or {}
            if "user_id" in args_dict:
                raise ValueError("Plan steps must not specify 'user_id'. user_id is server-injected.")

            if op == "set_profile":
                SetProfileArgs.model_validate(args_dict)
            elif op == "mark_item_complete":
                MarkItemCompleteArgs.model_validate(args_dict)
            elif op == "set_onboarding_status":
                SetOnboardingStatusArgs.model_validate(args_dict)
            elif op == "get_profile":
                GetProfileArgs.model_validate(args_dict)
            elif op == "get_progress":
                GetProgressArgs.model_validate(args_dict)
        elif t == "web_research":
            args_dict = self.args or {}
            if not args_dict and self.input:
                args_dict = {"query": self.input, "max_results": 3}
            if not args_dict:
                raise ValueError("Step for tool 'web_research' must specify 'args' with 'query' and optional 'max_results'.")
            validated_args = WebResearchStepArgs.model_validate(args_dict)
            self.args = validated_args.model_dump()
        return self


class AgentPlan(BaseModel):
    """Structured plan generated for an agent query."""

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Goal or objective of the plan",
    )
    steps: List[PlanStep] = Field(
        ...,
        min_length=1,
        max_length=8,
        description="List of 1 to 8 plan steps to execute sequentially",
    )

    @field_validator("goal")
    @classmethod
    def clean_goal(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Goal cannot be empty or solely whitespace.")
        return s


class ToolResult(BaseModel):
    """Result of executing an agent tool."""

    model_config = ConfigDict(extra="forbid")

    tool: str = Field(..., description="Tool that was executed")
    success: bool = Field(..., description="Whether execution completed without error")
    source: str = Field(default="sqlite", description="Data source provenance")
    data: Optional[Any] = Field(default=None, description="Retrieved structured payload")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class CheckItem(BaseModel):
    """Individual verification assertion result."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of the check")
    passed: bool = Field(..., description="Whether assertion passed")
    detail: str = Field(..., description="Explanation of verification outcome")


class Verification(BaseModel):
    """Deterministic verification report of agent execution and grounding."""

    model_config = ConfigDict(extra="forbid")

    passed: bool = Field(..., description="Overall verification outcome")
    checks: List[CheckItem] = Field(
        default_factory=list,
        description="List of individual verification checks",
    )
    unresolved: List[str] = Field(
        default_factory=list,
        description="Unresolved questions, missing facts, or failed operations",
    )


class AgentRequest(BaseModel):
    """User request payload for the agent endpoint with server-injected user_id."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="User question to process",
    )
    user_id: str = Field(
        default="demo-user",
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_\-]+$",
        description="User identifier injected from request context",
    )

    @field_validator("question")
    @classmethod
    def clean_question(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Question cannot be empty or solely whitespace.")
        return s


class AgentResponse(BaseModel):
    """Comprehensive response returned by the agent orchestrator."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Grounded final answer")
    plan: Optional[AgentPlan] = Field(default=None, description="Executed or attempted plan")
    tool_results: List[ToolResult] = Field(
        default_factory=list,
        description="Results from executed tools",
    )
    verification: Verification = Field(..., description="Deterministic verification report")
    processing_time_ms: float = Field(..., description="Total pipeline latency in milliseconds")
    llm_provider: Optional[str] = Field(default=None, description="Active LLM provider used ('ollama', 'nebius')")
    llm_model: Optional[str] = Field(default=None, description="Active LLM model ID")
    llm_calls_count: Optional[int] = Field(default=None, description="Count of LLM calls executed during processing")
    llm_latency_ms: Optional[List[float]] = Field(
        default=None, description="Per-call latency in milliseconds for LLM calls executed during processing"
    )
    user_id: Optional[str] = Field(default=None, description="Active user ID context")
    memory_used: Optional[bool] = Field(default=None, description="Whether persistent memory tool was utilized")
    memory_writes: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Confirmed memory write operations executed"
    )
    web_used: Optional[bool] = Field(default=None, description="Whether web research tool was executed")
    sources: Optional[List[Dict[str, str]]] = Field(
        default=None, description="Sources cited from external web research"
    )
    web_calls_count: Optional[int] = Field(default=None, description="Count of web research calls executed")

    @property
    def verified(self) -> bool:
        return bool(self.verification and self.verification.passed)

