from __future__ import annotations

from acp import RequestPermissionResponse
from acp.schema import AllowedOutcome
import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from vibe.acp.acp_agent_loop import AcpSessionLoop, VibeAcpAgentLoop
from vibe.acp.utils import ToolOption
from vibe.core.tools.base import ToolPermission
from vibe.core.tools.builtins.bash import BashArgs, BashToolConfig
from vibe.core.types import ApprovalResponse


@pytest.mark.asyncio
async def test_acp_allow_always_uses_bash_command_pattern(
    acp_agent_loop: VibeAcpAgentLoop, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = "session-bash-pattern"
    agent_loop = build_test_agent_loop(
        config=build_test_vibe_config(
            system_prompt_id="tests",
            include_project_context=False,
            include_prompt_detail=False,
        )
    )
    acp_agent_loop.sessions[session_id] = AcpSessionLoop(
        id=session_id, agent_loop=agent_loop
    )

    async def request_permission(**_kwargs) -> RequestPermissionResponse:
        return RequestPermissionResponse(
            outcome=AllowedOutcome(
                outcome="selected", option_id=ToolOption.ALLOW_ALWAYS
            )
        )

    monkeypatch.setattr(acp_agent_loop.client, "request_permission", request_permission)

    approval_callback = acp_agent_loop._create_approval_callback(session_id)
    response, feedback = await approval_callback(
        "bash",
        BashArgs(
            command="uv run pytest tests/tools/test_bash.py -q",
            command_pattern=["uv", "run", "pytest"],
        ),
        "call-bash-pattern",
    )

    bash_config = agent_loop.tool_manager.get_tool_config("bash")

    assert response is ApprovalResponse.YES
    assert feedback is None
    assert isinstance(bash_config, BashToolConfig)
    assert bash_config.permission is ToolPermission.ASK
    assert "uv run pytest" in bash_config.command_patterns
