from __future__ import annotations

from tests.conftest import build_test_agent_loop, build_test_vibe_app, build_test_vibe_config
from vibe.core.tools.base import ToolPermission
from vibe.core.tools.builtins.bash import BashArgs, BashToolConfig


def test_bash_approval_uses_command_pattern_when_available() -> None:
    agent_loop = build_test_agent_loop(
        config=build_test_vibe_config(
            system_prompt_id="tests",
            include_project_context=False,
            include_prompt_detail=False,
        )
    )
    app = build_test_vibe_app(agent_loop=agent_loop, config=agent_loop.config)

    app._persist_tool_approval_selection(
        "bash",
        BashArgs(
            command="uv run pytest tests/tools/test_bash.py -q",
            command_pattern=["uv", "run", "pytest"],
        ),
    )

    bash_config = agent_loop.tool_manager.get_tool_config("bash")

    assert isinstance(bash_config, BashToolConfig)
    assert bash_config.permission is ToolPermission.ASK
    assert "uv run pytest" in bash_config.command_patterns


def test_bash_approval_falls_back_to_tool_permission_without_command_pattern() -> None:
    agent_loop = build_test_agent_loop(
        config=build_test_vibe_config(
            system_prompt_id="tests",
            include_project_context=False,
            include_prompt_detail=False,
        )
    )
    app = build_test_vibe_app(agent_loop=agent_loop, config=agent_loop.config)

    app._persist_tool_approval_selection(
        "bash",
        BashArgs(command="git status --short"),
    )

    bash_config = agent_loop.tool_manager.get_tool_config("bash")

    assert isinstance(bash_config, BashToolConfig)
    assert bash_config.permission is ToolPermission.ALWAYS
    assert bash_config.command_patterns == []
