"""Tests for speculative Discord voice STT helpers."""

import pytest

from gateway.voice_speculative import (
    should_start_speculative,
    voice_texts_compatible,
)


def test_should_start_speculative_requires_length_and_words():
    cfg = {"min_chars": 12, "min_words": 3}
    assert should_start_speculative("hi there", cfg) is False
    assert should_start_speculative("what is the weather", cfg) is True


def test_voice_texts_compatible_prefix_extension():
    assert voice_texts_compatible(
        "what is the weather",
        "what is the weather in seattle",
    )


def test_voice_texts_compatible_rejects_unrelated():
    assert not voice_texts_compatible(
        "tell me a joke",
        "what is the weather in seattle",
    )


def test_voice_defer_uses_session_set_not_stale_metadata():
    """Finalize clears session defer before the agent finishes.

    Auto-TTS gating must follow the runner session set, not event metadata
    left over from the speculative partial turn.
    """
    from types import SimpleNamespace

    runner = SimpleNamespace(
        voice_defer_audio_for_session=lambda _key: False,
    )
    session_key = "discord:1265311103748014195"
    event_metadata = {"voice_defer_audio": True, "voice_speculative_partial": True}

    defer = runner.voice_defer_audio_for_session(session_key)

    assert event_metadata.get("voice_defer_audio") is True
    assert defer is False


def test_speculative_held_reply_skips_should_send_voice_reply_gate():
    """Held replies must be stored even when base auto-TTS owns voice input."""
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._voice_defer_audio_sessions = {"discord:1265311103748014195"}
    runner._voice_held_replies = {}
    runner._voice_mode = {"discord:1265311103748014195": "all"}
    runner.adapters = {}

    event = MessageEvent(
        text="partial",
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="1265311103748014195",
            user_id="1",
            user_name="test",
        ),
        message_type=MessageType.VOICE,
        metadata={"voice_speculative_partial": True, "voice_defer_audio": True},
    )
    session_key = "discord:1265311103748014195"
    response = "Hello from speculative turn"

    assert runner._should_send_voice_reply(event, response, []) is False
    if (
        runner.voice_defer_audio_for_session(session_key)
        and event.metadata.get("voice_speculative_partial")
        and response
        and not response.startswith("Error:")
    ):
        runner._voice_held_replies[session_key] = (response, event)

    assert session_key in runner._voice_held_replies


@pytest.mark.anyio
async def test_finalize_delivers_held_even_when_running_agent_slot_occupied():
    """Finalize can run while the handler is still unwinding."""
    from unittest.mock import AsyncMock, MagicMock

    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._voice_held_replies = {
        "discord:1265311103748014195": ("held reply", MagicMock()),
    }
    runner._voice_defer_audio_sessions = {"discord:1265311103748014195"}
    runner._running_agents = {"discord:1265311103748014195": MagicMock()}
    runner._voice_speculative_config = lambda: {"compatible_ratio": 0.72}
    runner._deliver_held_voice_reply = AsyncMock()

    from gateway.voice_speculative import VoiceUtteranceState

    state = VoiceUtteranceState(
        started_text="hey there",
        session_key="discord:1265311103748014195",
        speculative_started=True,
    )

    await runner._finalize_speculative_voice_turn(
        1265311103748014191,
        210968513685356545,
        "hey there friend",
        state,
    )

    runner._deliver_held_voice_reply.assert_awaited_once_with(
        "discord:1265311103748014195"
    )
    assert "discord:1265311103748014195" not in runner._voice_defer_audio_sessions
