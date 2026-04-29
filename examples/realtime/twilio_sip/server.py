"""Minimal FastAPI server for handling OpenAI Realtime SIP calls with Twilio."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import websockets
from fastapi import FastAPI, HTTPException, Request, Response
from openai import APIStatusError, AsyncOpenAI, InvalidWebhookSignatureError

from agents.realtime.config import RealtimeSessionModelSettings
from agents.realtime.items import (
    AssistantAudio,
    AssistantMessageItem,
    AssistantText,
    InputText,
    UserMessageItem,
)
from agents.realtime.model_inputs import RealtimeModelSendRawMessage
from agents.realtime.openai_realtime import OpenAIRealtimeSIPModel
from agents.realtime.runner import RealtimeRunner
from agents.realtime import RealtimeRawModelEvent
from agents.realtime.model_events import RealtimeModelRawServerEvent

from .agents import WELCOME_MESSAGE, get_starting_agent

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("twilio_sip_example")


# ---------------------------------------------------------------------------
# Configuration for consent/disclosure and greeting messages
# Set CONSENT_MESSAGE to None or empty string to skip consent and go directly to greeting
# ---------------------------------------------------------------------------
CONSENT_MESSAGE: str | None = os.getenv("CONSENT_MESSAGE", None)
GREETING_MESSAGE: str = os.getenv("GREETING_MESSAGE", WELCOME_MESSAGE)


def _get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


OPENAI_API_KEY = _get_env("OPENAI_API_KEY")
OPENAI_WEBHOOK_SECRET = _get_env("OPENAI_WEBHOOK_SECRET")

client = AsyncOpenAI(api_key=OPENAI_API_KEY, webhook_secret=OPENAI_WEBHOOK_SECRET)

# Build the multi-agent graph (triage + specialist agents) from agents.py.
assistant_agent = get_starting_agent()

app = FastAPI()

# Track background tasks so repeated webhooks do not spawn duplicates.
active_call_tasks: dict[str, asyncio.Task[None]] = {}


DEFAULT_TURN_DETECTION_CONFIG: dict[str, Any] = {
    "type": "server_vad",
    "silence_duration_ms": 500,
    "threshold": 0.5,
    "prefix_padding_ms": 300,
}


async def send_session_update(session: Any, turn_detection: dict[str, Any] | None) -> None:
    """Enable or disable turn detection by sending a session.update raw message.

    Args:
        session: The active RealtimeSession.
        turn_detection: The turn detection config dict, or None to disable.
    """
    await session.model.send_event(
        RealtimeModelSendRawMessage(
            message={
                "type": "session.update",
                "other_data": {
                    "session": {
                        "type": "realtime",
                        "audio": {
                            "input": {
                                "turn_detection": turn_detection,
                            },
                        },
                    },
                },
            },
        ),
    )


async def send_response_create(session: Any, instructions: str) -> None:
    """Send a response.create event to trigger speech generation.

    Args:
        session: The active RealtimeSession.
        instructions: The instructions for what the model should say.
    """
    await session.model.send_event(
        RealtimeModelSendRawMessage(
            message={
                "type": "response.create",
                "other_data": {
                    "response": {
                        "instructions": instructions,
                    },
                },
            },
        )
    )


# ---------------------------------------------------------------------------
# First Response Manager (self-contained, from first_response.py logic)
# ---------------------------------------------------------------------------
class FirstResponseHandler:
    """Manages the first bot response including consent and greeting flows.

    Handles:
    - Consent message delivery with turn detection toggling
    - Listening for consent completion events
    - Initiating the greeting after consent (or directly if no consent is configured)
    """

    def __init__(
        self,
        session: Any,
        consent_message: str | None,
        greeting_message: str,
        call_logger: logging.Logger,
    ) -> None:
        self._session = session
        self._consent_message = consent_message
        self._greeting_message = greeting_message
        self._logger = call_logger
        self._consent_initiated: bool = False
        self._first_response_complete: bool = False

    async def handle_event(self, event: Any) -> None:
        """Event handler that restores turn detection after consent response completes."""
        if self._consent_initiated:
            return

        # Check for consent completion events (voice or text channel)
        if not (
            isinstance(event, RealtimeRawModelEvent)
            and isinstance(event.data, RealtimeModelRawServerEvent)
            and event.data.data.get("type") in ("output_audio_buffer.stopped", "response.output_text.done")
        ):
            return

        # Re-enable turn detection after consent message is spoken
        self._logger.info("Consent response completed - re-enabling turn detection and initiating greeting.")
        await send_session_update(self._session, DEFAULT_TURN_DETECTION_CONFIG)
        self._consent_initiated = True
        await self._initiate_greeting()

    async def _initiate_greeting(self) -> None:
        """Send the greeting message."""
        await send_response_create(
            self._session,
            instructions=(
                f"Say exactly the following greeting only once: {self._greeting_message}. "
                f"If you were interrupted while saying it, do NOT repeat it. "
                f"Proceed to the next step instead."
            ),
        )

    async def trigger_first_response(self) -> None:
        """Trigger the first bot response by delivering the consent message or greeting directly."""
        if self._consent_message:
            # Disable turn detection during consent so user cannot interrupt
            self._logger.info("Disabling turn detection for consent message delivery.")
            await send_session_update(self._session, None)

            # Deliver the consent/disclosure message
            await send_response_create(
                self._session,
                instructions=(
                    f'Say exactly the following disclosure, word for word, with nothing else added: '
                    f'"{self._consent_message}"'
                ),
            )
        else:
            # No consent message - initiate greeting directly
            self._logger.info("No consent message configured - initiating greeting directly.")
            await self._initiate_greeting()
            self._first_response_complete = True

    @property
    def needs_event_handling(self) -> bool:
        """Returns True if we need to listen for consent completion events."""
        return bool(self._consent_message) and not self._consent_initiated


async def accept_call(call_id: str) -> None:
    """Accept the incoming SIP call and configure the realtime session."""

    # The starting agent uses static instructions, so we can forward them directly to the accept
    # call payload. If someone swaps in a dynamic prompt, fall back to a sensible default.
    instructions_payload = (
        assistant_agent.instructions
        if isinstance(assistant_agent.instructions, str)
        else "You are a helpful triage agent for ABC customer service."
    )

    try:
        # AsyncOpenAI does not yet expose high-level helpers like client.realtime.calls.accept, so
        # we call the REST endpoint directly via client.post(). Keep this until the SDK grows an
        # async helper.
        await client.post(
            f"/realtime/calls/{call_id}/accept",
            body={
                "type": "realtime",
                "model": "gpt-realtime-1.5",
                "instructions": instructions_payload,
            },
            cast_to=dict,
        )
    except APIStatusError as exc:
        if exc.status_code == 404:
            # Twilio occasionally retries webhooks after the caller hangs up; treat as a no-op so
            # the webhook still returns 200.
            logger.warning(
                "Call %s no longer exists when attempting accept (404). Skipping.", call_id
            )
            return

        detail = exc.message
        if exc.response is not None:
            try:
                detail = exc.response.text
            except Exception:  # noqa: BLE001
                detail = str(exc.response)

        logger.error("Failed to accept call %s: %s %s", call_id, exc.status_code, detail)
        raise HTTPException(status_code=500, detail="Failed to accept call") from exc

    logger.info("Accepted call %s", call_id)


async def observe_call(call_id: str) -> None:
    """Attach to the realtime session and log conversation events."""

    runner = RealtimeRunner(assistant_agent, model=OpenAIRealtimeSIPModel())

    try:
        initial_model_settings: RealtimeSessionModelSettings = {
            "turn_detection": DEFAULT_TURN_DETECTION_CONFIG,
        }
        async with await runner.run(
            model_config={
                "call_id": call_id,
                "initial_model_settings": initial_model_settings,
            }
        ) as session:
            # Initialize the first response handler for consent/greeting flow
            first_response_handler = FirstResponseHandler(
                session=session,
                consent_message=CONSENT_MESSAGE,
                greeting_message=GREETING_MESSAGE,
                call_logger=logger,
            )

            # Trigger the first response (consent or greeting)
            await first_response_handler.trigger_first_response()

            async for event in session:
                # Handle consent completion if needed (re-enables turn detection and triggers greeting)
                if first_response_handler.needs_event_handling:
                    await first_response_handler.handle_event(event)

                if event.type == "history_added":
                    item = event.item
                    if isinstance(item, UserMessageItem):
                        for user_content in item.content:
                            if isinstance(user_content, InputText) and user_content.text:
                                logger.info("Caller: %s", user_content.text)
                    elif isinstance(item, AssistantMessageItem):
                        for assistant_content in item.content:
                            if (
                                isinstance(assistant_content, AssistantText)
                                and assistant_content.text
                            ):
                                logger.info("Assistant (text): %s", assistant_content.text)
                            elif (
                                isinstance(assistant_content, AssistantAudio)
                                and assistant_content.transcript
                            ):
                                logger.info(
                                    "Assistant (audio transcript): %s",
                                    assistant_content.transcript,
                                )
                elif event.type == "error":
                    logger.error("Realtime session error: %s", event.error)

    except websockets.exceptions.ConnectionClosedError:
        # Callers hanging up causes the WebSocket to close without a frame; log at info level so it
        # does not surface as an error.
        logger.info("Realtime WebSocket closed for call %s", call_id)
    except Exception as exc:  # noqa: BLE001 - demo logging only
        logger.exception("Error while observing call %s", call_id, exc_info=exc)
    finally:
        logger.info("Call %s ended", call_id)
        active_call_tasks.pop(call_id, None)


def _track_call_task(call_id: str) -> None:
    existing = active_call_tasks.get(call_id)
    if existing:
        if not existing.done():
            logger.info(
                "Call %s already has an active observer; ignoring duplicate webhook delivery.",
                call_id,
            )
            return
        # Remove completed tasks so a new observer can start for a fresh call.
        active_call_tasks.pop(call_id, None)

    task = asyncio.create_task(observe_call(call_id))
    active_call_tasks[call_id] = task


@app.post("/openai/webhook")
async def openai_webhook(request: Request) -> Response:
    body = await request.body()

    try:
        event = client.webhooks.unwrap(body, request.headers)
    except InvalidWebhookSignatureError as exc:
        raise HTTPException(status_code=400, detail="Invalid webhook signature") from exc

    if event.type == "realtime.call.incoming":
        call_id = event.data.call_id
        await accept_call(call_id)
        _track_call_task(call_id)
        return Response(status_code=200)

    # Ignore other webhook event types for brevity.
    return Response(status_code=200)


@app.get("/")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
