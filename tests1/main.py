import asyncio
import base64
import json
import os
from pathlib import Path
from typing import AsyncIterable

from dotenv import load_dotenv
from fastapi import FastAPI, Query, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.run_config import RunConfig
from google.adk.events.event import Event
from google.adk.runners import Runner
# from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
from google.adk.memory import VertexAiMemoryBankService
import global_queue
from google.adk.sessions import VertexAiSessionService

from bajaj_agents.insurance_agent import agent
# from .bajaj_agents.renewal_agent.agent import root_agent
from vertexai import agent_engines
agent_engine = agent_engines.create()
agent_engine_id = agent_engine.resource_name.split("/")[-1]

memory_bank_service = VertexAiMemoryBankService(
    project="onyx-syntax-451614-t2",
    location="us-central1",
    agent_engine_id=agent_engine_id
)
agent_id = agent_engine.name

AGENT_MAP = {"bagic agent": agent}

APP_NAME = "BAGIC App example"
session_service = VertexAiSessionService(
    project="onyx-syntax-451614-t2",
    location="us-central1",
    agent_engine_id=agent_engine_id

)
# session_service = InMemorySessionService()
# session_service_2 = InMemorySessionService()
load_dotenv()


async def start_agent_session(session_id, agent, is_audio=False):
    """Starts an agent session"""

    # Create a Session
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=session_id,
        # session_id=session_id,
    )
    vertex_session_id = session.id  

    current_agent = agent

    # Create a Runner
    runner = Runner(
        app_name=APP_NAME,
        agent=current_agent,
        session_service=session_service,
        memory_service=memory_bank_service
    )


    modality = "AUDIO" if is_audio else "TEXT"

    speech_config = types.SpeechConfig(
        language_code="en-IN",
        voice_config=types.VoiceConfig(
            # Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, and Zephyr
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
        ),
    )

    config = {"response_modalities": [modality], "speech_config": speech_config}

    if is_audio:
        config["output_audio_transcription"] = {}
        config["input_audio_transcription"] = {}

    run_config = RunConfig(**config)

    # Create a LiveRequestQueue for this session
    live_request_queue = LiveRequestQueue()


    # Start agent session
    live_events = runner.run_live(
        session=session,
        live_request_queue=live_request_queue,
        run_config=run_config,
    )


    return live_events, live_request_queue


# NEW: Handler for Agent 1 (your original logic)
async def handle_agent1_events(websocket: WebSocket, live_events: AsyncIterable[Event]):
    """Agent 1 to client communication"""
    async for event in live_events:
        if event is None:
            continue

        # If the turn complete or interrupted, send it
        if event.turn_complete or event.interrupted:
            message = {
                "turn_complete": event.turn_complete,
                "interrupted": event.interrupted,
            }
            await websocket.send_text(json.dumps(message))
            print(f"[AGENT 1 TO CLIENT]: {message}")
            continue

        # Read the Content and its first Part
        part = event.content and event.content.parts and event.content.parts[0]
        if not part:
            continue

        # Make sure we have a valid Part
        if not isinstance(part, types.Part):
            continue

        if event.content.role == "user" and part.text and not event.partial:
            message = {
                "mime_type": "text/plain",
                "data": part.text,
                "role": "user",
            }
            await websocket.send_text(json.dumps(message))
            print(f"[AGENT 1 TO CLIENT]: text/plain: {part.text}")

        if event.content.role == "model" and part.text and not event.partial:
            message = {
                "mime_type": "text/plain",
                "data": part.text,
                "role": "model",
            }
            await websocket.send_text(json.dumps(message))
            print(f"[AGENT 1 TO CLIENT]: text/plain: {part.text}")

        # If it's audio, send Base64 encoded audio data
        is_audio = (
            part.inline_data
            and part.inline_data.mime_type
            and part.inline_data.mime_type.startswith("audio/pcm")
        )
        if is_audio:
            audio_data = part.inline_data and part.inline_data.data
            if audio_data:
                message = {
                    "mime_type": "audio/pcm",
                    "data": base64.b64encode(audio_data).decode("ascii"),
                    "role": "model",
                }
                await websocket.send_text(json.dumps(message))
                print(f"[AGENT 1 TO CLIENT]: audio/pcm: {len(audio_data)} bytes.")


async def client_to_agent_messaging(
    websocket: WebSocket,
    live_request_queue: LiveRequestQueue
):
    """Client to agent communication"""
    while True:
        # Decode JSON message
        message_json = await websocket.receive_text()
        message = json.loads(message_json)
        mime_type = message["mime_type"]
        data = message["data"]
        role = message.get("role", "user")  # Default to 'user' if role is not provided

        # Send the message to the agent
        if mime_type == "text/plain":
            # Send a text message
            content = types.Content(role=role, parts=[types.Part.from_text(text=data)])
            live_request_queue.send_content(content=content)
            print(f"[CLIENT TO AGENT 1]: {data}")
        elif mime_type == "audio/pcm":
            # Send audio data
            decoded_data = base64.b64decode(data)

            # Send audio to both agents
            live_request_queue.send_realtime(
                types.Blob(data=decoded_data, mime_type=mime_type)
            )
            # live_request_queue_2.send_realtime(
            #     types.Blob(data=decoded_data, mime_type=mime_type)
            # )
            print(
                f"[CLIENT TO AGENTS]: audio/pcm: {len(decoded_data)} bytes to both agents"
            )

        else:
            raise ValueError(f"Mime type not supported: {mime_type}")


app = FastAPI()

app.mount("/static", StaticFiles(directory=Path("fe/dist")), name="root")
app.mount("/assets", StaticFiles(directory=Path("fe/dist/assets")), name="assets")


@app.get("/")
async def root():
    """Serves the index.html"""
    return FileResponse(os.path.join(Path("fe/dist"), "index.html"))


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    is_audio: str = Query(...),
    agent: str = Query(...),
):
    """Client websocket endpoint"""
    await websocket.accept()
    print(f"Client #{session_id} connected, audio mode: {is_audio}")

    # Start agent sessions
    live_events, live_request_queue = (
        await start_agent_session(session_id, agent=agent, is_audio=is_audio == "true")
    )

    global_queue.global_live_request_queue = live_request_queue

    # Create a task for each concurrent operation
    agent1_to_client_task = asyncio.create_task(
        handle_agent1_events(websocket, live_events)
    )
    client_to_agents_task = asyncio.create_task(
        client_to_agent_messaging(websocket, live_request_queue)
    )

    # Run all tasks concurrently. This will only return when one of the
    # tasks finishes or raises an exception.
    await asyncio.gather(
        agent1_to_client_task, client_to_agents_task
    )

    # Disconnected
    print(f"Client #{session_id} disconnected")