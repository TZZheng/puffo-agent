from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Coroutine, Mapping, TypedDict

import websockets
import websockets.exceptions  # websockets>=16 lazy-loads submodules; bare `import websockets` doesn't bind `.exceptions`

from .encoding import base64url_encode, generate_nonce
from .http_client import PuffoCoreHttpClient
from .keystore import KeyStore, decode_secret
from .primitives import Ed25519KeyPair

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT = 5.0
MAX_BACKOFF = 30
INITIAL_BACKOFF = 1


def _http_to_ws(url: str) -> str:
    if url.startswith("https://"):
        return "wss://" + url[len("https://"):]
    if url.startswith("http://"):
        return "ws://" + url[len("http://"):]
    return url


def _now_ms() -> int:
    return int(time.time() * 1000)


class ServerDelivery(TypedDict):
    seq: int
    envelope: dict[str, Any]


class TransportOutcome(str, Enum):
    ACK = "ack"
    HOLD = "hold"


@dataclass(frozen=True)
class DeliveryResult:
    outcome: TransportOutcome
    envelope_id: str | None
    reason: str


MessageHandler = Callable[[ServerDelivery], Awaitable[TransportOutcome]]
EventHandler = Callable[[str, dict], Coroutine[Any, Any, None]]
CertHandler = Callable[[dict], Coroutine[Any, Any, None]]


class PuffoCoreWsClient:
    def __init__(
        self,
        server_url: str,
        keystore: KeyStore,
        slug: str,
        http_client: PuffoCoreHttpClient,
    ):
        self.ws_url = _http_to_ws(server_url.rstrip("/")) + "/subscribe"
        self.keystore = keystore
        self.slug = slug
        self.http_client = http_client
        self._ws: Any = None
        self._running = False
        self.session_id: str | None = None

        self.on_message: MessageHandler | None = None
        self.on_event: EventHandler | None = None
        self.on_cert_update: CertHandler | None = None
        # Fires after every (re)connect handshake, before catch-up.
        self.on_connect: Callable[[], Coroutine[Any, Any, None]] | None = None

    def _build_connect_frame(self) -> str:
        sess = self.keystore.load_session(self.slug)
        key = Ed25519KeyPair.from_secret_bytes(decode_secret(sess.subkey_secret_key))
        nonce = generate_nonce()
        timestamp = _now_ms()

        message = f"ws-connect\n{self.slug}\n{sess.subkey_id}\n{nonce}\n{timestamp}".encode()
        sig = key.sign(message)

        return json.dumps({
            "type": "connect",
            "slug": self.slug,
            "subkey_id": sess.subkey_id,
            "nonce": nonce,
            "timestamp": timestamp,
            "signature": base64url_encode(sig),
        })

    async def _handshake(self, ws: Any) -> str:
        frame = self._build_connect_frame()
        await ws.send(frame)

        raw = await asyncio.wait_for(ws.recv(), timeout=CONNECT_TIMEOUT)
        resp = json.loads(raw)
        if resp.get("type") != "connected":
            raise ConnectionError(f"Unexpected handshake response: {resp}")
        return resp["session_id"]

    async def _catchup(self) -> None:
        """Drain `/messages/pending` — rows the server held while we
        were offline. MUST run after ``_handshake`` (it registers the
        session); ordering is register → pending fetch → live listen.
        """
        try:
            data = await self.http_client.get("/messages/pending")
            messages = data.get("messages", [])
            # Log on every reconnect, even zero — a silent reconnect
            # should still leave one line (else it's indistinguishable
            # from a no-op).
            logger.info(
                "Catch-up: %d pending messages (session=%s)",
                len(messages), self.session_id or "<pre-handshake>",
            )
            if not messages:
                return
            # HTTP, not WS-frame (no pong until the listen loop → WS dies
            # mid-catch-up); chunked so progress survives a death.
            envelope_ids = []
            for item in messages:
                result = await self.dispatch_delivery(item)
                if result.outcome is TransportOutcome.ACK and result.envelope_id:
                    envelope_ids.append(result.envelope_id)
                if len(envelope_ids) >= 25:
                    await self._ack_http(envelope_ids)
                    envelope_ids = []
            if envelope_ids:
                await self._ack_http(envelope_ids)
        except Exception:
            logger.exception("Catch-up failed")

    async def _ack_http(self, envelope_ids: list[str]) -> None:
        try:
            await self.http_client.post(
                "/messages/ack", {"envelope_ids": list(envelope_ids)},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "catch-up HTTP ack failed (%d ids): %s", len(envelope_ids), exc,
            )

    async def _send_ack(self, envelope_ids: list[str]) -> None:
        if self._ws:
            await self._ws.send(json.dumps({
                "type": "ack",
                "envelope_ids": envelope_ids,
            }))

    async def dispatch_delivery(self, item: Mapping[str, Any]) -> DeliveryResult:
        """Validate and classify one Server delivery without sending an ACK.

        Both HTTP catch-up and live WebSocket delivery use this seam.  Callers
        may also reuse it for another durable delivery source.
        """
        if not isinstance(item, Mapping):
            return DeliveryResult(TransportOutcome.HOLD, None, "delivery is not a mapping")
        seq = item.get("seq")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq <= 0:
            return DeliveryResult(TransportOutcome.HOLD, None, "invalid server sequence")
        raw_envelope = item.get("envelope")
        if not isinstance(raw_envelope, Mapping):
            return DeliveryResult(TransportOutcome.HOLD, None, "invalid envelope")
        envelope = dict(raw_envelope)
        envelope_id = envelope.get("envelope_id")
        if not isinstance(envelope_id, str) or not envelope_id.strip():
            return DeliveryResult(TransportOutcome.HOLD, None, "invalid envelope id")
        if self.on_message is None:
            return DeliveryResult(TransportOutcome.HOLD, envelope_id, "no message handler")

        delivery: ServerDelivery = {"seq": seq, "envelope": envelope}
        try:
            outcome = await self.on_message(delivery)
        except Exception:
            logger.exception("on_message callback failed")
            return DeliveryResult(TransportOutcome.HOLD, envelope_id, "handler exception")
        if outcome is TransportOutcome.ACK:
            return DeliveryResult(TransportOutcome.ACK, envelope_id, "handler committed")
        if outcome is TransportOutcome.HOLD:
            return DeliveryResult(TransportOutcome.HOLD, envelope_id, "handler held")
        return DeliveryResult(TransportOutcome.HOLD, envelope_id, "unexpected handler result")

    async def _handle_frame(self, raw: str) -> None:
        msg = json.loads(raw)
        if not isinstance(msg, Mapping):
            logger.warning("ignoring malformed WebSocket frame")
            return
        msg_type = msg.get("type")

        if msg_type == "ping":
            await self._ws.send(json.dumps({"type": "pong"}))

        elif msg_type == "message":
            result = await self.dispatch_delivery(msg)
            if result.outcome is TransportOutcome.ACK and result.envelope_id:
                await self._send_ack([result.envelope_id])

        elif msg_type == "cert_update":
            if self.on_cert_update:
                try:
                    await self.on_cert_update(msg.get("entry", {}))
                except Exception:
                    logger.exception("on_cert_update callback failed")

        elif msg_type == "event":
            if self.on_event:
                try:
                    await self.on_event(msg.get("scope", ""), msg.get("event", {}))
                except Exception:
                    logger.exception("on_event callback failed")

    async def _listen_loop(self) -> None:
        async for raw in self._ws:
            await self._handle_frame(raw)

    async def connect_once(self) -> None:
        await self.http_client._ensure_subkey()
        async with websockets.connect(self.ws_url) as ws:
            self._ws = ws
            self.session_id = await self._handshake(ws)
            logger.info("[%s] WS connected, session=%s", self.slug, self.session_id)
            if self.on_connect:
                try:
                    await self.on_connect()
                except Exception:
                    logger.exception("on_connect callback failed")
            await self._catchup()
            await self._listen_loop()

    async def run(self) -> None:
        self._running = True
        backoff = INITIAL_BACKOFF
        while self._running:
            try:
                await self.connect_once()
                backoff = INITIAL_BACKOFF
            except (
                websockets.exceptions.ConnectionClosed,
                ConnectionError,
                OSError,
                asyncio.TimeoutError,
            ) as exc:
                if not self._running:
                    break
                logger.warning(
                    "[%s] WS disconnected (%s: %s), reconnecting in %ds",
                    self.slug, type(exc).__name__, exc, backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
            except Exception:
                if not self._running:
                    break
                logger.exception("[%s] WS unexpected error, reconnecting in %ds", self.slug, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
        self._ws = None
        self.session_id = None

    def stop(self) -> None:
        self._running = False
        if self._ws:
            asyncio.ensure_future(self._ws.close())
