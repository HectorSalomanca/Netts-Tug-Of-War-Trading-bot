"""
Event Bus — ZeroMQ-based Pub/Sub for Tug-Of-War EDA

Architecture:
  - scout_tape.py (PUBLISHER): broadcasts microstructure events on ipc:///tmp/tow_events.sock
  - engine_v2.py (SUBSCRIBER): listens for high-priority events and reacts in <100ms

Event Types:
  - TRAPPED_EXHAUSTION: OFI flip at breakout peak + volume spike → immediate fade opportunity
  - ICEBERG_DETECTED:   Large hidden order absorbing retail flow → institutional signal
  - STACKED_IMBALANCE:  3:1 aggressive/passive ratio across 3+ ticks → momentum signal
  - OFI_EXTREME:        OFI Z-score > 2.5 or < -2.5 → extreme imbalance
  - SPREAD_BLOW:        Spread widens >3x average → liquidity withdrawal (danger)

Transport: ZeroMQ IPC socket (no external server needed, ~10μs latency)
Fallback:  If ZeroMQ unavailable, events are silently dropped (polling still works)
"""

import json
import time
from datetime import datetime, timezone
from typing import Optional, Callable

try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False

# TCP socket (brokerless, zero-config, HFT-standard)
# Runs on localhost only — no external exposure
EVENT_SOCKET = "tcp://127.0.0.1:5555"

# Event type constants
EVT_TRAPPED_EXHAUSTION = "TRAPPED_EXHAUSTION"
EVT_ICEBERG_DETECTED   = "ICEBERG_DETECTED"
EVT_STACKED_IMBALANCE  = "STACKED_IMBALANCE"
EVT_OFI_EXTREME        = "OFI_EXTREME"
EVT_SPREAD_BLOW        = "SPREAD_BLOW"

# Priority levels (higher = more urgent)
PRIORITY = {
    EVT_TRAPPED_EXHAUSTION: 5,  # highest — immediate fade
    EVT_SPREAD_BLOW:        4,  # danger — pull orders
    EVT_ICEBERG_DETECTED:   3,  # institutional signal
    EVT_OFI_EXTREME:        3,  # extreme imbalance
    EVT_STACKED_IMBALANCE:  2,  # momentum confirmation
}


class EventPublisher:
    """
    Used by scout_tape.py to broadcast microstructure events.
    Non-blocking: if no subscribers, messages are silently dropped.
    """

    def __init__(self):
        self._socket = None
        if HAS_ZMQ:
            try:
                self._ctx = zmq.Context()
                self._socket = self._ctx.socket(zmq.PUB)
                self._socket.bind(EVENT_SOCKET)
                # Allow subscribers time to connect before first publish
                time.sleep(0.1)
                print(f"[EVENT_BUS] Publisher bound to {EVENT_SOCKET}")
            except Exception as e:
                print(f"[EVENT_BUS] Publisher init error (non-fatal): {e}")
                self._socket = None

    def publish(self, event_type: str, symbol: str, data: dict):
        """Broadcast an event. Non-blocking, fire-and-forget."""
        if not self._socket:
            return
        try:
            payload = {
                "event": event_type,
                "symbol": symbol,
                "priority": PRIORITY.get(event_type, 1),
                "ts": datetime.now(timezone.utc).isoformat(),
                **data,
            }
            # Topic-based filtering: prefix message with event type
            topic = f"{event_type}:{symbol}"
            self._socket.send_string(f"{topic} {json.dumps(payload)}", zmq.NOBLOCK)
        except zmq.ZMQError:
            pass  # subscriber not connected — silently drop

    def close(self):
        if self._socket:
            self._socket.close()


class EventSubscriber:
    """
    Used by engine_v2.py to receive microstructure events.
    Non-blocking poll: check for events without blocking the main loop.
    """

    def __init__(self, topics: Optional[list] = None):
        self._socket = None
        self._handlers: dict = {}
        if HAS_ZMQ:
            try:
                self._ctx = zmq.Context()
                self._socket = self._ctx.socket(zmq.SUB)
                self._socket.connect(EVENT_SOCKET)
                # Subscribe to specific topics or all
                if topics:
                    for t in topics:
                        self._socket.setsockopt_string(zmq.SUBSCRIBE, t)
                else:
                    self._socket.setsockopt_string(zmq.SUBSCRIBE, "")  # all events
                self._socket.setsockopt(zmq.RCVTIMEO, 0)  # non-blocking
                print(f"[EVENT_BUS] Subscriber connected to {EVENT_SOCKET}")
            except Exception as e:
                print(f"[EVENT_BUS] Subscriber init error (non-fatal): {e}")
                self._socket = None

    def register_handler(self, event_type: str, handler: Callable):
        """Register a callback for a specific event type."""
        self._handlers[event_type] = handler

    def poll(self, max_events: int = 50) -> list:
        """
        Non-blocking poll: drain up to max_events from the socket.
        Returns list of parsed event dicts, sorted by priority (highest first).
        """
        if not self._socket:
            return []

        events = []
        for _ in range(max_events):
            try:
                raw = self._socket.recv_string(zmq.NOBLOCK)
                # Parse: "TOPIC:SYMBOL {json_payload}"
                space_idx = raw.index(" ")
                payload = json.loads(raw[space_idx + 1:])
                events.append(payload)
            except zmq.Again:
                break  # no more messages
            except (ValueError, json.JSONDecodeError):
                continue

        # Sort by priority (highest first)
        events.sort(key=lambda e: e.get("priority", 0), reverse=True)

        # Dispatch to registered handlers
        for evt in events:
            handler = self._handlers.get(evt.get("event"))
            if handler:
                try:
                    handler(evt)
                except Exception as e:
                    print(f"[EVENT_BUS] Handler error for {evt.get('event')}: {e}")

        return events

    def close(self):
        if self._socket:
            self._socket.close()


# Singleton instances (lazy-initialized)
_publisher: Optional[EventPublisher] = None
_subscriber: Optional[EventSubscriber] = None


def get_publisher() -> EventPublisher:
    global _publisher
    if _publisher is None:
        _publisher = EventPublisher()
    return _publisher


def get_subscriber(topics: Optional[list] = None) -> EventSubscriber:
    global _subscriber
    if _subscriber is None:
        _subscriber = EventSubscriber(topics)
    return _subscriber
