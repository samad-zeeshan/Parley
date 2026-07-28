"""Prometheus metrics. One registry per app instance so tests can build many apps."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.booking_actions = Counter(
            "parley_booking_actions_total", "Booking API actions by action and outcome",
            ["action", "outcome"], registry=self.registry,
        )
        self.turns = Counter(
            "parley_agent_turns_total", "Dialogue turns by detected language variety and next action",
            ["dialect", "action"], registry=self.registry,
        )
        self.grounding_rejections = Counter(
            "parley_grounding_rejections_total", "LLM replies rejected for stating an ungrounded fact",
            registry=self.registry,
        )
        self.tool_rejections = Counter(
            "parley_tool_rejections_total", "Tool calls rejected by schema validation", ["tool"],
            registry=self.registry,
        )
        buckets = (0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0)
        self.stage_seconds = Histogram(
            "parley_stage_seconds", "Latency of each pipeline stage", ["stage"],
            buckets=buckets, registry=self.registry,
        )
