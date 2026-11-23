"""Transaction-correlated control communicator for TPE (F4).

Unlike the stock `FanOutCommunicator`, `TpeTransactionCommunicator` validates
`(instance_generation, job_id, op, op_id, expected_artifact_hash)` and
deduplicates by `responder_id` BEFORE counting responses, preventing late,
stale-generation, or duplicate responses from corrupting active transactions.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar

logger = logging.getLogger(__name__)

TReq = TypeVar("TReq")
TResp = TypeVar("TResp")


@dataclass(frozen=True)
class _ActiveWaiter:
    op: str
    op_id: str
    job_id: str
    instance_generation: int
    expected_artifact_hash: Optional[str]
    expected_responders: Optional[Set[str]]
    fan_out: int
    event: asyncio.Event
    responses: Dict[str, Any]


class TpeTransactionCommunicator(Generic[TReq, TResp]):
    """Serialized, generation- and op_id-correlated fan-out communicator."""

    def __init__(
        self,
        send: Callable[[TReq], None],
        fan_out: int,
        expected_responders: Optional[Set[str]] = None,
    ) -> None:
        self._send = send
        self._fan_out = fan_out
        self._expected_responders = (
            set(expected_responders) if expected_responders is not None else None
        )
        self._lock = asyncio.Lock()
        self._active: Optional[_ActiveWaiter] = None
        self.dropped_response_count: int = 0

    def set_fan_out(
        self, fan_out: int, expected_responders: Optional[Set[str]] = None
    ) -> None:
        self._fan_out = fan_out
        if expected_responders is not None:
            self._expected_responders = set(expected_responders)

    async def call(
        self, req: TReq, *, timeout_s: Optional[float] = None
    ) -> List[TResp]:
        async with self._lock:
            waiter = _ActiveWaiter(
                op=getattr(req, "op"),
                op_id=getattr(req, "op_id"),
                job_id=getattr(req, "job_id"),
                instance_generation=int(getattr(req, "instance_generation")),
                expected_artifact_hash=getattr(req, "expected_artifact_hash", None),
                expected_responders=self._expected_responders,
                fan_out=self._fan_out,
                event=asyncio.Event(),
                responses={},
            )
            self._active = waiter
            try:
                self._send(req)
                if timeout_s is not None:
                    await asyncio.wait_for(waiter.event.wait(), timeout=timeout_s)
                else:
                    await waiter.event.wait()
                return list(waiter.responses.values())
            finally:
                if self._active is waiter:
                    self._active = None

    def handle_recv(self, resp: TResp) -> bool:
        """Accept a response only if it matches the active transaction and is unique."""
        waiter = self._active
        if waiter is None:
            self.dropped_response_count += 1
            logger.debug("TPE communicator dropped late response (no active waiter).")
            return False

        resp_op = getattr(resp, "op", None)
        resp_op_id = getattr(resp, "op_id", None)
        resp_job_id = getattr(resp, "job_id", None)
        resp_gen = getattr(resp, "instance_generation", None)
        responder_id = str(getattr(resp, "responder_id", ""))

        if (
            resp_op != waiter.op
            or resp_op_id != waiter.op_id
            or resp_job_id != waiter.job_id
            or resp_gen != waiter.instance_generation
        ):
            self.dropped_response_count += 1
            logger.warning(
                "TPE communicator dropped mismatched response: "
                "got (op=%r, op_id=%r, job=%r, gen=%r), "
                "expected (op=%r, op_id=%r, job=%r, gen=%r)",
                resp_op,
                resp_op_id,
                resp_job_id,
                resp_gen,
                waiter.op,
                waiter.op_id,
                waiter.job_id,
                waiter.instance_generation,
            )
            return False

        if not responder_id:
            self.dropped_response_count += 1
            logger.warning("TPE communicator dropped response with empty responder_id.")
            return False

        if (
            waiter.expected_responders is not None
            and responder_id not in waiter.expected_responders
        ):
            self.dropped_response_count += 1
            logger.warning(
                "TPE communicator dropped response from unexpected responder %r.",
                responder_id,
            )
            return False

        if responder_id in waiter.responses:
            self.dropped_response_count += 1
            logger.warning(
                "TPE communicator dropped duplicate response from responder %r.",
                responder_id,
            )
            return False

        waiter.responses[responder_id] = resp
        if len(waiter.responses) >= waiter.fan_out:
            waiter.event.set()
        return True
