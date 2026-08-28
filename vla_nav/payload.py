"""Building the JSON that goes on ``drone/{id}/cmd/vla``.

The repo's own README is explicit that "Payload JSON phải khớp schema phía
drone. Các field trong docstring chỉ là stub mẫu" -- the drone-side schema is
authoritative and the examples are not it. So nothing here hard-codes a schema:
:func:`build` renders a *template*, and the default template is only a starting
point. Point ``--payload-template`` at a JSON file matching the real schema and
no code has to change.

Placeholders are ``{{name}}`` and are substituted with typed values, so
``{"vx": "{{vx}}"}`` yields a JSON number, not a string.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_PLACEHOLDER = re.compile(r"^\{\{(\w+)\}\}$")

# What we send when the drone side has not told us otherwise. Carries the
# command three ways -- velocity, displacement, and the instruction -- so the
# drone can use whichever it understands and ignore the rest.
DEFAULT_TEMPLATE: dict[str, Any] = {
    "ts": "{{ts_ms}}",
    "cmd_id": "{{cmd_id}}",
    "seq": "{{seq}}",
    "instruction": "{{instruction}}",
    "source": "{{source}}",
    "frame": "{{frame}}",
    "nav_vector": {
        "vx": "{{vx}}",
        "vy": "{{vy}}",
        "vz": "{{vz}}",
        "yaw_rate": "{{yaw_rate}}",
    },
    "displacement": {
        "dx": "{{dx}}",
        "dy": "{{dy}}",
        "dz": "{{dz}}",
        "dyaw": "{{dyaw}}",
    },
    "valid_for_s": "{{valid_for_s}}",
    "status": "{{status}}",
}


def load_template(path: str | Path | None) -> dict:
    if not path:
        return json.loads(json.dumps(DEFAULT_TEMPLATE))
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render(node: Any, ctx: dict[str, Any]) -> Any:
    """Substitute ``{{name}}`` leaves, preserving the value's real type."""
    if isinstance(node, dict):
        return {k: render(v, ctx) for k, v in node.items()}
    if isinstance(node, list):
        return [render(v, ctx) for v in node]
    if isinstance(node, str):
        m = _PLACEHOLDER.match(node)
        if m:
            key = m.group(1)
            if key not in ctx:
                raise KeyError(f"template placeholder {{{{{key}}}}} has no value")
            return ctx[key]
        # inline placeholders inside a longer string stay strings
        return re.sub(r"\{\{(\w+)\}\}",
                      lambda mm: str(ctx.get(mm.group(1), mm.group(0))), node)
    return node


def build(template: dict, *, vector, action, instruction: str, seq: int,
          ts_ms: int, frame: str, source: str, valid_for_s: float) -> dict:
    """Render one ``cmd/vla`` payload.

    ``vector`` is the safe body-frame velocity (already frame-converted by the
    caller if the drone wants FRD); ``action`` is the raw per-timestep
    displacement the network produced, kept so the drone side can integrate it
    itself if that is what its controller prefers.
    """
    ctx = {
        "ts_ms": ts_ms,
        "cmd_id": f"vla-{seq}",
        "seq": seq,
        "instruction": instruction,
        "source": source,
        "frame": frame,
        "vx": round(float(vector[0]), 4),
        "vy": round(float(vector[1]), 4),
        "vz": round(float(vector[2]), 4),
        "yaw_rate": round(float(vector[3]), 4),
        "dx": round(float(action[0]), 4),
        "dy": round(float(action[1]), 4),
        "dz": round(float(action[2]), 4),
        "dyaw": round(float(action[3]), 4),
        "valid_for_s": round(float(valid_for_s), 3),
        "status": "ok",
    }
    return render(template, ctx)


HOLD_OVERRIDE = {"ts": 0, "cmd_id": "vla-hold", "action": "hold"}


def hold_payload(seq: int, ts_ms: int, reason: str) -> dict:
    """What we publish to ``cmd/override`` when the loop must stop commanding."""
    return {"ts": ts_ms, "cmd_id": f"vla-hold-{seq}", "action": "hold",
            "reason": reason}
