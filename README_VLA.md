# VLA navigation for UAV-control

Closed-loop drone navigation driven by a **SmolVLA policy fine-tuned on
WareFly-VLA** and served through **vla.cpp** (arXiv 2606.08094), publishing a
navigation vector on this repo's existing `drone/{id}/cmd/vla` topic.

```
camera frame ─┐
              ├─► vla.cpp (GGUF, C++/ggml) ─► [dx, dy, dz, dyaw] ─► Navigator ─► cmd/vla
telemetry/state ┘        4-DoF displacement       safety limits      MQTT
```

Nothing in `MQTT/` was modified. The addition is `vla_nav/` (library),
`scripts/` (two entry points) and `tests/`.

## What the model outputs

The policy emits a **displacement per model timestep in the drone body frame**,
not a velocity:

| field | meaning | unit |
|---|---|---|
| `dx_local` | forward | m |
| `dy_local` | left | m |
| `dz` | up | m |
| `dyaw` | yaw, CCW positive | rad |

`Navigator` converts it with `v = d × fps`. **`--fps` must be the rate the
checkpoint was trained at**, because it scales every command — a 10 FPS
checkpoint driven with `--fps 1` commands one tenth of the intended speed.

| checkpoint | `--fps` |
|---|---|
| `smolvla_warefly_1fps-*.gguf` | `1` |
| `smolvla_warefly_10fps-*.gguf` | `10` |

**Use the 10 FPS checkpoint for real flight.** At 1 FPS the vertical channel is
almost always inside the deadband (`\|dz\| ≈ 0.004 m` → 0.004 m/s), so the
aircraft never changes altitude, and one command has to cover a whole second of
motion.

## Frames

Internally everything is **body FLU** (+forward, +left, +up, +CCW), which is
WareFly's convention. PX4/MAVLink body-NED is (+forward, +right, +down) with yaw
positive clockwise, so pass `--frame body_frd` and `adapters/base.py:to_body_ned`
flips y, z and yaw once, in one place. Getting this wrong flies the mirror image
of the intended path.

## Running it

```bash
# 0. deps -- no torch needed on the aircraft
pip install paho-mqtt pillow numpy tokenizers opencv-python

# 1. offline: recorded episode, no broker, no aircraft
python3 scripts/run_vla_nav.py --no-broker \
    --camera replay:<episode-uuid> --fps 1 \
    --gguf /path/to/smolvla_warefly_1fps-bf16.gguf

# 2. live telemetry in, payloads printed, NOTHING published
python3 scripts/run_vla_nav.py --camera meta:url --fps 10 \
    --gguf /path/to/smolvla_warefly_10fps-bf16.gguf -P "$MQTT_PASSWORD"

# 3. actually command the aircraft
python3 scripts/run_vla_nav.py ... --publish
```

`--publish` is required to transmit. Without it every payload is rendered,
printed and logged but never sent — check the schema and the numbers first.

Camera sources: `replay:<uuid>` (recorded episode) · `mqtt:<topic-suffix>` (JPEG
bytes or base64 in JSON) · `meta:<field>` (follow a URL from
`telemetry/image_meta`) · any RTSP/HTTP URL or device index via OpenCV.

## ⚠ The one thing that still needs your input

`MQTT/Test/README.md` says the drone-side schema is authoritative
("*Payload JSON phải khớp schema phía drone*"), and the broker
(`116.111.21.154:1883`) is not reachable from outside your network, so the real
`cmd/vla` and `telemetry/state` shapes could not be observed.

Both are therefore **configuration, not code**:

* **Command payload** — `--payload-template schema.json`, with `{{vx}}`-style
  placeholders substituted as typed JSON values. The default sends the vector
  three ways (velocity, displacement, instruction) so the drone can use whichever
  it understands:

  ```json
  {"ts": 1724800000000, "cmd_id": "vla-42", "seq": 42,
   "instruction": "follow the person in an orange hard hat",
   "source": "vla.cpp/smolvla-warefly", "frame": "body_flu",
   "nav_vector":   {"vx": 0.74, "vy": -0.02, "vz": 0.0, "yaw_rate": -0.07},
   "displacement": {"dx": 0.074, "dy": -0.002, "dz": 0.0, "dyaw": -0.007},
   "valid_for_s": 0.2, "status": "ok"}
  ```

* **Telemetry pose** — resolved by trying candidate paths (`x`, `position.x`,
  `local_position.x`, `pose.position.x`, …). Pin them with
  `--state-path x=local_position.x --state-path yaw=attitude.yaw`.

Send me one real `telemetry/state` message and the `cmd/vla` schema and this is a
config change, not a code change.

## Safety

Every vector passes through `Navigator` before it can reach the aircraft:

| guard | default | why |
|---|---|---|
| speed clamp | 1.5 m/s xy, 0.5 m/s z, 0.8 rad/s | horizontal is clamped as a **magnitude**, so a diagonal cannot reach √2 × max |
| deadband | 0.02 | crawl commands become hold |
| slew | 2.0 m/s², 1.0 m/s², 2.0 rad/s² | bounded change per tick, including the **first** tick after engaging |
| watchdog | 1.0 s | stale policy output decays to zero; a latched setpoint keeps flying |
| altitude guard | 0.8–20 m | blocks descent below the floor, climb past the ceiling |
| non-finite | always | a NaN never reaches the flight controller |
| readiness gate | heartbeat 5 s, state 2 s, battery 20% | loss of any of these stops commanding |
| stop | always | publishes `cmd/override {"action":"hold"}`, never just goes quiet |

`--publish` is off by default and prints a 3-second countdown when enabled.

## Verification

```bash
python3 tests/test_navigator.py    # 8 safety behaviours
python3 tests/test_mqtt_path.py    # 9 transport/payload checks (fake broker)
python3 tests/test_parity.py       # deployment path == validated offline path
```

Established on this machine:

* **Runtime parity.** vla.cpp bf16 vs PyTorch/LeRobot SmolVLA on the same 1,228
  held-out frames of the canonical split: `dx` MAE 0.4738 vs 0.4680, r 0.554 vs
  0.557; `dyaw` MAE 0.2457 vs 0.2433. The C++ runtime is the same policy.
* **Preprocessing parity.** The torch path and the torch-free path produce
  **bit-identical** actions end-to-end. This mattered: a naive OpenCV downsample
  differed from the validated PIL one by 0.21 (0–1 scale) and moved commanded
  velocity by up to 0.14 m/s. Only the exact-2× upsample is substituted now.
* **Tokenizer parity.** `tokenizers` gives ids identical to
  `transformers.AutoTokenizer`, so no torch is needed for language either.
* **Determinism.** With `fixed_noise_seed`, two runs are bit-identical — a
  necessary property for reproducing a flight from a log.

## Latency

| where | per network call | effective per frame | note |
|---|---|---|---|
| H100, CUDA | 26.4 ms (p90) | 7.4 ms → ~135 Hz | 4-step chunk: the network fires once per 4 frames |
| this host, CPU, 16 threads | 12–16 s | 3–4 s | measured while the GPU was busy; **not** flight-capable |

Closed-loop flight needs the CUDA (Jetson) or a quantised build — CPU bf16 is
two to three orders of magnitude too slow. Note that `Q8_0`/`Q4_0` currently fail
to load on this vla.cpp revision's CPU path (`gguf unsupported dtype 8` in
`read_to_f32`); they need re-testing on the CUDA build.

## Honest limitations

* The policy was trained **open-loop on recorded flights**. It has never seen the
  states its own mistakes produce, so closed-loop behaviour is not predicted by
  the offline metrics. Fly it tethered / in a cage first.
* On the 1 FPS benchmark, "repeat the previous action" scores higher than every
  published VLA we tested (dx r 0.858 vs 0.557). Much of the offline score is
  motion continuity, not visual understanding — another reason to treat the
  open-loop numbers as a floor, not a promise.
* `dz` is barely learned (r ≈ 0.06–0.09). Altitude should stay under the existing
  controller, not this policy.
* There is no obstacle avoidance anywhere in this path.
