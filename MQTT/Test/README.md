# MQTT drone tools (VinUI)

Scripts Python để subscribe telemetry / publish lệnh điều khiển drone qua MQTT (không cần ROS).

## Yêu cầu

```bash
pip install paho-mqtt
```

## Broker mặc định

| Tham số | Giá trị |
|---------|---------|
| Host | `116.111.21.154` |
| Port | `1883` |
| User | `mqtt` |
| Drone ID | `drone01` |

Đổi bằng `--host`, `--port`, `-u`, `-P`, `--drone-id`.

## Subscribe

```bash
python3 mqtt_drone_sub.py
python3 mqtt_drone_sub.py --drone-id drone01 --pretty
python3 mqtt_drone_sub.py --only state,battery,heartbeat
python3 mqtt_drone_sub.py --only ack,result --pretty
```

### Topic ngắn (`--only`)

| Short name | MQTT topic |
|------------|------------|
| `state` | `drone/{id}/telemetry/state` |
| `battery` | `drone/{id}/telemetry/battery` |
| `gps` | `drone/{id}/telemetry/gps` |
| `ekf_status` | `drone/{id}/telemetry/ekf_status` |
| `image_meta` | `drone/{id}/telemetry/image_meta` |
| `inference_ctx` | `drone/{id}/telemetry/inference_ctx` |
| `heartbeat` | `drone/{id}/status/heartbeat` |
| `online` | `drone/{id}/status/online` |
| `lwt` | `drone/{id}/status/lwt` |
| `health` | `drone/{id}/system/health` |
| `log` | `drone/{id}/system/log` |
| `ack` | `drone/{id}/cmd/ack` |
| `result` | `drone/{id}/cmd/result` |
| `cmd_vla` | `drone/{id}/cmd/vla` |
| `cmd_mission` | `drone/{id}/cmd/mission` |
| `cmd_override` | `drone/{id}/cmd/override` |

Không truyền `--only` → subscribe `drone/{id}/#`.

## Publish lệnh (từng topic một script)

Mỗi script bắt buộc có `--payload '{...}'` **hoặc** `--payload-file path.json`.

```bash
# Terminal 1: xem ack/result
python3 mqtt_drone_sub.py --only ack,result --pretty

# Terminal 2: gửi lệnh
python3 mqtt_drone_pub_vla.py --payload '{"ts":0,"cmd_id":"vla-1","instruction":"..."}'
python3 mqtt_drone_pub_mission.py --payload-file mission.json
python3 mqtt_drone_pub_override.py --payload '{"ts":0,"cmd_id":"ovr-1","action":"hold"}'
```

| Script | Topic publish |
|--------|----------------|
| `mqtt_drone_pub_vla.py` | `drone/{id}/cmd/vla` |
| `mqtt_drone_pub_mission.py` | `drone/{id}/cmd/mission` |
| `mqtt_drone_pub_override.py` | `drone/{id}/cmd/override` |

Payload JSON phải khớp schema phía drone. Các field trong docstring chỉ là stub mẫu.

## File trong repo

- `mqtt_drone_sub.py` — subscribe telemetry / status / cmd ack
- `mqtt_drone_pub_vla.py` — publish `cmd/vla`
- `mqtt_drone_pub_mission.py` — publish `cmd/mission`
- `mqtt_drone_pub_override.py` — publish `cmd/override`
