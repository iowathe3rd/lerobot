# Remote Inference with LeRobot

This guide explains how to run policies remotely on a GPU server while controlling your robot from a resource-constrained laptop or embedded board.

## Overview
LeRobot’s remote inference architecture separates:

- **Server**: Runs on a powerful machine with GPU. Hosts the policy inference endpoint.
- **Client**: Runs on your robot/laptop, streams observations, receives actions, and commands the robot.

![Remote Inference Diagram](../../media/lerobot-logo-thumbnail.png)

## Prerequisites

- Python >=3.8, PyTorch >=1.13
- Install LeRobot and ZeroMQ bindings on both machines:
  ```bash
  pip install lerobot[all] pyzmq
  ```
- Ensure network connectivity between client and server (open firewall ports).

## Server Setup

1. Choose or upload a pretrained policy to Hugging Face Hub or local checkpoint. E.g. `lerobot/diffusion_pusht`.
2. Launch the server:
   ```bash
   python lerobot/scripts/remote_server.py \
     --policy_path <POLICY_PATH> \
     --host 0.0.0.0 \
     --port 5555
   ```
3. Server logs will show:
   ```text
   Using device: cuda
   Server running on port 5555
   ```

## Client Setup

1. On the robot controller (laptop or embedded board), install requirements:
   ```bash
   pip install lerobot[robots] pyzmq
   ```
2. Edit the example script `examples/remote_inference_example.py`:
   - `SERVER_HOST`: IP of the server
   - `SERVER_PORT`: Matching port (default 5555)
   - `POLICY_PATH`: Hub ID or local path
   - Robot-specific fields: `ROBOT_ID`, `SERIAL_PORT`, etc.
3. Run the example:
   ```bash
   python examples/remote_inference_example.py
   ```
4. The script will:
   - Connect to the SO-100 arm
   - Stream observations at 30 Hz
   - Receive remote actions
   - Apply actions to the robot

## Customization

- **Different Robots**: Use any `lerobot.common.robots.<type>.config` and change import in the example.
- **Text Prompts**: Pass `text_prompt` to `RemoteInferenceConfig()` to use vision-language policies (e.g. SMoLvLA, PI0).
- **Control Frequency**: Adjust `busy_wait()` target.

## Troubleshooting

- **Timeouts**: Increase `--timeout_ms` on client/server.
- **Connection Refused**: Verify IP/port and network reachability.
- **Policy Errors**: Ensure the policy supports the robot’s observation/action shapes.

## Next Steps

- Integrate into your control stack or teleoperation UI
- Add logging, monitoring, or batching
- Scale to multi-robot or multi-server setups

---

*Generated with LeRobot v*`0.1.*`*.*
