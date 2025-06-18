#!/usr/bin/env python
"""
Script to start the remote inference server for LeRobot policies.
Usage:
  python lerobot/scripts/remote_server.py \
    --policy_path <path_or_hub_id> \
    [--host <host>] [--port <port>] [--timeout_ms <ms>]
"""
import logging
import argparse

from lerobot.remote.server import RemoteInferenceServer
from lerobot.remote.config import RemoteInferenceConfig


def main():
    parser = argparse.ArgumentParser(description="Run LeRobot remote inference server.")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host/IP to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=5555,
        help="Port for the server to listen on (default: 5555)"
    )
    parser.add_argument(
        "--policy_path", type=str, required=True,
        help="Path or Hub repo ID of the pretrained policy"
    )
    parser.add_argument(
        "--timeout_ms", type=int, default=1000,
        help="Server request timeout in milliseconds"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = RemoteInferenceConfig(
        server_host=args.host,
        server_port=args.port,
        timeout_ms=args.timeout_ms,
        policy_path=args.policy_path,
        robot_config=None  # not needed for server
    )
    server = RemoteInferenceServer(config)
    server.start()


if __name__ == "__main__":
    main()
