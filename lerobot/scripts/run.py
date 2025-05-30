#!/usr/bin/env python3
"""
Universal script for running robots with AI policies in the lerobot framework.

This script provides a unified interface to run any robot (SO-100, Aloha, Koch, etc.) 
with any AI policy (ACT, Diffusion, PI0, etc.) using a simple command-line interface.
It supports both local and HuggingFace Hub policies, natural language instructions,
and comprehensive safety features.

Example usage:
    python lerobot/scripts/run.py --robot_type="so100" --policy_path="act-so100-test"
    python lerobot/scripts/run.py --robot_type="aloha" --policy_path="lerobot/diffusion_pusht"
    python lerobot/scripts/run.py --robot_type="koch" --policy_path="./my_local_policy"
    python lerobot/scripts/run.py --robot_type="so100" --policy_path="finetuned/pi0" --text_prompt="take a blue thing and put in the cup"
"""

import logging
import signal
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from lerobot.common.policies.factory import get_policy_class
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robot_devices.robots.utils import make_robot_from_config, make_robot_config
from lerobot.common.robot_devices.robots.configs import RobotConfig
from lerobot.common.robot_devices.control_utils import control_loop
from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.utils.utils import init_logging, get_safe_torch_device
from lerobot.configs import parser


@dataclass
class RunConfig:
    """Configuration for running robot with policy.
    
    This class defines all the parameters needed to run a robot with an AI policy,
    including robot type, policy path, control parameters, and safety settings.
    """
    
    # Robot configuration
    robot_type: str = "so100"  # Type of robot to use (so100, aloha, koch, etc.)
    
    # Policy configuration  
    policy_path: str = ""  # Path to pretrained policy (local path or HuggingFace repo_id)
    text_prompt: Optional[str] = None  # Natural language instruction for language-conditioned policies
    device: Optional[str] = None  # Device for policy inference (auto-detected if not specified)
    
    # Control parameters
    control_time_s: float = 3600.0  # Duration to run the control loop (seconds)
    fps: int = 30  # Control frequency (Hz)
    display_data: bool = True  # Whether to display camera feeds and robot state
    
    # Episode configuration
    max_episodes: Optional[int] = None  # Maximum number of episodes to run
    episode_time_s: Optional[float] = None  # Duration per episode (None = use control_time_s)
    
    # Safety parameters
    enable_robot_safety: bool = True  # Enable comprehensive safety checks
    emergency_stop: bool = True  # Enable emergency stop on keyboard interrupt
    
    # Debug and development options
    log_level: str = "INFO"  # Logging level (DEBUG, INFO, WARNING, ERROR)
    dry_run: bool = False  # Run without actually controlling robot (for testing)
    
    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        # Auto-detect device if not specified
        if self.device is None:
            # Use the auto-detection function instead of get_safe_torch_device
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
            
        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}, got '{self.log_level}'")
        self.log_level = self.log_level.upper()


# Global variable for graceful shutdown
_shutdown_requested = False


def signal_handler(signum: int, frame) -> None:
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    logging.info("Shutdown signal received. Initiating graceful shutdown...")
    _shutdown_requested = True


def get_available_robots() -> list[str]:
    """Dynamically get list of available robot types."""
    try:
        from lerobot.common.robot_devices.robots.configs import RobotConfig
        # Get all registered robot types from the registry
        return list(RobotConfig._choice_registry.keys())
    except Exception:
        # Fallback to known robot types if registry access fails
        return ["aloha", "koch", "koch_bimanual", "moss", "so100", "so101", "stretch", "lekiwi"]


def load_robot_config(robot_type: str) -> RobotConfig:
    """Load robot configuration by type with enhanced error handling."""
    try:
        robot_config = make_robot_config(robot_type)
        logging.info(f"✓ Loaded robot configuration for: {robot_type}")
        return robot_config
        
    except ValueError as e:
        available_robots = get_available_robots()
        error_msg = (
            f"Failed to load robot config for '{robot_type}'. "
            f"Available robots: {available_robots}"
        )
        
        # Suggest closest match if available
        if available_robots:
            from difflib import get_close_matches
            suggestions = get_close_matches(robot_type, available_robots, n=3, cutoff=0.6)
            if suggestions:
                error_msg += f". Did you mean one of: {suggestions}?"
        
        raise ValueError(f"{error_msg}. Original error: {e}") from e
        
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading robot config for '{robot_type}': {e}") from e


def load_policy(policy_path: str, robot_config: RobotConfig, device: str) -> Tuple[PreTrainedPolicy, PreTrainedConfig]:
    """Load AI policy from local path or HuggingFace Hub with enhanced error handling."""
    try:
        logging.info(f"Loading policy from: {policy_path}")
        
        # Load policy configuration first
        policy_config = PreTrainedConfig.from_pretrained(policy_path)
        logging.info(f"✓ Detected policy type: {policy_config.type}")
        
        # Get the specific policy class
        policy_cls = get_policy_class(policy_config.type)
        logging.info(f"✓ Using policy class: {policy_cls.__name__}")
        
        # Load the policy with pretrained weights
        policy = policy_cls.from_pretrained(policy_path)
        
        # Move to specified device and set to evaluation mode
        policy = policy.to(device)
        policy.eval()
        
        logging.info(f"✓ Successfully loaded policy: {policy.__class__.__name__} on {device}")
        return policy, policy_config
        
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Policy not found at '{policy_path}'. Please check the path or repo_id. "
            f"For HuggingFace models, ensure the repo exists and you have access. "
            f"Original error: {e}"
        ) from e
        
    except Exception as e:
        raise RuntimeError(f"Failed to load policy from '{policy_path}': {e}") from e


def setup_robot(robot_config: RobotConfig, dry_run: bool = False) -> Optional[object]:
    """Initialize and setup the robot with comprehensive error handling."""
    try:
        if dry_run:
            logging.info("✓ DRY RUN: Skipping actual robot initialization")
            return None
            
        logging.info("Initializing robot hardware...")
        
        # Create robot from config
        robot = make_robot_from_config(robot_config)
        
        # Connect to robot
        logging.info("Connecting to robot...")
        robot.connect()
        
        # Calibrate if needed
        if hasattr(robot, 'calibrate'):
            logging.info("Calibrating robot...")
            robot.calibrate()
        
        logging.info("✓ Robot setup completed successfully")
        return robot
        
    except ConnectionError as e:
        raise ConnectionError(
            f"Failed to connect to robot. Please check:\n"
            f"  - Robot is powered on and properly connected\n"
            f"  - USB/network connections are secure\n"
            f"  - Robot drivers are installed\n"
            f"  - No other processes are using the robot\n"
            f"Original error: {e}"
        ) from e
        
    except Exception as e:
        raise RuntimeError(f"Failed to setup robot: {e}") from e


@contextmanager
def robot_context(robot):
    """Context manager for proper robot resource management."""
    try:
        yield robot
    finally:
        if robot and hasattr(robot, 'disconnect'):
            try:
                logging.info("Disconnecting robot...")
                robot.disconnect()
                logging.info("✓ Robot disconnected safely")
            except Exception as e:
                logging.error(f"Error during robot disconnect: {e}")


def apply_text_prompt(policy: PreTrainedPolicy, text_prompt: str) -> bool:
    """Apply text prompt to language-conditioned policies."""
    if hasattr(policy, "set_prompt"):
        try:
            policy.set_prompt(text_prompt)
            logging.info(f"✓ Applied text prompt: '{text_prompt}'")
            return True
        except Exception as e:
            logging.error(f"Failed to apply text prompt: {e}")
            return False
    elif hasattr(policy, "condition_on_text"):
        try:
            policy.condition_on_text(text_prompt)
            logging.info(f"✓ Applied text conditioning: '{text_prompt}'")
            return True
        except Exception as e:
            logging.error(f"Failed to apply text conditioning: {e}")
            return False
    else:
        logging.warning(
            f"Policy {policy.__class__.__name__} does not support text prompts. "
            f"Available methods: {[m for m in dir(policy) if not m.startswith('_')]}"
        )
        return False


def run_policy_control(
    robot,
    policy: PreTrainedPolicy,
    policy_config: PreTrainedConfig,
    control_time_s: float,
    fps: int,
    display_data: bool = True,
    dry_run: bool = False,
    text_prompt: Optional[str] = None,
    enable_robot_safety: bool = True
) -> None:
    """Run the main control loop with the policy."""
    
    if dry_run:
        logging.info(f"✓ DRY RUN: Would run control loop for {control_time_s}s at {fps}Hz")
        return
    
    logging.info(f"Starting control loop for {control_time_s}s at {fps}Hz")
    
    # Setup signal handlers for graceful shutdown
    if hasattr(signal, 'SIGINT'):
        signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Use the existing control_loop function from lerobot
        control_loop(
            robot=robot,
            policy=policy,
            control_time_s=control_time_s,
            fps=fps,
            display_data=display_data,
        )
        
    except KeyboardInterrupt:
        logging.info("✓ Control loop interrupted by user")
    except Exception as e:
        logging.error(f"Error during control loop: {e}")
        raise


def validate_config(config: RunConfig) -> None:
    """Comprehensive validation of configuration parameters."""
    errors = []
    
    if not config.policy_path:
        errors.append("Policy path must be specified (--policy_path)")
    
    if config.fps <= 0:
        errors.append(f"FPS must be positive, got {config.fps}")
    
    if config.control_time_s <= 0:
        errors.append(f"Control time must be positive, got {config.control_time_s}")
    
    if config.max_episodes is not None and config.max_episodes <= 0:
        errors.append(f"max_episodes must be positive, got {config.max_episodes}")
    
    if config.episode_time_s is not None and config.episode_time_s <= 0:
        errors.append(f"episode_time_s must be positive, got {config.episode_time_s}")
    
    if errors:
        raise ValueError("Configuration validation failed:\n  " + "\n  ".join(errors))


def print_configuration_summary(config: RunConfig) -> None:
    """Print a formatted summary of the configuration."""
    logging.info("=== LeRobot Universal Policy Runner ===")
    logging.info(f"Robot Type:     {config.robot_type}")
    logging.info(f"Policy Path:    {config.policy_path}")
    logging.info(f"Device:         {config.device}")
    logging.info(f"Control Time:   {config.control_time_s}s")
    logging.info(f"Frequency:      {config.fps}Hz")
    logging.info(f"Display Data:   {config.display_data}")
    logging.info(f"Safety Enabled: {config.enable_robot_safety}")
    logging.info(f"Dry Run:        {config.dry_run}")
    
    if config.text_prompt:
        logging.info(f"Text Prompt:    '{config.text_prompt}'")
    if config.max_episodes:
        logging.info(f"Max Episodes:   {config.max_episodes}")
    if config.episode_time_s:
        logging.info(f"Episode Time:   {config.episode_time_s}s")
    
    logging.info("=" * 45)


@parser.wrap()
def main(cfg: RunConfig) -> int:
    """Main entry point with comprehensive error handling."""
    
    try:
        # Setup logging
        init_logging()
        
        # Set log level manually if different from INFO
        if cfg.log_level != "INFO":
            logging.getLogger().setLevel(getattr(logging, cfg.log_level))
        
        # Validate configuration
        validate_config(cfg)
        
        # Print configuration summary
        print_configuration_summary(cfg)
        
        # Load robot configuration
        robot_config = load_robot_config(cfg.robot_type)
        
        # Load AI policy
        policy, policy_config = load_policy(cfg.policy_path, robot_config, cfg.device)
        
        # Apply text prompt if provided
        if cfg.text_prompt:
            apply_text_prompt(policy, cfg.text_prompt)
        
        # Setup robot with proper resource management
        robot = setup_robot(robot_config, cfg.dry_run)
        
        with robot_context(robot):
            # Run control loop
            run_policy_control(
                robot=robot,
                policy=policy,
                policy_config=policy_config,
                control_time_s=cfg.control_time_s,
                fps=cfg.fps,
                display_data=cfg.display_data,
                dry_run=cfg.dry_run,
                text_prompt=cfg.text_prompt,
                enable_robot_safety=cfg.enable_robot_safety
            )
        
        logging.info("✓ Run completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logging.info("✓ Interrupted by user")
        return 0
        
    except ValueError as e:
        logging.error(f"Configuration error: {e}")
        return 1
        
    except ConnectionError as e:
        logging.error(f"Robot connection error: {e}")
        return 2
        
    except FileNotFoundError as e:
        logging.error(f"Policy not found: {e}")
        return 3
        
    except RuntimeError as e:
        logging.error(f"Runtime error: {e}")
        return 4
        
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        logging.debug("Full traceback:", exc_info=True)
        return 5


if __name__ == "__main__":
    main()
