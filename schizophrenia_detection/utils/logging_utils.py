"""
Logging configuration utilities
"""

import os
import numpy as np
import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(
    log_file=None, level=logging.INFO, console_logging=True, format_string=None
):
    """
    Setup logging configuration

    Args:
        log_file (str): Path to log file
        level: Logging level
        console_logging (bool): Whether to log to console
        format_string (str): Custom format string
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Add console handler if requested
    if console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if log file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def setup_logger(name, log_file=None, level=logging.INFO, console_logging=True):
    """
    Setup a logger with the specified name

    Args:
        name (str): Logger name
        log_file (str): Path to log file
        level: Logging level
        console_logging (bool): Whether to log to console

    Returns:
        Logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Add console handler if requested
    if console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if log file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name, log_file=None, level=logging.INFO, console_logging=True):
    """
    Get a logger with the specified name

    Args:
        name (str): Logger name
        log_file (str): Path to log file
        level: Logging level
        console_logging (bool): Whether to log to console

    Returns:
        Logger object
    """
    logger = logging.getLogger(name)

    # If logger already has handlers, return it
    if logger.handlers:
        return logger

    # Setup logging
    setup_logging(log_file, level, console_logging)

    return logger


def log_function_call(func):
    """
    Decorator to log function calls

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """

    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")

        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} raised {type(e).__name__}: {str(e)}")
            raise

    return wrapper


def log_execution_time(func):
    """
    Decorator to log function execution time

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """

    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = datetime.now()

        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.error(
                f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}"
            )
            raise

    return wrapper


class ProgressLogger:
    """
    Class for logging progress of long-running tasks
    """

    def __init__(self, total_steps, logger=None, log_interval=10):
        """
        Initialize progress logger

        Args:
            total_steps (int): Total number of steps
            logger: Logger object
            log_interval (int): Interval for logging progress
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.log_interval = log_interval
        self.logger = logger if logger else logging.getLogger(__name__)
        self.start_time = datetime.now()

    def update(self, step=1, message=None):
        """
        Update progress

        Args:
            step (int): Number of steps completed
            message (str): Optional message to log
        """
        self.current_step += step

        # Log progress at specified intervals
        if (
            self.current_step % self.log_interval == 0
            or self.current_step >= self.total_steps
        ):
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            progress = self.current_step / self.total_steps * 100

            # Estimate remaining time
            if self.current_step > 0:
                avg_time_per_step = elapsed_time / self.current_step
                remaining_steps = self.total_steps - self.current_step
                eta_seconds = avg_time_per_step * remaining_steps
                eta = f"ETA: {eta_seconds:.1f}s"
            else:
                eta = "ETA: Unknown"

            log_message = f"Progress: {self.current_step}/{self.total_steps} ({progress:.1f}%) - {eta}"
            if message:
                log_message += f" - {message}"

            self.logger.info(log_message)

    def finish(self, message=None):
        """
        Finish progress tracking

        Args:
            message (str): Optional message to log
        """
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        log_message = (
            f"Completed {self.total_steps} steps in {elapsed_time:.2f} seconds"
        )
        if message:
            log_message += f" - {message}"

        self.logger.info(log_message)


def log_system_info(logger=None):
    """
    Log system information

    Args:
        logger: Logger object
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    import platform
    import psutil

    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python version: {platform.python_version()}")
    logger.info(f"  CPU count: {psutil.cpu_count()}")
    logger.info(f"  Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")

    # Log package versions
    try:
        import numpy

        logger.info(f"  NumPy version: {numpy.__version__}")
    except ImportError:
        logger.warning("  NumPy not installed")

    try:
        import tensorflow

        logger.info(f"  TensorFlow version: {tensorflow.__version__}")
    except ImportError:
        logger.warning("  TensorFlow not installed")

    try:
        import torch

        logger.info(f"  PyTorch version: {torch.__version__}")
    except ImportError:
        logger.warning("  PyTorch not installed")


def log_model_info(model, logger=None):
    """
    Log model information

    Args:
        model: Neural network model
        logger: Logger object
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Model Information:")

    # Count parameters
    try:
        total_params = model.count_params()
        trainable_params = sum([np.prod(p.shape) for p in model.trainable_weights])
        non_trainable_params = total_params - trainable_params

        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Non-trainable parameters: {non_trainable_params:,}")
    except:
        logger.warning("  Could not count parameters")

    # Log model summary
    try:
        import io
        import sys

        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        model.summary()
        sys.stdout = old_stdout

        # Log summary
        summary = buffer.getvalue()
        for line in summary.split("\n"):
            if line.strip():
                logger.info(f"  {line}")
    except:
        logger.warning("  Could not generate model summary")


def create_log_file_name(prefix, suffix=None):
    """
    Create a log file name with timestamp

    Args:
        prefix (str): Prefix for the log file name
        suffix (str): Suffix for the log file name

    Returns:
        str: Log file name
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if suffix:
        return f"{prefix}_{timestamp}_{suffix}.log"
    else:
        return f"{prefix}_{timestamp}.log"


def configure_tensorflow_logging(level="1"):
    """
    Configure TensorFlow logging

    Args:
        level (str): TensorFlow logging level ('0' = all, '1' = INFO, '2' = WARNING, '3' = ERROR)
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = level


def configure_matplotlib_logging(level="WARNING"):
    """
    Configure Matplotlib logging

    Args:
        level (str): Matplotlib logging level
    """
    import matplotlib

    matplotlib.set_loglevel(level)
