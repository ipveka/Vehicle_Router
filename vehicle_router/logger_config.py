"""
Advanced Logging Configuration Module

This module provides comprehensive logging functionality for the Vehicle Router application,
supporting both the main CLI application and the Streamlit web interface with separate
log files, rotation, and detailed formatting.

Features:
- Separate log directories for main and app components
- Log rotation to prevent file size issues
- Structured logging with timestamps and context
- Console and file output with different log levels
- Session-based logging for web application
- Performance tracking and debugging support

Usage:
    from vehicle_router.logger_config import setup_main_logging, setup_app_logging
    
    # For main.py
    logger = setup_main_logging(log_level="INFO")
    
    # For streamlit app
    logger = setup_app_logging(session_id="user_123", log_level="INFO")
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import uuid


class CustomFormatter(logging.Formatter):
    """
    Custom formatter with color support and enhanced formatting
    """
    
    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def __init__(self, include_colors: bool = True, include_thread: bool = False):
        self.include_colors = include_colors
        self.include_thread = include_thread
        
        # Base format
        base_format = '[%(levelname)s] %(asctime)s - %(name)s'
        
        if include_thread:
            base_format += ' - Thread:%(thread)d'
            
        base_format += ' - %(message)s'
        
        super().__init__(base_format, datefmt='%Y-%m-%d %H:%M:%S')
    
    def format(self, record):
        """Format log record with optional colors"""
        if self.include_colors and hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            # Add colors for console output
            color = self.COLORS.get(record.levelname, '')
            reset = self.RESET
            
            # Temporarily modify the levelname for coloring
            original_levelname = record.levelname
            record.levelname = f"{color}{record.levelname}{reset}"
            
            result = super().format(record)
            
            # Restore original levelname
            record.levelname = original_levelname
            
            return result
        else:
            return super().format(record)


class PerformanceLogger:
    """
    Performance tracking utility for optimization operations
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_times: Dict[str, float] = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation"""
        import time
        self.start_times[operation] = time.time()
        self.logger.info(f"⏱️  Started: {operation}")
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and log duration"""
        import time
        if operation not in self.start_times:
            self.logger.warning(f"No start time recorded for operation: {operation}")
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        self.logger.info(f"✅ Completed: {operation} (Duration: {duration:.2f}s)")
        del self.start_times[operation]
        return duration
    
    def log_memory_usage(self, context: str = "") -> None:
        """Log current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.logger.debug(f"🧠 Memory usage{' - ' + context if context else ''}: {memory_mb:.1f} MB")
        except ImportError:
            self.logger.debug("psutil not available - memory tracking disabled")


def setup_main_logging(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    enable_performance_tracking: bool = True
) -> logging.Logger:
    """
    Set up comprehensive logging for the main CLI application
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file (bool): Whether to log to file
        log_to_console (bool): Whether to log to console
        enable_performance_tracking (bool): Whether to enable performance tracking
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create main logger
    logger = logging.getLogger('vehicle_router.main')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create logs directory structure
    project_root = Path(__file__).parent.parent
    main_logs_dir = project_root / "logs" / "main"
    main_logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for this execution
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # File handler with rotation
    if log_to_file:
        log_filename = main_logs_dir / f"main_{timestamp}.log"
        
        # Use rotating file handler to prevent large files
        file_handler = logging.handlers.RotatingFileHandler(
            log_filename,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # File gets all logs
        
        # File formatter (no colors)
        file_formatter = CustomFormatter(include_colors=False, include_thread=True)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Create a symlink to latest log for easy access
        latest_link = main_logs_dir / "latest.log"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        try:
            latest_link.symlink_to(log_filename.name)
        except OSError:
            # Windows might not support symlinks, just copy the filename
            pass
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Console formatter (with colors)
        console_formatter = CustomFormatter(include_colors=True, include_thread=False)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Suppress noisy loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    # Log startup information
    logger.info("=" * 80)
    logger.info("🚛 VEHICLE ROUTER - MAIN APPLICATION LOGGING INITIALIZED")
    logger.info("=" * 80)
    logger.info(f"📅 Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"📂 Log directory: {main_logs_dir}")
    logger.info(f"📊 Log level: {log_level.upper()}")
    logger.info(f"🖥️  Console logging: {'Enabled' if log_to_console else 'Disabled'}")
    logger.info(f"📄 File logging: {'Enabled' if log_to_file else 'Disabled'}")
    
    if log_to_file:
        logger.info(f"📝 Log file: {log_filename}")
    
    # Add performance tracking capability
    if enable_performance_tracking:
        logger.perf = PerformanceLogger(logger)
        logger.info("⏱️  Performance tracking: Enabled")
    
    logger.info("-" * 80)
    
    return logger


def setup_app_logging(
    session_id: Optional[str] = None,
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = False,  # Usually disabled for Streamlit
    enable_performance_tracking: bool = True
) -> logging.Logger:
    """
    Set up comprehensive logging for the Streamlit web application
    
    Args:
        session_id (Optional[str]): Unique session identifier for this user session
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file (bool): Whether to log to file
        log_to_console (bool): Whether to log to console (usually False for Streamlit)
        enable_performance_tracking (bool): Whether to enable performance tracking
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Generate session ID if not provided
    if session_id is None:
        session_id = str(uuid.uuid4())[:8]
    
    # Create app logger with session-specific name
    logger_name = f'vehicle_router.app.{session_id}'
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create logs directory structure
    project_root = Path(__file__).parent.parent
    app_logs_dir = project_root / "logs" / "app"
    app_logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for this session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # File handler with rotation
    if log_to_file:
        log_filename = app_logs_dir / f"app_{session_id}_{timestamp}.log"
        
        # Use rotating file handler to prevent large files
        file_handler = logging.handlers.RotatingFileHandler(
            log_filename,
            maxBytes=5*1024*1024,  # 5MB (smaller for web app)
            backupCount=3,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # File gets all logs
        
        # File formatter (no colors, include session info)
        file_formatter = logging.Formatter(
            f'[%(levelname)s] %(asctime)s - Session:{session_id} - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Create a symlink to latest session log
        latest_link = app_logs_dir / f"session_{session_id}_latest.log"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        try:
            latest_link.symlink_to(log_filename.name)
        except OSError:
            # Windows might not support symlinks
            pass
    
    # Console handler (optional for Streamlit)
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Console formatter (with colors and session info)
        console_formatter = logging.Formatter(
            f'[%(levelname)s] %(asctime)s - Session:{session_id} - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Suppress noisy loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('streamlit').setLevel(logging.WARNING)
    logging.getLogger('tornado').setLevel(logging.WARNING)
    
    # Log startup information
    logger.info("=" * 80)
    logger.info("🌐 VEHICLE ROUTER - STREAMLIT APP LOGGING INITIALIZED")
    logger.info("=" * 80)
    logger.info(f"📅 Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🆔 Session ID: {session_id}")
    logger.info(f"📂 Log directory: {app_logs_dir}")
    logger.info(f"📊 Log level: {log_level.upper()}")
    logger.info(f"🖥️  Console logging: {'Enabled' if log_to_console else 'Disabled'}")
    logger.info(f"📄 File logging: {'Enabled' if log_to_file else 'Disabled'}")
    
    if log_to_file:
        logger.info(f"📝 Log file: {log_filename}")
    
    # Add performance tracking capability
    if enable_performance_tracking:
        logger.perf = PerformanceLogger(logger)
        logger.info("⏱️  Performance tracking: Enabled")
    
    # Store session info in logger for easy access
    logger.session_id = session_id
    logger.session_start = datetime.now()
    
    logger.info("-" * 80)
    
    return logger


def log_system_info(logger: logging.Logger) -> None:
    """
    Log system information for debugging purposes
    
    Args:
        logger: Logger instance to use for output
    """
    import platform
    import sys
    
    logger.info("🖥️  SYSTEM INFORMATION:")
    logger.info(f"   Platform: {platform.platform()}")
    logger.info(f"   Python version: {sys.version}")
    logger.info(f"   Architecture: {platform.architecture()[0]}")
    logger.info(f"   Processor: {platform.processor()}")
    
    try:
        import psutil
        logger.info(f"   CPU cores: {psutil.cpu_count()}")
        logger.info(f"   Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    except ImportError:
        logger.debug("psutil not available - extended system info unavailable")


def log_optimization_start(logger: logging.Logger, config: Dict[str, Any]) -> None:
    """
    Log optimization configuration and start information
    
    Args:
        logger: Logger instance
        config: Optimization configuration dictionary
    """
    logger.info("🚀 OPTIMIZATION STARTING:")
    logger.info(f"   Method: {config.get('optimizer_type', 'standard')}")
    logger.info(f"   Depot: {config.get('depot_location', 'N/A')}")
    logger.info(f"   Real distances: {config.get('use_real_distances', False)}")
    logger.info(f"   Timeout: {config.get('solver_timeout', 'N/A')}s")
    
    if config.get('optimizer_type') == 'genetic':
        logger.info(f"   GA Population: {config.get('ga_population', 'N/A')}")
        logger.info(f"   GA Generations: {config.get('ga_generations', 'N/A')}")
        logger.info(f"   GA Mutation: {config.get('ga_mutation', 'N/A')}")
    
    if 'cost_weight' in config and 'distance_weight' in config:
        logger.info(f"   Weights: Cost={config['cost_weight']:.2f}, Distance={config['distance_weight']:.2f}")


def cleanup_old_logs(max_age_days: int = 7) -> None:
    """
    Clean up old log files to prevent disk space issues
    
    Args:
        max_age_days (int): Maximum age of log files to keep in days
    """
    import time
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    logs_dir = project_root / "logs"
    
    if not logs_dir.exists():
        return
    
    current_time = time.time()
    cutoff_time = current_time - (max_age_days * 24 * 60 * 60)
    
    deleted_count = 0
    for log_file in logs_dir.rglob("*.log"):
        if log_file.stat().st_mtime < cutoff_time:
            try:
                log_file.unlink()
                deleted_count += 1
            except OSError:
                pass
    
    if deleted_count > 0:
        print(f"🧹 Cleaned up {deleted_count} old log files") 