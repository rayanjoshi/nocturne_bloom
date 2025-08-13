import logging
import sys
from pathlib import Path
import colorlog
from datetime import datetime
from typing import Dict, Optional

class ProjectLogger:
    _loggers: Dict[str, logging.Logger] = {}
    _log_dir: Optional[Path] = None
    _base_config_set = False
    @classmethod
    def setup_project_logging(
        cls,
        log_level: str = "INFO",
        log_dir: str = "logs",
        console_output: bool = True,
        file_output: bool = True,
    ):
        if cls._base_config_set:
            return
        
        script_dir = Path(__file__).parent  # /path/to/repo/scripts
        project_root = script_dir.parent     # /path/to/repo/
        cls._log_dir = Path(project_root / log_dir).resolve()
        cls._log_dir.mkdir(parents=True, exist_ok=True)

        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        root_logger.handlers.clear()
        
        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s %(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'white',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                },
        )

        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handlers = []
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_handler.setFormatter(console_formatter)
            handlers.append(console_handler)
        if file_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = cls._log_dir / f"nvda_predictor_{timestamp}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)
        
        for handler in handlers:
            root_logger.addHandler(handler)
        
        cls._base_config_set = True
        
        setup_logger = cls.get_logger("logging_setup")
        setup_logger.info("=" * 60)
        setup_logger.info("PROJECT LOGGING INITIALIZED")
        setup_logger.info("=" * 60)
        setup_logger.info(f"Log Level: {log_level}")
        setup_logger.info(f"Log Directory: {cls._log_dir.absolute()}")
        setup_logger.info(f"Console Output: {console_output}")
        setup_logger.info(f"File Output: {file_output}")
        if file_output:
            setup_logger.info(f"Log File: {log_file.name}")
        setup_logger.info("=" * 60)

    @classmethod
    def get_logger(cls, name: str = None) -> logging.Logger:
        """
        Get a logger instance for a specific module/component.
        
        Args:
            name: Logger name. If None, uses the calling module's name.
        
        Returns:
            Configured logger instance
        """
        if not cls._base_config_set:
            # Auto-setup with defaults if not already configured
            cls.setup_project_logging()
        
        if name is None:
            # Get the calling module's name
            import inspect
            frame = inspect.currentframe().f_back
            name = frame.f_globals.get('__name__', 'unknown')
        
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
            
            # Log the logger creation
            logger.info(f"📝 Logger '{name}' initialized")
        
        return cls._loggers[name]
    
    @classmethod
    def log_function_start(cls, func_name: str, **kwargs) -> logging.Logger:
        """Helper to log function start with parameters."""
        logger = cls.get_logger()
        logger.info("=" * 50)
        logger.info(f" STARTING: {func_name}")
        logger.info("=" * 50)
        
        if kwargs:
            logger.info("Parameters:")
            for key, value in kwargs.items():
                # Don't log sensitive information
                if any(sensitive in key.lower() for sensitive in ['password', 'token', 'key', 'secret']):
                    value = "***HIDDEN***"
                logger.info(f"  {key}: {value}")
        
        return logger
    
    @classmethod
    def log_function_end(cls, func_name: str, success: bool = True, **results) -> logging.Logger:
        """Helper to log function completion."""
        logger = cls.get_logger()
        
        if success:
            logger.info("=" * 50)
            logger.info(f"COMPLETED: {func_name}")
        else:
            logger.error("=" * 50)
            logger.error(f"FAILED: {func_name}")
        
        if results:
            logger.info("Results:")
            for key, value in results.items():
                logger.info(f"  {key}: {value}")
        
        logger.info("=" * 50)
        return logger

# Convenience functions for easy importing
def get_logger(name: str = None) -> logging.Logger:
    """Get a project logger instance."""
    return ProjectLogger.get_logger(name)

def setup_logging(**kwargs):
    """Set up project-wide logging configuration."""
    return ProjectLogger.setup_project_logging(**kwargs)

def log_function_start(func_name: str, **kwargs) -> logging.Logger:
    """Log function start with parameters."""
    return ProjectLogger.log_function_start(func_name, **kwargs)

def log_function_end(func_name: str, success: bool = True, **results) -> logging.Logger:
    """Log function completion."""
    return ProjectLogger.log_function_end(func_name, success, **results)
