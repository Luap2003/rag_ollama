import typing
import logging
import colorlog

def setup_logger(log_level=logging.INFO) -> logging.Logger:
    """
    Sets up the logger with colored log levels and a specific format.
    
    Args:
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("ChromaDBProcessor")
    logger.setLevel(log_level)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    
    # Define log colors
    log_colors = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
    
    # Create a ColoredFormatter without fixed-width padding for levelname
    formatter = colorlog.ColoredFormatter(
        fmt="%(log_color)s%(levelname)s:%(reset)s\t[%(asctime)s] %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        log_colors=log_colors
    )
    
    # Add formatter to the handler
    ch.setFormatter(formatter)
    
    # Add handler to the logger if it's not already added
    if not logger.handlers:
        logger.addHandler(ch)
    
    return logger