import sys
from loguru import logger
from rich.logging import RichHandler

def setup_logger():
    """Sets up a production-ready logger using Loguru and Rich."""
    # Remove default handler
    logger.remove()

    # Add Rich handler for beautiful console output
    logger.add(
        RichHandler(rich_tracebacks=True, markup=True),
        format="{message}",
        level="INFO",
    )

    # Add file handler for production logs
    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="10 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )

    return logger

# Initialize logger
logger = setup_logger()
