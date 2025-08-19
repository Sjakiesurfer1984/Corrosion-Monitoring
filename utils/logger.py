# utils/logger.py

import logging
import sys

# Create a consistent logger for the whole project
logger = logging.getLogger("corrosion_project")
logger.setLevel(logging.DEBUG)  # Use DEBUG during dev, change to INFO or WARNING in prod

# Check if handlers are already attached (avoid duplicate logs)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    
    # ASCII-safe formatter (no fancy arrows, emojis etc.)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
