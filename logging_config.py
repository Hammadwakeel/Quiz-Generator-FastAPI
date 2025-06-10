import logging

# Configure a module‐level logger for RAG components
logger = logging.getLogger("app.rag")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
