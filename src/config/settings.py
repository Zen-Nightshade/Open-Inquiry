from dotenv import load_dotenv
from repo_paths import PROVIDERS_ENV, get_db_path

# Primary Groq models verified available on the active endpoint:
GROQ_LARGE = "groq_gpt_oss_120b"
GROQ_SMALL = "groq_gpt_oss_20b"
GROQ_QWEN = "groq_qwen3_8_27b"


def init_env():
    """Load API credentials and configuration from PROVIDERS_ENV."""
    load_dotenv(PROVIDERS_ENV)

