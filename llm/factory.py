import os
from pathlib import Path
from dotenv import load_dotenv

from . import LoadbyGoogle
from . import LoadbyHFLocalonly
from . import LoadbyHuggingFace
from . import LoadbyLlamaCpp
from . import LoadbyNvidia
from . import LoadbyOllama
from . import LoadbyOpenRouter

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"


def load_config(env_path=None):
    env_path = env_path or ENV_PATH
    print(f"Loading .env from: {env_path}")
    load_dotenv(dotenv_path=env_path)


def load_llm(by, model=None):
    if by is None:
        return None

    context_size = os.getenv("CONTEXT_WINDOWS_SIZE", "4096")

    match by:
        case "HuggingFace":
            return LoadbyHuggingFace.load_llm((
                model or os.getenv("HF_MODEL"),
                os.getenv("HF_TOKEN"),
                context_size,
            ))
        case "HFLocalonly":
            return LoadbyHFLocalonly.load_llm((
                model or os.getenv("HF_MODEL"),
                os.getenv("HF_TOKEN"),
                context_size,
            ))
        case "Ollama":
            return LoadbyOllama.load_llm((
                model or os.getenv("OLLAMA_MODEL"),
                os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                context_size,
            ))
        case "LlamaCpp":
            return LoadbyLlamaCpp.load_llm((
                model or os.getenv("LC_MODEL_PATH"),
                context_size,
            ))
        case "OpenRouter_API":
            return LoadbyOpenRouter.load_llm((
                os.getenv("OPENROUTER_API_KEY"),
                model or os.getenv("OPENROUTER_MODEL"),
                os.getenv("OPENROUTER_BASE_URL"),
                context_size,
            ))
        case "Google_API":
            return LoadbyGoogle.load_llm((
                os.getenv("GOOGLE_API_KEY"),
                model or os.getenv("GOOGLE_MODEL"),
                context_size,
            ))
        case "Nvidia_API":
            return LoadbyNvidia.load_llm((
                os.getenv("NVIDIA_API_KEY"),
                model or os.getenv("NVIDIA_MODEL"),
                context_size,
            ))
        case _:
            return None


def set_langsmith(project_name="agent_skills"):
    os.environ["LANGSMITH_PROJECT"] = project_name
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")