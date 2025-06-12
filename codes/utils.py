from termcolor import *
import os
from dotenv import load_dotenv

def log(prompt, color="white"):
    cprint(prompt, color=color, force_color=True)


def log_error(prompt, color="red"):
    log(prompt, color)

#
# def load_env_vars():
#     # Load .env file first
#     load_dotenv()
#
#     # Define your environment schema
#     REQUIRED_ENV_VARS = [
#         "API_KEY",
#     ]
#
#     OPTIONAL_ENV_VARS_WITH_DEFAULTS = {
#         "DEBUG": "false",
#         "TIMEOUT": "10",  # seconds
#     }
#
#     def load_and_validate_env():
#         # Load required variables
#         for var in REQUIRED_ENV_VARS:
#             value = os.getenv(var)
#             if not value:
#                 raise EnvironmentError(f"Missing required environment variable: {var}")
#             os.environ[var] = value  # Ensure it's present in os.environ
#
#         # Load optional variables with defaults
#         for var, default in OPTIONAL_ENV_VARS_WITH_DEFAULTS.items():
#             if os.getenv(var) is None:
#                 os.environ[var] = default
#
#         # Optional: log loaded vars for debug (excluding secrets)
#         print("Environment variables loaded successfully.")
#
#     # Call it once at app startup
#     load_and_validate_env()
#     pass
#
#
#
