import os
import sys

from dotenv import load_dotenv
from codes.GlobalVars import *
from codes.utils import *

CONFIG = None


def get_config():
    if CONFIG is None:
        init()
    return CONFIG


def init(env_file=None):
    global CONFIG
    load_dotenv(env_file)
    config = {}
    config = translation(config, env_file)
    CONFIG = config
    return config


def translation_pr(config, env_file=None):
    if config is None:
        config = {}
    if TR_PR not in config:
        if TR_PR in os.environ:
            tr_pr = os.environ[TR_PR]
            if tr_pr is not None:
                config[TR_PR] = tr_pr
        else:
            tr_pr = os.getenv(TR_PR)
            if tr_pr is not None:
                config[TR_PR] = tr_pr
            else:
                log_error("Fatal Error: Translation Provider not set")
                sys.exit(1)
        pass
    return config


def translation_model(config, env_file=None):
    if config is None:
        print("true")
        config = {}
    if TR_MODEL not in config:
        if TR_MODEL in os.environ:
            tr_m = os.environ[TR_MODEL]
            if tr_m is not None:
                config[TR_MODEL] = tr_m
        else:
            tr_m = os.getenv(TR_MODEL)
            if tr_m is not None:
                config[TR_MODEL] = tr_m
            else:
                log_error("Fatal Error: Translation Model not set")
                sys.exit(1)
        pass
    return config


def translation_api_url(config, env_file=None):
    if config is None:
        config = {}
    if TR_API_URL not in config:
        if TR_API_URL in os.environ:
            tr_api_url = os.environ[TR_API_URL]
            if tr_api_url is not None:
                config[TR_API_URL] = tr_api_url
        else:
            tr_api_url = os.getenv(TR_API_URL)
            if tr_api_url is not None:
                config[TR_API_URL] = tr_api_url
            else:
                log_error("Fatal Error: Translation API url not set")
                sys.exit(1)
        pass
    return config


def translation_api_key(config, env_file=None):
    if config is None:
        config = {}
    if TR_API_KEY not in config:
        if TR_API_KEY in os.environ:
            tr_api_key = os.environ[TR_API_KEY]
            if tr_api_key is not None:
                config[TR_API_KEY] = tr_api_key
        else:
            tr_api_key = os.getenv(TR_API_KEY)
            if tr_api_key is not None:
                config[TR_API_KEY] = tr_api_key
            else:
                log_error("Fatal Error: Translation API key not set")
                sys.exit(1)
        pass
    return config


def translation(config, env_file=None):
    config = translation_pr(config, env_file)
    print(config)
    config = translation_model(config, env_file)
    print(config)
    config = translation_api_url(config, env_file)
    print(config)
    config = translation_api_key(config, env_file)
    print(config)
    return config
