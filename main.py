from fastapi import FastAPI
from pydantic import BaseModel
import typing as tp
import logging
import json
import uvicorn
import colorama
from colorama import Fore, Style
import typing as tp

import auth
from models import (
    GenerativeModel,
    YandexGptModel,
    YandexGptModelConfig,
    QwenModel,
    QwenModelConfig,
)


# tokens and auth
with open('config.json') as f:
    CONFIG = json.loads(f.read())

auth.load_tokens(CONFIG['tokens_path'])

# models
ya_config = YandexGptModelConfig(CONFIG['er_file'], **CONFIG['yandexgpt_secrets'])
qwen_config = QwenModelConfig(CONFIG['er_file'])

model_names = {k: k for k in CONFIG['available_models']}
generative_models: tp.List[GenerativeModel] = {
    'yandexgpt': YandexGptModel(ya_config),
    'qwen': QwenModel(qwen_config),
}

# logger
logger = logging.getLogger('rag_logger')
logger.setLevel(logging.DEBUG)
logger.propagate = True
colorama.init(autoreset=True)

if not logger.hasHandlers():
    LOG_COLORS = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    class ColoredFormatter(logging.Formatter):
        def format(self, record):
            log_color = LOG_COLORS.get(record.levelno, Fore.WHITE)
            record.levelname = f"{log_color}{record.levelname}{Style.RESET_ALL}"
            record.msg = f"{record.msg}"
            return super().format(record)
    
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = ColoredFormatter('[%(asctime)s]: %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


app = FastAPI()


@app.get('/ping')
async def ping() -> tp.Dict[str, tp.Any]:
    logger.info(f'ping request')
    return {
        'status': 'success',
        'response': {'message': 'pong'},
    }


class Query(BaseModel):
    token: str
    message: str
    model: str
    max_len: tp.Optional[int] = None


@app.post('/generate')
async def generate(query: Query) -> tp.Dict[str, tp.Any]:
    user = auth.get_user(query.token)

    if user is None:
        logger.info(f'generate request failed: wrong token "{query.token}"')
        return {
            'status': 'failed',
            'message': 'wrong token',
        }
    
    model_name = model_names.get(query.model, None)

    if model_name is None:
        logger.info(f'generate request failed: wrong model "{query.model}"')
        return {
            'status': 'failed',
            'message': 'wrong model',
        }
    
    model = generative_models[model_name]
    
    answer = model.generate(query.message, query.max_len)

    if query.max_len and len(answer) > query.max_len:
        answer = answer[:query.max_len]

    logger.info(
        f'generate request successed: '
        f'user={user}, '
        f'answer_len={len(answer)}, '
        f'input="{query.message[:min(50, len(query.message))]}"'
    )
    return {
        'status': 'success',
        'response': {'message': answer},
    }


@app.post('/test/generate')
async def test_generate(query: Query) -> tp.Dict[str, tp.Any]:
    user = auth.get_user(query.token)

    if user is None:
        logger.info(f'generate request failed: wrong token "{query.token}"')
        return {
            'status': 'failed',
            'message': 'wrong token',
        }
    
    model_name = model_names.get(query.model, None)

    if model_name is None:
        logger.info(f'generate request failed: wrong model "{query.model}"')
        return {
            'status': 'failed',
            'message': 'wrong model',
        }
    
    answer = 'Lorem Ipsum is simply dummy text of the printing and typesetting industry.'

    if query.max_len and len(answer) > query.max_len:
        answer = answer[:query.max_len]

    logger.info(
        f'test generate request successed: '
        f'user={user}, '
        f'answer_len={len(answer)}, '
        f'input="{query.message[:min(50, len(query.message))]}"'
    )
    return {
        'status': 'success',
        'response': {'message': answer},
    }


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
