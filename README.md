# Aorta disease conclusion API

API для написания заключения по заболеваниям аорты

**Вход:** "диаметр `восходящей` аорты `4.3 см`"

**Ответ:**
```txt
Заключение:
<Диагноз>
<Лечение>
<Рекомендации>
```

## Requirements

Создание и активация venv:
```bash
python3 -m venv rag_api
source rag_api/bin/activate
```

Установка зависимостей:
```bash
pip3 install -r requirements.txt
```

## Start

```bash
python3 main.py
```
