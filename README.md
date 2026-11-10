# Hoplite OpenAI-compatible proxy

Прокси принимает запросы OpenAI Chat Completions (`POST /v1/chat/completions`, включая SSE-стриминг) и выполняет их в облачных агентах [Hoplite](https://hoplite.sh). Клиент настраивается так:

```
base_url = http://localhost:8000/v1
api_key  = sk-hop-…   (ключ выдаёт прокси)
```

Внутри каждый ответ — это ран агента Hoplite в облачной песочнице, а не вызов LLM. Агент сам читает репозиторий проекта, запускает команды и правит код. Один ответ занимает от десятков секунд до нескольких минут, поэтому для длинных задач лучше включать `stream: true`.

## Как это работает

```
клиент ──sk-hop──▶ прокси ──hop_…──▶ api.hoplite.sh
                     │  POST /api/threads            (новый тред: вся история в prompt)
                     │  POST /api/threads/{id}/messages (продолжение разговора)
                     │  GET  /api/events/stream      (SSE: черновики ответа → дельты OpenAI)
                     │  GET  /api/threads/{id}/run-state, /messages, /usage
```

- **Ключи.** Каждый `sk-hop-…` привязан к ключу Hoplite (`hop_…` или `hop_svc_…`) и проекту. В SQLite хранятся только bcrypt-хеш секретной части `sk-hop` и Hoplite-ключ, зашифрованный Fernet.
- **Треды.** Клиенты OpenAI каждый раз присылают всю историю. Прокси запоминает отпечаток «история + выданный ответ». Если следующий запрос продолжает эту историю, новые сообщения дописываются в тот же тред Hoplite: агент работает в той же песочнице и видит те же файлы. Регенерация, правка истории или параллельная ветка создают новый тред, куда вся история уходит одним prompt.
- **Стриминг.** Hoplite отдаёт полные снимки текущего ответа («черновики»). Прокси превращает их в дельты `delta.content`. Пока поднимается песочница, каждые 15 с идёт `: keep-alive`. В конце приходят `finish_reason`, `usage` (если указан `stream_options.include_usage`) и `[DONE]`.
- **Инструменты клиента (tool bridge).** Агентные клиенты вроде opencode присылают свои инструменты (`read`, `edit`, `bash`…). Прокси описывает их агенту Hoplite текстовым протоколом, а ответ агента с блоком `<tool_calls>` возвращает клиенту как обычные OpenAI `tool_calls`. Клиент выполняет инструменты **у себя локально** и присылает результаты, а прокси дописывает их в тот же тред. Невалидные вызовы и попытки агента работать в своей песочнице прокси ловит сам и просит агента исправиться.
- **Вопросы агента.** Если агент задаёт вопрос (ран в статусе `waiting`), вопрос с вариантами дописывается в ответ. Следующее сообщение клиента отправляется как ответ на него. При `QUESTION_POLICY=auto` прокси отвечает сам.
- **Повторы и обрывы.** Повтор того же запроса (openai SDK ретраит после 600 с таймаута) подключается к уже идущему рану, а не запускает второго агента. Если клиент отключился, ран ждёт 60 с, не придёт ли повтор, и затем останавливается (`POST /stop`).
- **Безопасность.** Новые треды создаются с `autoMerge: false` и лимитом трат `DEFAULT_SPEND_LIMIT_MICROS` ($1 по умолчанию, можно задать для каждого ключа).

## Быстрый старт (Windows, PowerShell)

```powershell
# 1. venv вне OneDrive (синхронизация мешает venv и SQLite)
python -m venv $env:USERPROFILE\.venvs\hoplite-proxy
& $env:USERPROFILE\.venvs\hoplite-proxy\Scripts\Activate.ps1
pip install -e ".[dev]"

# 2. конфигурация
copy .env.example .env
python -m app.cli gen-secrets      # вставьте MASTER_KEY и ENCRYPTION_KEYS в .env
#    в .env также задайте HOPLITE_API_KEY=hop_… (и, если проектов несколько, HOPLITE_PROJECT_ID)

# 3. запуск (или двойной клик по start-proxy.cmd: окно с логами, закрыть окно = остановить)
python -m app                       # http://127.0.0.1:8000, документация API: /docs

# 4. выпустить ключ для клиента (Hoplite-ключ читается из .env, в историю команд не попадает)
python -m app.cli create-key --name my-agent
```

Чтобы узнать ID проектов и моделей, запустите `python scripts/probe_hoplite.py list`: скрипт только читает данные.

## Подключение клиентов

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-hop-…", timeout=1800)
stream = client.chat.completions.create(
    model="hoplite-default",   # или id модели Hoplite: gpt-6-sol, claude-sonnet-5, … (см. /v1/models)
    messages=[{"role": "user", "content": "Добавь healthcheck-эндпоинт и тесты к нему"}],
    stream=True,
    stream_options={"include_usage": True},
)
for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="")
```

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-hop-…" -H "Content-Type: application/json" \
  -d '{"model":"hoplite-default","stream":true,"messages":[{"role":"user","content":"hi"}]}'
```

Дополнительные заголовки:

| Заголовок | Назначение |
|---|---|
| `X-Hoplite-Thread-Id: thr_…` (запрос) | дописать сообщение в конкретный тред |
| `X-Hoplite-New-Thread: 1` (запрос) | всегда начинать новый тред |
| `X-Hoplite-Thread-Id`, `X-Hoplite-Run-Id` (ответ) | в каком треде и ране выполнен ответ |
| `Idempotency-Key` (запрос) | повтор с тем же ключом получает тот же ответ |

Ответ без стриминга содержит расширение `hoplite: {thread_id, run_id, cost_micros}`. Обычные OpenAI-клиенты его игнорируют.

### opencode

Добавьте провайдер в `C:\Users\<вы>\.config\opencode\opencode.jsonc` (глобально) или в `opencode.json` в папке проекта:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "hoplite": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Hoplite",
      "options": { "baseURL": "http://127.0.0.1:8000/v1", "apiKey": "sk-hop-…" },
      "models": {
        "claude-sonnet-5-5": { "name": "Claude Sonnet 5.5 (Hoplite)" },
        "claude-haiku-5-5": { "name": "Claude Haiku 5.5 (Hoplite)" }
      }
    }
  }
}
```

Дальше всё как обычно: запустите `opencode` и выберите модель `hoplite/…` в `/models`, или выполните `opencode run -m hoplite/claude-haiku-5-5 "…"`.
- Агент Hoplite решает, что делать, а файлы читает и правит сам opencode у вас на диске.
- Каждый шаг агента (ответ с вызовами инструментов) — это один ран Hoplite: около 7–15 секунд и 25–40 тысяч входных токенов.
- Проверено на opencode 1.18.21: чтение, создание и правка файлов, параллельные вызовы.
- Если вызываете `opencode run` из скрипта, закрывайте ему stdin (`< /dev/null`). Иначе он ждёт ввода и не отправляет запрос.

## Админ-API

Все запросы требуют `Authorization: Bearer $MASTER_KEY` (или `X-Admin-Key`).

```bash
# выпустить ключ (project_id можно не указывать, если у Hoplite-ключа один проект)
curl -s http://localhost:8000/admin/keys -H "Authorization: Bearer $MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"hoplite_api_key":"hop_…","project_id":"proj_…","name":"ci-agent",
       "default_model":"claude-sonnet-5","spend_limit_micros":2000000,"max_concurrency":2}'
# → 201 {"id":"…","key":"sk-hop-…", …}  (ключ показывается один раз)

curl -s http://localhost:8000/admin/keys -H "Authorization: Bearer $MASTER_KEY"            # список без секретов
curl -s -X PATCH http://localhost:8000/admin/keys/<id> -H "Authorization: Bearer $MASTER_KEY" \
  -H "Content-Type: application/json" -d '{"hoplite_api_key":"hop_svc_новый"}'          # ротация Hoplite-ключа
curl -s -X DELETE http://localhost:8000/admin/keys/<id> -H "Authorization: Bearer $MASTER_KEY"  # отзыв
```

- В `DELETE /admin/keys/{…}` можно передать id или полный `sk-hop-…`, как в ТЗ. Лучше передавать id: URL попадает в логи прокси-серверов (в логах самого прокси путь маскируется).
- `PATCH` меняет Hoplite-ключ, проект, модель и лимиты без перевыпуска `sk-hop`. Это нужно, потому что сервисные ключи `hop_svc_` живут 90 дней.
- При создании и изменении ключа прокси проверяет Hoplite-ключ и проект. Отключить проверку можно параметром `"verify": false`.

## Маппинг OpenAI → Hoplite

| OpenAI | Hoplite |
|---|---|
| `model` | id модели Hoplite, алиас из `MODEL_ALIASES` или `hoplite-default` (модель проекта). Неизвестная модель → 404 `model_not_found` со списком доступных |
| `messages` | Новый тред получает `prompt` из `<instructions>` (system/developer), `<conversation_history>` и последнего запроса. Продолжение получает только новые сообщения. Одно сообщение уходит без обёртки |
| `reasoning_effort` | `reasoning.mode` (`none` … `xhigh`, `max`) |
| `stream`, `stream_options.include_usage` | SSE в формате OpenAI |
| `tools`, `functions`, `tool_choice`, `parallel_tool_calls` | tool bridge: инструменты клиента передаются агенту, его вызовы возвращаются как `tool_calls` с `finish_reason: "tool_calls"`, результаты (`role: tool`) уходят в тот же тред. С `TOOLS_MODE=reject` такие запросы получают 400 |
| `temperature`, `max_tokens`, `top_p`, `n`, `stop`, `seed`, `response_format`, … | у Hoplite нет аналогов: поля игнорируются, их имена пишутся в лог |
| картинки, аудио и файлы в `content` | заменяются на `[image omitted]` и т. п. (в v1) |

| Итог рана Hoplite | Ответ OpenAI |
|---|---|
| `completed` | `finish_reason: "stop"`; текст — все сообщения ассистента за ран |
| `waiting` (вопрос) | `"stop"`, текст вопроса в ответе |
| исчерпан лимит трат треда | `"length"`, если текст уже есть, иначе 402 `insufficient_quota` |
| закончились кредиты / контекст / rate limit | 402 / 400 `context_length_exceeded` / 429 |
| прочие сбои | 502 `run_failed` с причиной из `/failure` |
| превышен `RUN_TIMEOUT_SECONDS` | ран останавливается, клиент получает 504 `run_timeout` |

`usage` показывает токены агента за весь ран (все шаги модели, включая работу с инструментами), а не только видимый текст. Hoplite может досчитывать их с небольшой задержкой.

Ошибки всегда приходят в формате OpenAI: `{"error": {"message", "type", "param", "code"}}`. Что Hoplite отвечает на ошибки:

| Ответ Hoplite | Что получает клиент |
|---|---|
| 429 | 429 с `Retry-After` |
| 401 или 403 | 502 `upstream_auth_error`: ключ клиента верный, но проблема в Hoplite-ключе, к которому он привязан |
| 5xx и сетевые ошибки | повторяются с тем же `Idempotency-Key`, затем 502 или 503 |

## Конфигурация

Все переменные с пояснениями — в [.env.example](.env.example). Основные:

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `MASTER_KEY` | — | доступ к `/admin/*`; если ключ короче 32 символов, админка отключена |
| `ENCRYPTION_KEYS` | — | Fernet-ключи (основной — первый); ротация: `python -m app.cli rotate-encryption` |
| `THREAD_REUSE` | `fingerprint` | `off` — новый тред на каждый запрос |
| `QUESTION_POLICY` | `surface` | `auto` — прокси сам отвечает на вопросы агента |
| `DEFAULT_SPEND_LIMIT_MICROS` | `1000000` | лимит трат нового треда ($1); `0` — без лимита |
| `RUN_TIMEOUT_SECONDS` | `1800` | максимальная длительность рана |
| `STOP_ON_DISCONNECT`, `DISCONNECT_GRACE_SECONDS` | `true`, `60` | остановка рана после ухода клиента |
| `MODEL_ALIASES` | `{}` | JSON, например `{"gpt-4o": "gpt-6-sol"}` |
| `MAX_CONCURRENCY_PER_KEY` | `4` | одновременных ранов на ключ (сверх лимита — 429) |
| `INCLUDE_REASONING` | `false` | сообщения агента типа thinking уходят в `reasoning_content` |
| `TOOLS_MODE` | `bridge` | `reject` — отказывать (400) запросам с `tools` |
| `TOOL_RESULT_MAX_CHARS` | `100000` | обрезать слишком длинные результаты инструментов |
| `TOOL_REPAIR_ATTEMPTS` | `2` | сколько раз просить агента исправить невалидный вызов |

## Ограничения

- **Вызовы инструментов эмулируются.** Агент Hoplite пишет вызовы текстом, а прокси их проверяет. Обычно это работает с первого раза; если нет, прокси просит исправить, на это до 2 попыток.
  - Агентный клиент тратит один ран Hoplite на каждый шаг, поэтому выходит медленнее, чем прямой доступ к модели.
  - Быстрее был бы MCP-мост (инструменты клиента как MCP-сервер для Hoplite), но ему нужен публичный URL до прокси.
- **Каждый ответ — это ран агента** (минуты, а не секунды). В чат-интерфейсах лучше отключить вспомогательные запросы (генерацию заголовков, тегов, подсказок): каждый из них запускает агента и тратит деньги.
- **Таймаут клиента должен быть не меньше `RUN_TIMEOUT_SECONDS`.** Иначе используйте стриминг.
- **Не входит в v1:** картинки (Hoplite принимает вложения через presign, можно добавить), `/v1/responses`, `/v1/embeddings`.
- **Один процесс.** Реестр активных ранов, лимиты параллельности и кэши живут в памяти, поэтому запускайте прокси с одним воркером. Async-сервер с этой нагрузкой справляется.
- **Формат SSE-черновиков Hoplite не задокументирован.** Формат, который ожидает прокси, сверен с живым API в ноябре 2026: `event: draft` с полями `messageId/runId/streamId/revision/content`, затем `draft.removed` (подробности — в `app/hoplite/events.py`). Если Hoplite его поменяет, проверьте поток командой `python scripts/probe_hoplite.py run`.

## Рекомендации по Hoplite-ключу

Для прокси лучше подходит сервисный ключ (Settings → Workspace → API keys → Service accounts) с минимальными правами: Projects — Read, Threads — Write, Model keys — Read, Usage — Read. Сервисные ключи живут 90 дней, ротация — через `PATCH /admin/keys/{id}`. Если ключ попадал в чат, письмо или лог, перевыпустите его.

## Разработка

```powershell
pytest                      # 76 тестов на фейковом Hoplite (tests/fake_hoplite.py), без сети
ruff check .
python -m tests.fake_hoplite --port 9100   # локальный фейковый Hoplite с настоящим стримингом (ключ hop_dev_key)
#   затем в .env: HOPLITE_BASE_URL=http://127.0.0.1:9100 и выпустите ключ на hop_dev_key
python scripts/probe_hoplite.py list       # живой Hoplite: только чтение
python scripts/probe_hoplite.py run        # живой Hoplite: 1 тред, лимит $0.25, сырые SSE-кадры в scripts/out/
$env:PROXY_API_KEY="sk-hop-…"; python scripts/smoke_openai.py   # openai SDK против запущенного прокси
```

Структура: `app/openai_api` — эндпоинты OpenAI, `app/admin` — админ-API, `app/hoplite` — клиент и SSE, `app/service` — промпты, модели, треды, раны, `app/storage` — SQLite, `app/security` — ключи и шифрование.

## Docker

```bash
docker compose up -d --build     # порт 127.0.0.1:8000, база в томе proxy-data
docker compose exec proxy python -m app.cli create-key --name my-agent   # HOPLITE_API_KEY берётся из .env
```
