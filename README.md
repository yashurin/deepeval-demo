# Local DeepEval demo

A small customer-support bot answers from a fixed store policy. [DeepEval](https://deepeval.com) scores those answers. [Ollama](https://ollama.com) serves one open model, `qwen3.5:2b` by default, and that same model is both the bot and the judge (`deepeval.models.OllamaModel`).

No API keys. The first `docker compose up` downloads the default model into a Docker volume.

## What you need

- Docker with Compose v2
- About 8 GB of RAM for the Docker VM (10 GB is more comfortable if you compare two models)
- About 5 GB of free disk for the image and `qwen3.5:2b` (about 3 GB). `phi4-mini` adds about 2.5 GB

On a Mac, Docker runs Ollama on CPU. There is no GPU inside this container.

## Run it

This one command starts Ollama, waits until it is healthy, pulls `qwen3.5:2b` if the volume does not already have it, scores the return-policy question, and prints the report:

```bash
cp .env.example .env          # optional; every variable has a default
docker compose up --build --abort-on-container-exit
```

`--abort-on-container-exit` stops Ollama when the report is printed. The model stays in the `ollama_models` volume.

Pull the default model without scoring:

```bash
docker compose run --rm deepeval-app python -m app.pull
```

Extra tags are arguments: `python -m app.pull phi4-mini`.

## Web UI

A Gradio page lets you type a prompt, generate one answer, and score it. `docker compose up` starts it with Ollama.

```bash
docker compose up --build
```

Open **http://localhost:7860**. Leave that tab open. The log line Gradio used to print, `http://0.0.0.0:7860`, is not a page browsers can open.

Do not add `--abort-on-container-exit` if you want the page. That flag stops every container when the one-shot eval exits, so the browser then has nothing to connect to.

Leave the context box empty for a plain answer. Paste policy excerpts, one per paragraph, to score faithfulness and contextual relevancy. Correctness also needs an expected answer. Relevancy and helpfulness are selected by default.

The process listens on port 7860 inside the container. Set `UI_PORT` in `.env` when that port is taken on the host, and open `http://localhost:` plus that number.

Each selected metric is another local generation. On CPU, one click can take several minutes, and the page runs one click at a time.

## Run the full check

Eleven questions and five metrics (answer relevancy, faithfulness, G-Eval correctness, G-Eval helpfulness, and contextual relevancy):

```bash
docker compose run --rm deepeval-app python -m app.run_evals
```

`python -m app.evaluate` is the same command. Eight questions include a policy excerpt (retrieval context). One of those is a follow-up turn. Three are plain questions, and those skip faithfulness and contextual relevancy, which need that context.

Start Ollama first if it is not already running:

```bash
docker compose up -d ollama
docker compose run --rm deepeval-app python -m app.evaluate
```

Exit 1 when a score misses its threshold:

```bash
docker compose run --rm deepeval-app python -m app.evaluate --fail-on-threshold
```

Ask the bot without scoring:

```bash
docker compose run --rm deepeval-app python -m app.chat "What is your return policy?"
```

The pytest-style version of the quick check:

```bash
docker compose run --rm deepeval-app deepeval test run app/pytest_example.py
```

## Compare two models

`--compare` asks a second Ollama model the same questions and prints one score column per model. The judge stays `OLLAMA_MODEL`, so the columns share one scorer. Set `JUDGE_MODEL` only when you want a different scorer.

Try it on the single returns question first. The first run also pulls `phi4-mini`:

```bash
docker compose run --rm deepeval-app python -m app.run_evals --compare phi4-mini --quick
```

The full suite against both models:

```bash
docker compose run --rm deepeval-app python -m app.run_evals --compare phi4-mini
```

Or set `COMPARE_MODEL=phi4-mini` in `.env` and leave the flag off. Between steps the app asks Ollama to unload the model that is not in use, so the two weight files are not kept in RAM together.

## Switch models

Edit `.env` and run again:

```bash
OLLAMA_MODEL=qwen3.5:4b
```

The next run pulls that tag and uses it as both the bot and the judge. Weights you already pulled stay in the volume.

One run, without editing the file:

```bash
docker compose run --rm -e OLLAMA_MODEL=phi4-mini deepeval-app python -m app.evaluate --quick
```

`OLLAMA_THINK=false` turns off qwen3.5's thinking mode. Thinking text lands in the middle of the JSON the judge must return. Set `OLLAMA_THINK=omit` for a model that rejects the flag.

## Add your own test cases

Edit `app/sample_data.py` and append a `Golden`. The container mounts `./app`, so the next `docker compose run` picks it up without a rebuild.

A plain question:

```python
Golden(
    name="capital-of-japan",
    question="What is the capital of Japan?",
    expected_output="The capital of Japan is Tokyo.",
)
```

A RAG question. Reuse a policy constant from `app/support_bot.py`, or add a new string there and to `POLICY_CHUNKS` so `python -m app.chat` can retrieve it too:

```python
Golden(
    name="gift-card",
    question="Do gift cards expire?",
    expected_output="Gift cards never expire.",
    retrieval_context=(GIFT_CARD_POLICY,),
)
```

A follow-up. `history` is earlier `(customer, assistant)` turns. The bot answers only `question`. DeepEval still receives one test case; the earlier turn is part of `input`:

```python
Golden(
    name="return-window-followup",
    question="What if I already wore them once?",
    expected_output="Worn items are not eligible for return.",
    retrieval_context=(RETURNS_POLICY,),
    history=(
        (
            "What is your return policy if the shoes don't fit?",
            "Unworn items can be returned within 30 days.",
        ),
    ),
)
```

Keep `returns` as the first golden. `--quick` and the pytest example use that case.

## Resource use

| | Disk | RAM while it runs | Time on a CPU-only Mac |
| --- | --- | --- | --- |
| Quick check (`docker compose up`) | about 3 GB for `qwen3.5:2b`, kept in the volume | a few GB at `NUM_CTX=8192` | a few minutes for one question and one metric |
| Full check | same weights | same | much longer: 11 questions and 5 metrics, and each metric is another local generation |
| `--compare phi4-mini` | about 2.5 GB more | one small model at a time | about twice the matching single-model run, plus the extra download the first time |

Give the Docker VM about 8 GB for one small model. Use 10 GB or more when you compare, or when you switch to a larger tag such as `qwen3.5:4b`. Lower `NUM_CTX` in `.env` if the VM runs out of memory. A 2B judge is noisy. Treat the scores as a way to see the framework work.

## Example output

The numbers below are an illustration of the report shape. A real run fills them from your local judge, and they will differ.

```
Local DeepEval demo
  Ollama:           http://ollama:11434
  Model under test: qwen3.5:2b, phi4-mini
  Judge:            qwen3.5:2b (deepeval.models.OllamaModel)
  Metrics:          correctness
  JSON summary:     /app/results/summary.json

=== Answers from qwen3.5:2b ===

[returns]
Question: What is your return policy if the shoes don't fit?
Retrieved policy:
  - Customers may return unworn items within 30 days of delivery for a full refund. Return shipping is free.
Answer:   Unworn items can be returned within 30 days of delivery for a full refund, and return shipping is free.

Summary (qwen3.5:2b: correctness)
  returns / Correctness: pass (score 0.80, threshold 0.5)
    reason: The answer states the same refund and the same 30-day window.

Score table
case     metric       qwen3.5:2b  phi4-mini
-------  -----------  ----------  ---------
returns  Correctness  0.80 pass   0.40 fail

JSON summary: /app/results/summary.json
Overall: fail
```

On the host that JSON file is `results/summary.json`. Pass `--json results/compare.json` to write somewhere else. Each row has the case, metric, model, score, threshold, pass/fail, reason, and error.

```json
{
  "judge_model": "qwen3.5:2b",
  "models": ["qwen3.5:2b", "phi4-mini"],
  "metrics": ["correctness"],
  "passed": false,
  "rows": [
    {
      "case": "returns",
      "metric": "Correctness",
      "model": "qwen3.5:2b",
      "score": 0.8,
      "threshold": 0.5,
      "passed": true,
      "reason": "The answer states the same refund and the same 30-day window.",
      "error": null
    }
  ]
}
```

DeepEval also writes its own report files in the same `results/` folder.

## What the evaluation is looking at

Each question becomes one `LLMTestCase`:

| Field | Where it comes from |
| --- | --- |
| `input` | The customer question. A follow-up also includes the earlier turns |
| `actual_output` | The bot's answer from Ollama |
| `expected_output` | The golden answer in `app/sample_data.py` |
| `retrieval_context` | The policy excerpt on a RAG case. Empty for a plain question |

Metrics, all scored from 0 to 1 by the local model:

- **Answer relevancy** compares the answer with the question.
- **Faithfulness** checks the answer against `retrieval_context`. A fluent answer that invents a policy fails here.
- **Correctness** is a G-Eval metric. It compares the answer with `expected_output`.
- **Helpfulness** is a G-Eval metric. It asks whether the answer is specific, on topic, and usable. An honest "I don't know" can pass. Filler does not.
- **Contextual relevancy** checks whether the retrieved excerpt is about the question. It does not grade the answer.

The Antarctica question is not in the policy. The bot is shown the return policy anyway and is instructed to say it does not know. Faithfulness can pass on that abstention while contextual relevancy fails, because the excerpt is about returns.

`express-and-returns` is shown two excerpts at once. `return-window-followup` continues the returns conversation and asks about shoes that were already worn.

A dropped connection, an HTTP 5xx, an empty answer, or judge JSON that does not match the schema is tried again. `OLLAMA_RETRIES` is the number of extra attempts (default 2).

Thresholds live in `.env`.

## Layout

```
docker-compose.yml    Ollama service and the evaluation app
Dockerfile             Python image for the app
requirements.txt
.env.example
app/support_bot.py     bot + tiny keyword retriever
app/ollama_client.py   HTTP calls for the model under test, with retries
app/judge.py           DeepEval OllamaModel subclass (the judge)
app/sample_data.py     11 questions and expected answers
app/dataset.py         re-exports sample_data, for older imports
app/generate.py        asks the model and builds LLMTestCase objects
app/evaluate.py        metrics, deepeval.evaluate(), table, JSON summary
app/run_evals.py       command line, including --compare
app/pull.py            download a model and exit
app/ui.py              Gradio page at http://localhost:7860
app/chat.py            one question, no metrics
app/pytest_example.py  the same check via assert_test
```

`app/` is mounted into the container, so edits apply on the next `docker compose run` without a rebuild.

## Troubleshooting

**The first start sits on "Pulling qwen3.5:2b".** That download is about 3 GB. Later runs reuse the volume. `docker compose up` pulls it for you. To pull without scoring: `docker compose run --rm deepeval-app python -m app.pull`.

**The container exits with a timeout.** The image sets `DEEPEVAL_DISABLE_TIMEOUTS=1` because a 2B model on CPU can spend more than DeepEval's default 180 seconds on one metric. Leave that set.

**JSON errors from the judge.** Keep `OLLAMA_THINK=false`. If you pointed `OLLAMA_MODEL` at a model that errors on that flag, set `OLLAMA_THINK=omit`.

**Docker runs out of memory.** Raise the Docker VM memory, or lower `NUM_CTX` (default `8192`) in `.env`. Comparison mode unloads the idle model, and it still needs headroom while one model is loaded.

**Port 11434 is already taken.** A local Ollama app often listens there. In `.env` set `OLLAMA_PORT=11435` and run Compose again. From your machine the container is then `http://localhost:11435`. The app container still talks to `http://ollama:11434` on the Compose network.

**You want a GPU.** On a Linux host with the NVIDIA Container Toolkit, add a GPU reservation to the `ollama` service. Docker Desktop on macOS does not pass a GPU into this container.
