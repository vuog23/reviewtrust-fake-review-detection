# ReviewTrust

ReviewTrust is a fake-product-review analysis app. It accepts review text or an image and presents the result in three readable stages.

![ReviewTrust pipeline](images/Landing.png)

## Pipeline

1. **Image reading (optional)** — `qwen/qwen3.6-27b` on Groq Cloud transcribes review text from PNG, JPEG, or WebP images.
2. **Classification** — a calibrated ModernBERT model, running locally, predicts deceptive or non-deceptive probabilities and rejects low-confidence decisions as uncertain.
3. **Evidence extraction** — GLiNER2, running locally, highlights suspicious phrases and displays each entity probability.
4. **Explanation** — `qwen/qwen3.6-27b` on Groq Cloud explains the classifier result using evidence found in the review.

![ReviewTrust analysis](images/Demo.png)

## Requirements

- Python 3.10+
- A Groq Cloud API key for the explanation and image-reading stages
- Internet access for the first GLiNER2 download, and for every Groq request
- An NVIDIA CUDA GPU is optional; the local classifier and entity models also run on CPU
- The fine-tuned classifier checkpoint in `models/` (see below)

## The classifier checkpoint is not in this repository

`config.yaml` loads the calibrated classifier from `models/` — `model.safetensors`, its
tokenizer, and the `temperature.json` produced by calibration. That directory is gitignored:
the checkpoint is roughly 600 MB, well past GitHub's 100 MB per-file limit. GLiNER2 downloads
itself into `cache/` on first run, but the classifier cannot, because it is a fine-tune of
`answerdotai/ModernBERT-base` rather than a published model.

A fresh clone therefore starts and serves the page, but the pipeline fails to initialize:
`/api/health` reports `pipeline_ready: false` and every analysis returns 503. Supply your own
`models/` directory to make it run.

## Configure the API key

Both hosted stages read the `GROQ_API_KEY` environment variable. Get a key from
[console.groq.com/keys](https://console.groq.com/keys) and put it in a `.env` file at the project root:

```
GROQ_API_KEY=gsk_your_key_here
```

Copy `.env.example` to get started. `.env` is gitignored; `config.yaml` is not, so the key must never
go there — `api_key_env` holds the *name* of the variable, not the value.

`.env` is loaded at startup and deliberately overrides an existing environment variable of the same
name, so a key exported in some shell months ago cannot silently shadow the project's own. The server
logs `Loaded from .env: GROQ_API_KEY` when it does this. An ordinary environment variable works too if
you would rather not use a file.

Without a key the server still starts and the classification and entity stages still run, but
`/api/health` reports `degraded` and the explanation and image-reading stages return a 503 naming the
variable to set.

The model id and request settings live under `reasoner_inference` and `ocr_inference` in `config.yaml`.

`qwen/qwen3.6-27b` is a Groq **preview** model, which carries two operational consequences:

- Groq may retire it without notice. If requests start failing with "Groq does not serve a model named…",
  pick a current id from [console.groq.com/docs/models](https://console.groq.com/docs/models).
- It gets capacity limited. Groq answers `503 … is currently over capacity` under load, and the app
  surfaces that as a 503 telling you to retry. This was observed during development and cleared on retry.
  The exact provider message is always written to the server log.

### Rate limits

Groq meters tokens per minute, and the free tier is small — 8,000 TPM on the key this was developed
against. Measured against that key:

| Request                | Tokens                                             | Wall clock (unthrottled) |
| ---------------------- | -------------------------------------------------- | ------------------------ |
| `/api/analyze`       | ~3,300 (the system prompt alone is ~2,100)         | ~2.8 s end to end        |
| `/api/analyze-image` | ~4,600 (transcription ~1,400 + explanation ~3,200) | —                       |

That is roughly **two analyses per minute** before the budget runs out. Beyond it Groq returns 429 and the
client retries with backoff, so a throttled request looks like one that takes 45–100 seconds rather than
one that fails. The server logs `Groq … used N prompt + M completion tokens` on every call, so compare
that against your plan's limit whenever requests feel slow. To buy headroom: raise the limit on your Groq
plan, shorten `SYSTEM_PROMPT` in `src/utils.py`, or lower `ocr_inference.max_image_edge`.

## Tuning entity extraction

The entity stage highlights concrete risk signals — payment offers, rating requests, contact details —
so a reviewer can see *why* something looks wrong. It is a separate stage from the classifier, and the
explanation model is not shown its output.

Its failure mode is not missing the manipulation but flagging ordinary retail vocabulary alongside it,
which buries the signal. `ner_inference.label_thresholds` in `config.yaml` therefore sets a cut per
label rather than one global number, because the labels have very different noise profiles: measured on
genuine reviews, `order_or_transaction` fires on *"price"* and *"Shipping"* around 0.65, while a genuine
instruction to post a review only reaches about 0.23. A single threshold either admits the first or
rejects the second.

Two related settings: `model_floor` is what the model itself is asked for (kept low, so filtering
happens against the tuned cuts), and `min_span_chars` drops one-word fragments. Overlapping spans are
resolved so the same characters are not highlighted twice.

The **Entity detail** slider shifts every label threshold by the same amount, so the relative tuning is
preserved wherever you put it.

## Run

```
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Open [http://127.0.0.1:8001](http://127.0.0.1:8001).

On Windows the app registers the Conda and Torch CUDA DLL directories at startup, so run it
from the same environment the dependencies were installed into.

## Usage

- Paste a product review and select **Analyze review**.
- Or upload a review image and select **Extract text & analyze**.
- Read the classification, highlighted evidence, and explanation on the right.

The GLiNER2 download is cached in `cache/`. Deleting that folder is safe while the server is
stopped; it is fetched again on the next run. `models/` is different — it holds the fine-tuned
classifier and nothing re-downloads it.

## Tests

```
python -m pytest -q
```

The suite runs entirely offline — the hosted stages are exercised through an injected fake client, so no
API key and no network access are needed to run it.

ReviewTrust is a decision-support tool; its output is not proof of fraud.
