# Shared scaffolding

The lessons are written to be run, not just read. This folder holds the small amount of shared setup they assume, so you are not guessing versions on every file.

```text
code/
├── requirements.txt   Python dependencies for stages 1-21
├── package.json       Node/TypeScript dependencies for the API and UI examples
├── .env.example       every environment variable the lessons reference
└── README.md          this file
```

## Python setup

```bash
cd code
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you are only working through the early stages, install less. The file is grouped by stage so you can comment out what you do not need yet — PyTorch and the training libraries are heavy, and you do not need them for stage 7 onward.

## Node setup

```bash
cd code
npm install
```

Node 20 or newer. The TypeScript examples use ESM, `zod` for validation and the platform `fetch`.

## Environment variables

```bash
cp .env.example .env
```

Then fill in the keys for whichever provider you are using. You do not need all of them. The lessons that call a provider are written so that the rest of the system still runs without it (a fake client is provided in the lesson) — so you can read the whole repo before paying for anything.

Never commit `.env`. It is in `.gitignore` for a reason. If you leak a key, revoke it at the provider; deleting the commit is not enough.

## Running a lesson's code

Most lessons are self-contained: copy the snippet into `scratch.py` or `scratch.ts` and run it. The `ship-*` files list their own file layout, because those are meant to become real repositories of their own rather than living inside this one.

## GPU notes

Stages 4, 5, 13 and 16 want a GPU for anything beyond a toy run.

- Google Colab (free tier) is enough for stages 4 and 5.
- A single rented GPU hour is usually enough for the stage 13 LoRA project on a small model.
- Stage 16's serving exercises need a GPU with enough memory for the model you pick; the lesson shows you how to compute the requirement before you rent anything.

Everything else in this repo runs on a laptop CPU.
