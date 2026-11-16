# 3. Exploratory Data Analysis

> **Stage 03 · Data Engineering** — asking questions of a dataset before you let a model answer them
>
> **You'll be able to:** run a structured exploration that ends in three written findings and the charts that support them.
>
> **Prerequisites:** [Cleaning Messy Data](02-data-cleaning.md) · [Probability and Bayes](../02-math-and-ml-basics/03-probability-bayes.md)

## Why this exists

The instinct when you get a new dataset is to plot everything. That produces twenty
charts and no conclusions. EDA is not plotting — it is asking a small number of
specific questions and letting the data answer them, with the chart as evidence
rather than decoration.

The failure it prevents is expensive: teams build a retrieval pipeline or a classifier,
then discover on launch day that one class is 97% of the data, or that every document
from one source is empty, or that the "current" data ends eight months ago. Each of
those is visible in ten minutes of structured EDA.

Mental model: **EDA is a loop — question, shape of the data, chart, sentence. Repeat
until the sentences stop changing.**

## The mental model

```text
              +-----------------------------+
 question --> | 1. univariate               |  one column at a time
              |    range, nulls, shape      |  histogram, value counts
              +--------------+--------------+
                             |
              +--------------v--------------+
              | 2. bivariate                |  two columns at a time
              |    relationship, leakage    |  scatter, box by category
              +--------------+--------------+
                             |
              +--------------v--------------+
              | 3. temporal                 |  does it drift?
              |    volume and mix over time |  line by day/month
              +--------------+--------------+
                             |
                             v
                  3 written findings + the
                  chart that proves each one
```

## Core concepts

### Start with the question, not the chart

Write the question down first: "Do support tickets get resolved faster for one
product line?" Now the chart is chosen for you — resolution time by product line, a
box plot or a groupby summary. A chart with no question behind it cannot be wrong,
which means it cannot be useful.

### Univariate first: distribution, not average

A mean hides everything. `df["latency_ms"].describe()` plus a histogram tells you
whether you have a long tail, a bimodal split (two user populations), or a hard cap at
some default value. For text data the univariate view is token count per document:
`df["text"].str.split().str.len().describe()`, which tells you how many chunks you will
actually need and how many empty documents you have.

### Bivariate: look for leakage and for confounds

A relationship between a feature and your target can be a real signal, a proxy for a
group, or leakage from the future. Sort by time and check whether the relationship
holds in the earlier half only. In LLM work the common bivariate question is "does
answer quality depend on document length?" — plot score against length and look for a
trend before you conclude the retriever is bad.

### Temporal: volume and mix over time

Group by day or month and plot two lines: row count, and the share of each category.
Volume catches pipeline gaps (a flat zero on three days is an outage, not calm). Mix
catches drift — a new client sending differently shaped records will change the mix
long before anyone reads an error log.

### End with sentences, not charts

An exploration is done when you can write three findings, each one falsifiable and
each one attached to a chart. "48% of documents are under 40 tokens, so a 512-token
chunk size will leave most chunks mostly padding" is a finding. "The distribution looks
interesting" is not.

## Code

A compact exploration that produces the three views and prints the numbers you would
put in a findings paragraph.

```python
from __future__ import annotations

import pandas as pd

df = pd.read_parquet("documents.parquet")   # columns: doc_id, source, text, created_at

# --- Univariate: how long is a document, really? ---
df["n_tokens"] = df["text"].str.split().str.len()
print(df["n_tokens"].describe(percentiles=[0.5, 0.9, 0.99]).round(1))
print("empty docs:", int((df["n_tokens"] == 0).sum()))
print("share under 40 tokens:", round(float((df["n_tokens"] < 40).mean()), 3))

# --- Bivariate: is length different by source? ---
by_source = df.groupby("source")["n_tokens"].agg(["count", "median", "max"])
print(by_source.sort_values("count", ascending=False).head())

# --- Temporal: volume and mix per month ---
df["month"] = pd.to_datetime(df["created_at"]).dt.to_period("M")
volume = df.groupby("month").size()                       # rows per month
mix = (df.groupby(["month", "source"]).size()
         .unstack(fill_value=0)
         .pipe(lambda t: t.div(t.sum(axis=1), axis=0)))   # share per month
print(volume.tail(), mix.tail().round(2))
```

The chart half — two panels, one for distribution and one for drift. Matplotlib is
enough; save the figure next to your findings.

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_parquet("documents.parquet")
df["n_tokens"] = df["text"].str.split().str.len()
df["month"] = pd.to_datetime(df["created_at"]).dt.to_period("M").astype(str)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# Long tails are unreadable on a linear axis; clip the view, not the data.
ax1.hist(df["n_tokens"].clip(upper=df["n_tokens"].quantile(0.99)), bins=40)
ax1.set_title("Token count per document (clipped at p99)")
ax1.set_xlabel("tokens")

df.groupby(["month", "source"]).size().unstack(fill_value=0).plot(ax=ax2)
ax2.set_title("Documents per month by source")
ax2.set_ylabel("documents")

fig.tight_layout()
fig.savefig("eda_documents.png", dpi=120)
```

## Real use case

A legal-document assistant was retrieving badly for one customer. The team's first
hypothesis was the embedding model, and they spent a week testing alternatives. EDA
found it in twenty minutes: grouping token counts by `source` showed that one
customer's PDFs were scanned images, so the extracted `text` was empty for 61% of
their documents. The retriever was working correctly on the 39% that had any text at
all.

The temporal view added the second half of the story — the empty share jumped on a
specific month, which matched a change in that customer's scanning workflow. Two
groupby calls replaced a week of model experiments, and the fix was OCR at ingestion,
not a better embedder.

| Question | Chart | Finding |
|---|---|---|
| How long is a document? | Histogram of tokens | Most documents are short; chunk size was oversized |
| Does one source differ? | Token count by source | One source 61% empty — scanned PDFs |
| When did it change? | Volume and mix by month | Empty share jumped in one month |

## Common mistakes

- **Mistake:** plotting before writing a question — **Instead:** write the question as a sentence, then pick the one chart that answers it.
- **Mistake:** reporting only `mean()` — **Instead:** report median and the 90th/99th percentile; long tails are the norm in real data.
- **Mistake:** clipping or dropping outliers to make a chart look clean — **Instead:** clip the axis view and say so in the caption; never change the data.
- **Mistake:** concluding from a bivariate plot with no time check — **Instead:** split by time and confirm the relationship holds in the earlier period.
- **Mistake:** ending with charts and no sentences — **Instead:** finish with three falsifiable findings, each tied to one chart.

## Checkpoint

1. Take any text dataset and produce the univariate token-count summary plus the empty-document count. State in one sentence what chunk size it implies.
2. Add a bivariate view: does a proxy for quality (length, or a score you have) differ by source? Use a groupby summary, not a scatter, and justify the choice.
3. Build the monthly volume-and-mix table for your dataset and write down one thing it tells you that neither univariate view did.

**Self-check:** the token summary tells you the median and tail of document length, and the share below your intended chunk size — that share is how much of your corpus would sit mostly empty inside a chunk.

## Go deeper

- [Cleaning Messy Data](02-data-cleaning.md) — the fixes that follow from what EDA reveals.
- [Dataset Preparation](08-dataset-prep.md) — turn these findings into splits and balance decisions.
- [Data Pipelines](06-data-pipelines.md) — run the volume-and-mix check on every load so drift is caught automatically.
- [Probability and Bayes](../02-math-and-ml-basics/03-probability-bayes.md) — the language for talking about distributions precisely.

---
[← Previous: Cleaning Messy Data](02-data-cleaning.md) · [Stage index](README.md) · [Next: SQL on PostgreSQL](04-sql-postgresql.md)
