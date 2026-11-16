# Stage 03 — Data Engineering

Every LLM application is downstream of data: the CSV a business analyst exports, the
JSONL you fine-tune on, the Parquet files your retrieval index is built from. This
stage teaches the data work an AI engineer actually does day to day — loading,
cleaning, querying, validating and shaping data — before any model enters the
picture. It sits after Stage 02 because you now know what a matrix and a gradient
are, so an array is no longer a mystery, and before Stage 04 because a model trained
on dirty data teaches you nothing about models.

**Time:** 2-3 weeks part-time · **You need first:** Stage 02 — Math and ML Basics · **You leave with:** a clean, validated, versioned dataset plus a data card

## What you will be able to do

- Load, slice, reshape and join real tabular data with NumPy and Pandas, and read any array by its shape instead of by hope.
- Clean a messy export — duplicates, mixed types, missing values, broken dates — and justify each decision.
- Explore a dataset and produce the three or four charts that actually answer a question.
- Write PostgreSQL with CTEs, window functions and JSON columns, and know when SQL beats Pandas.
- Move data through an idempotent, incremental pipeline in Parquet and JSONL.
- Validate every record with pydantic and assemble a fine-tuning or eval dataset that survives dedup, splitting and contamination checks.

## Topics

| # | Topic | What it gives you |
|---|-------|-------------------|
| 1 | [NumPy and Pandas](01-numpy-pandas.md) | Arrays, shapes, vectorisation, DataFrames, joins and groupby |
| 2 | [Cleaning Messy Data](02-data-cleaning.md) | Duplicates, types, missing values, dates, outliers, an audit trail |
| 3 | [Exploratory Data Analysis](03-eda.md) | Questions first, then charts, then a written finding |
| 4 | [SQL on PostgreSQL](04-sql-postgresql.md) | Joins, CTEs, window functions, JSON columns, indexes |
| 5 | [Parquet and JSONL](05-parquet-jsonl.md) | Columnar storage and the line-delimited format of LLM data |
| 6 | [Data Pipelines](06-data-pipelines.md) | Idempotency, incremental loads, orchestration, failure recovery |
| 7 | [Schema Validation with Pydantic](07-validation.md) | Records that cannot be malformed past the boundary |
| 8 | [Dataset Preparation](08-dataset-prep.md) | Dedup, splits, contamination, class and length balance |
| 9 | [Ship: A Clean Dataset with a Data Card](09-ship-clean-dataset.md) | End-to-end build of a documented, validated dataset |

## The shipping project

[09-ship-clean-dataset.md](09-ship-clean-dataset.md) — you take a deliberately messy
raw export, run it through a validated pipeline and emit a clean Parquet plus JSONL
dataset with a data card. You prove it works with a validation script that fails on
purpose, a reproducibility check that produces byte-identical output on a second run,
and a short data card another engineer could trust.

## How to know you are done

- You can state the shape of any array in your own code before running it, and you are right.
- Your cleaning script is re-runnable: a second run on the same input changes nothing.
- Every record in your output dataset passes a pydantic model, and a bad record raises instead of slipping through.
- Your dataset ships with a data card listing source, size, splits, known gaps and licence.
- You can write the same aggregation in Pandas and in SQL, and say which one you would ship.

---
[← Back to roadmap](../../README.md) · [Next stage: Deep Learning](../04-deep-learning/README.md)
