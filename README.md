# LLM-Powered Drink Recommendation (Starbucks Menu)

Turn a free-text request like *"need a pick me up, just black coffee, under
45 g sugar"* into a ranked list of menu items that actually satisfy it.

The system combines an LLM for **understanding** the request with
deterministic filtering for **correctness** and embeddings for **ranking**,
so hard constraints (calories, dairy, price) are never violated by a
"creative" model.

## Pipeline

```
free-text query
      │
      ▼
Stage 1  LLM constraint extraction  (JSON-schema structured output, few-shot)
      │   → {category, temperature, max_calories, max_sugar, max_price,
      │      dairy_free, vegan, caffeine_level}
      ▼
Stage 2  Hard filtering on the product table
      │   (rules, not the LLM: a 250-cal limit is always a 250-cal limit)
      ▼
Stage 3  Embedding similarity ranking of remaining candidates
          (text-embedding-3-large; product text enriched with derived tags)
      │
      ▼
ranked recommendations
```

**Why this split:** LLMs are good at parsing messy language ("oat milk or
something dairy free", "lots of caffeine") but unreliable at enforcing
numeric limits. Letting the LLM only extract a strict schema, and doing the
filtering in code, makes every constraint auditable and every failure
traceable to one stage.

## Data

| File | Rows | Contents |
|---|---|---|
| `products.csv` | 115 | Menu items with category, temperature, caffeine, calories, sugar, protein, dairy / nut / gluten flags, vegan flag, description, price |
| `queries_train.csv` | 100 | Natural-language queries with labeled constraints and a ranked list of relevant products |
| `queries_test.csv` | 100 | Queries without labels |

## What the notebook does

`recommendation_pipeline.ipynb` contains the improvement experiments on top
of the base pipeline:

1. **Constraint schema and normalization**: valid values, alias mapping
   (e.g. "frap" → `frappuccino`, "decaf" → caffeine `none`).
2. **Structured LLM extraction** via an OpenAI-compatible API with a strict
   JSON schema, so outputs always parse and only valid enum values appear.
3. **Few-shot example selection**: the two most constraint-dense training
   queries per category, plus hand-picked hard cases (20 examples total).
4. **Per-field error analysis** of extracted constraints against labels.
5. **Filter tracing**: shows how many candidates each constraint removes,
   for debugging empty or over-filtered results.
6. **Ranking with enriched product text** and graded NDCG evaluation.
7. **Hard-case analyzer**: surfaces the lowest-NDCG queries and compares
   predicted vs. relevant products side by side.
8. **Tuning**: caffeine-level thresholds (grid search with an 80/20
   train/validation split) and the weight between full-text and tag
   embeddings.

## Results (training queries)

| Stage | Metric | Result |
|---|---|---|
| Constraint extraction | Per-field accuracy, 100 queries | 100% on all 8 fields |
| Ranking | Graded NDCG, 100 queries | 0.989 |
| Caffeine thresholds | NDCG on 20% held-out validation split | 0.998 (low ≤ 60 mg, medium ≤ 140 mg) |

Error analysis pointed at where to tune: the lowest-scoring queries were
"cappuccino or latte … high caffeine" requests, where the labeled answers
were high-caffeine Americanos while the embedding ranker favored lattes that
matched the wording. Caffeine-level thresholds were then tuned against
held-out queries rather than hand-set.

**Caveats.** These are training-set numbers. The few-shot examples are drawn
from the same 100 queries, so extraction accuracy is optimistic, and the
test queries have no labels in this repo. A fair next step is to evaluate
extraction with few-shot examples excluded from the evaluation set.

## Run

```bash
pip install openai python-dotenv pandas numpy scikit-learn
```

Create a `.env` file:

```
OPENROUTER_API_KEY=your_key
```

Then open `recommendation_pipeline.ipynb`. Stage 2–3 cells read Stage 1
outputs from `artifacts/<query_id>/stage1_constraints.json`, produced by the
base pipeline (not included in this repo).

## Tech

Python · pandas · OpenAI-compatible LLM API (structured outputs) ·
text-embedding-3-large · scikit-learn · NDCG evaluation
