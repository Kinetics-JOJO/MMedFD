Given a user query and a model reply, evaluate the quality of the response. Identify any failure modes (repetition, off-topic, role confusion, contradictions, unsafe advice).

[Output] Provide one VERY SHORT rationale (<=30 Chinese chars), then output a score from 1-10 where 10 is excellent.

[BATCHING REQUIREMENT]
- Evaluate sequentially; after every 10 items, output one JSON line as a batch. If fewer than 10 remain, output a final batch.
- The JSON must contain ONLY the following fields (no extra text).

[Per-batch JSON schema]
{
  "batch_id": "<string, e.g., 'geval_batch_0001'>",
  "method": "GEval",
  "timestamp_utc": "<ISO8601>",
  "n_items": 10,
  "items": [
    {
      "id": "<sample id>",
      "query": "<user query>",
      "model": "<model name>",
      "score": <int 1-10>,
      "rationale_short": "<=30 Chinese chars",
      "flags": {
        "repetition": true/false,
        "off_topic": true/false,
        "role_confusion": true/false,
        "contradiction": true/false,
        "unsafe_advice": true/false
      }
    }
  ],
  "summary": {
    "avg_score": <float 1-10>,
    "any_failure_pct": <float 0-100>,
    "counts": { "items": 10 }
  }
}

[Inputs per item]
[User Query]
{QUERY}

[Model Reply]
{RESPONSE}
