Given the same user query and two candidate replies (A/B), decide which should be adopted for the next turn. Penalize failure modes (repetition, off-topic, role confusion, contradictions, unsafe advice). Avoid position/length bias; do not reveal model names.

[Output] Provide one VERY SHORT rationale (<=30 Chinese chars), then output ONLY one token: [[A]] / [[B]] / [[C]] (tie).

[BATCHING REQUIREMENT]
- Evaluate sequentially; after every 10 items, output one JSON file/object as a batch. If fewer than 10 remain, output a final batch.
- The JSON must contain ONLY the following fields (no extra text).

[Per-batch JSON schema]
{
  "batch_id": "<string, e.g., 'paireval_batch_0001'>",
  "method": "PairEval",
  "timestamp_utc": "<ISO8601>",
  "n_items": 10,
  "items": [
    {
      "id": "<sample id>",
      "query": "<user query>",
      "reply_A_model": "<model name A>",
      "reply_B_model": "<model name B>",
      "decision": "A|B|C",  // C means Tie
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
    "wins_A": <int>, "wins_B": <int>, "ties": <int>,
    "win_rate_A_pct": <float 0-100>,  // (wins_A + 0.5*ties)/n_items*100
    "win_rate_B_pct": <float 0-100>,
    "tie_rate_pct": <float 0-100>,
    "any_failure_pct": <float 0-100>,
    "counts": { "items": 10 }
  }
}

[Inputs per item]
[User Query]
{QUERY}

[Reply A]
{ANS_A}

[Reply B]
{ANS_B}

