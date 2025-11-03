\# Prompts used by the memory pipeline



\## 1) summarize\_for\_context

You compress content to ≤2 sentences, preserving: names, dates, decisions, counters,

and explicit constraints. Do not add opinions. Output plain text.



\## 2) kg\_extract (triples + evidence)

Extract `(head, relation, tail)` from the text. Use ontology terms where possible.

Return JSON with fields:

\[

&nbsp; {

&nbsp;   "h": "string", "r": "RELATION", "t": "string",

&nbsp;   "evidence": {"source":"string","url":"string","timestamp":"ISO8601","confidence":0.0-1.0,"snippet":"string"}

&nbsp; }

]



\## 3) event\_classify\_and\_score

Classify the event as one of: Chaos | Foundation | Glow.

Score each dimension on 0..10: error\_severity, novelty, foundation\_weight, rlhf\_weight.

Return JSON:

{

&nbsp; "class": "Glow",

&nbsp; "scores": {"error\_severity": 2, "novelty": 8, "foundation\_weight": 6, "rlhf\_weight": 5},

&nbsp; "rationale": "One concise sentence."

}



\## 4) recall\_critique

Given: query, top-k retrieved snippets (with sources).

Task: verify factual alignment with persona/facts; flag conflicts; suggest one missing invariant if any.

Output a bullet list with ✅/⚠️ markers.



