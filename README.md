# StayFair — Fair Resolve

English-only classroom prototype. Fictional reservations and illustrative policies; not affiliated with Airbnb. No actual refunds, emails, identity verification or authentication. The three tabs share one browser session. Closing or refreshing the session can lose cases. Do not use real personal evidence.

## Update on Windows
1. Stop the old app with Ctrl+C.
2. Extract this ZIP into your existing Group Project folder and replace the files. Keep your .venv folder.
3. In PowerShell inside that folder:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
$env:GEMINI_API_KEY="YOUR_KEY"
$env:LLM_MODEL="gemini-3.1-flash-lite"
$env:PORT="7861"
& ".\.venv\Scripts\python.exe" airbnb_app.py
```

Open http://127.0.0.1:7861. Use the model that successfully passed your own connection test; availability varies. Never put your key in code or share it.

## Guest refund walkthrough
Select HM-1001, Guest, and enter Alicia. Report no hot water for three nights, set nights affected to 3, use a recent discovery time (UTC), tick AI consent and submit. The system requests the other party response. Switch to Host, change the name to Ben, keep the same case selected, choose Confirms it, and explain that the boiler failed for three nights and you agree to the partial refund. Tick consent and submit. With a successful live investigation and all checks passing, the outcome is US$225. Both parties can see the result and appeal with a reason.

## Host dispute walkthrough
Start a new case, select Host / Ben, describe a cracked TV and enter US$300 compensation. Upload photos and a repair receipt. Switch to Guest / Alicia and submit a disputed response. Staff can open all evidence, review the policy and record a reasoned decision or information request. Damage responsibility requires specialist review in this prototype.

## Reliability
Each live request has a 120-second timeout and up to three attempts for temporary errors, with short backoff. Override with AI_TIMEOUT_MS if needed. Successful evidence observations are reused within the case when its file bytes and consent are unchanged. Retry AI review in Staff workspace retries without adding another customer message. A failed live investigation cannot automatically resolve a case. Errors are displayed in staff view with the configured key redacted. Both parties must consent before the combined record is sent to AI. Files remain available locally to staff even if AI fails.

## Implementation
Python, Gradio 6, Google Gen AI SDK, Pydantic, TF-IDF/cosine policy retrieval, Pillow and pypdf. A fixed workflow invokes LLM analysis and deterministic policy/calculation/routing functions; it is not an autonomous tool-choosing agent. Confidence is assessed through explicit eligibility/evidence/agreement/risk checks rather than an invented probability.

## Verification
Run `python -m unittest -v test_workflow`. The tests use mocked AI responses; they do not establish live model availability or validate real-world policy.
