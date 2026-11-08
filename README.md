# StayFair — AI-Assisted Airbnb Dispute Resolution

**BC3415 AI in Accounting & Finance | Educational group prototype**

StayFair is a Python/Gradio demonstration of an AI-assisted workflow for guest refund requests and host damage disputes. It combines Google Gemini investigation, retrieval of relevant *synthetic* policy passages using TF-IDF and cosine similarity, deterministic checks, and specialist review for uncertain or higher-risk cases.

**Disclaimer:** This is a student prototype, not affiliated with Airbnb. All booking records and policy rules are fictional or illustrative. The application does not process actual refunds or payments, contact customers, verify identities, or send external messages. Do not enter real personal information or sensitive evidence.

## Features

- **Customer view:** Guests and hosts submit support requests, respond to claims, attach evidence, and review case updates.
- **Staff workspace:** Review the case summary, attached evidence, retrieved policy passages, escalation reasons, and recorded decisions.
- **Workflow:** Follow the case from intake through investigation, information gathering, specialist review, and outcome.
- **AI investigation:** When a valid Gemini API key and appropriate consent are available, the model analyzes statements and optional uploaded images and produces structured findings. PDFs remain for staff review.
- **Policy retrieval:** TF-IDF and cosine similarity retrieve relevant passages from a small synthetic policy library. Deterministic logic determines proposed calculations and escalation conditions.
- **Fallback:** If live AI is unavailable, the application shows a clearly labeled local fallback rather than presenting it as an AI-generated result.

## Requirements

- Python 3.12 recommended
- A Google Gemini API key for live AI analysis (the interface and local fallback can work without one)

Dependencies are defined in `requirements.txt`.

## Run locally

From the repository folder:

```bash
python -m venv .venv
```

Activate the virtual environment:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

For live Gemini analysis, set `GEMINI_API_KEY` as an environment variable. For example, in **Windows PowerShell**:

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

Or in **macOS/Linux**:

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

Do not commit credentials or a `.env` file containing credentials to GitHub. You may also configure `LLM_MODEL` and `LLM_FALLBACK_MODEL` with supported Gemini model names accessible to your account; model availability can change.

Start the application:

```bash
python airbnb_app.py
```

By default it listens on port `10000` at `0.0.0.0`; open `http://localhost:10000` on your own computer. Set `PORT` to use a different port. The web-hosting platform may supply `PORT` automatically.

## Demo walkthrough

**Guest refund example:** Open **Customer view** → select **No hot water — confirmed facts** → click **Load example** → review the form and submit the request with the appropriate AI consent. This example uses the fictional booking HM-1001 (Alicia / Ben). A valid live AI response is required for live-AI analysis; resulting routing depends on available evidence and checks.

**Host damage dispute:** Load **Damaged TV — US$300, disputed**. Review how the claim, the other party's account, policy matches, and supporting evidence lead to staff review rather than automatic assignment of responsibility.

**Specialist workflow:** Use **Staff workspace** to inspect the case, see why it was escalated, request information, retry a failed AI review, or record a simulated decision. The **Workflow** tab shows progress through the case.

## Technical design

- **Interface:** Gradio 6
- **Language and libraries:** Python, Google Gen AI SDK, Pydantic, scikit-learn, Pillow, pypdf
- **Policy retrieval:** TF-IDF vectorization and cosine similarity over a small synthetic policy library
- **LLM:** Gemini structured-output investigation, including image observations where applicable
- **Decision support:** Fixed business rules for timing, amount calculation, evidence needs, safety/fraud escalation, and human review

The application uses a predefined workflow rather than an autonomous agent that independently selects tools. Suggested monetary outcomes are simulated only.

## Reliability and limitations

- Live AI requests have a configurable total-case budget controlled by `AI_TOTAL_SECONDS` (default 45 seconds, capped at 90 seconds); individual request timeouts can be configured with `AI_TIMEOUT_MS` (default 20,000 ms). Up to two model attempts may occur, with a configured fallback model on a temporary failure.
- Gemini requests and model availability depend on the connected API project, quotas, and network connectivity.
- Case state is stored in the Gradio browser session, not a persistent database. Refreshing or opening a new session may lose earlier cases. The three tabs are shared views without authentication; do not use the prototype with real personal data.
- Attached images can be analyzed with consent; PDFs are only retained for staff review.
- The sample policies and fixed refund percentages are classroom assumptions and are not Airbnb's actual entitlements.

## Deploying

A `Dockerfile` in the repository root can be used by container-based hosting providers. Configure `GEMINI_API_KEY` and any optional AI variables as **private environment variables on the hosting platform**. The public domain must forward to the actual `PORT` on which Gradio starts. Do not publish credentials.
