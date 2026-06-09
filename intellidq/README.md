# IntelliDQ — AI-Powered Data Quality Tool
**Gurleen Kaur · Capgemini DTSP 2025**

## Overview
IntelliDQ is a Flask-based web application that automates data quality (DQ) checks on uploaded CSV/Excel files. It applies rule-based DQ checks aligned to the **Government Data Quality Framework** and uses **Isolation Forest** (machine learning) for anomaly detection.

## Features
- 📁 Drag-and-drop file upload (CSV, XLSX, XLS — up to 50MB)
- ✓ **Completeness** — missing value detection per column
- ◈ **Uniqueness** — duplicate row detection
- ⬡ **Validity** — mixed type and negative value checks
- ≡ **Consistency** — case-variant and format inconsistencies
- ⚡ **Anomaly Detection** — Isolation Forest on numeric columns
- 📊 Interactive dashboard with dimension scores
- 🤖 AI-generated summary (Microsoft Foundry integration hook)
- 📥 Export as CSV or JSON

## Setup & Run

### Requirements
```
Python 3.9+
```

### Install dependencies
```bash
pip install flask pandas scikit-learn openpyxl numpy
```

### Run the app
```bash
cd intellidq
python app.py
```

Then open: **http://localhost:5000**

## Microsoft Foundry Integration
The AI summary panel is designed to connect to Microsoft Foundry's LLM API.
To integrate, replace the rule-based summary in `renderAISummary()` (templates/index.html)
with a call to your Foundry endpoint — pass in `d` (the DQ results JSON) as context.

Example endpoint call (add to `/upload` route in `app.py`):
```python
# Foundry LLM call (pseudo-code)
summary = foundry_client.chat(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": f"Summarise this data quality report: {json.dumps(dq_results)}"
    }]
)
```

## Project Structure
```
intellidq/
├── app.py              # Flask backend + DQ engine
├── templates/
│   └── index.html      # Full UI (upload, dashboard, export)
├── uploads/            # Temporary result cache (auto-created)
└── README.md
```

## GovDQ Framework Alignment
| Dimension     | Implementation |
|---------------|---------------|
| Completeness  | Missing value % per column |
| Uniqueness    | Duplicate row detection |
| Validity      | Type checks, negative value flags |
| Consistency   | Case/format variant detection |
| Accuracy      | Isolation Forest anomaly flagging |
| Timeliness    | (Future: date column staleness checks) |
