# AI Agent for Intelligent Job Search and Resume Optimization

This project implements a single LLM-based AI agent for job filtering, ranking, and resume tailoring.

The agent:
- loads a candidate profile and job dataset
- uses LLM reasoning to decide which tool to call
- filters jobs
- ranks the filtered jobs
- selects the best matching job
- tailors the resume summary and exactly two experience bullet points

This project was built for the AI for Engineers assignment: `AI Agent for Intelligent Job Search and Resume Optimization`.

## Project Structure

```text
JobSearch_Agent/
├── main.py
├── requirements.txt
├── README.md
├── data/
│   ├── jobs_output.csv
│   ├── candidate_profile.json
│   └── base_resume.txt
├── tools/
│   ├── filtering.py
│   ├── ranking.py
│   └── tailoring_resume.py
├── scripts/
│   └── scrape_jobs.py
├── tests/
│   ├── test_filtering.py
│   ├── test_ranking.py
│   └── test_main.py
└── artifacts/
```

## Requirements

- Python 3.10+
- A Gemini API key for the main agent
- A Groq API key for the resume tailoring tool
- A SerpAPI key only if you want to regenerate the job dataset

## Installation

From the project folder:

```bash
cd /Users/archana/Documents/SP2026/AI_For_Engg/Assignment2/JobSearch_Agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Set both API keys before running:

```bash
export GEMINI_API_KEY='your_gemini_api_key'
export GROQ_API_KEY='your_groq_api_key'
```

You can verify that they are set with:

```bash
python - <<'PY'
import os
print("GEMINI_API_KEY set:", bool(os.getenv("GEMINI_API_KEY")))
print("GROQ_API_KEY set:", bool(os.getenv("GROQ_API_KEY")))
PY
```

If you want to regenerate the dataset from the scraping script, also set:

```bash
export SERPAPI_KEY='your_serpapi_key'
```

## How to Run

Run the full agent with:

```bash
python main.py
```

## Expected Agent Flow

When the script runs, the agent should:

1. Load the dataset and candidate profile.
2. Use `filtering_tool` to narrow the job list.
3. Use `ranking_tool` to score and rank filtered jobs.
4. Use `resume_tailoring_tool` for the selected best job.
5. Produce a final recommendation and tailored resume output.

In the terminal, you should see:
- agent reasoning for each turn
- tool calls in sequence
- filtered jobs
- ranked jobs
- tailored summary
- tailored experience bullets

## Output Files

After a successful run, the following files are written to `artifacts/`:

- `trace.json`
  - full reasoning trace and tool calls
- `tailored_resume.json`
  - structured tailored resume output
- `tailored_resume.txt`
  - plain-text tailored resume output

Additional CSV artifacts are also written for filtered and ranked job outputs.

## Input Files

The main inputs are:

- `data/jobs_output.csv`
  - AI/ML job postings used by the agent
- `data/candidate_profile.json`
  - candidate skills, years of experience, and preferred locations
- `data/base_resume.txt`
  - base resume used for resume tailoring

These files can also be regenerated using `scripts/scrape_jobs.py`.

## Main Files

- `main.py`
  - orchestrates the full LLM-driven agent loop
- `tools/filtering.py`
  - applies rule-based filtering
- `tools/ranking.py`
  - scores and ranks filtered jobs
- `tools/tailoring_resume.py`
  - rewrites the resume summary and two experience bullet points
- `scripts/scrape_jobs.py`
  - generates `data/jobs_output.csv`, `data/candidate_profile.json`, and `data/base_resume.txt`

## Tests

Run the test files individually if needed:

```bash
pytest tests/test_filtering.py -q
pytest tests/test_ranking.py -q
pytest tests/test_main.py -q
```

If `pytest` has issues in a local Anaconda/Python environment, the main script can still be validated by running:

```bash
python main.py
```

## Optional Dataset Regeneration

To regenerate the dataset and supporting input files, run:

```bash
python scripts/scrape_jobs.py
```

This script writes:
- `data/jobs_output.csv`
- `data/candidate_profile.json`
- `data/base_resume.txt`

This step is optional for grading. The main agent can be run directly with the files already included in `data/`.

## Notes

- The project currently uses `google.generativeai` for Gemini integration.
- You may see a deprecation warning from Google about that SDK. The script still runs, but the package has been deprecated upstream.
- Resume tailoring depends on the Groq API being available and the `GROQ_API_KEY` being set.
- Dataset scraping depends on SerpAPI being available and the `SERPAPI_KEY` being set.

## Demo Summary

This project satisfies the assignment pipeline by showing:
- single-agent LLM reasoning
- tool-calling
- job filtering
- job ranking
- final best-job recommendation
- resume tailoring output
