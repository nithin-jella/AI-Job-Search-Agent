# AI Job Search Agent

## Overview

AI Job Search Agent is an LLM-powered project designed to support intelligent job discovery, role matching, and resume customization. The system evaluates a candidate profile against available job postings, filters relevant opportunities, ranks them based on fit, and generates a tailored resume summary with optimized experience bullet points.

This project demonstrates how a single AI agent can combine reasoning, tool usage, structured data processing, and resume personalization in one workflow.

## Key Capabilities

The agent is designed to:

* Load candidate details and job posting records
* Interpret the user request using LLM-based reasoning
* Apply filtering logic to narrow down relevant jobs
* Rank shortlisted roles based on candidate-job alignment
* Identify the strongest job match
* Generate a tailored resume summary
* Rewrite exactly two experience bullet points for the selected role
* Save final results and execution traces as project deliverables

## Project Purpose

This project was developed as part of the AI for Engineers assignment focused on building an intelligent job search and resume optimization agent. It highlights practical use of AI agents for career support, job matching, and personalized resume generation.

## Project Structure

```text
AI-Job-Search-Agent/
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
└── deliverables/
```

## Requirements

Before running the project, make sure the following are available:

* Python 3.10 or above
* Gemini API key for the main agent workflow
* Groq API key for resume tailoring
* SerpAPI key only if you want to regenerate the job dataset

## Installation

Clone the repository and move into the project folder:

```bash
git clone YOUR_REPOSITORY_LINK
cd AI-Job-Search-Agent
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Environment Setup

Set the required API keys before running the application:

```bash
export GEMINI_API_KEY="your_gemini_api_key"
export GROQ_API_KEY="your_groq_api_key"
```

To confirm that the keys are available in your environment, run:

```bash
python - <<'PY'
import os

print("GEMINI_API_KEY set:", bool(os.getenv("GEMINI_API_KEY")))
print("GROQ_API_KEY set:", bool(os.getenv("GROQ_API_KEY")))
PY
```

If you want to regenerate the job dataset using the scraping script, also set:

```bash
export SERPAPI_KEY="your_serpapi_key"
```

## How to Run

Run the full agent workflow using:

```bash
python main.py
```

## Expected Workflow

When executed, the agent performs the following steps:

1. Loads the job dataset and candidate profile
2. Uses LLM reasoning to decide the next tool action
3. Filters jobs based on candidate requirements
4. Scores and ranks filtered jobs
5. Selects the best-matching role
6. Generates a customized resume summary
7. Produces two tailored experience bullet points
8. Saves the final outputs for review

The terminal output displays the agent reasoning, tool calls, filtered job results, ranked job matches, and tailored resume content.

## Project Deliverables

After a successful run, the system writes output files to the `deliverables/` folder.

Main deliverables include:

* `trace.json`
  Stores the agent reasoning trace and tool execution sequence

* `tailored_resume.json`
  Contains the structured resume tailoring output

* `tailored_resume.txt`
  Provides a plain-text version of the tailored resume content

Additional CSV outputs may also be generated for filtered and ranked job results.

## Project Data Sources

The project uses the following input files:

* `data/jobs_output.csv`
  Contains AI and machine learning job postings used by the agent

* `data/candidate_profile.json`
  Stores candidate skills, experience level, and preferred job locations

* `data/base_resume.txt`
  Provides the original resume content used for tailoring

These files can be used directly or regenerated through the scraping script.

## Main Components

### `main.py`

Controls the complete agent workflow, including input loading, LLM reasoning, tool selection, and final output generation.

### `tools/filtering.py`

Applies rule-based filtering to identify job postings that match candidate preferences.

### `tools/ranking.py`

Scores and ranks filtered jobs based on skill match and experience alignment.

### `tools/tailoring_resume.py`

Generates a tailored resume summary and two customized experience bullet points for the selected job.

### `scripts/scrape_jobs.py`

Optional script used to regenerate job postings, candidate profile data, and base resume input files.

## Testing

Run individual test files using:

```bash
pytest tests/test_filtering.py -q
pytest tests/test_ranking.py -q
pytest tests/test_main.py -q
```

If there are local environment issues with `pytest`, the project can still be validated by running:

```bash
python main.py
```

## Optional Dataset Regeneration

To recreate the dataset and supporting input files, run:

```bash
python scripts/scrape_jobs.py
```

This script generates:

* `data/jobs_output.csv`
* `data/candidate_profile.json`
* `data/base_resume.txt`

This step is optional because the project can run with the included input files.

## Notes

* The agent uses Gemini for LLM-based reasoning.
* Resume tailoring depends on the Groq API.
* Dataset regeneration depends on SerpAPI.
* If API keys are missing, parts of the workflow may not run as expected.
* Some SDK warnings may appear depending on the installed package versions.

## Demo Summary

This project demonstrates an end-to-end AI agent pipeline that includes:

* Single-agent LLM reasoning
* Tool-based decision execution
* Job filtering and ranking
* Best-fit job recommendation
* Resume summary generation
* Tailored experience bullet creation
* Structured output tracking

## Tech Stack

Python, Gemini API, Groq API, SerpAPI, Pandas, Pytest, LLM Agents, Resume Optimization, Job Matching
