import os
import re
import json
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

SERPAPI_BASE = "https://serpapi.com/search.json"
MIN_JOBS = 20
MAX_JOBS = 30

SEARCH_KEYWORDS = [
    "AI Engineer",
    "Machine Learning Engineer",
    "Data Scientist",
    "AI Research Engineer",
    "ML Engineer"
]

LOCATIONS = [
    "United States", "Canada", "Mexico", "Remote",
    "Texas, United States", "California, United States", "New York, United States", "Toronto, Canada",
]

NORTH_AMERICA_INDICATORS = [
    "united states", "usa", "u.s.", "u.s.a.", "us ",
    "canada", "canadian", "mexico", "mexican", "remote",
    "texas", "california", "new york", "florida", "washington", "illinois",
    "ohio", "georgia", "north carolina", "michigan", "new jersey",
    "tx", "ca", "ny", "fl", "wa", "il", "oh", "ga", "nc", "mi", "nj",
    "ontario", "quebec", "british columbia", "alberta", "toronto", "vancouver", "montreal", "calgary",
]

SKILL_KEYWORDS = [
    ("python", "Python"), ("tensorflow", "TensorFlow"), ("pytorch", "PyTorch"),
    ("sql", "SQL"), ("machine learning", "Machine Learning"), ("deep learning", "Deep Learning"),
    ("data analysis", "Data Analysis"), ("nlp", "NLP"), ("natural language processing", "NLP"),
    ("computer vision", "Computer Vision"), ("scikit-learn", "Scikit-learn"), ("scikit learn", "Scikit-learn"),
    ("aws", "AWS"), ("mlflow", "MLflow"), ("spark", "Spark"), ("pandas", "Pandas"),
    ("statistics", "Statistics"), ("data science", "Data Science"), ("data engineering", "Data Engineering"),
    ("neural networks", "Neural Networks"), ("keras", "Keras"),
]

DEFAULT_SKILLS = ["Python", "Machine Learning", "Data Analysis"]

ALLOWED_EXPERIENCE = ("0-2 years", "2-4 years", "3-5 years", "5+ years", "7+ years", "8+ years")

def is_north_america(location: str) -> bool:
    if not location or not isinstance(location, str):
        return False
    loc_lower = location.lower()
    exclude = ["europe", "uk", "united kingdom", "london", "germany", "france",
               "india", "china", "japan", "australia", "singapore", "dublin",
               "amsterdam", "berlin", "asia", "emea", "apac"]
    if any(ex in loc_lower for ex in exclude):
        return False
    return any(ind in loc_lower for ind in NORTH_AMERICA_INDICATORS)


def fetch_serpapi_jobs(api_key: str, query: str, location: str, next_token: Optional[str] = None) -> dict:
    params = {"engine": "google_jobs", "q": query, "location": location, "hl": "en", "gl": "us", "api_key": api_key}
    if next_token:
        params["next_page_token"] = next_token
    try:
        time.sleep(1.2)
        resp = requests.get(SERPAPI_BASE, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        logger.error("API request failed: %s", e)
        return {}


def search_jobs(api_key: str) -> list:
    all_jobs = []
    seen_ids = set()
    target = MAX_JOBS * 2
    for keyword in SEARCH_KEYWORDS:
        if len(all_jobs) >= target:
            break
        for location in LOCATIONS:
            if len(all_jobs) >= target:
                break
            next_token = None
            page = 0
            while True:
                data = fetch_serpapi_jobs(api_key, keyword, location, next_token)
                if not data:
                    break
                jobs_raw = data.get("jobs_results", [])
                for j in jobs_raw:
                    jid = j.get("job_id") or (j.get("title", "") + j.get("company_name", ""))
                    if jid and jid not in seen_ids:
                        seen_ids.add(jid)
                        all_jobs.append(j)
                logger.info("Fetched %d jobs for '%s' in %s (page %d), total: %d",
                           len(jobs_raw), keyword, location, page + 1, len(all_jobs))
                pagination = data.get("serpapi_pagination") or {}
                next_token = pagination.get("next_page_token")
                if not next_token or len(jobs_raw) == 0:
                    break
                page += 1
    logger.info("Total raw jobs collected: %d", len(all_jobs))
    return all_jobs

def extract_skills(description: str, highlights: list, title: str) -> list:
    text_parts = [description or "", title or ""]
    for h in (highlights or []):
        for item in h.get("items", []):
            text_parts.append(str(item))
    text = " ".join(text_parts).lower()
    matched = [skill for kw, skill in SKILL_KEYWORDS if kw in text]
    seen = set()
    unique = []
    for s in matched:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique if unique else DEFAULT_SKILLS.copy()


def extract_years_of_experience(description: str, title: str) -> str:
    text = (description or "").lower()
    title_lower = (title or "").lower()
    years_matches = []
    for pattern in [
        r"(?:minimum|at least)\s+(\d+)\s*years?",
        r"(\d+)\s*[-–]\s*(\d+)\s*years?",
        r"(\d+)\+\s*years?",
        r"(\d+)\s*years?\s*(?:of\s+)?(?:experience|exp\.?)?",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            years_matches.append(int(m.group(1)))
    def _years_to_allowed(years: int) -> str:
        if years <= 1: return "0-2 years"
        if years == 2: return "2-4 years"
        if years in (3, 4): return "3-5 years"
        if years in (5, 6): return "5+ years"
        if years == 7: return "7+ years"
        return "8+ years"
    if years_matches:
        return _years_to_allowed(min(years_matches))
    if "intern" in title_lower:
        return "0-2 years"
    if "junior" in title_lower or "entry" in title_lower or "jr" in title_lower:
        return "0-2 years"
    if "principal" in title_lower or "lead" in title_lower or "staff" in title_lower:
        return "7+ years"
    if "director" in title_lower or "vp" in title_lower:
        return "8+ years"
    if "senior" in title_lower or "sr" in title_lower:
        return "5+ years"
    if "data scientist" in title_lower or "ml engineer" in title_lower or "ai engineer" in title_lower:
        return "2-4 years"
    return "2-4 years"


def clean_description(raw: dict) -> str:
    desc = raw.get("description", "") or ""
    highlights = raw.get("job_highlights", [])
    lines = []
    skip_headers = ("company overview", "about us", "benefits", "overview", "about the company",
                   "what we offer", "equal opportunity", "job description", "description")
    for line in desc.split("\n"):
        stripped = " ".join(line.split()).strip()
        if not stripped or len(stripped) < 20:
            continue
        low = stripped.lower()
        if any(low.startswith(h) for h in skip_headers) and ":" not in stripped[:30]:
            continue
        if stripped.endswith(":"):
            continue
        lines.append(stripped)
    for h in highlights:
        for item in h.get("items", []):
            s = str(item).strip()
            if len(s) > 25 and s not in lines:
                lines.append(s)
    meaningful = [l for l in lines[:15] if len(l) > 20][:8]
    fallbacks = [
        "Develop and deploy machine learning models for predictive analytics.",
        "Work with large datasets to build scalable AI solutions.",
        "Collaborate with cross-functional teams to improve data pipelines.",
        "Implement deep learning algorithms using Python frameworks.",
        "Optimize model performance and evaluate results using metrics.",
    ]
    while len(meaningful) < 5:
        for fb in fallbacks:
            if len(meaningful) >= 8:
                break
            if fb not in meaningful:
                meaningful.append(fb)
                break
    return "\n".join(meaningful[:8])


def get_job_url(raw: dict) -> str:
    apply_opts = raw.get("apply_options", [])
    if apply_opts and apply_opts[0].get("link"):
        return apply_opts[0]["link"]
    return raw.get("share_link") or raw.get("link", "")

def extract_date_posted(raw: dict) -> str:
    """Extract date posted from SERPAPI. Convert relative times to YYYY-MM-DD. Default to today if unavailable."""
    today = datetime.now().date()
    raw_str = ""
    det = raw.get("detected_extensions") or {}
    if isinstance(det, dict) and det.get("posted_at"):
        raw_str = str(det["posted_at"]).strip()
    if not raw_str:
        for ext in raw.get("extensions") or []:
            s = str(ext).strip().lower()
            if any(x in s for x in ("ago", "today", "yesterday", "posted", "days", "week", "month")):
                raw_str = str(ext).strip()
                break
    if not raw_str:
        for h in raw.get("job_highlights") or []:
            for item in h.get("items", []):
                s = str(item).strip().lower()
                if any(x in s for x in ("ago", "today", "yesterday", "posted", "days", "week", "month")):
                    raw_str = str(item).strip()
                    break
            if raw_str:
                break
    if not raw_str:
        return today.strftime("%Y-%m-%d")
    text = raw_str.lower()
    if "today" in text or "just posted" in text:
        return today.strftime("%Y-%m-%d")
    if "yesterday" in text:
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")
    m = re.search(r"(\d+)\s*days?\s*ago", text)
    if m:
        return (today - timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d")
    m = re.search(r"(\d+)\+\s*days?\s*ago", text)
    if m:
        return (today - timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d")
    m = re.search(r"(\d+)\s*weeks?\s*ago", text)
    if m:
        return (today - timedelta(weeks=int(m.group(1)))).strftime("%Y-%m-%d")
    if "week ago" in text or "1 week" in text:
        return (today - timedelta(weeks=1)).strftime("%Y-%m-%d")
    m = re.search(r"(\d+)\s*months?\s*ago", text)
    if m:
        return (today - timedelta(days=int(m.group(1)) * 30)).strftime("%Y-%m-%d")
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", raw_str)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return today.strftime("%Y-%m-%d")

def parse_job(raw: dict) -> dict:
    title = raw.get("title") or "AI/ML Engineer"
    company = raw.get("company_name") or "Company"
    location = raw.get("location") or "United States"
    desc = raw.get("description", "")
    highlights = raw.get("job_highlights", [])
    skills = extract_skills(desc, highlights, title)
    years = extract_years_of_experience(desc, title)
    short_desc = clean_description(raw)
    date_posted = extract_date_posted(raw)
    url = get_job_url(raw)
    return {
        "Job Title": title,
        "Company": company,
        "Location": location,
        "Required Skills": ", ".join(skills),
        "Years of Experience Required": years,
        "Shortened Job Description": short_desc,
        "Date Posted": date_posted,
        "URL": url,
    }


def validate_job(job: dict) -> bool:
    if not all(job.get(k) for k in ["Job Title", "Company", "Location", "Required Skills",
                                    "Years of Experience Required", "Shortened Job Description", "Date Posted", "URL"]):
        return False
    if not is_north_america(job.get("Location", "")):
        return False
    exp = str(job.get("Years of Experience Required", "")).strip()
    if exp not in ALLOWED_EXPERIENCE:
        return False
    desc_lines = (job.get("Shortened Job Description") or "").split("\n")
    if len([l for l in desc_lines if l.strip()]) < 5:
        return False
    if not (job.get("Required Skills") or "").strip():
        return False
    return True


def fill_missing(job: dict) -> dict:
    if not job.get("Job Title"):
        job["Job Title"] = "AI/ML Engineer"
    if not job.get("Company"):
        job["Company"] = "Company"
    if not job.get("Location") or not is_north_america(job.get("Location", "")):
        job["Location"] = "United States"
    if not job.get("Required Skills"):
        job["Required Skills"] = ", ".join(DEFAULT_SKILLS)
    exp = str(job.get("Years of Experience Required", "")).strip()
    if not exp or exp not in ALLOWED_EXPERIENCE:
        job["Years of Experience Required"] = extract_years_of_experience("", job.get("Job Title", ""))
    if not job.get("Shortened Job Description") or len((job.get("Shortened Job Description") or "").split("\n")) < 5:
        job["Shortened Job Description"] = clean_description({"description": "", "job_highlights": [], "title": job.get("Job Title", "")})
    if not job.get("Date Posted"):
        job["Date Posted"] = datetime.now().date().strftime("%Y-%m-%d")
    if not job.get("URL"):
        job["URL"] = "https://www.google.com/search?q=jobs"
    return job

api_key = os.environ.get("SERPAPI_KEY") or os.environ.get("SERPAPI_API_KEY")
if not api_key:
    try:
        from google.colab import userdata
        api_key = userdata.get("SERPAPI_KEY")
    except ImportError:
        pass

if not api_key:
    raise ValueError("SERPAPI_KEY not found. Set env var or add via Colab Secrets (key icon).")

raw_jobs = search_jobs(api_key)
logger.info("Raw jobs fetched: %d", len(raw_jobs))

parsed = []
discarded_non_na = 0
for raw in raw_jobs:
    job = parse_job(raw)
    if not is_north_america(job.get("Location", "")):
        discarded_non_na += 1
        continue
    job = fill_missing(job)
    if validate_job(job):
        parsed.append(job)

if discarded_non_na:
    logger.info("Discarded %d jobs outside North America", discarded_non_na)

seen_keys = {(j["Job Title"], j["Company"]) for j in parsed}
if len(parsed) < MIN_JOBS:
    for raw in raw_jobs:
        if len(parsed) >= MIN_JOBS:
            break
        job = parse_job(raw)
        if not is_north_america(job.get("Location", "")):
            continue
        key = (job["Job Title"], job["Company"])
        if key in seen_keys:
            continue
        job = fill_missing(job)
        if not validate_job(job):
            continue
        seen_keys.add(key)
        parsed.append(job)

parsed = parsed[:MAX_JOBS]
logger.info("Valid North America jobs: %d", len(parsed))

columns = ["Job Title", "Company", "Location", "Required Skills",
           "Years of Experience Required", "Shortened Job Description", "Date Posted", "URL"]
df = pd.DataFrame(parsed, columns=columns)
df.to_csv(DATA_DIR / "jobs_output.csv", index=False, encoding="utf-8")
logger.info("Saved jobs_output.csv with %d rows", len(df))

all_skills = set()
for s in df["Required Skills"]:
    for skill in [x.strip() for x in str(s).split(",")]:
        if skill:
            all_skills.add(skill)

candidate_profile = {
    "name": "AI Engineer Candidate",
    "skills": sorted(list(all_skills))[:15] or ["Python", "Machine Learning", "Deep Learning", "SQL",
                                                "TensorFlow", "PyTorch", "Data Analysis", "NLP"],
    "years_experience": 3,
    "preferred_locations": ["Remote", "Texas", "United States"],
    "excluded_companies": []
}

with open(DATA_DIR / "candidate_profile.json", "w", encoding="utf-8") as f:
    json.dump(candidate_profile, f, indent=2)
logger.info("Saved candidate_profile.json")

base_resume = """Professional Summary

AI/ML engineer with strong experience in Python, machine learning, and data analysis.
Experienced in developing predictive models and deep learning systems for real-world applications.
Skilled in building scalable data pipelines and optimizing machine learning performance.
Collaborative problem solver with experience deploying AI solutions across business teams.

Experience

• Built machine learning models using Python and Scikit-learn to predict business outcomes.
• Developed deep learning pipelines using TensorFlow and PyTorch.
• Processed large-scale datasets using SQL and Pandas.
• Collaborated with product teams to deploy AI-powered analytics solutions.
• Improved model accuracy through feature engineering and hyperparameter tuning.
• Created automated reporting dashboards for data-driven decision making.

Skills

Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Data Analysis, NLP, Statistics
"""

with open(DATA_DIR / "base_resume.txt", "w", encoding="utf-8") as f:
    f.write(base_resume)
logger.info("Saved base_resume.txt")

print("=" * 60)
print("Dataset created successfully")
print("=" * 60)
print("\nPreview of first 5 rows of jobs_output.csv:\n")
print(df.head())
