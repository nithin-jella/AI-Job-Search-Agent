import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from tools.filtering import filter_jobs


def _candidate():
    return {
        "skills": ["python", "tensorflow", "pytorch", "sql", "aws"],
        "preferred_locations": ["texas", "austin", "dallas", "houston"],
        "years_experience": 4,
        "min_skill_matches_k": 2,
    }


def _jobs_df():
    rows = [
        {
            "title": "AI Engineer",
            "company": "A",
            "location": "Austin, TX",
            "skills": "['Python','TensorFlow','SQL']",
            "salary": "",
            "job_url": "u1",
            "description": "good fit",
            "job_posted_date": "5 days ago",
        },
        {
            "title": "ML Engineer",
            "company": "B",
            "location": "Remote - US",
            "skills": "['Python','AWS']",
            "salary": "",
            "job_url": "u2",
            "description": "remote fit",
            "job_posted_date": "2 days ago",
        },
        {
            "title": "Data Engineer",
            "company": "C",
            "location": "New York, NY",
            "skills": "['Python','SQL','Airflow']",
            "salary": "",
            "job_url": "u3",
            "description": "location mismatch",
            "job_posted_date": "1 day ago",
        },
        {
            "title": "Backend Engineer",
            "company": "D",
            "location": "Dallas, TX",
            "skills": "['Java']",
            "salary": "",
            "job_url": "u4",
            "description": "skill mismatch",
            "job_posted_date": "7 days ago",
        },
    ]
    return pd.DataFrame(rows)


def test_filter_jobs_applies_location_and_skill_filters():
    df = _jobs_df()
    filtered, trace = filter_jobs(df, _candidate())

    # Should keep: Austin + Remote rows
    assert len(filtered) == 2
    assert set(filtered["job_url"]) == {"u1", "u2"}

    # Skill overlap count added and satisfies K
    assert "skill_overlap_count" in filtered.columns
    assert (filtered["skill_overlap_count"] >= 2).all()

    # Trace sanity checks
    assert trace["input_rows"] == 4
    assert trace["after_location_filter"] == 3
    assert trace["after_skill_filter"] == 2
    assert "location_match_or_remote" in trace["filters_applied"]
    assert "skill_overlap_count >= 2" in trace["filters_applied"]


def test_filter_jobs_is_deterministic():
    df = _jobs_df()
    candidate = _candidate()

    filtered1, trace1 = filter_jobs(df, candidate)
    filtered2, trace2 = filter_jobs(df, candidate)

    assert trace1 == trace2
    assert filtered1.reset_index(drop=True).equals(filtered2.reset_index(drop=True))
