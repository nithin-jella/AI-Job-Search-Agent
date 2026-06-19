import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from tools.filtering import filter_jobs
from tools.ranking import rank_jobs


def _candidate():
    return {
        "skills": ["python", "tensorflow", "pytorch", "sql", "aws"],
        "preferred_locations": ["texas", "austin", "dallas", "houston"],
        "years_experience": 4,
        "min_skill_matches_k": 1,
    }


def _jobs_df():
    rows = [
        {
            "title": "AI Engineer",
            "company": "A",
            "location": "Austin, TX",
            "skills": "['Python','TensorFlow','PyTorch','SQL']",
            "salary": "",
            "job_url": "u1",
            "description": "strong fit",
            "job_posted_date": "5 days ago",
        },
        {
            "title": "ML Ops Engineer",
            "company": "B",
            "location": "Dallas, TX",
            "skills": "['Python','AWS']",
            "salary": "",
            "job_url": "u2",
            "description": "medium fit",
            "job_posted_date": "3 days ago",
        },
        {
            "title": "Data Analyst",
            "company": "C",
            "location": "Remote - US",
            "skills": "['Python']",
            "salary": "",
            "job_url": "u3",
            "description": "lower fit",
            "job_posted_date": "1 day ago",
        },
    ]
    return pd.DataFrame(rows)


def test_rank_jobs_outputs_sorted_scores_and_rationale():
    df = _jobs_df()
    filtered, _ = filter_jobs(df, _candidate())
    ranked = rank_jobs(filtered, _candidate(), top_k=10)

    assert len(ranked) == 3
    assert list(ranked["rank"]) == [1, 2, 3]
    assert (ranked["final_score"].diff().fillna(0) <= 0).all()

    required_cols = {
        "rank",
        "title",
        "company",
        "location",
        "skills",
        "job_url",
        "skill_overlap_count",
        "skill_overlap_ratio",
        "location_score",
        "final_score",
        "rationale",
    }
    assert required_cols.issubset(set(ranked.columns))
    assert ranked["rationale"].str.contains("Skill overlap").all()


def test_rank_jobs_is_deterministic_and_respects_top_k():
    df = _jobs_df()
    filtered, _ = filter_jobs(df, _candidate())

    ranked1 = rank_jobs(filtered, _candidate(), top_k=2)
    ranked2 = rank_jobs(filtered, _candidate(), top_k=2)

    assert len(ranked1) == 2
    assert ranked1.reset_index(drop=True).equals(ranked2.reset_index(drop=True))
