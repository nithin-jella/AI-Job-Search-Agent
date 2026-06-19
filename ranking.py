import pandas as pd
from typing import Dict


def _normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    # Normalize the assignment CSV headers to the names used by the ranking logic.
    colmap = {
        "Job Title": "title",
        "Company": "company",
        "Location": "location",
        "Required Skills": "skills",
        "Shortened Job Description": "description",
        "URL": "job_url",
        "Date Posted": "job_posted_date",
        "Years of Experience Required": "years_experience_required",
    }
    rename_map = {c: colmap[c] for c in data.columns if c in colmap}
    return data.rename(columns=rename_map)


def _location_score(job_location: str, preferred_locations: list[str]) -> float:
    loc = (job_location or "").lower()
    if "remote" in loc:
        return 0.8
    return 1.0 if any(p.lower() in loc for p in preferred_locations) else 0.2


def rank_jobs(filtered_df: pd.DataFrame, candidate_profile: Dict, top_k: int = 10) -> pd.DataFrame:
    """
    Scores:
    - skill_overlap_ratio
    - location_score
    - experience_score
    """
    df = _normalize_columns(filtered_df.copy())
    if df.empty:
        cols = [
            "rank",
            "title",
            "company",
            "location",
            "skills",
            "description",
            "job_url",
            "skill_overlap_count",
            "skill_overlap_ratio",
            "location_score",
            "experience_score",
            "final_score",
            "rationale",
        ]
        return pd.DataFrame(columns=cols)

    candidate_skills = {s.lower() for s in candidate_profile.get("skills", [])}
    preferred_locations = candidate_profile.get("preferred_locations", [])
    candidate_years = float(candidate_profile.get("years_experience", 0))

    # Skill overlap
    if "skill_overlap_count" not in df.columns:
        df["skill_overlap_count"] = df["skills_list"].apply(
            lambda job_skills: len(candidate_skills.intersection(set(job_skills)))
        )
    denom = max(len(candidate_skills), 1)
    df["skill_overlap_ratio"] = df["skill_overlap_count"] / denom

    # Location score
    df["location_score"] = df["location"].apply(
        lambda x: _location_score(x, preferred_locations)
    )

    # Experience score
    df["years_experience_required"] = pd.to_numeric(
        df.get("years_experience_required"), errors="coerce"
    )

    def _exp_score(req):
        if pd.isna(req):
            return 0.6
        req = float(req)
        if candidate_years >= req:
            return 1.0
        gap = req - candidate_years
        if gap <= 1:
            return 0.7
        if gap <= 2:
            return 0.4
        return 0.1

    # Higher scores mean the candidate is closer to the target experience level.
    df["experience_score"] = df["years_experience_required"].apply(_exp_score)

    # Final score
    df["final_score"] = (
        0.50 * df["skill_overlap_ratio"] +
        0.20 * df["location_score"] +
        0.30 * df["experience_score"]
    )

    # Optional rationale per row
    df["rationale"] = df.apply(
        lambda r: (
            f"Skill overlap: {int(r['skill_overlap_count'])} ({r['skill_overlap_ratio']:.2f}), "
            f"location score: {r['location_score']:.2f}, "
            f"experience score: {r['experience_score']:.2f}"
        ),
        axis=1,
    )

    ranked = df.sort_values("final_score", ascending=False).head(top_k).copy()
    ranked["rank"] = range(1, len(ranked) + 1)

    # Keep description in the ranked output so downstream tailoring has the job context.
    cols = [
        "rank", "title", "company", "location", "skills", "description", "job_url",
        "skill_overlap_count", "skill_overlap_ratio", "location_score",
        "experience_score",
        "final_score", "rationale",
    ]
    return ranked[cols].reset_index(drop=True)
