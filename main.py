import sys
import json
import os
from pathlib import Path

import pandas as pd
import google.generativeai as genai
from google.generativeai import protos

PROJECT_ROOT = Path(__file__).resolve().parent
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Set it in your shell environment.")

genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"

print("Gemini configured successfully.")


JOBS_CSV = PROJECT_ROOT / "data" / "jobs_output.csv"
CANDIDATE_JSON = PROJECT_ROOT / "data" / "candidate_profile.json"
BASE_RESUME = PROJECT_ROOT / "data" / "base_resume.txt"

sys.path.append(str(PROJECT_ROOT / "tools"))

from filtering import filter_jobs
from ranking import rank_jobs
from tailoring_resume import (
    ResumeTailoringTool,
    extract_resume_parts,
    select_best_bullets,
)

print("Imported filter_jobs, rank_jobs, and resume tailoring helpers successfully.")


df = pd.read_csv(JOBS_CSV)

with open(CANDIDATE_JSON, "r", encoding="utf-8") as f:
    candidate = json.load(f)

print("Dataset shape:", df.shape)
print(
    "Candidate loaded:",
    candidate.get("name", "Candidate"),
    f"| years={candidate.get('years_experience')}",
    f"| skills={len(candidate.get('skills', []))}",
    f"| preferred_locations={len(candidate.get('preferred_locations', []))}",
)


def filtering_tool_wrapper(df: pd.DataFrame, candidate: dict) -> dict:
    """
    Wraps the GitHub filter_jobs() tool and returns JSON-serializable output.
    """
    filtered_df, trace = filter_jobs(df, candidate)

    preview_cols = [
        c
        for c in [
            "title",
            "company",
            "location",
            "skills",
            "job_url",
            "skill_overlap_count",
            "years_experience_required",
        ]
        if c in filtered_df.columns
    ]

    preview_rows = filtered_df[preview_cols].head(10).fillna("").to_dict(orient="records")

    return {
        "filtered_count": int(len(filtered_df)),
        "trace": trace,
        "filtered_preview": preview_rows,
        "filtered_df": filtered_df,
    }


def ranking_tool_wrapper(filtered_df: pd.DataFrame, candidate: dict, top_k: int = 10) -> dict:
    ranked_df = rank_jobs(filtered_df, candidate, top_k=top_k)

    # Keep the original job description attached so the tailoring step has full context.
    join_keys = [c for c in ["job_url", "title", "company", "location"] if c in ranked_df.columns]
    metadata_cols = [c for c in join_keys + ["description"] if c in filtered_df.columns]
    if "description" in metadata_cols and join_keys:
        ranked_df = ranked_df.merge(
            filtered_df[metadata_cols].drop_duplicates(subset=join_keys),
            on=join_keys,
            how="left",
        )

    preview_cols = [
        c
        for c in [
            "rank",
            "title",
            "company",
            "location",
            "job_url",
            "skill_overlap_count",
            "skill_overlap_ratio",
            "location_score",
            "experience_score",
            "final_score",
            "rationale",
        ]
        if c in ranked_df.columns
    ]

    preview_rows = ranked_df[preview_cols].head(10).fillna("").to_dict(orient="records")

    best_job = preview_rows[0] if preview_rows else None

    return {
        "ranked_count": int(len(ranked_df)),
        "top_jobs_preview": preview_rows,
        "best_job": best_job,
        "ranked_df": ranked_df,
    }


def resume_tailoring_tool_wrapper(
    ranked_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    candidate: dict,
    job_rank: int = 1,
) -> dict:
    if ranked_df is None or ranked_df.empty:
        raise ValueError("No ranked jobs available. Run ranking_tool first.")

    if not BASE_RESUME.exists():
        raise FileNotFoundError(f"Base resume not found: {BASE_RESUME}")

    try:
        selected_rank = max(1, int(job_rank))
    except (TypeError, ValueError):
        selected_rank = 1

    # If the requested rank is missing, just use the top-ranked job.
    selected_matches = ranked_df[ranked_df["rank"] == selected_rank]
    if selected_matches.empty:
        selected_matches = ranked_df.head(1)

    selected_job = selected_matches.iloc[0].to_dict()

    if not selected_job.get("description") and filtered_df is not None and not filtered_df.empty:
        matched_rows = filtered_df
        if selected_job.get("job_url") and "job_url" in filtered_df.columns:
            matched_rows = filtered_df[filtered_df["job_url"] == selected_job["job_url"]]
        elif all(k in filtered_df.columns for k in ["title", "company", "location"]):
            matched_rows = filtered_df[
                (filtered_df["title"] == selected_job.get("title", ""))
                & (filtered_df["company"] == selected_job.get("company", ""))
                & (filtered_df["location"] == selected_job.get("location", ""))
            ]
        if not matched_rows.empty and "description" in matched_rows.columns:
            selected_job["description"] = matched_rows.iloc[0]["description"]

    resume_summary, bullets = extract_resume_parts(str(BASE_RESUME))
    selected_bullets = select_best_bullets(selected_job, bullets, top_k=2)

    tailor = ResumeTailoringTool()
    result = tailor.tailor_resume(
        candidate_profile=candidate,
        selected_job=selected_job,
        resume_summary=resume_summary,
        selected_bullets=selected_bullets,
    )
    result["job_rank"] = int(selected_job.get("rank", selected_rank))
    return result


filtering_fn = protos.FunctionDeclaration(
    name="filtering_tool",
    description=(
        "Apply rule-based filtering to the job dataset using the candidate profile. "
        "Use this first when you need to narrow the dataset."
    ),
    parameters=protos.Schema(
        type=protos.Type.OBJECT,
        properties={},
    ),
)

ranking_fn = protos.FunctionDeclaration(
    name="ranking_tool",
    description=(
        "Rank the already filtered jobs using skill alignment, location, and experience fit. "
        "Use this after filtering when you need to choose the best job."
    ),
    parameters=protos.Schema(
        type=protos.Type.OBJECT,
        properties={
            "top_k": protos.Schema(
                type=protos.Type.INTEGER,
                description="How many ranked jobs to return",
            )
        },
    ),
)

tailoring_fn = protos.FunctionDeclaration(
    name="resume_tailoring_tool",
    description=(
        "Tailor the base resume for one ranked job by rewriting the summary and exactly two "
        "experience bullet points. Use this after ranking."
    ),
    parameters=protos.Schema(
        type=protos.Type.OBJECT,
        properties={
            "job_rank": protos.Schema(
                type=protos.Type.INTEGER,
                description="Rank number of the job to tailor the resume for",
            )
        },
    ),
)

gemini_tools = protos.Tool(function_declarations=[filtering_fn, ranking_fn, tailoring_fn])


model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    tools=[gemini_tools],
    system_instruction="""
You are a single LLM-based AI job search agent.

Your task:
1. Analyze the candidate profile and the available job dataset.
2. Decide which tool to call and explain why.
3. You must use filtering_tool, ranking_tool, and resume_tailoring_tool before making a final recommendation.
4. After each tool call, briefly explain what you learned from the output.
5. End by recommending exactly one best job and explain why it is the best fit.
6. Include the tailored summary and the two tailored bullet points in the final answer.

Rules:
- Do not pretend a tool ran if it did not.
- Use only the provided dataset and tool outputs.
- Keep reasoning concise but clear.
- When calling ranking_tool, request the full shortlist, not just one row.
- Use resume_tailoring_tool on the final selected ranked job.
""",
)

print("Gemini agent initialized.")


def run_agent(df: pd.DataFrame, candidate: dict):
    chat = model.start_chat()

    filtered_df_cache = None
    ranked_df_cache = None
    tailored_resume_cache = None
    used_filtering = False
    used_ranking = False
    used_tailoring = False
    trace = []

    dataset_preview = df.head(8).fillna("").to_dict(orient="records")

    user_prompt = f"""
Candidate profile:
{json.dumps(candidate, indent=2)}

Dataset info:
- Total jobs: {len(df)}
- Sample rows:
{json.dumps(dataset_preview, indent=2)}

Your goal:
Find the best matching job for this candidate.

Remember:
- Decide which tool to use and why.
- Use filtering_tool, ranking_tool, and resume_tailoring_tool before giving a final answer.
- Ask ranking_tool for the full ranked shortlist so all filtered jobs can be compared.
- Use resume_tailoring_tool on the best-ranked job after ranking.
"""

    response = chat.send_message(user_prompt)

    turn = 0
    while True:
        turn += 1
        print("\n" + "─" * 70)
        print(f"[Agent Turn {turn}]")
        print("─" * 70)

        parts = response.candidates[0].content.parts
        reasoning_texts = []

        for part in parts:
            if hasattr(part, "text") and part.text and part.text.strip():
                print("\nAgent reasoning:\n")
                print(part.text.strip())
                reasoning_texts.append(part.text.strip())

        fn_calls = [
            part.function_call
            for part in parts
            if hasattr(part, "function_call") and part.function_call and part.function_call.name
        ]

        if not fn_calls:
            trace.append(
                {
                    "turn": turn,
                    "reasoning": reasoning_texts,
                    "tool_calls": [],
                }
            )
            # Keep the chat going until all required tools have been used at least once.
            if not (used_filtering and used_ranking and used_tailoring):
                print("\nAgent stopped before using all required tools. Requesting continuation.")
                response = chat.send_message(
                    "You must use filtering_tool, ranking_tool, and resume_tailoring_tool before finishing. "
                    "Continue by calling the missing required tool(s)."
                )
                continue
            print("\n" + "=" * 70)
            print("AGENT FINISHED")
            print("=" * 70)
            break

        function_responses = []
        turn_trace = {
            "turn": turn,
            "reasoning": reasoning_texts,
            "tool_calls": [],
        }

        for fn_call in fn_calls:
            tool_name = fn_call.name
            tool_args = dict(fn_call.args)

            print(f"\nTool requested: {tool_name}")
            print("Arguments:", tool_args)

            if tool_name == "filtering_tool":
                result = filtering_tool_wrapper(df, candidate)
                filtered_df_cache = result["filtered_df"]
                used_filtering = True

                payload = {
                    "filtered_count": result["filtered_count"],
                    "trace": result["trace"],
                    "filtered_preview": result["filtered_preview"],
                }
                print(
                    f"Filter result: {payload['filtered_count']} jobs kept "
                    f"(location={payload['trace']['after_location_filter']}, "
                    f"skills={payload['trace']['after_skill_filter']}, "
                    f"experience={payload['trace']['after_experience_filter']})"
                )

            elif tool_name == "ranking_tool":
                if filtered_df_cache is None:
                    payload = {"error": "ranking_tool called before filtering_tool. Filter first."}
                else:
                    requested_top_k = tool_args.get("top_k", 10)
                    try:
                        top_k = max(1, int(requested_top_k))
                    except (TypeError, ValueError):
                        top_k = 10
                    result = ranking_tool_wrapper(filtered_df_cache, candidate, top_k=top_k)
                    ranked_df_cache = result["ranked_df"]
                    used_ranking = True

                    payload = {
                        "ranked_count": result["ranked_count"],
                        "top_jobs_preview": result["top_jobs_preview"],
                        "best_job": result["best_job"],
                    }
                    if payload["best_job"] is not None:
                        best = payload["best_job"]
                        print(
                            f"Ranking result: {payload['ranked_count']} jobs ranked | "
                            f"top job: {best['title']} at {best['company']} "
                            f"(score={best['final_score']:.2f})"
                        )

            elif tool_name == "resume_tailoring_tool":
                if ranked_df_cache is None:
                    payload = {
                        "error": "resume_tailoring_tool called before ranking_tool. Rank jobs first."
                    }
                else:
                    requested_job_rank = tool_args.get("job_rank", 1)
                    try:
                        job_rank = max(1, int(requested_job_rank))
                    except (TypeError, ValueError):
                        job_rank = 1

                    result = resume_tailoring_tool_wrapper(
                        ranked_df=ranked_df_cache,
                        filtered_df=filtered_df_cache,
                        candidate=candidate,
                        job_rank=job_rank,
                    )
                    tailored_resume_cache = result
                    used_tailoring = True

                    payload = {
                        "job_rank": result["job_rank"],
                        "selected_job_title": result["selected_job_title"],
                        "company": result["company"],
                        "tailored_summary": result["tailored_summary"],
                        "tailored_bullets": result["tailored_bullets"],
                        "reasoning": result["reasoning"],
                    }
                    print(
                        f"Tailoring result: rank {payload['job_rank']} | "
                        f"{payload['selected_job_title']} at {payload['company']}"
                    )

            else:
                payload = {"error": f"Unknown tool: {tool_name}"}

            turn_trace["tool_calls"].append(
                {
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "tool_response": payload,
                }
            )
            function_responses.append(
                protos.Part(
                    function_response=protos.FunctionResponse(
                        name=tool_name,
                        response={"result": json.dumps(payload)},
                    )
                )
            )

        trace.append(turn_trace)
        response = chat.send_message(function_responses)

    return {
        "filtered_df": filtered_df_cache,
        "ranked_df": ranked_df_cache,
        "tailored_resume": tailored_resume_cache,
        "trace": trace,
    }


agent_outputs = run_agent(df, candidate)

artifacts_dir = PROJECT_ROOT / "artifacts"
artifacts_dir.mkdir(exist_ok=True)
# Save the full trace so it is easy to inspect the agent's decisions after the run.
with open(artifacts_dir / "trace.json", "w", encoding="utf-8") as f:
    json.dump(agent_outputs["trace"], f, indent=2)

if agent_outputs["tailored_resume"] is not None:
    with open(artifacts_dir / "tailored_resume.json", "w", encoding="utf-8") as f:
        json.dump(agent_outputs["tailored_resume"], f, indent=2)
    tailored = agent_outputs["tailored_resume"]
    tailored_resume_text = (
        f"Target Job: {tailored['selected_job_title']} - {tailored['company']}\n\n"
        "Tailored Resume Summary\n"
        f"{tailored['tailored_summary']}\n\n"
        "Tailored Experience Bullets\n"
        f"1. {tailored['tailored_bullets'][0]}\n"
        f"2. {tailored['tailored_bullets'][1]}\n"
    )
    with open(artifacts_dir / "tailored_resume.txt", "w", encoding="utf-8") as f:
        f.write(tailored_resume_text)


if agent_outputs["filtered_df"] is not None:
    print("\nFiltered Jobs:")
    for idx, row in agent_outputs["filtered_df"].reset_index(drop=True).iterrows():
        print(f"{idx + 1}. {row['title']} - {row['company']} ({row['location']})")
else:
    print("\nNo filtered output available.")


if agent_outputs["ranked_df"] is not None:
    print("Final ranked_df rows:", len(agent_outputs["ranked_df"]))
else:
    print("No ranked output available.")


print("\nTop Ranked Jobs:")
for _, row in agent_outputs["ranked_df"].iterrows():
    print(f"{row['rank']}. {row['title']} - {row['company']} ({row['location']})")


if agent_outputs["tailored_resume"] is not None:
    tailored = agent_outputs["tailored_resume"]
    print("\nTailored Resume Summary:")
    print(tailored["tailored_summary"])
    print("\nTailored Experience Bullets:")
    for idx, bullet in enumerate(tailored["tailored_bullets"], start=1):
        print(f"{idx}. {bullet}")
else:
    print("\nNo tailored resume output available.")
