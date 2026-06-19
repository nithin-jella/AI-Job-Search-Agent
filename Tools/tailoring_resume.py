import os
import re
import json
from typing import Dict, List, Tuple, Any

from groq import Groq


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def tokenize(text: str) -> set:
    return set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-\+\.#]*\b", str(text).lower()))


def extract_resume_parts(resume_file_path: str) -> Tuple[str, List[str]]:
    """
    Extract:
    - professional summary
    - all experience bullets
    from base_resume.txt
    """
    with open(resume_file_path, "r", encoding="utf-8") as f:
        text = f.read()

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Pull out the summary block before the Experience section.
    summary_lines = []
    in_summary = False
    for line in lines:
        if line.lower() == "professional summary":
            in_summary = True
            continue
        if line.lower() == "experience":
            break
        if in_summary:
            summary_lines.append(line)

    resume_summary = " ".join(summary_lines).strip()

    # Grab the bullet points from the Experience section only.
    bullets = []
    in_experience = False
    for line in lines:
        if line.lower() == "experience":
            in_experience = True
            continue
        if line.lower() == "skills":
            break
        if in_experience and (line.startswith("•") or line.startswith("-")):
            bullets.append(line.lstrip("•- ").strip())

    if len(bullets) < 2:
        raise ValueError("Need at least 2 experience bullets in base_resume.txt")

    return resume_summary, bullets


def select_best_bullets(selected_job: Dict[str, Any], bullets: List[str], top_k: int = 2) -> List[str]:
    """
    Select the best matching bullets using overlap with:
    - title
    - skills
    - description
    """
    job_text = " ".join([
        str(selected_job.get("title", "")),
        str(selected_job.get("skills", "")),
        str(selected_job.get("description", "")),
    ])
    job_tokens = tokenize(job_text)

    # Start from bullets that already overlap with the job instead of rewriting random ones.
    scored = []
    for bullet in bullets:
        bullet_tokens = tokenize(bullet)
        overlap = len(job_tokens & bullet_tokens)
        scored.append((overlap, bullet))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = [b for _, b in scored[:top_k]]

    if len(best) < 2:
        raise ValueError("Could not select 2 matching bullets.")

    return best


class ResumeTailoringTool:
    def __init__(self, api_key: str = None, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found.")
        self.client = Groq(api_key=self.api_key)
        self.model = model

    def _build_prompt(
        self,
        candidate_profile: Dict[str, Any],
        selected_job: Dict[str, Any],
        resume_summary: str,
        selected_bullets: List[str],
    ) -> str:
        return f"""
You are the Resume Tailoring Tool inside an AI job search agent.

Task:
1. Rewrite the professional summary to better match the selected job.
2. Rewrite EXACTLY TWO experience bullet points to better match the selected job.

Priority:
Match the rewritten content primarily to:
- Job Title
- Required Skills
- Shortened Job Description

STRICT RULES:
- Use ONLY information already present in:
  1. candidate profile
  2. original resume summary
  3. original selected experience bullet points
- Do NOT invent years of experience.
- Do NOT invent companies, projects, tools, frameworks, metrics, achievements, deployments, or responsibilities not supported by the inputs.
- Do NOT introduce new technologies that are not present in the candidate profile or resume.
- Keep the summary concise and professional.
- Keep each bullet resume-ready, specific, and aligned to the job.
- Return ONLY valid JSON. No markdown fences. No extra text.

Return JSON in exactly this format:
{{
  "tailored_summary": "string",
  "tailored_bullets": [
    "string",
    "string"
  ],
  "reasoning": "short explanation"
}}

Candidate Profile:
{json.dumps(candidate_profile, indent=2)}

Selected Job:
Title: {selected_job.get("title", "")}
Company: {selected_job.get("company", "")}
Location: {selected_job.get("location", "")}
Skills: {selected_job.get("skills", "")}
Description: {selected_job.get("description", "")}
Rationale: {selected_job.get("rationale", "")}

Original Resume Summary:
{resume_summary}

Selected Experience Bullet 1:
{selected_bullets[0]}

Selected Experience Bullet 2:
{selected_bullets[1]}
""".strip()

    def _extract_json(self, raw_text: str) -> Dict[str, Any]:
        cleaned = raw_text.strip()
        # The model sometimes wraps JSON in code fences, so strip them first.
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1

        if start == -1 or end == 0:
            raise ValueError(f"Could not find JSON in model output:\n{raw_text}")

        cleaned = cleaned[start:end]
        return json.loads(cleaned)

    def tailor_resume(
        self,
        candidate_profile: Dict[str, Any],
        selected_job: Dict[str, Any],
        resume_summary: str,
        selected_bullets: List[str],
    ) -> Dict[str, Any]:
        if len(selected_bullets) != 2:
            raise ValueError("selected_bullets must contain exactly 2 bullets.")

        prompt = self._build_prompt(
            candidate_profile=candidate_profile,
            selected_job=selected_job,
            resume_summary=resume_summary,
            selected_bullets=selected_bullets,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
        )

        raw_output = response.choices[0].message.content.strip()
        parsed = self._extract_json(raw_output)

        if "tailored_summary" not in parsed or "tailored_bullets" not in parsed:
            raise ValueError(f"Missing required keys in model output: {parsed}")

        if not isinstance(parsed["tailored_bullets"], list) or len(parsed["tailored_bullets"]) != 2:
            raise ValueError("Model must return exactly 2 tailored bullets.")

        # Return a compact payload that the main agent loop can log and save directly.
        return {
            "tool_name": "Resume Tailoring Tool",
            "selected_job_title": selected_job.get("title", ""),
            "company": selected_job.get("company", ""),
            "selected_original_bullets": selected_bullets,
            "tailored_summary": normalize_text(parsed["tailored_summary"]),
            "tailored_bullets": [normalize_text(b) for b in parsed["tailored_bullets"]],
            "reasoning": normalize_text(parsed.get("reasoning", "")),
        }

