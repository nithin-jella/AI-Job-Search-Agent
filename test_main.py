import json
import runpy
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd


def _install_fake_gemini(monkeypatch):
    google_module = ModuleType("google")
    generativeai_module = ModuleType("google.generativeai")

    class FakeSchema:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeFunctionDeclaration:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeTool:
        def __init__(self, function_declarations):
            self.function_declarations = function_declarations

    class FakeFunctionResponse:
        def __init__(self, name, response):
            self.name = name
            self.response = response

    class FakePart:
        def __init__(self, text=None, function_call=None, function_response=None):
            self.text = text
            self.function_call = function_call
            self.function_response = function_response

    class FakeFunctionCall:
        def __init__(self, name, args=None):
            self.name = name
            self.args = args or {}

    class FakeResponse:
        def __init__(self, parts):
            self.candidates = [SimpleNamespace(content=SimpleNamespace(parts=parts))]

    class FakeChat:
        def __init__(self):
            self.turn = 0

        def send_message(self, message):
            if self.turn == 0:
                self.turn += 1
                return FakeResponse(
                    [
                        FakePart(text="I will filter the dataset first."),
                        FakePart(function_call=FakeFunctionCall("filtering_tool", {})),
                    ]
                )
            if self.turn == 1:
                self.turn += 1
                return FakeResponse(
                    [
                        FakePart(text="Now I will rank the filtered jobs."),
                        FakePart(function_call=FakeFunctionCall("ranking_tool", {"top_k": 10})),
                    ]
                )
            if self.turn == 2:
                self.turn += 1
                return FakeResponse(
                    [
                        FakePart(text="I will tailor the resume for the top-ranked job."),
                        FakePart(function_call=FakeFunctionCall("resume_tailoring_tool", {"job_rank": 1})),
                    ]
                )

            return FakeResponse(
                [
                    FakePart(
                        text="The best job is AI Engineer because it has the strongest overall match."
                    )
                ]
            )

    class FakeGenerativeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start_chat(self):
            return FakeChat()

    def fake_configure(**kwargs):
        return None

    protos = SimpleNamespace(
        FunctionDeclaration=FakeFunctionDeclaration,
        Schema=FakeSchema,
        Tool=FakeTool,
        Part=FakePart,
        FunctionResponse=FakeFunctionResponse,
        Type=SimpleNamespace(OBJECT="object", INTEGER="integer"),
    )

    generativeai_module.configure = fake_configure
    generativeai_module.GenerativeModel = FakeGenerativeModel
    generativeai_module.protos = protos
    google_module.generativeai = generativeai_module

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.generativeai", generativeai_module)


def _install_fake_groq(monkeypatch):
    groq_module = ModuleType("groq")

    class FakeGroqResponse:
        def __init__(self):
            content = json.dumps(
                {
                    "tailored_summary": "AI/ML engineer with strong Python, TensorFlow, SQL, and AWS experience aligned to AI engineering roles.",
                    "tailored_bullets": [
                        "Built machine learning models using Python, TensorFlow, and SQL to support production-ready AI applications.",
                        "Developed deep learning pipelines and collaborated with teams to deploy AI-powered analytics solutions.",
                    ],
                    "reasoning": "Matched the summary and bullets to the selected role using only resume and candidate details.",
                }
            )
            self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]

    class FakeGroqCompletions:
        def create(self, **kwargs):
            return FakeGroqResponse()

    class FakeGroqChat:
        def __init__(self):
            self.completions = FakeGroqCompletions()

    class FakeGroq:
        def __init__(self, api_key):
            self.api_key = api_key
            self.chat = FakeGroqChat()

    groq_module.Groq = FakeGroq
    monkeypatch.setitem(sys.modules, "groq", groq_module)


def _stage_temp_project(tmp_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    project_root = tmp_path / "JobSearch_Agent"
    (project_root / "tools").mkdir(parents=True)
    (project_root / "data").mkdir()

    shutil.copy(repo_root / "main.py", project_root / "main.py")
    shutil.copy(repo_root / "tools" / "filtering.py", project_root / "tools" / "filtering.py")
    shutil.copy(repo_root / "tools" / "ranking.py", project_root / "tools" / "ranking.py")
    shutil.copy(
        repo_root / "tools" / "tailoring_resume.py",
        project_root / "tools" / "tailoring_resume.py",
    )
    shutil.copy(repo_root / "data" / "base_resume.txt", project_root / "data" / "base_resume.txt")

    jobs_df = pd.DataFrame(
        [
            {
                "Job Title": "AI Engineer",
                "Company": "Alpha",
                "Location": "Austin, TX",
                "Required Skills": "['Python','TensorFlow','SQL']",
                "Years of Experience Required": "3 years",
                "Shortened Job Description": "Strong ML role",
                "URL": "https://example.com/alpha",
            },
            {
                "Job Title": "ML Engineer",
                "Company": "Beta",
                "Location": "Remote - US",
                "Required Skills": "['Python','AWS']",
                "Years of Experience Required": "5 years",
                "Shortened Job Description": "Remote MLOps role",
                "URL": "https://example.com/beta",
            },
            {
                "Job Title": "Backend Engineer",
                "Company": "Gamma",
                "Location": "Boston, MA",
                "Required Skills": "['Java']",
                "Years of Experience Required": "4 years",
                "Shortened Job Description": "Not a fit",
                "URL": "https://example.com/gamma",
            },
        ]
    )
    jobs_df.to_csv(project_root / "data" / "jobs_output.csv", index=False)

    candidate = {
        "name": "Test Candidate",
        "skills": ["Python", "TensorFlow", "SQL", "AWS"],
        "preferred_locations": ["Austin", "Texas", "Remote"],
        "years_experience": 5,
        "min_skill_matches_k": 1,
    }
    (project_root / "data" / "candidate_profile.json").write_text(
        json.dumps(candidate), encoding="utf-8"
    )

    return project_root


def test_main_runs_agent_loop_and_writes_trace(tmp_path, monkeypatch):
    _install_fake_gemini(monkeypatch)
    _install_fake_groq(monkeypatch)
    project_root = _stage_temp_project(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    globals_after_run = runpy.run_path(str(project_root / "main.py"), run_name="__main__")

    artifacts_trace = project_root / "artifacts" / "trace.json"
    tailored_resume = project_root / "artifacts" / "tailored_resume.json"
    assert artifacts_trace.exists()
    assert tailored_resume.exists()

    trace = json.loads(artifacts_trace.read_text(encoding="utf-8"))
    assert len(trace) == 4
    assert trace[0]["tool_calls"][0]["tool_name"] == "filtering_tool"
    assert trace[1]["tool_calls"][0]["tool_name"] == "ranking_tool"
    assert trace[2]["tool_calls"][0]["tool_name"] == "resume_tailoring_tool"

    agent_outputs = globals_after_run["agent_outputs"]
    assert agent_outputs["filtered_df"] is not None
    assert agent_outputs["ranked_df"] is not None
    assert agent_outputs["tailored_resume"] is not None
    assert len(agent_outputs["tailored_resume"]["tailored_bullets"]) == 2
    assert len(agent_outputs["ranked_df"]) >= 1
