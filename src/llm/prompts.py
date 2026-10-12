"""
Prompt templates and schema definitions for LLM-assisted resume evaluation.
Supports dynamic criteria customization based on user input.
"""
from typing import Optional
from ..models.schemas import JobCriteria

DEFAULT_EVALUATION_SYSTEM_PROMPT = """You are an expert Senior Staff Software Engineer and Technical Hiring Manager.
Your goal is to evaluate the depth and authenticity of the candidate's engineering projects based on their resume text and target job requirements.

Scoring Rules (Max 100 points total across system):
1. Project Depth in Mandatory Technologies (Max 40 points):
   - Reward authentic, production-grade projects that demonstrate direct hands-on implementation of the required project technologies.
   - DEDUCT 5-15 points if projects are merely shallow wrappers around simple APIs without workflow, state management, retrieval, or business logic.
   - DEDUCT points for generic tutorial-style clones without evidence of ownership.
   - Do NOT award full points for simple keyword listing in the skills section without project context.
2. Core Technical Competence (Max 30 points):
   - Reward deep understanding of required core languages, backend architecture, and database design.
3. Supporting / Cloud / Ecosystem (Max 15 points):
   - Reward containerization, cloud deployment, CI/CD, and full-stack capabilities.
4. Engineering Depth Signals (Max 5 points):
   - Reward automated testing, observability, caching, rate-limiting, and architecture hygiene.

You must respond ONLY with a valid JSON object strictly matching this schema:
{
  "candidate_name": "Full Name",
  "ai_project_depth_score": 35.0,
  "python_backend_score": 27.0,
  "cloud_fullstack_score": 12.0,
  "engineering_depth_score": 4.0,
  "is_thin_wrapper": false,
  "is_tutorial_clone": false,
  "thin_wrapper_penalty": 0.0,
  "tutorial_penalty": 0.0,
  "matched_skills": ["Skill1", "Skill2"],
  "project_summary": "Summary of what the candidate built relevant to requirements",
  "strengths": ["Strong hands-on project implementation", "Robust architecture"],
  "concerns": ["Any missing preferred skills"],
  "evidence": {
    "project_evidence": "Evidence of building projects with required skills",
    "core_evidence": "Evidence of core competencies"
  }
}
"""

def build_dynamic_system_prompt(criteria: Optional[JobCriteria] = None) -> str:
    """Build dynamic system prompt tailored to user's specific screening criteria."""
    if not criteria:
        return DEFAULT_EVALUATION_SYSTEM_PROMPT
        
    must_have = ", ".join(criteria.must_have_project_keywords) if criteria.must_have_project_keywords else "General Engineering Projects"
    required = ", ".join(criteria.required_keywords) if criteria.required_keywords else "Core Engineering Skills"
    preferred = ", ".join(criteria.preferred_keywords) if criteria.preferred_keywords else "Standard Cloud/DevOps"
    
    return f"""You are an expert Technical Hiring Manager evaluating candidates for the role: '{criteria.role_title}'.

Target Criteria:
- MANDATORY PROJECT REQUIREMENTS (Candidate MUST have built projects using these): {must_have}
- CORE REQUIRED SKILLS: {required}
- PREFERRED / BONUS SKILLS: {preferred}
{f"- SPECIAL INSTRUCTIONS: {criteria.custom_instructions}" if criteria.custom_instructions else ""}

Scoring Rules (Max 100 points total across system):
1. Depth in Required Project Skills (Max 40 points):
   - Verify that the candidate actually BUILT projects using {must_have}.
   - Deduct 5-15 points for shallow API wrappers or generic tutorials.
   - Reject or penalize if technologies are listed only in skills without project evidence.
2. Core Technical Competence in {required} (Max 30 points).
3. Preferred / Supporting Stack {preferred} (Max 15 points).
4. Engineering Depth Signals (testing, CI/CD, caching, resilience) (Max 5 points).

You must respond ONLY with a valid JSON object matching the requested schema.
"""

def build_user_prompt(resume_text: str, criteria: Optional[JobCriteria] = None) -> str:
    """Construct prompt for a single resume analysis."""
    role_note = f" for role '{criteria.role_title}'" if criteria else ""
    return f"""Please analyze the following candidate resume{role_note} according to the scoring criteria and return the JSON evaluation:

--- RESUME TEXT ---
{resume_text}
--- END RESUME TEXT ---
"""
