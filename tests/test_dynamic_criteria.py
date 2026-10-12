"""
Unit tests for dynamic user requirements filtering, project-evidence verification,
and explainable acceptance/rejection remarks.
"""
import unittest
from src.models.schemas import ParsedResume, JobCriteria
from src.screening.filter import (
    evaluate_dynamic_eligibility,
    verify_project_experience
)
from src.pipeline import process_single_resume, run_screening_pipeline

class TestDynamicCriteriaAndProjectVerification(unittest.TestCase):

    def test_accept_candidate_with_verified_project_experience(self):
        """Candidate who actually built a project using required technology must be accepted."""
        text = """
        Alex Chen
        Email: alex.chen@example.com
        Skills: Python, TypeScript, Docker
        Projects:
        - Autonomous Agent System: Built stateful multi-agent workflows using LangGraph and FastAPI with Pinecone vector search.
        """
        parsed = ParsedResume(
            file_path="alex.pdf",
            raw_text=text,
            candidate_name="Alex Chen",
            projects_text="Built stateful multi-agent workflows using LangGraph and FastAPI with Pinecone vector search."
        )

        criteria = JobCriteria(
            role_title="AI Engineer",
            required_keywords=["Python"],
            must_have_project_keywords=["LangGraph"],
            preferred_keywords=["Docker"]
        )

        is_eligible, rejections, acceptances, evidence, missing_proj, missing_req, matched = evaluate_dynamic_eligibility(parsed, criteria)
        
        self.assertTrue(is_eligible)
        self.assertEqual(len(rejections), 0)
        self.assertIn("LangGraph", evidence)
        self.assertTrue(any("LangGraph" in rem for rem in acceptances))

    def test_reject_candidate_with_skill_in_skills_list_only_no_project(self):
        """
        CRUCIAL TEST: Candidate who lists 'LangGraph' in skills list, but never built
        any project or experience using it, MUST be rejected for lacking project evidence.
        """
        text = """
        Jordan Smith
        Email: jordan@example.com
        Skills: Python, Django, PostgreSQL, LangGraph, Docker
        Work Experience:
        - Backend Engineer: Maintained Django REST APIs, optimized SQL queries, wrote unit tests.
        Projects:
        - Weather App: Simple weather tracker using Flask and OpenWeatherMap API.
        """
        parsed = ParsedResume(
            file_path="jordan.pdf",
            raw_text=text,
            candidate_name="Jordan Smith",
            projects_text="Weather App: Simple weather tracker using Flask and OpenWeatherMap API."
        )

        criteria = JobCriteria(
            role_title="Agentic Engineer",
            required_keywords=["Python"],
            must_have_project_keywords=["LangGraph"],
            preferred_keywords=["PostgreSQL"]
        )

        is_eligible, rejections, acceptances, evidence, missing_proj, missing_req, matched = evaluate_dynamic_eligibility(parsed, criteria)
        
        self.assertFalse(is_eligible)
        self.assertIn("LangGraph", missing_proj)
        self.assertTrue(any("Lacks required project experience with 'LangGraph'" in r for r in rejections))

    def test_reject_missing_required_core_keyword(self):
        """Candidate missing mandatory core skill must be rejected with explicit remark."""
        text = """
        David Miller
        Email: david@example.com
        Skills: Java, Spring Boot, MySQL
        Projects:
        - Order Service: Built order processing backend in Spring Boot.
        """
        parsed = ParsedResume(
            file_path="david.pdf",
            raw_text=text,
            candidate_name="David Miller"
        )

        criteria = JobCriteria(
            role_title="Python Engineer",
            required_keywords=["Python"],
            must_have_project_keywords=["FastAPI"]
        )

        is_eligible, rejections, _, _, missing_proj, missing_req, _ = evaluate_dynamic_eligibility(parsed, criteria)
        self.assertFalse(is_eligible)
        self.assertIn("Python", missing_req)
        self.assertTrue(any("Missing required core skill: 'Python'" in r for r in rejections))

    def test_process_single_resume_populates_explainable_remarks(self):
        """Ensure CandidateResult contains remarks_accepted and project summaries."""
        text = """
        Elena Rostova
        Email: elena@example.com
        Skills: Python, FastAPI, LangChain, Qdrant, Docker
        Projects:
        - Legal RAG Assistant: Implemented document retrieval pipeline using LangChain, Qdrant vector database, and FastAPI backend.
        """
        import tempfile
        import os
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
            f.write(text)
            temp_path = f.name

        try:
            criteria = JobCriteria(
                role_title="RAG Systems Engineer",
                required_keywords=["Python"],
                must_have_project_keywords=["LangChain"],
                preferred_keywords=["Docker"]
            )
            candidate, failure, eligible = process_single_resume(temp_path, criteria)

            self.assertTrue(eligible)
            self.assertIsNotNone(candidate)
            self.assertTrue(len(candidate.remarks_accepted) > 0)
            self.assertTrue(any("LangChain" in rem for rem in candidate.remarks_accepted))
            self.assertIn("Python", candidate.matched_skills)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
