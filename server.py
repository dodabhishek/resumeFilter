#!/usr/bin/env python3
"""
Production-ready Web & REST API server for AI Resume Screening & Filtering.
Supports:
    GET  /              (Modern Web UI)
    POST /api/upload    (Multipart file/folder upload)
    POST /api/screen    (Dynamic criteria screening)
    GET  /api/results   (Retrieve latest screening results)
    GET  /health        (Health check)
"""
import os
import json
import uuid
import email
import argparse
import logging
from email.policy import default
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, List, Dict, Any

from src.pipeline import run_screening_pipeline
from src.models.schemas import JobCriteria, ScreeningOutput

logger = logging.getLogger("ResumeScreener.Server")

# In-memory latest results store
LATEST_RESULTS: Dict[str, Any] = {
    "status": "No screening run executed yet",
    "batch_summary": {
        "total_resumes": 0,
        "successfully_parsed": 0,
        "eligible": 0,
        "rejected": 0,
        "execution_time_seconds": 0.0
    },
    "ranked_candidates": [],
    "rejected_candidates": []
}

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
UPLOAD_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "uploads")
os.makedirs(UPLOAD_BASE_DIR, exist_ok=True)


class ScreeningAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for serving Web UI and Screening endpoints."""

    def _send_json(self, status_code: int, payload: dict):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(payload, indent=2).encode('utf-8'))

    def _send_html(self, file_path: str):
        if not os.path.exists(file_path):
            self._send_json(404, {"error": "UI file not found"})
            return
        with open(file_path, 'rb') as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_OPTIONS(self):
        """Handle CORS pre-flight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        url_path = self.path.split('?')[0]

        if url_path in ("/", "/index.html"):
            index_path = os.path.join(STATIC_DIR, "index.html")
            self._send_html(index_path)
        elif url_path in ("/api/results", "/results"):
            self._send_json(200, LATEST_RESULTS)
        elif url_path in ("/health", "/api/v1/health"):
            self._send_json(200, {"status": "healthy", "service": "Resume Screening & UI Engine"})
        elif url_path.startswith("/static/"):
            rel_path = url_path[len("/static/"):]
            safe_path = os.path.normpath(os.path.join(STATIC_DIR, rel_path))
            if safe_path.startswith(STATIC_DIR) and os.path.exists(safe_path) and os.path.isfile(safe_path):
                with open(safe_path, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                if safe_path.endswith('.css'):
                    self.send_header("Content-Type", "text/css")
                elif safe_path.endswith('.js'):
                    self.send_header("Content-Type", "application/javascript")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self._send_json(404, {"error": "File not found"})
        else:
            self._send_json(404, {"error": f"Endpoint not found: {url_path}"})

    def do_POST(self):
        global LATEST_RESULTS
        url_path = self.path.split('?')[0]

        # 1. File / Folder Upload Endpoint
        if url_path == "/api/upload":
            content_type = self.headers.get('Content-Type', '')
            if 'multipart/form-data' not in content_type:
                self._send_json(400, {"error": "Expected multipart/form-data"})
                return

            content_length = int(self.headers.get('Content-Length', 0))
            if content_length <= 0:
                self._send_json(400, {"error": "Empty upload payload"})
                return

            try:
                body_bytes = self.rfile.read(content_length)
                # Parse multipart message via standard library
                raw_headers = f"Content-Type: {content_type}\r\n\r\n".encode('latin1')
                msg = email.message_from_bytes(raw_headers + body_bytes, policy=default)

                session_id = str(uuid.uuid4())[:8]
                session_dir = os.path.join(UPLOAD_BASE_DIR, session_id)
                os.makedirs(session_dir, exist_ok=True)

                saved_files: List[str] = []
                for part in msg.iter_parts():
                    filename = part.get_filename()
                    if filename:
                        # Clean filename
                        safe_filename = os.path.basename(filename)
                        file_bytes = part.get_payload(decode=True)
                        if file_bytes and len(file_bytes) > 0:
                            target_file = os.path.join(session_dir, safe_filename)
                            with open(target_file, 'wb') as f:
                                f.write(file_bytes)
                            saved_files.append(safe_filename)

                logger.info(f"Received {len(saved_files)} file(s) for upload session: {session_id}")
                self._send_json(200, {
                    "session_id": session_id,
                    "uploaded_files": saved_files,
                    "count": len(saved_files)
                })
            except Exception as e:
                logger.error(f"Error handling upload: {e}")
                self._send_json(500, {"error": f"Upload failed: {str(e)}"})
            return

        # 2. Dynamic Screening Endpoint
        elif url_path in ("/api/screen", "/screen"):
            content_length = int(self.headers.get('Content-Length', 0))
            payload = {}
            if content_length > 0:
                try:
                    payload = json.loads(self.rfile.read(content_length).decode('utf-8'))
                except Exception as e:
                    self._send_json(400, {"error": f"Invalid JSON payload: {e}"})
                    return

            # Determine input source
            session_id = payload.get("session_id")
            input_dir = payload.get("input_dir")

            target_dir = None
            if session_id:
                target_dir = os.path.join(UPLOAD_BASE_DIR, session_id)
                if not os.path.exists(target_dir):
                    self._send_json(400, {"error": f"Upload session not found: {session_id}"})
                    return
            elif input_dir:
                target_dir = input_dir
            else:
                target_dir = "./resumes"

            if not os.path.exists(target_dir):
                self._send_json(400, {"error": f"Directory not found: {target_dir}"})
                return

            # Build dynamic criteria
            criteria_payload = payload.get("criteria", {})
            criteria = JobCriteria(
                role_title=criteria_payload.get("role_title", "Software Engineer"),
                must_have_project_keywords=criteria_payload.get("must_have_project_keywords", []),
                required_keywords=criteria_payload.get("required_keywords", []),
                preferred_keywords=criteria_payload.get("preferred_keywords", []),
                custom_instructions=criteria_payload.get("custom_instructions", "")
            )

            try:
                output = run_screening_pipeline(input_dir=target_dir, criteria=criteria)
                LATEST_RESULTS = {
                    "batch_summary": output.summary.model_dump(),
                    "criteria": criteria.model_dump(),
                    "ranked_candidates": [c.model_dump(exclude_none=True) for c in output.ranked_candidates],
                    "rejected_candidates": [c.model_dump(exclude_none=True) for c in output.rejected_candidates]
                }
                self._send_json(200, LATEST_RESULTS)
            except Exception as e:
                logger.error(f"Screening execution error: {e}")
                self._send_json(500, {"error": f"Screening error: {str(e)}"})
            return

        else:
            self._send_json(404, {"error": f"Endpoint not found: {url_path}"})


def run_server(port: int = 8000, host: str = "0.0.0.0"):
    """Start HTTP server."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, ScreeningAPIHandler)
    print(f"\n=======================================================")
    print(f" 🚀 RESUMEFILTER AI SERVER RUNNING")
    print(f" • Web UI Interface : http://localhost:{port}")
    print(f" • Screening API    : http://localhost:{port}/api/screen")
    print(f" • Health Check     : http://localhost:{port}/health")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume Screening Web & API Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    args = parser.parse_args()
    run_server(port=args.port, host=args.host)
