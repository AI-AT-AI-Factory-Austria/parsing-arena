"""LlamaParse engine against the real SDK with a mocked HTTP transport."""

import asyncio
import functools
import json

import httpx
import llama_cloud
import pytest

from app.config import Settings
from app.engines.llamaparse import LlamaParseEngine, LlamaParseError, page_markdown
from tests.conftest import SAMPLE_PDF

JOB = {"id": "job-1", "project_id": "proj-1"}


class FakeApi:
    def __init__(self, *, statuses=("PENDING", "RUNNING", "COMPLETED"), pages=None, auth_ok=True):
        self.statuses = list(statuses)
        self.pages = pages or []
        self.auth_ok = auth_ok
        self.upload: httpx.Request | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if not self.auth_ok:
            return httpx.Response(401, json={"detail": "Invalid API key"})
        if request.method == "POST" and request.url.path == "/api/v2/parse/upload":
            self.upload = request
            return httpx.Response(200, json={**JOB, "status": "PENDING"})
        if request.method == "GET" and request.url.path == "/api/v2/parse/job-1":
            if "expand" in request.url.params:
                return httpx.Response(
                    200,
                    json={
                        "job": {**JOB, "status": "COMPLETED", "usage": {"credits": 45.0}},
                        "markdown": {"pages": self.pages},
                    },
                )
            status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            job = {**JOB, "status": status}
            if status == "FAILED":
                job["error_message"] = "document is encrypted"
            return httpx.Response(200, json={"job": job})
        return httpx.Response(404, json={"detail": "not found"})


def run_engine(monkeypatch, api: FakeApi, *, page_numbers, page_count=2):
    transport = httpx.MockTransport(api)
    monkeypatch.setattr(
        llama_cloud,
        "AsyncLlamaCloud",
        functools.partial(
            llama_cloud.AsyncLlamaCloud,
            http_client=httpx.AsyncClient(transport=transport),
            max_retries=0,
        ),
    )
    settings = Settings(
        llama_cloud_api_key="llx-test",
        llamaparse_poll_interval_s=0,
        local_inference=False,
    )
    events: list[dict] = []

    async def report(event: dict) -> None:
        events.append(event)

    output = asyncio.run(
        LlamaParseEngine(settings).parse(
            SAMPLE_PDF,
            filename="report.pdf",
            media_type="application/pdf",
            page_numbers=page_numbers,
            page_count=page_count,
            report=report,
        )
    )
    return output, events


def _configuration(request: httpx.Request) -> dict:
    body = request.content.decode("utf-8", "replace")
    marker = 'name="configuration"'
    start = body.index(marker) + len(marker)
    payload = body[start:].split("\r\n\r\n", 1)[1].split("\r\n--", 1)[0]
    return json.loads(payload)


def test_subset_request_and_result(monkeypatch):
    api = FakeApi(pages=[{"page_number": 2, "success": True, "markdown": "# Outlook"}])
    output, events = run_engine(monkeypatch, api, page_numbers=[2])

    config = _configuration(api.upload)
    assert config["tier"] == "agentic_plus"
    assert config["version"] == "latest"
    assert config["disable_cache"] is True
    assert config["page_ranges"] == {"target_pages": "2"}
    assert 'filename="report.pdf"' in api.upload.content.decode("utf-8", "replace")

    assert [e["phase"] for e in events] == ["uploading", "queued", "parsing"]
    assert [(p.page_number, p.markdown) for p in output.pages] == [(2, "# Outlook")]
    assert output.meta["credits"] == 45.0
    assert output.meta["job_id"] == "job-1"


def test_whole_document_sends_no_page_ranges(monkeypatch):
    api = FakeApi(
        pages=[
            {"page_number": 1, "success": True, "markdown": "one"},
            {"page_number": 2, "success": True, "markdown": "two"},
        ]
    )
    output, _ = run_engine(monkeypatch, api, page_numbers=[1, 2])
    assert "page_ranges" not in _configuration(api.upload)
    assert output.markdown == "one\n\n---\n\ntwo"


def test_renumbered_subset_maps_back(monkeypatch):
    api = FakeApi(pages=[{"page_number": 1, "success": True, "markdown": "only"}])
    output, _ = run_engine(monkeypatch, api, page_numbers=[2])
    assert [(p.page_number, p.markdown) for p in output.pages] == [(2, "only")]


def test_separate_header_and_footer_are_kept_for_display_and_judging(monkeypatch):
    api = FakeApi(
        pages=[
            {
                "page_number": 1,
                "success": True,
                "header": "GLOBAL REPORT 2024 | ABOUT | DATA",
                "markdown": "## Enterprise cybersecurity\n\nReport content.",
                "footer": "22",
            }
        ]
    )
    output, _ = run_engine(monkeypatch, api, page_numbers=[2])
    full_page = (
        "GLOBAL REPORT 2024 | ABOUT | DATA\n\n## Enterprise cybersecurity\n\nReport content.\n\n22"
    )
    assert output.pages[0].page_number == 2
    assert output.pages[0].markdown == full_page
    assert output.markdown == full_page
    assert output.meta["page_assembly"] == "header_body_footer_v1"
    assert output.meta["page_regions"] == [
        {
            "page_number": 2,
            "header": "GLOBAL REPORT 2024 | ABOUT | DATA",
            "footer": "22",
        }
    ]


@pytest.mark.parametrize(
    "body,header,footer,expected",
    [
        ("  Body text.\n", None, None, "Body text."),
        ("Body", " ", "\n", "Body"),
        ("", "Report title", "Page 1", "Report title\n\nPage 1"),
        ("# Report\n\nBody\n\nPage 1", "# Report", "Page 1", "# Report\n\nBody\n\nPage 1"),
        (
            "Report   title\n\nBody\n\nPage\n1",
            "Report title",
            "Page 1",
            "Report   title\n\nBody\n\nPage\n1",
        ),
        (
            "Body mentions Report title inside.",
            "Report title",
            None,
            "Report title\n\nBody mentions Report title inside.",
        ),
        ("20240 is a number", "2024", None, "2024\n\n20240 is a number"),
    ],
)
def test_page_regions_preserve_text_without_duplicate_edges(body, header, footer, expected):
    assert page_markdown(body, header, footer) == expected


def test_failed_page_is_reported(monkeypatch):
    api = FakeApi(
        pages=[
            {"page_number": 1, "success": True, "markdown": "ok"},
            {"page_number": 2, "success": False, "error": "timeout"},
        ]
    )
    output, _ = run_engine(monkeypatch, api, page_numbers=[1, 2])
    assert output.pages[1].error == "timeout"
    assert output.markdown == "ok"


def test_bad_key_gets_region_hint(monkeypatch):
    with pytest.raises(LlamaParseError, match="region-specific"):
        run_engine(monkeypatch, FakeApi(auth_ok=False), page_numbers=[1, 2])


def test_failed_job_raises(monkeypatch):
    api = FakeApi(statuses=("PENDING", "FAILED"))
    with pytest.raises(LlamaParseError, match="document is encrypted"):
        run_engine(monkeypatch, api, page_numbers=[1, 2])
