"""Page-weighted batch metrics; document boundaries stay intact for parsing/judging."""

from collections import defaultdict

from .catalog import get_model
from .judge import RUBRIC_VERSION
from .models import Experiment, ModelRun
from .schemas import ModelAggregateOut


def combined_status(statuses: list[str]) -> str:
    if all(s == "queued" for s in statuses):
        return "queued"
    if any(s in ("queued", "loading", "running") for s in statuses):
        return "running"
    return "failed" if all(s == "failed" for s in statuses) else "completed"


def aggregate_models(experiments: list[Experiment]) -> list[ModelAggregateOut]:
    grouped: dict[str, list[tuple[Experiment, ModelRun]]] = defaultdict(list)
    for experiment in experiments:
        for run in experiment.runs:
            grouped[run.model_id].append((experiment, run))
    results = []
    for model_id, entries in grouped.items():
        total = completed = failed = timed = 0
        seconds = 0.0
        verdicts = {"excellent": 0, "pass": 0, "fail": 0}
        for experiment, run in entries:
            selected = set(experiment.pages)
            total += len(selected)
            pages = {p["page_number"]: p for p in run.page_results or []}
            successful = {n for n in selected if n in pages and not pages[n].get("error")}
            completed += len(successful)
            if run.status in ("completed", "failed"):
                failed += len(selected - successful)
            # The timer covers a document's selected pages, including any page-level
            # failures. Weight by processed pages, never by the number of documents.
            processed = len(selected & pages.keys())
            if run.status == "completed" and run.parse_seconds is not None and processed:
                seconds += run.parse_seconds
                timed += processed
            judgment = run.judgment
            if judgment and judgment.rubric_version == RUBRIC_VERSION:
                judged = {p["page_number"]: p for p in judgment.page_results or []}
                for n in successful & judged.keys():
                    verdict = judged[n].get("verdict")
                    if verdict in verdicts:
                        verdicts[verdict] += 1
        judged_count = sum(verdicts.values())
        model = get_model(model_id)
        results.append(ModelAggregateOut(
            model_id=model_id,
            model_name=model.name if model else model_id,
            kind=model.kind if model else "open",
            status=combined_status([r.status for _, r in entries]),
            pages_total=total,
            pages_completed=completed,
            pages_failed=failed,
            timed_pages=timed,
            parse_seconds=round(seconds, 3) if timed else None,
            seconds_per_page=round(seconds / timed, 3) if timed else None,
            judged_pages=judged_count,
            excellent=verdicts["excellent"], passed=verdicts["pass"], failed=verdicts["fail"],
            pass_rate=round(100 * (verdicts["excellent"] + verdicts["pass"]) / judged_count, 1)
            if judged_count else None,
        ))
    return results
