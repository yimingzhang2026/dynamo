# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check ranking fairness, evidence validation and the frozen combined dataset."""

import hashlib
import json
from pathlib import Path

import pytest
from build_leaderboards import (
    check_report,
    descending_ranks,
    export,
    load_diagnostics,
    load_static_latency,
    make_report,
    ranking,
    render,
    validate_record,
    validate_static_latency,
    write_report,
)
from compact_results import compact
from test_evidence import completed_record, static_record

ROOT = Path(__file__).parent


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_report_check_reproduces_saved_bytes_and_rejects_modified_output(tmp_path):
    report = make_report(ROOT / "inputs")
    write_report(report, tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    check_report({**report, "generated_at": "different rebuild time"}, tmp_path)
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    (tmp_path / "index.html").write_text("modified")
    with pytest.raises(ValueError, match="index.html"):
        check_report(report, tmp_path)


def row(value, status="completed"):
    return {
        "status": status,
        "metrics": {"goodput_per_gpu": value, "good_rate": 0.75, "gpu_hours": 1},
    }


def test_each_case_has_equal_weight_despite_different_score_scales():
    rows = {
        "c1-mpc": row(10000),
        "c1-keda": row(1),
        "c2-mpc": row(0.01),
        "c2-keda": row(0.02),
    }
    result = ranking(rows, ["c1", "c2"], ["mpc", "keda"])
    assert {r["mean_rank"] for r in result["rows"]} == {1.5}
    assert {r["rank"] for r in result["rows"]} == {1.5}
    assert {r["wins"] for r in result["rows"]} == {1}


def test_missing_or_failed_case_is_excluded_for_every_controller():
    rows = {
        "c1-mpc": row(2),
        "c1-keda": row(1),
        "c2-mpc": row(900),
        "c2-keda": row(800, "failed"),
        "c3-mpc": row(1000),
    }
    result = ranking(rows, ["c1", "c2", "c3"], ["mpc", "keda"])
    assert result["shared_cases"] == ["c1"]
    assert result["excluded_cases"] == ["c2", "c3"]
    assert [(r["planner"], r["mean_rank"], r["completed"]) for r in result["rows"]] == [
        ("mpc", 1, 3),
        ("keda", 2, 1),
    ]


def test_zero_scores_are_real_results_and_ties_split_rank_and_win_credit():
    assert descending_ranks({"mpc": 4, "keda": 4, "reactive": 1, "planner": 0}) == {
        "mpc": 1.5,
        "keda": 1.5,
        "reactive": 3,
        "planner": 4,
    }
    result = ranking({"c-mpc": row(0), "c-keda": row(0)}, ["c"], ["mpc", "keda"])
    assert [r["mean_rank"] for r in result["rows"]] == [1.5, 1.5]
    assert [r["wins"] for r in result["rows"]] == [0.5, 0.5]


def test_no_shared_cases_is_unranked():
    result = ranking({"c-mpc": row(10)}, ["c"], ["mpc", "keda"])
    assert not result["shared_cases"]
    assert all(r["mean_rank"] is None and r["rank"] is None for r in result["rows"])


@pytest.fixture
def valid_inputs():
    return completed_record()


def test_a_complete_original_result_validates(valid_inputs):
    assert validate_record(*valid_inputs)["status"] == "completed"


@pytest.mark.parametrize(
    "field,value",
    [
        ("goodput_per_gpu", float("nan")),
        ("goodput_per_gpu", -1),
        ("goodput_per_gpu", 9999),
        ("good_rate", 1.01),
        ("gpu_hours", 0),
    ],
)
def test_invalid_metrics_are_rejected(valid_inputs, field, value):
    valid_inputs[0]["report"]["results"][0]["metrics"][field] = value
    assert validate_record(*valid_inputs)["status"] == "invalid"


@pytest.mark.parametrize(
    "change", ["trace", "slo", "requests", "window", "truncated", "topology", "workers"]
)
def test_incomparable_runs_cannot_enter_ranks(valid_inputs, change):
    result = valid_inputs[0]["report"]["results"][0]
    if change == "trace":
        result["evaluation"]["trace"] = {"sha256": "wrong"}
    elif change == "slo":
        result["evaluation"]["sla"]["ttft_ms"] += 1
    elif change == "requests":
        result["raw_metrics"]["attempted_requests"] -= 1
    elif change == "window":
        result["raw_metrics"]["measurement"]["end_s"] -= 1
    elif change == "truncated":
        result["evaluation"]["max_requests"] = 100
    elif change == "topology":
        result["runtime"]["topology"] = "incorrect"
    elif change == "workers":
        result["runtime"]["worker_limits"]["max_total"] += 1
    assert validate_record(*valid_inputs)["status"] == "invalid"


def test_failed_attempt_does_not_leak_partial_scores(valid_inputs):
    record = valid_inputs[0]
    record["saved_status"] = "failed"
    record["report"]["results"][0]["status"] = "failed"
    result = validate_record(*valid_inputs)
    assert result["status"] == "failed"
    assert result["metrics"] == {}


def test_streaming_extraction_keeps_scoring_and_hashes_entire_file(tmp_path):
    data = {
        "summary": {"status": "ok"},
        "results": [
            {
                "metrics": {"goodput_per_gpu": 0.123},
                "raw_metrics": {"attempted_requests": 7},
                "runtime": {"topology": "agg", "timeline": [1, 2, 3]},
                "evaluation": {
                    "trace": {"sha256": "tracehash"},
                    "arrival_timestamps": [1, 2, 3],
                },
                "request_history": ["ignored"] * 10000,
            },
            {"metrics": {"goodput_per_gpu": 4}},
        ],
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(data))
    result, fingerprint = compact(path)
    assert fingerprint == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["results"] == [
        {
            "metrics": {"goodput_per_gpu": 0.123},
            "raw_metrics": {"attempted_requests": 7},
            "runtime": {"topology": "agg"},
            "evaluation": {"trace": {"sha256": "tracehash"}},
        },
        {"metrics": {"goodput_per_gpu": 4}},
    ]


def test_streaming_extraction_rejects_truncated_json(tmp_path):
    from ijson import IncompleteJSONError

    path = tmp_path / "broken.json"
    path.write_text('{"results":[{"metrics":{"goodput_per_gpu":1}},')
    with pytest.raises(IncompleteJSONError):
        compact(path)


def test_latest_evidence_covers_all_cases_without_scoring_pending_replacements(
    tmp_path,
):
    report = make_report(ROOT / "inputs")
    assert sum(report["counts"].values()) == report["expected"] == 360
    assert sum(report["evidence_counts"].values()) == report["evidence_expected"] == 360
    assert report["scheduled_runs"] == 300
    assert report["evidence_scheduled_runs"] == 300
    assert report["knowline_runs"] == 60
    assert report["static_baselines"] == sum(
        r["status"] == "completed" for r in report["static_latency"]["rows"].values()
    )
    export(report, tmp_path)
    validation = json.loads((tmp_path / "validation.json").read_text())
    assert validation["knowline_results_validated"] == sum(
        case["rows"]["knowline31b"]["status"] == "completed"
        for page in report["pages"].values()
        for case in page["cases"]
    )
    assert report["experiment_setup"]["budget_mismatches"] == {}
    for page in report["pages"].values():
        for case in page["cases"]:
            assert case["rows"]["static"]["status"] in {"completed", "pending"}
            for planner, saved in case["rows"].items():
                if saved["status"] == "pending":
                    assert saved["metrics"] == {}
                    assert saved["source_sha256"] is None
                elif planner != "static":
                    assert (
                        saved["runtime"]["gpu_budget"]
                        == report["experiment_setup"]["rows"][case["case_id"]][
                            "gpu_budget"
                        ]
                    )
            assert "jev" not in case["rows"]
        for view in page["views"].values():
            assert len(view["core"]["rows"]) == 6
            assert view["expected"] == len(view["case_ids"]) * 6
            expected = {
                c["case_id"]
                for c in page["cases"]
                if c["case_id"] in view["case_ids"]
                and all(r["status"] == "completed" for r in c["rows"].values())
            }
            assert set(view["core"]["shared_cases"]) == expected
            assert sum(r["wins"] for r in view["core"]["rows"]) == pytest.approx(
                len(expected)
            )
            if expected:
                assert sum(
                    r["mean_rank"] for r in view["core"]["rows"]
                ) == pytest.approx(21)


def test_no_generated_ranking_ever_receives_jev(monkeypatch):
    import build_leaderboards

    calls = []

    def checked_ranking(rows, case_ids, planners):
        assert "jev" not in planners
        calls.append(tuple(planners))
        return ranking(rows, case_ids, planners)

    monkeypatch.setattr(build_leaderboards, "ranking", checked_ranking)
    report = make_report(ROOT / "inputs")
    assert len(calls) == 6 and "jev" not in report["planners"]
    assert "comparison" not in report


def test_timeline_import_rejects_evidence_from_another_result(tmp_path):
    from publish_latest import refresh_diagnostics

    for path in (ROOT / "inputs").glob("*.json"):
        (tmp_path / path.name).write_bytes(path.read_bytes())
    report = make_report(tmp_path)
    evidence = json.loads((tmp_path / "diagnostic-evidence.json").read_text())["runs"]
    key = next(iter(evidence))
    supplied = tmp_path / "supplied"
    supplied.mkdir()
    (supplied / "timeline.json").write_text(
        json.dumps({**evidence[key], "job_id": key, "source_sha256": "wrong-result"})
    )
    with pytest.raises(ValueError, match="Timeline does not match current result"):
        refresh_diagnostics(tmp_path, report, [supplied])


@pytest.mark.parametrize("change", ["duplicate", "missing", "unexpected"])
def test_final_results_require_exactly_one_success_per_job(tmp_path, change):
    for path in (ROOT / "inputs").glob("*.json"):
        (tmp_path / path.name).write_bytes(path.read_bytes())
    path = tmp_path / "results.json"
    evidence = json.loads(path.read_text())
    if change == "duplicate":
        evidence["records"][-1] = evidence["records"][0]
    elif change == "missing":
        evidence["records"].pop()
    else:
        evidence["records"][-1]["job_id"] = "unknown"
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="exactly one final result"):
        make_report(tmp_path)


@pytest.fixture
def knowline_inputs():
    return completed_record("knowline31b")


@pytest.mark.parametrize(
    "change", ["provider", "model", "revision", "errors", "calls", "label"]
)
def test_knowline_requires_verified_independent_decision_model(knowline_inputs, change):
    record, case, job = knowline_inputs
    row = record["report"]["results"][0]
    if change == "provider":
        row["runtime"]["autoscaler_config"]["provider"] = "jev"
    elif change == "model":
        row["runtime"]["autoscaler_config"]["model"] = "another-model"
    elif change == "revision":
        record["decision_model"]["revision"] = "different-weights"
    elif change == "errors":
        record["decision_summary"]["errors"] = 1
    elif change == "calls":
        record["decision_summary"]["calls"] += 1
    elif change == "label":
        row["autoscaler"] = "JEV"
    assert validate_record(record, case, job)["status"] == "invalid"


def test_knowline_participates_in_case_ranking():
    values = {"c-static": row(2), "c-knowline31b": row(3)}
    result = ranking(values, ["c"], ["static", "knowline31b"])
    assert [(r["planner"], r["rank"]) for r in result["rows"]] == [
        ("knowline31b", 1),
        ("static", 2),
    ]


@pytest.mark.parametrize("replayed", [0, 7])
def test_knowline_singleton_forced_decisions_and_continuation_validate(replayed):
    record, case, job = completed_record(
        "knowline31b", "D-LC-S04", policy={"resources": {"gpu_budget_multiplier": 1}}
    )
    assert job["resources"]["feasible_fleet_count"] == 1
    record["decision_summary"].update(calls=0, forced_decisions=10)
    record["report"]["results"][0]["runtime"]["jev"].update(
        calls=0, forced_decisions=10 - replayed, replayed_decisions=replayed
    )
    assert validate_record(record, case, job)["status"] == "completed"
    record["decision_summary"]["forced_decisions"] += 1
    assert validate_record(record, case, job)["status"] == "invalid"


def test_knowline_multiple_feasible_fleets_require_model_calls():
    record, case, job = completed_record("knowline31b", "A-HT-L01")
    assert job["resources"]["feasible_fleet_count"] > 1
    record["decision_summary"].update(calls=0, forced_decisions=10)
    record["report"]["results"][0]["runtime"]["jev"].update(
        calls=0, forced_decisions=10
    )
    assert validate_record(record, case, job)["status"] == "invalid"


def test_static_participates_in_case_ranks_and_changes_the_overall_winner():
    rows = {
        "c1-mpc": row(10),
        "c1-static": row(20),
        "c2-mpc": row(9),
        "c2-static": row(12),
    }
    result = ranking(rows, ["c1", "c2"], ["mpc", "static"])
    assert [
        (r["planner"], r["rank"], r["mean_rank"], r["wins"]) for r in result["rows"]
    ] == [("static", 1, 1, 2), ("mpc", 2, 2, 0)]


def test_embedded_data_cannot_close_script_tag():
    value = "</script><script>alert(1)</script>&"
    html = render({"dataset": value}, "<script>__REPORT_DATA__</script>", "agg")
    assert html.count("</script>") == 1
    assert (
        json.loads(html[len("<script>") : -len("</script>")])["report"]["dataset"]
        == value
    )


def test_current_static_results_supply_every_tab(tmp_path):
    report = make_report(ROOT / "inputs")
    reruns = report["static_latency"]["rows"]
    assert len(reruns) == 60
    for page in report["pages"].values():
        for case in page["cases"]:
            current = case["rows"]["static"]
            assert current == reruns[case["case_id"]]
            assert current["trace_sha256"] == case["trace_sha256"]
            if current["status"] == "completed":
                assert current["completed_requests"] == case["requests"]
                assert current["metrics"]["mean_ttft_ms"] >= 0
            else:
                assert current["metrics"] == {}
    export(report, tmp_path)
    import csv

    with (tmp_path / "static-latency-runs.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 60
    for row in rows:
        source = reruns[row["case_id"]]
        assert row["status"] == source["status"]
        if source["status"] == "completed":
            assert float(row["good_rate"]) == source["metrics"]["good_rate"]
        else:
            assert row["good_rate"] == ""
    proof = json.loads((tmp_path / "validation.json").read_text())["static_latency"]
    assert (
        proof["validated_runs"] == report["static_baselines"]
        and proof["input_manifest"]["sha256"]
    )


@pytest.mark.parametrize(
    "change",
    [
        "trace",
        "slo",
        "requests",
        "window",
        "latency",
        "allocation",
        "scaling",
        "receipt",
        "score",
    ],
)
def test_static_rerun_rejects_incompatible_evidence(change):
    case = json.loads((ROOT / "inputs/catalog.json").read_text())["cases"][0]
    entry = static_record(case)
    saved = entry["result"]
    if change == "trace":
        saved["inputs"]["trace_sha256"] = "wrong"
    elif change == "slo":
        entry["config"]["slo_profiles"][0]["ttft_ms"] += 1
    elif change == "requests":
        saved["raw_metrics"]["completed_requests"] -= 1
    elif change == "window":
        saved["raw_metrics"]["measurement"]["end_s"] -= 1
    elif change == "latency":
        saved["metrics"]["mean_itl_ms"] = float("nan")
    elif change == "allocation":
        saved["metrics"]["gpu_hours"] += 1
    elif change == "scaling":
        saved["raw_metrics"]["scale_events"] = 1
    elif change == "receipt":
        entry["source_sha256"] = "wrong"
    elif change == "score":
        saved["metrics"]["good_rate"] = 0.123
    with pytest.raises(ValueError, match="Static rerun"):
        validate_static_latency(entry, case)


@pytest.mark.parametrize("change", ["duplicate", "missing"])
def test_static_reruns_require_exact_coverage(tmp_path, change):
    evidence = json.loads((ROOT / "inputs/static-latency.json").read_text())
    if change == "duplicate":
        evidence["records"][-1] = evidence["records"][0]
    else:
        evidence["records"].pop()
    (tmp_path / "static-latency.json").write_text(json.dumps(evidence))
    cases = {
        c["case_id"]: c
        for c in json.loads((ROOT / "inputs/catalog.json").read_text())["cases"]
    }
    with pytest.raises(ValueError, match="exactly one Static latency rerun"):
        load_static_latency(tmp_path, cases)


def test_diagnoses_cover_all_red_runs_with_verified_sources(tmp_path):
    report = make_report(ROOT / "inputs")
    diagnoses = report["diagnostics"]["rows"]
    assert diagnoses
    assert all(d["source_sha256"] for d in diagnoses.values())
    assert all(
        d["basis"] in {"timeline", "summary_inference"} for d in diagnoses.values()
    )
    completed = {
        row["job_id"]
        for page in report["pages"].values()
        for case in page["cases"]
        for row in case["rows"].values()
        if row["status"] == "completed"
    }
    assert set(diagnoses) <= completed
    export(report, tmp_path)
    proof = json.loads((tmp_path / "validation.json").read_text())["diagnostics"]
    assert proof["validated_red_results"] == len(diagnoses)
    assert len(proof["input_manifest"]) == 2


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "source",
        "trace",
        "empty",
        "basis",
        "policy",
        "evidence_source",
        "evidence_metric",
    ],
)
def test_diagnoses_reject_missing_or_stale_evidence(tmp_path, change):
    report = json.loads((ROOT / "leaderboards.json").read_text())
    data = json.loads((ROOT / "inputs/red-diagnoses.json").read_text())
    evidence = json.loads((ROOT / "inputs/diagnostic-evidence.json").read_text())
    key = next(iter(data["rows"]))
    diagnosis = data["rows"][key]
    if change == "missing":
        del data["rows"][key]
    elif change == "extra":
        data["rows"]["A-HT-S01-keda"] = diagnosis.copy()
    elif change == "source":
        diagnosis["source_sha256"] = "wrong"
    elif change == "trace":
        diagnosis["trace_sha256"] = "wrong"
    elif change == "empty":
        diagnosis["summary"] = " "
    elif change == "basis":
        diagnosis["basis"] = "invalid-basis"
    elif change == "policy":
        data["policy"]["mean_severe_ratio"] = 3
    elif change == "evidence_source":
        evidence["runs"][key]["source_sha256"] = "wrong"
    elif change == "evidence_metric":
        evidence["runs"][key]["metrics"]["mean_itl_ms"] += 1
    (tmp_path / "red-diagnoses.json").write_text(json.dumps(data))
    (tmp_path / "diagnostic-evidence.json").write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="Diagnos"):
        load_diagnostics(tmp_path, report)
