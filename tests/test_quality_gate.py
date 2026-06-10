from __future__ import annotations

from openfreqbench.quality import _release_file_checks


def test_release_file_checks_detect_open_software_artifacts(tmp_path) -> None:
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text("name: CI\n", encoding="utf-8")
    (tmp_path / ".github" / "ISSUE_TEMPLATE").mkdir(parents=True)
    (tmp_path / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml").write_text(
        "name: Bug report\n",
        encoding="utf-8",
    )
    (tmp_path / "paper").mkdir()
    (tmp_path / "paper" / "paper.md").write_text("# Paper\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "AI_USAGE_DISCLOSURE.md").write_text("# AI\n", encoding="utf-8")
    (tmp_path / ".zenodo.json").write_text("{}\n", encoding="utf-8")

    checks = _release_file_checks(tmp_path)

    assert checks == {
        "ci_workflow_present": True,
        "issue_templates_present": True,
        "software_paper_present": True,
        "ai_disclosure_present": True,
        "zenodo_metadata_present": True,
    }


def test_release_file_checks_fail_closed_when_artifacts_are_missing(tmp_path) -> None:
    checks = _release_file_checks(tmp_path)

    assert checks == {
        "ci_workflow_present": False,
        "issue_templates_present": False,
        "software_paper_present": False,
        "ai_disclosure_present": False,
        "zenodo_metadata_present": False,
    }
