import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "run_paper_suite.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("paper_suite_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_is_valid_and_case_ids_are_unique():
    runner = load_runner()
    manifest = runner.load_manifest(ROOT / "paper_suite.toml")
    identifiers = [case["id"] for case in manifest["case"]]
    assert len(identifiers) == len(set(identifiers))
    assert {"rising_bubble", "wreckhouse", "appendix_temporal"} <= set(
        identifiers
    )


def test_every_command_uses_existing_python_entry_point(tmp_path):
    runner = load_runner()
    manifest = runner.load_manifest(ROOT / "paper_suite.toml")
    substitutions = runner.context(tmp_path)
    for case in manifest["case"]:
        for stage in case["stage"]:
            command = runner.expand(stage["command"], substitutions)
            assert Path(command[0]).name == "python3"
            assert (ROOT / command[1]).is_file(), (case["id"], stage["name"])


def test_stage_completion_requires_all_declared_outputs(tmp_path):
    runner = load_runner()
    substitutions = {"output_root": str(tmp_path)}
    stage = {"outputs": ["{output_root}/a", "{output_root}/b"]}
    assert not runner.stage_complete(stage, substitutions)
    (tmp_path / "a").touch()
    assert not runner.stage_complete(stage, substitutions)
    (tmp_path / "b").touch()
    assert runner.stage_complete(stage, substitutions)
