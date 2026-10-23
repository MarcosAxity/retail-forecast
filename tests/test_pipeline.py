"""El orquestador y los scripts deben estar completos y ser sintácticamente válidos."""
import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _steps():
    spec = importlib.util.spec_from_file_location("run_pipeline", ROOT / "run_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.STEPS


def test_every_step_has_a_valid_script():
    ids = []
    for step_id, name, *_ in _steps():
        path = ROOT / "scripts" / f"{name}.py"
        assert path.exists(), f"falta {path}"
        ast.parse(path.read_text(encoding="utf-8"))
        assert name.startswith(step_id + "_")
        ids.append(step_id)
    assert ids == sorted(ids) and len(set(ids)) == len(ids)


def test_no_orphan_scripts():
    in_pipeline = {f"{name}.py" for _, name, *_ in _steps()}
    on_disk = {p.name for p in (ROOT / "scripts").glob("[0-9]*.py")}
    assert on_disk == in_pipeline
