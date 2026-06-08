from pathlib import Path


def test_project_structure_exists():
    root = Path(__file__).resolve().parents[1]

    assert (root / "data").exists()
    assert (root / "src").exists()
    assert (root / "notebooks").exists()
    assert (root / "configs").exists()
