from recommender.config import PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR


def test_project_paths_are_rooted_in_repository() -> None:
    """The application must resolve paths independently of the shell directory."""
    assert (PROJECT_ROOT / "README.md").is_file()
    assert RAW_DATA_DIR == PROJECT_ROOT / "data" / "raw"
    assert PROCESSED_DATA_DIR == PROJECT_ROOT / "data" / "processed"
