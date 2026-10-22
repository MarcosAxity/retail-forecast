import pytest

from src.data.load import load_calendar, load_stores, load_transactions
from src.data.preprocess import build_panel


@pytest.fixture(scope="session")
def panel():
    return build_panel(load_transactions(), load_stores(), load_calendar())
