import pytest

from fplopt.config import load_config
from fplopt.features.build import build_feature_table
from synthetic import make_league


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def league():
    return make_league(seed=0)


@pytest.fixture(scope="session")
def table(league, cfg):
    return build_feature_table(league, cfg)
