# (c) Copyright Riverlane 2020-2025.
import pytest
import stim


@pytest.fixture(scope="function")
def empty_circuit():
    return stim.Circuit()
