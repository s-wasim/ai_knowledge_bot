from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_session():
    """Return a mock SQLAlchemy session."""
    session = MagicMock()
    session.add.return_value = None
    session.flush.return_value = None
    session.commit.return_value = None
    return session


@pytest.fixture
def mock_session_factory(mock_session):
    """Return a mock session factory."""
    factory = MagicMock()
    factory.return_value = mock_session
    return factory
