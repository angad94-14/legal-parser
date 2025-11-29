"""Test that basic setup is working"""
import pytest
from src.utils.config import settings


def test_environment_loaded():
    """Test that environment variables are loaded"""
    assert settings.environment in ["development", "staging", "production"]


def test_api_key_exists():
    """Test that OpenAI API key is configured"""
    assert settings.openai_api_key is not None
    assert len(settings.openai_api_key) > 0


def test_imports():
    """Test that key packages can be imported"""
    import fastapi
    import openai
    import langchain
    import pdfplumber

    assert fastapi is not None
    assert openai is not None
    assert langchain is not None
    assert pdfplumber is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])