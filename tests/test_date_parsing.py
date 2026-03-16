import pytest
from datetime import datetime, timedelta
from agent import AIAgent
from config import Config
from storage import Storage

# Mock dependencies
class MockConfig:
    ANTHROPIC_API_KEY = "fake_key"
    CLAUDE_MAX_RETRIES = 1
    CLAUDE_BASE_DELAY = 0
    CLAUDE_MAX_TOKENS = 100
    SONNET_MODEL = "claude-sonnet-4-6"
    HAIKU_MODEL = "claude-haiku-4-5-20251001"
    CONTEXT_TOKEN_BUDGET = 12000
    NOTION_TOKEN = ""

class MockStorage:
    pass

@pytest.fixture
def agent():
    return AIAgent(MockConfig(), MockStorage())

def test_parse_when_relative(agent):
    """Test relative time parsing"""
    dt = agent._parse_when("in 10 minutes")
    assert dt is not None
    # Allow small delta for execution time
    expected = datetime.now() + timedelta(minutes=10)
    diff = abs((dt - expected).total_seconds())
    assert diff < 5

def test_parse_when_absolute_today(agent):
    """Test absolute time parsing for today"""
    # If we say a time that is in the future today
    future_hour = (datetime.now().hour + 2) % 24
    if future_hour < datetime.now().hour: # Wrapped around to tomorrow
        return # Skip this edge case for simple test
        
    dt = agent._parse_when(f"at {future_hour}:00")
    assert dt is not None
    assert dt.hour == future_hour
    assert dt.day == datetime.now().day

def test_parse_when_tomorrow(agent):
    """Test tomorrow parsing"""
    dt = agent._parse_when("tomorrow at 9am")
    assert dt is not None
    assert dt.day == (datetime.now() + timedelta(days=1)).day
    assert dt.hour == 9

def test_parse_when_complex(agent):
    """Test more complex natural language"""
    dt = agent._parse_when("friday at noon")
    assert dt is not None
    assert dt.hour == 12
    # Weekday check (Friday is 4)
    assert dt.weekday() == 4
    assert dt > datetime.now()
