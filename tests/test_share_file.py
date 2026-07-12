from types import SimpleNamespace

import pytest

from agent import AIAgent


def make_agent(tmp_path):
    agent = AIAgent.__new__(AIAgent)
    agent.config = SimpleNamespace(
        SHARE_DIR=tmp_path, SHARE_URL_BASE="https://x.test/docs/shared"
    )
    return agent


def test_share_file_writes_and_returns_url(tmp_path):
    agent = make_agent(tmp_path)
    url = agent._share_file("My Report!.html", "<h1>hi</h1>")
    name = url.rsplit("/", 1)[1]
    assert url.startswith("https://x.test/docs/shared/")
    assert name.startswith("My-Report-") and name.endswith(".html")
    assert (tmp_path / name).read_text() == "<h1>hi</h1>"


def test_share_file_rejects_bad_extension(tmp_path):
    agent = make_agent(tmp_path)
    with pytest.raises(ValueError):
        agent._share_file("script.sh", "echo hi")


def test_share_file_neutralizes_path_traversal(tmp_path):
    agent = make_agent(tmp_path)
    url = agent._share_file("../../etc/passwd.txt", "x")
    name = url.rsplit("/", 1)[1]
    assert (tmp_path / name).exists()
    assert "etc" not in str(tmp_path / name).replace(str(tmp_path), "")
