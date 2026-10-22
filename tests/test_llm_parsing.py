import pytest

from app.llm.base import parse_json


@pytest.mark.parametrize("text", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    'Sure! Here you go:\n{"a": 1}\nLet me know.',
    'noise {not json} then {"a": 1}',
])
def test_parse_json_tolerates_wrappers(text):
    assert parse_json(text) == {"a": 1}


def test_parse_json_raises_without_object():
    with pytest.raises(ValueError):
        parse_json("no json here")
