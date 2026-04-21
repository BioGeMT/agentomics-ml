import sys
import unittest
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models.test import TestModel

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


class SimpleOutput(BaseModel):
    value: str


class TestStackedOutputValidators(unittest.TestCase):
    def _make_agent_with_two_validators(self, called: list[str], first_raises: bool = False, second_raises: bool = False):
        agent = Agent(
            model=TestModel(custom_output_args={"value": "hello"}),
            output_type=SimpleOutput,
            retries=0,
        )

        @agent.output_validator
        def validator_first(result: SimpleOutput) -> SimpleOutput:
            called.append("first")
            if first_raises:
                raise ModelRetry("first validator rejected")
            return result

        @agent.output_validator
        def validator_second(result: SimpleOutput) -> SimpleOutput:
            called.append("second")
            if second_raises:
                raise ModelRetry("second validator rejected")
            return result

        return agent

    def test_both_validators_are_called(self):
        called: list[str] = []
        agent = self._make_agent_with_two_validators(called)
        result = agent.run_sync("test prompt")
        self.assertEqual(result.output.value, "hello")
        self.assertIn("first", called)
        self.assertIn("second", called)

    def test_validators_called_in_registration_order(self):
        called: list[str] = []
        agent = self._make_agent_with_two_validators(called)
        agent.run_sync("test prompt")
        self.assertEqual(called, ["first", "second"])

    def test_first_validator_raising_prevents_second_from_running(self):
        called: list[str] = []
        agent = self._make_agent_with_two_validators(called, first_raises=True)
        with self.assertRaises(Exception):
            agent.run_sync("test prompt")
        self.assertIn("first", called)
        self.assertNotIn("second", called)

    def test_second_validator_raising_does_not_affect_first(self):
        called: list[str] = []
        agent = self._make_agent_with_two_validators(called, second_raises=True)
        with self.assertRaises(Exception):
            agent.run_sync("test prompt")
        self.assertIn("first", called)
        self.assertIn("second", called)


if __name__ == "__main__":
    unittest.main()
