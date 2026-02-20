import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
from openai import AsyncOpenAI, InternalServerError

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def make_500_response() -> httpx.Response:
    body = json.dumps({
        "error": {
            "message": "error parsing tool call: raw='...'",
            "type": "api_error",
            "param": None,
            "code": None
        }
    }).encode()
    return httpx.Response(
        status_code=500,
        headers={"content-type": "application/json"},
        content=body,
    )


class TestRetryTransport(unittest.IsolatedAsyncioTestCase):

    async def test_500_retries_3_times_and_raises(self):
        """500 on every attempt: SDK should retry 3 times total then raise InternalServerError."""
        inner_transport = MagicMock(spec=httpx.AsyncBaseTransport)
        inner_transport.handle_async_request = AsyncMock(return_value=make_500_response())

        client = AsyncOpenAI(
            base_url="http://fake-host",
            api_key="test",
            http_client=httpx.AsyncClient(transport=inner_transport, timeout=5),
        )

        with self.assertRaises(InternalServerError):
            await client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "hello"}]
            )

        self.assertEqual(inner_transport.handle_async_request.call_count, 3)


if __name__ == "__main__":
    unittest.main()