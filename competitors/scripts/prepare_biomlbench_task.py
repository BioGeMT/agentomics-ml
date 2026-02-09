#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

import aiohttp
import fsspec.implementations.http as fsspec_http
from biomlbench.cli import main as biomlbench_main


def _patch_fsspec_http_client() -> None:
    original_init = fsspec_http.HTTPFileSystem.__init__

    def patched_init(self, *args, **kwargs):
        client_kwargs = dict(kwargs.get("client_kwargs") or {})
        client_kwargs.setdefault("trust_env", True)
        client_kwargs.setdefault(
            "timeout",
            aiohttp.ClientTimeout(total=900, connect=120, sock_read=900),
        )
        kwargs["client_kwargs"] = client_kwargs
        return original_init(self, *args, **kwargs)

    fsspec_http.HTTPFileSystem.__init__ = patched_init


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    _patch_fsspec_http_client()

    sys.argv = [
        "biomlbench",
        "prepare",
        "-t",
        args.task_id,
        "--data-dir",
        args.data_dir,
    ]
    biomlbench_main()


if __name__ == "__main__":
    main()
