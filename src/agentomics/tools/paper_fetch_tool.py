from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel
from pydantic_ai import Tool
from pydantic_ai.mcp import MCPServerStreamableHTTP

from agentomics.utils.config import Config


class RetrievedPaper(BaseModel):
    title: str
    year: int | None
    file_path: str

@dataclass(frozen=True)
class PaperSearchResult:
    paper_id: str
    title: str
    year: int | None

    @classmethod
    def from_asta(cls, paper: dict[str, Any]) -> PaperSearchResult:
        return cls(
            paper_id=paper["paperId"],
            title=paper["title"],
            year=paper.get("year"),
        )

def create_paper_fetch_tool(config: Config) -> Tool[Any]:
    async def paper_fetch(query: str) -> list[RetrievedPaper]:
        """
        Find and download open-access scientific papers relevant to a research question.
        Save available PDFs. Each result includes the paper's title,
        publication year, and local file_path.

        Args:
            query: One focused scientific topic or question to search for.
        """
        search_results = await search_asta_papers(
            query=query,
            limit=config.paper_fetch_max_results,
            timeout=config.paper_fetch_timeout,
        )
        return await download_papers(
            papers=search_results,
            destination=config.fetched_papers_dir,
            timeout=config.paper_fetch_timeout,
        )

    return Tool(paper_fetch, name="paper_fetch")

async def search_asta_papers(
    query: str, limit: int, timeout: float
) -> list[PaperSearchResult]:
    headers = {}
    # An optional Asta key enables higher rate limits.
    if api_key := os.getenv("ASTA_API_KEY"):
        headers["x-api-key"] = api_key

    asta_mcp = MCPServerStreamableHTTP(
        url="https://asta-tools.allen.ai/mcp/v1",
        headers=headers,
        timeout=timeout,
    )
    result = await asta_mcp.direct_call_tool(
        "search_papers_by_relevance",
        {
            "keyword": query,
            "fields": "title,year",
            "limit": limit,
        },
    )
    if isinstance(result, str):
        result = json.loads(result)
    return [PaperSearchResult.from_asta(paper) for paper in result]

async def download_papers(
    papers: list[PaperSearchResult], destination: Path, timeout: float
) -> list[RetrievedPaper]:
    destination.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; Agentomics/1.0; scientific paper retrieval)",
        },
    ) as client:
        downloaded_papers = []
        papers_to_download = []
        for paper in papers:
            pdf_path = destination / f"{paper.paper_id}.pdf"
            if _is_pdf_file(pdf_path):
                downloaded_papers.append(
                    RetrievedPaper(
                        title=paper.title,
                        year=paper.year,
                        file_path=str(pdf_path),
                    )
                )
                continue

            papers_to_download.append(paper)

        metadata_by_paper_id = await _get_semantic_scholar_metadata(
            client,
            [paper.paper_id for paper in papers_to_download],
        )

        for paper in papers_to_download:
            pdf_path = destination / f"{paper.paper_id}.pdf"
            candidates = await resolve_pdf_candidates(
                client,
                paper,
                metadata_by_paper_id.get(paper.paper_id, {}),
            )
            for candidate in candidates:
                try:
                    response = await client.get(candidate)
                    response.raise_for_status()
                    if not response.content.startswith(b"%PDF-"):
                        continue
                    pdf_path.write_bytes(response.content)
                except (httpx.HTTPError, OSError):
                    continue

                downloaded_papers.append(
                    RetrievedPaper(
                        title=paper.title,
                        year=paper.year,
                        file_path=str(pdf_path),
                    )
                )
                break

    return downloaded_papers

async def resolve_pdf_candidates(
    client: httpx.AsyncClient,
    paper: PaperSearchResult,
    metadata: dict[str, Any],
) -> list[str]:
    candidates: list[str] = []

    external_ids = metadata.get("externalIds") or {}
    metadata_doi = external_ids.get("DOI")
    resolved_doi = metadata_doi
    arxiv_id = external_ids.get("ArXiv") or _arxiv_id_from_doi(metadata_doi)
    pmcid = external_ids.get("PubMedCentral")
    pdf_url = (metadata.get("openAccessPdf") or {}).get("url") or None

    candidates.extend(await _unpaywall_pdf_urls(client, metadata_doi))

    if arxiv_id:
        arxiv_id = quote(arxiv_id, safe="/.")
        candidates.append(f"https://arxiv.org/pdf/{arxiv_id}")

    pmcid = pmcid or await _find_pmcid(client, metadata_doi)
    if pmcid is None:
        europe_pmc_doi, pmcid = await _find_europe_pmc_ids(client, paper.title)
        resolved_doi = metadata_doi or europe_pmc_doi
        # Europe PMC may identify a preprint with a different DOI.
        if metadata_doi and europe_pmc_doi and europe_pmc_doi != metadata_doi:
            candidates.extend(await _biorxiv_pdf_urls(client, europe_pmc_doi))

    if pmcid:
        pmcid = str(pmcid).upper()
        if not pmcid.startswith("PMC"):
            pmcid = f"PMC{pmcid}"
        candidates.extend(
            [
                f"https://europepmc.org/articles/{pmcid}?pdf=render",
                "https://europepmc.org/backend/ptpmcrender.fcgi"
                f"?accid={pmcid}&blobtype=pdf",
            ]
        )

    candidates.extend(await _biorxiv_pdf_urls(client, resolved_doi))

    if pdf_url:
        candidates.append(pdf_url)

    candidates.extend(await _crossref_pdf_urls(client, resolved_doi))

    return list(dict.fromkeys(candidates))

async def _get_semantic_scholar_metadata(
    client: httpx.AsyncClient,
    paper_ids: list[str],
) -> dict[str, dict[str, Any]]:
    if not paper_ids:
        return {}

    headers = {"Accept": "application/json"}
    if api_key := os.getenv("S2_API_KEY"):
        headers["x-api-key"] = api_key

    for attempt in range(3):
        try:
            response = await client.post(
                "https://api.semanticscholar.org/graph/v1/paper/batch",
                params={"fields": "externalIds,openAccessPdf"},
                headers=headers,
                json={"ids": paper_ids},
            )
            if response.status_code != 429:
                response.raise_for_status()
                metadata = response.json()
                break
        except (httpx.HTTPError, ValueError):
            return {}

        if attempt == 2:
            return {}
        await asyncio.sleep(3)

    return {
        paper["paperId"]: paper
        for paper in metadata
        if paper and paper.get("paperId")
    }

async def _unpaywall_pdf_urls(
    client: httpx.AsyncClient,
    doi: str | None,
) -> list[str]:
    email = os.getenv("UNPAYWALL_EMAIL")
    if not doi or not email:
        return []

    try:
        response = await client.get(
            f"https://api.unpaywall.org/v2/{quote(doi, safe='/')}",
            params={"email": email},
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError):
        return []

    locations = [data.get("best_oa_location"), *(data.get("oa_locations") or [])]
    return [
        location["url_for_pdf"]
        for location in locations
        if location and location.get("url_for_pdf")
    ]

def _arxiv_id_from_doi(doi: str | None) -> str | None:
    prefix = "10.48550/arxiv."
    if doi and doi.lower().startswith(prefix.lower()):
        return doi[len(prefix):]
    return None

async def _find_pmcid(client: httpx.AsyncClient, doi: str | None) -> str | None:
    if doi is None:
        return None

    params = {
        "ids": doi,
        "idtype": "doi",
        "format": "json",
        "tool": "agentomics",
    }

    try:
        response = await client.get(
            "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/",
            params=params,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        records = response.json().get("records") or []
    except (httpx.HTTPError, ValueError):
        return None

    return records[0].get("pmcid") if records else None

async def _find_europe_pmc_ids(
    client: httpx.AsyncClient,
    title: str,
) -> tuple[str | None, str | None]:
    try:
        response = await client.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={
                "query": f'TITLE:"{title}"',
                "format": "json",
                "resultType": "lite",
                "pageSize": 1,
            },
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        results = response.json().get("resultList", {}).get("result") or []
    except (httpx.HTTPError, ValueError):
        return None, None

    if not results:
        return None, None

    result = results[0]
    returned_title = result.get("title", "").casefold().rstrip(".")
    if returned_title != title.casefold().rstrip("."):
        return None, None

    return result.get("doi"), result.get("pmcid")

async def _biorxiv_pdf_urls(
    client: httpx.AsyncClient,
    doi: str | None,
) -> list[str]:
    if doi is None or not doi.lower().startswith("10.1101/"):
        return []

    encoded_doi = quote(doi, safe="/")
    for server in ("biorxiv", "medrxiv"):
        try:
            response = await client.get(
                f"https://api.biorxiv.org/details/{server}/{encoded_doi}/na/json",
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            if response.json().get("collection"):
                return [
                    f"https://www.{server}.org/content/{encoded_doi}.full.pdf",
                ]
        except (httpx.HTTPError, ValueError):
            continue

    return []

async def _crossref_pdf_urls(
    client: httpx.AsyncClient,
    doi: str | None,
) -> list[str]:
    if not doi:
        return []

    try:
        response = await client.get(
            f"https://api.crossref.org/works/{quote(doi, safe='')}",
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        metadata = response.json()["message"]
    except (httpx.HTTPError, ValueError, KeyError):
        return []

    return [
        link["URL"]
        for link in metadata.get("link", [])
        if link.get("URL") and link.get("content-type") == "application/pdf"
    ]

def _is_pdf_file(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as pdf_file:
        return pdf_file.read(5) == b"%PDF-"
