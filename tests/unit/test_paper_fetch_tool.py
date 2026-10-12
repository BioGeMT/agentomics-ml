import asyncio
from io import BytesIO

import httpx
import pytest
from pypdf import PdfWriter

from agentomics.tools import paper_fetch_tool
from agentomics.tools.paper_fetch_tool import PaperSearchResult, RetrievedPaper, download_papers


@pytest.mark.parametrize(
    "external_ids,pdf_url,resolver_url,resolver_response",
    [
        pytest.param(
            {"ArXiv": "2501.12948"}, "https://arxiv.org/pdf/2501.12948",
            None, None, id="arxiv",
        ),
        pytest.param(
            {"PubMedCentral": "123456"},
            "https://europepmc.org/articles/PMC123456?pdf=render",
            None, None, id="europe-pmc",
        ),
        pytest.param(
            {"DOI": "10.1101/2024.01.01.123456"},
            "https://www.biorxiv.org/content/10.1101/2024.01.01.123456.full.pdf",
            "https://api.biorxiv.org/details/biorxiv/10.1101/2024.01.01.123456/na/json",
            {"collection": [{"doi": "10.1101/2024.01.01.123456"}]}, id="biorxiv",
        ),
        pytest.param(
            {"DOI": "10.1101/2024.01.01.123456"},
            "https://www.medrxiv.org/content/10.1101/2024.01.01.123456.full.pdf",
            "https://api.biorxiv.org/details/medrxiv/10.1101/2024.01.01.123456/na/json",
            {"collection": [{"doi": "10.1101/2024.01.01.123456"}]}, id="medrxiv",
        ),
        pytest.param(
            {}, "https://publisher.example/semantic-scholar.pdf",
            None, None, id="semantic-scholar",
        ),
        pytest.param(
            {"DOI": "10.1234/paper"}, "https://repository.example/unpaywall.pdf",
            "https://api.unpaywall.org/v2/10.1234/paper",
            {"best_oa_location": {"url_for_pdf": "https://repository.example/unpaywall.pdf"}},
            id="unpaywall",
        ),
        pytest.param(
            {"DOI": "10.1234/paper"}, "https://publisher.example/crossref.pdf",
            "https://api.crossref.org/works/10.1234%2Fpaper",
            {"message": {"link": [{
                "URL": "https://publisher.example/crossref.pdf",
                "content-type": "application/pdf",
            }]}}, id="crossref",
        ),
    ],
)
def test_download_papers_saves_pdf_from_resource(
    external_ids, pdf_url, resolver_url, resolver_response, tmp_path, monkeypatch,
):
    monkeypatch.delenv("S2_API_KEY", raising=False)
    monkeypatch.setenv("UNPAYWALL_EMAIL", "researcher@example.org")
    paper = PaperSearchResult(paper_id="paper-id", title="Sequence models", year=2025)
    metadata = {"paperId": paper.paper_id, "externalIds": external_ids}
    if resolver_url is None and not external_ids:
        metadata["openAccessPdf"] = {"url": pdf_url}

    pdf = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(pdf)
    pdf_bytes = pdf.getvalue()

    def respond(request):
        if request.method == "POST" and request.url.path == "/graph/v1/paper/batch":
            return httpx.Response(200, json=[metadata])
        if str(request.url) == pdf_url:
            return httpx.Response(200, content=pdf_bytes)
        if str(request.url.copy_with(query=None)) == resolver_url:
            return httpx.Response(200, json=resolver_response)
        return httpx.Response(404)

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        paper_fetch_tool.httpx, "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs),
    )

    destination = tmp_path / "papers"
    result = asyncio.run(download_papers([paper], destination, timeout=1))

    saved_pdf = destination / "paper-id.pdf"
    assert result == [RetrievedPaper(title=paper.title, year=paper.year, file_path=str(saved_pdf))]
    assert saved_pdf.read_bytes() == pdf_bytes


@pytest.mark.parametrize(
    "paper,pdf_url,status,body",
    [
        pytest.param(
            PaperSearchResult(
                paper_id="f36f5e1578017e53df01a95ecc002a44906de433",
                title="miRBench: novel benchmark datasets for microRNA binding site prediction",
                year=2025,
            ),
            "https://academic.oup.com/bioinformatics/article-pdf/41/Supplement_1/i542/63745693/btaf233.pdf",
            403, b"Forbidden", id="oup-forbidden",
        ),
        pytest.param(
            PaperSearchResult(
                paper_id="2c3c17026138fee75aa0bf662dd25bbab601899b",
                title="GraphTar: applying word2vec and graph neural networks to miRNA target prediction",
                year=2023,
            ),
            "https://bmcbioinformatics.biomedcentral.com/counter/pdf/10.1186/s12859-023-05564-x",
            200, b"<html>Verify you are human</html>", id="bmc-html-response",
        ),
    ],
)
def test_download_papers_omits_paper_when_pdf_is_inaccessible(
    paper, pdf_url, status, body, tmp_path, monkeypatch,
):
    # Reproduce previously observed responses without depending on live availability.
    monkeypatch.delenv("S2_API_KEY", raising=False)
    monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)
    requested_urls = []

    def respond(request):
        requested_urls.append(str(request.url))
        if request.method == "POST" and request.url.path == "/graph/v1/paper/batch":
            return httpx.Response(200, json=[{
                "paperId": paper.paper_id, "openAccessPdf": {"url": pdf_url},
            }])
        if str(request.url) == pdf_url:
            return httpx.Response(status, content=body)
        return httpx.Response(404)

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        paper_fetch_tool.httpx, "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs),
    )

    destination = tmp_path / "papers"
    result = asyncio.run(download_papers([paper], destination, timeout=1))

    assert result == []
    assert pdf_url in requested_urls
    assert list(destination.iterdir()) == []
