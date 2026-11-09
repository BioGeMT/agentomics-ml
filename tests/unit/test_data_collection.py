import pytest
from pydantic import ValidationError

from agentomics.agents.data_collection import DataRequest, Review


@pytest.mark.parametrize(
    ("task_type", "classes"),
    [("classification", ["benign"]), ("regression", ["low", "high"])],
)
def test_request_rejects_classes_that_do_not_fit_the_task_type(task_type, classes):
    with pytest.raises(ValidationError):
        DataRequest(task_type=task_type, description="d", inputs=["i"], label="l", classes=classes)


def test_review_that_does_not_accept_must_list_an_issue():
    with pytest.raises(ValidationError):
        Review(accepted=False)
