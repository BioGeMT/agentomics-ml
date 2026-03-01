from pydantic import BaseModel, Field

class KnowledgeIntegration(BaseModel):
    queries: list = Field(
        description="""
        List of specific queries you want to answer using the knowledge base. These should be directly relevant to your task. Can be an empty list if you don't have useful queries.
        """
    )

def get_knowledge_integration_prompt():
    return """\nYour first task: define a series of queries, that will retrieve knowledge from the database that contains information relevant to the task and the dataset.
    These queries should be designed to retrieve information that is relevant and useful for your task.    
    """