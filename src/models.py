from pydantic import BaseModel

class ThesisTitle(BaseModel):
    title: str

class ChatQuery(BaseModel):
    query: str
    context: str

class Query(BaseModel):
    question: str

class ContinueGenerateQuery(BaseModel):
    query: str
    context: str
    previous_response: str