from fastapi import HTTPException
from fastapi.responses import JSONResponse
from models import ChatQuery, ContinueGenerateQuery

async def chat_with_document(chat_query: ChatQuery, llm):
    try:
        prompt = f"Context: {chat_query.context}\n\nQuestion: {chat_query.query}\n\nAnswer:"
        response = llm.complete(prompt)
        
        return JSONResponse(content={"response": str(response)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def continue_generate(continue_query: ContinueGenerateQuery, llm):
    try:
        prompt = f"Based on the previous response: '{continue_query.previous_response}' and given context: {continue_query.context}, continue the answer for the query: {continue_query.query}."
        response = llm.complete(prompt)
        
        return JSONResponse(content={"response": str(response)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))