import os
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from llama_index.core import Document, VectorStoreIndex
from models import Query

pdf_index = None

async def upload_pdf(file: UploadFile = File(...)):
    global pdf_index
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF file.")
    
    with open(file.filename, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        reader = PdfReader(file.filename)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        document = Document(text=text)
        pdf_index = VectorStoreIndex.from_documents([document])
        
        return JSONResponse(content={"message": "PDF uploaded and indexed successfully"})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
    finally:
        os.remove(file.filename)

async def query_pdf(query: Query, llm):
    global pdf_index
    
    if pdf_index is None:
        raise HTTPException(status_code=400, detail="No PDF has been uploaded and indexed yet.")
    
    try:
        query_engine = pdf_index.as_query_engine(llm=llm)
        response = query_engine.query(query.question)
        
        return JSONResponse(content={"answer": str(response)})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")