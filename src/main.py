from fastapi import FastAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from pinecone import Pinecone
from transformers import AutoTokenizer, BitsAndBytesConfig
from fastapi import FastAPI, UploadFile, File
from llama_index.llms.huggingface import HuggingFaceLLM
from pyngrok import ngrok

from pdf_chat import upload_pdf, query_pdf
from document_chat import chat_with_document, continue_generate
from related_documents import get_related_documents
from models import ThesisTitle, ChatQuery, Query, ContinueGenerateQuery

import os
from dotenv import load_dotenv

load_dotenv()


app = FastAPI()
hf_key= os.getenv("HF_KEY")

# Initialize Pinecone and embedding model
pinecone_key = os.getenv("PINECONE_KEY")
pc = Pinecone(api_key=pinecone_key)
index_name = 'lmitd2'
pinecone_index = pc.Index(index_name)

embed_model = "Alibaba-NLP/gte-large-en-v1.5"
Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model, trust_remote_code=True)

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
index = VectorStoreIndex.from_vector_store(vector_store)

def get_llm():
    quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    #load_in_4bit=True,
    #bnb_4bit_compute_dtype=torch.float16,
    #bnb_4bit_quant_type="nf4",
    #bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        token=hf_key,
    )

    stopping_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    
    llm = HuggingFaceLLM(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        model_kwargs={
            "token": hf_key,
            #"torch_dtype": torch.float16,
            "quantization_config":quantization_config
        },
        generate_kwargs={
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.9,
            "pad_token_id": tokenizer.eos_token_id, 
        },
        tokenizer_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        tokenizer_kwargs={"token": hf_key},
        stopping_ids=stopping_ids,
    )
    Settings.llm = llm
    return llm, tokenizer

llm, tokenizer = get_llm()

@app.post("/upload-pdf/")
async def api_upload_pdf(file: UploadFile = File(...)):
    return await upload_pdf(file)

@app.post("/query-pdf/")
async def api_query_pdf(query: Query):
    return await query_pdf(query, llm)

@app.post("/chat/")
async def api_chat_with_document(chat_query: ChatQuery):
    return await chat_with_document(chat_query,llm)

@app.post("/related_documents/")
async def api_get_related_documents(thesis: ThesisTitle):
    return await get_related_documents(thesis, index)

@app.post("/continue-generate/")
async def api_continue_generate(continue_query: ContinueGenerateQuery):
    return await continue_generate(continue_query, llm)

if __name__ == "__main__":
    ngrok_token = os.getenv("NGROK_TOKEN")
    ngrok.set_auth_token(ngrok_token)
    ngrok_tunnel = ngrok.connect(8000)
    print('Public URL:', ngrok_tunnel.public_url)
    import nest_asyncio
    nest_asyncio.apply()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)