from fastapi import HTTPException
from models import ThesisTitle

async def get_related_documents(thesis: ThesisTitle, index):
    try:
        related_documents = index.as_retriever(similarity_top_k=3).retrieve(thesis.title)
        
        results = []
        for doc in related_documents:
            text = doc.text
            url_start = text.find("url")
            url_end = text.find("\n", url_start)
            url = text[url_start + len("url:"):url_end].strip() if url_start != -1 else "No URL"

            abstrak_start = text.find("Abstrak:")
            abstrak_text = text[abstrak_start + len("Abstrak:"):].strip()
            kata_kata = abstrak_text.split()
            abstrak_20_kata = ' '.join(kata_kata[:20]) + '.....'

            author_start = text.find("Author:")
            judul_start = text.find("Judul:")
            judul = text[judul_start + len("Judul:"):author_start].strip()

            results.append({
                "judul": judul,
                "abstrak": abstrak_20_kata,
                "url": url
            })

        return {"related_documents": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))