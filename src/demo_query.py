import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

INDEX_DIR = "faiss_index"

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an analyst assistant. Use ONLY the context to answer.\n"
        "If the context is insufficient, say you don't know.\n\n"
        "{context}\n\n"
        "Question: {question}\n\n"
        "Answer with concise bullets.\n"
        "When you make a claim, append citations in brackets, e.g., [Title — Source, Date].\n"
        "If multiple sources support a point, include both."
    ),
)

def main():
    # Ensuring API key is set
    assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY in your environment."

    embeddings = OpenAIEmbeddings()  # optionally: model="text-embedding-3-small"

    vs = FAISS.load_local(
        INDEX_DIR,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vs.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )

    q = "Summarize recent deals that mention contingent value rights (CVR)."
    res = qa.invoke({"query": q})

    print("\nQ:", q)
    print("\nA:", res["result"])

    print("\nCitations:")
    seen = set()
    for d in res["source_documents"]:
        m = d.metadata or {}
        title = m.get("title") or "(untitled)"
        date  = m.get("date") or "n.d."
        url   = m.get("source_url") or "(no url)"
        key = (title, url, date)
        if key in seen: 
            continue
        seen.add(key)
        print(f"- {title} ({url}, {date})")

if __name__ == "__main__":
    main()
