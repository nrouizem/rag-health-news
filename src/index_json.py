import json
from pathlib import Path
import datetime as dt
import tiktoken

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

JSON_PATH = "data/processed/records.json"  # list[dict] with your keys
INDEX_DIR = "faiss_index"
CHUNK_SIZE, CHUNK_OVERLAP = 900, 120

enc = tiktoken.get_encoding("cl100k_base")
tok = lambda s: len(enc.encode(s or ""))

def norm_date(s):
    if not s: return ""
    try:
        return dt.date.fromisoformat(str(s)[:10]).isoformat()
    except Exception:
        return str(s)

def first_url(sources):
    if not sources: return None
    if isinstance(sources, str): return sources
    if isinstance(sources, (list, tuple)): 
        for u in sources:
            if isinstance(u, str) and u.startswith("http"):
                return u
    return None

def coalesce_title(item):
    acq = item.get("acquirer") or ""
    tgt = item.get("acquired_company") or item.get("company") or ""
    kind = item.get("type") or "Deal"
    bits = [b for b in [acq, kind, tgt] if b]
    return " – ".join(bits) if bits else "Deal"

def main():
    items = json.load(open(JSON_PATH, "r", encoding="utf-8"))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        length_function=tok, separators=["\n\n","\n",". "," "]
    )

    chunk_texts, metadatas = [], []

    for i, it in enumerate(items):
        title = coalesce_title(it)
        date  = norm_date(it.get("date_of_deal_announcement"))
        url   = first_url(it.get("sources"))
        companies = [c for c in [it.get("acquirer"), it.get("acquired_company")] if c]
        deal_type = it.get("type") or ""
        th_area   = ", ".join(it.get("therapeutic_areas") or []) if isinstance(it.get("therapeutic_areas"), list) else (it.get("therapeutic_areas") or "")
        advisors_legal = it.get("legal_advisors") or ""
        advisors_fin   = it.get("financial_advisors") or ""
        value_mil      = it.get("total_consideration_mil")

        body = it.get("article_text") or ""  # embed the full article/press text
        header = (
            f"Title: {title}\n"
            f"Date: {date}\n"
            f"Companies: {', '.join(companies)}\n"
            f"Deal Type: {deal_type}\n"
            f"Therapeutic Areas: {th_area}\n"
            f"Advisors (Legal): {advisors_legal}\n"
            f"Advisors (Financial): {advisors_fin}\n"
            f"Disclosed Value (USD mil): {value_mil}\n"
            f"Source: {url}\n\n"
        )
        full = header + body

        # chunk as needed
        if tok(full) <= CHUNK_SIZE:
            chunk_texts.append(full)
            metadatas.append({
                "id": f"row_{i}",
                "title": title,
                "date": date,
                "source_url": url,
                "companies": companies,
                "deal_type": deal_type,
                "therapeutic_areas": th_area,
                "advisors_legal": advisors_legal,
                "advisors_financial": advisors_fin,
                "value_mil": value_mil,
            })
        else:
            parts = splitter.split_text(full)
            for j, part in enumerate(parts):
                chunk_texts.append(part)
                metadatas.append({
                    "id": f"row_{i}",
                    "chunk": j,
                    "title": title,
                    "date": date,
                    "source_url": url,
                    "companies": companies,
                    "deal_type": deal_type,
                    "therapeutic_areas": th_area,
                    "advisors_legal": advisors_legal,
                    "advisors_financial": advisors_fin,
                    "value_mil": value_mil,
                })

    embeddings = OpenAIEmbeddings()
    vs = FAISS.from_texts(chunk_texts, embedding=embeddings, metadatas=metadatas)
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    vs.save_local(INDEX_DIR)
    print(f"Saved FAISS with {len(chunk_texts)} chunks from {len(items)} records at {INDEX_DIR}/")

if __name__ == "__main__":
    main()
