import os
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.ext.asyncio import create_async_engine
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

raw_url = os.getenv("DATABASE_URL")
if not raw_url:
    raise ValueError("HATA: Lütfen .env dosyasında DATABASE_URL değişkenini ayarlayın!")

if raw_url.startswith("postgres://"):
    DEFAULT_DB_URL = raw_url.replace("postgres://", "postgresql://", 1)
else:
    DEFAULT_DB_URL = raw_url



_vectorstores = {}

def get_relevant_schema(query: str, db_url=DEFAULT_DB_URL, k=3):
    global _vectorstores
    
    if db_url not in _vectorstores:
        engine = create_engine(db_url)
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        
        docs = []
        for table_name in table_names:
            columns = [f"{col['name']} ({col['type']})" for col in inspector.get_columns(table_name)]
            table_info = f"Table: {table_name}\nColumns: {', '.join(columns)}"
            docs.append(Document(page_content=table_info, metadata={"table_name": table_name}))
            
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _vectorstores[db_url] = FAISS.from_documents(docs, embeddings)
        
    vs = _vectorstores[db_url]
    # FAISS içindeki doküman sayısını alıp k ile kıyasla
    fetch_k = min(k, vs.index.ntotal)
    if fetch_k == 0:
        return "Veritabanında tablo bulunmuyor."
        
    relevant_docs = vs.similarity_search(query, k=fetch_k)
    schema_str = ""
    for doc in relevant_docs:
        schema_str += f"- {doc.page_content}\n"
    return schema_str

_value_vectorstores = {}

def get_relevant_values(query: str, db_url=DEFAULT_DB_URL, k=3):
    global _value_vectorstores
    if db_url not in _value_vectorstores:
        engine = create_engine(db_url)
        inspector = inspect(engine)
        docs = []
        
        with engine.connect() as conn:
            for table_name in inspector.get_table_names():
                for col in inspector.get_columns(table_name):
                    col_type = str(col['type']).upper()
                    if "VARCHAR" in col_type or "TEXT" in col_type or "STRING" in col_type:
                        try:
                            # Tırnak işareti ile güvenli kolon/tablo sorgusu
                            res = conn.execute(text(f'SELECT DISTINCT "{col["name"]}" FROM "{table_name}" LIMIT 50'))
                            for row in res:
                                val = row[0]
                                if val:
                                    docs.append(Document(
                                        page_content=f"Değer: '{val}' (Tablo: {table_name}, Kolon: {col['name']})",
                                        metadata={"table": table_name, "column": col['name']}
                                    ))
                        except Exception:
                            pass
        
        if not docs:
            docs.append(Document(page_content="Kategorik veri bulunamadı.", metadata={}))
            
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _value_vectorstores[db_url] = FAISS.from_documents(docs, embeddings)
        
    relevant_docs = _value_vectorstores[db_url].similarity_search(query, k=k)
    val_str = ""
    for doc in relevant_docs:
        val_str += f"- {doc.page_content}\n"
    return val_str

def get_current_dialect(db_url=DEFAULT_DB_URL):
    engine = create_engine(db_url)
    name = engine.dialect.name
    mapping = {
        "sqlite": "SQLite",
        "postgresql": "PostgreSQL",
        "mysql": "MySQL",
        "mssql": "SQL Server",
        "oracle": "Oracle"
    }
    return mapping.get(name, name.capitalize())

def get_async_url(db_url):
    if db_url.startswith("sqlite:///"):
        return db_url.replace("sqlite:///", "sqlite+aiosqlite:///")
    elif db_url.startswith("postgresql://"):
        return db_url.replace("postgresql://", "postgresql+asyncpg://")
    elif db_url.startswith("mysql://"):
        return db_url.replace("mysql://", "mysql+aiomysql://")
    return db_url

async def execute_sql_async(query: str, db_url=DEFAULT_DB_URL):
    async_url = get_async_url(db_url)
    engine = create_async_engine(async_url)
    try:
        async with engine.connect() as conn:
            # İşlemi başlat
            trans = await conn.begin()
            try:
                result = await conn.execute(text(query))
                if result.returns_rows:
                    columns = list(result.keys())
                    rows = [list(row) for row in result.fetchall()]
                    res = {"columns": columns, "rows": rows}
                else:
                    res = {"columns": [], "rows": []}
            finally:
                # Ne olursa olsun geri al (Değişiklikleri kaydetmeyi reddet)
                await trans.rollback()
            return True, res
    except Exception as e:
        return False, str(e)
    finally:
        await engine.dispose()


