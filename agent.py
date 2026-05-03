from typing import TypedDict, Optional, Annotated
import operator
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from database import execute_sql_async, get_relevant_schema, get_relevant_values

class ExecuteSQLQuery(BaseModel):
    """Veritabanında SQL sorgusunu çalıştırmak için kullanılacak araç."""
    query: str = Field(..., description="Çalıştırılacak tam SQL sorgusu. Sadece geçerli SQL kodu olmalı.")

# Graph state definition
class AgentState(TypedDict):
    query: str
    intent: Optional[str]
    sql_query: Optional[str]
    sql_valid: Optional[bool]
    sql_error: Optional[str]
    sql_result: Optional[str]
    final_answer: Optional[str]
    error_count: int
    dialect: Optional[str] # Örneğin: "SQLite", "PostgreSQL", "MySQL"
    chat_history: Annotated[list, operator.add]

# Initialize model (Ollama llama3.1)
llm = ChatOllama(model="llama3.1", temperature=0)

async def intent_parser(state: AgentState):
    """
    Kullanıcı veritabanı sorgusu mu istiyor (sql) yoksa genel sohbet mi (chat)?
    """
    # Sadece son 10 mesajı alıp context limit aşımını önle
    recent_history = state.get("chat_history", [])[-10:]
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
    
    sys_msg = f"""Sen bir Intent Parser'sın. 
Kullanıcının mesajını analiz et ve sadece JSON formatında dön:
{{"intent": "sql"}} veya {{"intent": "chat"}}

Eğer kullanıcı veri analizi, istatistik, kayıt arama veya sistemde tutulan bir bilgiyi sorguluyorsa "sql" seç. 
Eğer sadece selamlaşma, hal hatır sorma veya veri/tablo gerektirmeyen genel bir sohbet ise "chat" seç.

Geçmiş Sohbet:
{history_str}
"""
    response = await llm.ainvoke([
        SystemMessage(content=sys_msg),
        HumanMessage(content=state["query"])
    ])
    
    content = response.content.lower()
    intent = "sql" if "sql" in content else "chat"
        
    return {"intent": intent, "error_count": state.get("error_count", 0)}

async def sql_generator(state: AgentState):
    """
    Kullanıcının sorusu için SQL üret.
    """
    dialect = state.get("dialect", "SQLite")
    dynamic_schema = get_relevant_schema(state["query"])
    dynamic_values = get_relevant_values(state["query"])
    
    schema = f"""
    Kullanılan Veritabanı: {dialect}
    
    Soru ile İlgili Olabilecek Tablolar (Semantic RAG):
{dynamic_schema}

    Veritabanında Sorunla Eşleşebilecek Örnek Veriler (Value RAG):
{dynamic_values}
    
    Not: Tarih hesaplamaları yaparken {dialect} veritabanına ait yerleşik fonksiyonları kullan.
    """
    
    error_msg = f"Daha önceki denemende şu hatayı aldık: {state.get('sql_error')}\nLütfen SQL sorgusunu düzelterek tekrar yaz." if state.get("sql_error") else ""
    
    recent_history = state.get("chat_history", [])[-10:]
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
    
    sys_msg = f"""Sen uzman bir {dialect} veritabanı geliştiricisisin. 
Kullanıcının isteğine uygun {dialect} sorgusu üret.
Mutlaka `ExecuteSQLQuery` aracını (tool) çağırarak SQL sorgunu argüman olarak ilet.

{schema}

Örnekler:
Soru: Toplam kaç sipariş var?
SQL: SELECT COUNT(*) FROM orders;
Soru: Fiyatı en yüksek olan ürün hangisi?
SQL: SELECT name FROM products ORDER BY price DESC LIMIT 1;
Soru: Bugün kaç sipariş geldi?
SQL: SELECT COUNT(*) FROM orders WHERE order_date = CURRENT_DATE;

Geçmiş Sohbet Bağlamı:
{history_str}

{error_msg}
"""
    llm_with_tools = llm.bind_tools([ExecuteSQLQuery])
    response = await llm_with_tools.ainvoke([
        SystemMessage(content=sys_msg),
        HumanMessage(content=state["query"])
    ])
    
    if response.tool_calls:
        sql_query = response.tool_calls[0]["args"]["query"]
    else:
        # Fallback (Eğer LLM aracı kullanmayı reddedip düz metin dönerse)
        sql_query = response.content.strip().replace("```sql", "").replace("```", "").strip()
        
    return {"sql_query": sql_query}

async def sql_validator(state: AgentState):
    """
    SQL güvenli mi (sadece SELECT mi içeriyor) kontrol et.
    """
    sql_query = state.get("sql_query", "")
    
    if not sql_query.upper().strip().startswith("SELECT"):
        return {
            "sql_valid": False, 
            "sql_error": "Sadece SELECT sorgularına izin verilir. Lütfen sadece SELECT içeren bir SQL üret.",
            "error_count": state.get("error_count", 0) + 1
        }
        
    # Tehlikeli komut kontrolü
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "GRANT", "TRUNCATE"]
    if any(keyword in sql_query.upper() for keyword in forbidden):
        return {
            "sql_valid": False, 
            "sql_error": "Veri değiştiren komutlar yasaktır. Sadece okuma yapabilirsin.",
            "error_count": state.get("error_count", 0) + 1
        }
        
    return {"sql_valid": True, "sql_error": None}

async def execute_query_node(state: AgentState):
    """
    Geçerli SQL'i çalıştır.
    """
    sql_query = state["sql_query"]
    success, result = await execute_sql_async(sql_query)
    
    if not success:
        return {
            "sql_valid": False, 
            "sql_error": f"Veritabanı Hatası: {result}",
            "error_count": state.get("error_count", 0) + 1
        }
        
    return {"sql_result": str(result), "sql_valid": True, "sql_error": None}

async def result_explainer(state: AgentState):
    """
    SQL sonuçlarını doğal dille açıklayan node.
    """
    recent_history = state.get("chat_history", [])[-10:]
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
    
    if state.get("intent") == "chat":
        response = await llm.ainvoke([
            SystemMessage(content=f"Sen yardımsever bir asistansın.\nGeçmiş Sohbet:\n{history_str}"),
            HumanMessage(content=state["query"])
        ])
        return {
            "final_answer": response.content,
            "chat_history": [{"role": "user", "content": state["query"]}, {"role": "assistant", "content": response.content}]
        }
        
    sys_msg = f"""Sen sonuçları yorumlayan bir uzmansın. 
Kullanıcının sorusu ve veritabanından dönen sonucu analiz ederek kullanıcıya güzel bir Türkçe açıklama yap.
Ham SQL sonucunu okunaklı bir listeye veya paragrafa çevir.

ÖNEMLİ KURALLAR:
1. KESİNLİKLE uydurma (halüsinasyon) veri üretme. Müşteri ismi, yaş, meslek gibi veritabanı sonucunda OLMAYAN hiçbir bilgiyi kafandan uydurma.
2. Sadece sana verilen "Veritabanı Sonucu" kısmındaki gerçek verileri kullan.
3. Eğer kullanıcının sorduğu bilgi veritabanından dönmemişse (örneğin müşteri adları tabloda yoksa), "Veritabanında bu bilgi bulunmuyor" diyerek dürüstçe belirt.
4. ÇOK ÖNEMLİ: Cevabını kısa, net ve tek parça halinde ver. Aynı bilgiyi farklı cümlelerle veya maddeleme yaparak tekrar etme. Doğrudan sonuca odaklan.

Geçmiş Sohbet Bağlamı:
{history_str}
"""
    user_msg = f"Soru: {state['query']}\nVeritabanı Sonucu: {state.get('sql_result', 'Sonuç yok.')}"
    
    # Hata olduysa halüsinasyonu önlemek için LLM'i atla ve kesin hata mesajı dön.
    if state.get("error_count", 0) > 3 or state.get("sql_valid") is False:
        err_reply = "Üzgünüm, bu soruyu cevaplamak için uygun veritabanı sorgusunu oluştururken sistemsel teknik bir hata oluştu. Lütfen soruyu daha basit veya farklı bir şekilde sormayı dener misin?"
        return {
            "final_answer": err_reply,
            "chat_history": [{"role": "user", "content": state["query"]}, {"role": "assistant", "content": err_reply}]
        }
        
    user_msg = f"Soru: {state['query']}\nVeritabanı Sonucu: {state.get('sql_result', 'Sonuç yok.')}"
    
    response = await llm.ainvoke([
        SystemMessage(content=sys_msg),
        HumanMessage(content=user_msg)
    ])
    
    return {
        "final_answer": response.content,
        "chat_history": [{"role": "user", "content": state["query"]}, {"role": "assistant", "content": response.content}]
    }

# Edge functions
def route_after_intent(state: AgentState):
    if state["intent"] == "sql":
        return "sql_generator"
    return "result_explainer"

def route_after_sql_validator(state: AgentState):
    if state.get("sql_valid"):
        return "execute_query"
    return "sql_generator"

def route_after_execute(state: AgentState):
    if state.get("sql_valid") is False:
        if state.get("error_count", 0) > 3:
            return "result_explainer" # Çok hata olduysa döngüyü kır
        return "sql_generator"
    return "result_explainer"

# Graph builder
workflow = StateGraph(AgentState)

workflow.add_node("intent_parser", intent_parser)
workflow.add_node("sql_generator", sql_generator)
workflow.add_node("sql_validator", sql_validator)
workflow.add_node("execute_query", execute_query_node)
workflow.add_node("result_explainer", result_explainer)

workflow.set_entry_point("intent_parser")

workflow.add_conditional_edges("intent_parser", route_after_intent)
workflow.add_edge("sql_generator", "sql_validator")
workflow.add_conditional_edges("sql_validator", route_after_sql_validator)
workflow.add_conditional_edges("execute_query", route_after_execute)
workflow.add_edge("result_explainer", END)

memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)
