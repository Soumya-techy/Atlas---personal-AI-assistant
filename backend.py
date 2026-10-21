import os
from dotenv import load_dotenv
from functools import lru_cache
import requests
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from typing import TypedDict, Annotated,Union, Literal,Optional
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from ddgs import DDGS
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from pinecone import Pinecone,ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import base64
from email.mime.text import MIMEText
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
from langgraph.types import interrupt
from langchain_core.messages import BaseMessage
from youtube_transcript_api import YouTubeTranscriptApi
import re
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.postgres import PostgresStore
import uuid
from datetime import datetime, timezone
load_dotenv()

os.getenv('HF_TOKEN')
pinecone_client=Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index_name='project-db'
if not pinecone_client.has_index(index_name):
      pinecone_client.create_index(
            name=index_name,
            dimension=1024,
            metric='dotproduct',
            spec=ServerlessSpec(cloud='aws',region='us-east-1')
      )
index=pinecone_client.Index(index_name)
mongo_db_client=MongoClient(os.getenv('MONGODB_URI'))
checkpoint=MongoDBSaver(mongo_db_client,db_name='Agent_memory',checkpoint_collection_name='thread_checkpoints')
pool = ConnectionPool(
    os.getenv("POSTGRESQL_URI"),
    max_size=10,
    kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    open=True,
)
store=PostgresStore(pool)
store.setup()
@lru_cache(maxsize=1)
def get_embedding_model():
    embeding_model=HuggingFaceEmbeddings(model='BAAI/bge-large-en-v1.5')
    return embeding_model
@lru_cache(maxsize=1)
def get_reranker_model():
    reranking_model=HuggingFaceCrossEncoder(model_name='BAAI/bge-reranker-v2-m3')
    return reranking_model
splitter=RecursiveCharacterTextSplitter(chunk_size=900,chunk_overlap=120)

@lru_cache(maxsize=1)
def get_hybrid_retreiver():
    hybrid=PineconeHybridSearchRetriever(embeddings=embeding_model,index=index,sparse_encoder=BM25Encoder.default(),top_k=15,alpha=0.5)
    return hybrid
@lru_cache(maxsize=1)
def get_main_retreiver():
    compressor=CrossEncoderReranker(model=get_reranker_model(),top_n=7)
    main_retreiver=ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=get_hybrid_retreiver())
    return main_retreiver


def ingest_document(file_path):
    chunks=splitter.split_documents(PyPDFLoader(file_path).load())
    get_hybrid_retreiver().add_texts(
        texts=[c.page_content for c in chunks],
        metadatas=[{"source": os.path.basename(file_path), "page": c.metadata.get("page", 0)} for c in chunks]
    )


@tool
def rag_tool(query:str) -> str:
        """
    Retrieve relevant information from the PDF document.

    Use this tool when the user asks factual or conceptual questions
    that may be answered using the stored PDF documents.

    Args:
        query: The question or search query used to retrieve PDF content.
    """ 
        formated_content=[]
        answers=get_main_retreiver().invoke(query)
        for content in answers:
            formated_content.append(content.page_content)
        return "\n\n".join(formated_content)

@tool
def web_search(query:str) ->list[dict]:
        """
    Use this tool when the user's query requires something that requires a web search or information retrieved
    from the rag tool is not enough to answer the user's query.

    Args:
    query: The question or search query you want to do web search for
    """
        web=DDGS()
        web_search_result=[]
        search=web.text(query, max_results=5)
        for content in search:
              web_search_result.append(
                    {'title': content['title'],
                     'url' : content['href'],
                     'content' : content['body']}
              )
        return web_search_result

SCOPES = [
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.readonly'
]
def get_gmail_service():
    creds = None
    # token.json stores the user's access and refresh tokens
    import os
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('gmail_api_secrets.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

@tool
def send_gmail_tool(to: str, subject: str, body: str) -> str:
    """
    Use this tool when you are asked to send any email
    
    Args:
        to (str): Recipient email address
        subject (str): Subject of the email
        body (str): Plain text body of the email
    """
    decision=interrupt({
        'type':'approval',
        'to':to,
        'subject': subject,
        'body': body,
        'instruction':'Approve this action? yes/no'
        })
    if not decision.get('approved', False):
        return "The email action was rejected by user"
    try:
        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject
        
        # Encode to base64url format required by Gmail API
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        body_payload = {'raw': raw_message}
        service = get_gmail_service()
        result = service.users().messages().send(userId="me", body=body_payload).execute()
        return f"Email sent successfully. Message ID: {result['id']}"
        
    except Exception as e:
        return f"Failed to send email: {str(e)}"


def _decode(data: str) -> str:
    data += "=" * (-len(data) % 4)  # Gmail sometimes drops base64 padding
    return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
 
 
def _walk_parts(payload: dict):
    """Yield the payload and every nested part, depth-first."""
    yield payload
    for part in payload.get("parts", []) or []:
        yield from _walk_parts(part)
 
 
def _extract_bodies(payload: dict):
    """Return (plain_text, html) found anywhere in the email."""
    plain, html = "", ""
    for part in _walk_parts(payload):
        data = part.get("body", {}).get("data")
        if not data:
            continue
        mime = part.get("mimeType", "")
        if mime == "text/plain" and not plain:
            plain = _decode(data)
        elif mime == "text/html" and not html:
            html = _decode(data)
    return plain, html
 
 
def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    for img in soup.find_all("img"):  # alt text sometimes holds the message
        alt = (img.get("alt") or "").strip()
        img.replace_with(f" {alt} " if len(alt) > 3 else " ")
    text = soup.get_text(separator="\n")
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()
 
 
# Marketing emails pad the preview text with hundreds of invisible characters
ZERO_WIDTH = re.compile(r"[\u00ad\u034f\u061c\u180e\u200b-\u200f\u2060-\u2064\u2066-\u206f\ufeff]")
ODD_SPACES = re.compile(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")
 
 
def _clean_text(text: str, limit: int = 3000) -> str:
    text = ZERO_WIDTH.sub("", text)             # remove invisible padding
    text = ODD_SPACES.sub(" ", text)            # exotic spaces -> normal space
    text = re.sub(r"https?://\S+", "[link]", text)  # tracking URLs are noise
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:limit]
 
 
# ---------- the tool ----------
@tool
def read_inbox_emails_tool(max_results: int = 5, query: str = "is:unread") -> Union[list, str]:
    """
    Fetch emails from the Inbox that match a Gmail search query.
 
    Args:
        max_results: Maximum number of emails to fetch (default 5).
        query: Gmail search query string. Examples:
               - "is:unread" (default)
               - "from:boss@company.com"
               - "from:boss@company.com is:unread"
 
    Each email has: id, from, subject, body, and image_only. If image_only is
    True, the real content is a picture (poster/flyer), so tell the user to
    open the email in Gmail instead of guessing at its contents.
    """
    try:
        service = get_gmail_service()
 
        results = service.users().messages().list(
            userId="me",
            labelIds=["INBOX"],
            q=query,
            maxResults=max_results,
        ).execute()
 
        messages = results.get("messages", [])
        if not messages:
            return f"No messages found matching the criteria: '{query}'"
 
        parsed_emails = []
        for msg in messages:
            message = service.users().messages().get(
                userId="me", id=msg["id"], format="full"
            ).execute()
 
            payload = message.get("payload", {})
            headers = payload.get("headers", [])
 
            subject = next((h["value"] for h in headers if h["name"].lower() == "subject"), "(No Subject)")
            sender = next((h["value"] for h in headers if h["name"].lower() == "from"), "(Unknown Sender)")
            snippet = message.get("snippet", "")
 
            plain, html = _extract_bodies(payload)
            body = _clean_text(plain)
            if len(body) < 80 and html:  # plain part missing or useless
                body = _clean_text(_html_to_text(html))
 
            image_only = len(body) < 80 and "<img" in html.lower()
 
            parsed_emails.append({
                "id": msg["id"],
                "from": sender,
                "subject": subject,
                "body": body or snippet,
                "image_only": image_only,
            })
 
        return parsed_emails
 
    except Exception as e:
        return f"Failed to retrieve emails: {str(e)}"
@tool
def yt_video_finder(q:str,max_results:int=5)->list[dict]:
    """ Use this tool when you need to find some youtube video for the user's query
     Args:
        q: the query or topic or video title you want to search on youtube,
         max_results: the total number of videos you want to fetch (default is 5) """
    result=[]
    parameters={
        'q':q,
        'part':'snippet',
        'type':'video',
        'maxResults':max_results,
        'key':os.getenv('YT_API_KEY')
    }
    try: 
        request=requests.get('https://www.googleapis.com/youtube/v3/search',params=parameters)
        request.raise_for_status()
        response=request.json()
        for item in response.get('items',{}):
            x={
                'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                'title': item['snippet']['title'],
                'description': item['snippet']['description']          
            }
            result.append(x)
        return result
    except requests.exceptions.HTTPError as code:
        if request.status_code==400:
            print("Bad request")
        elif request.status_code==429:
            print("Limit exceeded")
        elif request.status_code==401:
            print("Error in api key")
        else:
            print(code)

@tool
def yt_video_summarizer(url:str)->list[dict]:
    """ Use this tool to find the transcript of any youtube video to summarize or other tasks
    Args:
      url: the url of the youtube video"""
    a,b=url.split('=')
    yt=YouTubeTranscriptApi()
    response=yt.fetch(b,languages=['en','en-IN'])
    result=[]
    for transcript in response:
        x={
            'text':transcript.text,
            'start':transcript.start,
            'duration':transcript.duration
        }
        result.append(x)
    return(result)



 
namespace=('admin','memories')
# Creating nodes
class state(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

class decider(BaseModel):
    action:Literal['ADD','UPDATE','DELETE','NONE']
    fact:Optional[str]=None
    key:Optional[str]=None



tools=[web_search,rag_tool,send_gmail_tool,read_inbox_emails_tool,yt_video_finder,yt_video_summarizer]
tool_node=ToolNode(tools)
chat_model=ChatGoogleGenerativeAI(
    model='gemini-3.6-flash',
    api_key=os.getenv('GEMINI_API_KEY'),
    streaming=True
)

llm=chat_model.bind_tools(tools)

decider_llm=ChatGroq(
    model='openai/gpt-oss-120b',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0,
).with_structured_output(decider, method='json_mode')



def chat_node(state:state):
    memories="\n".join(f"{m.value['fact']}" for m in store.search(namespace)) or "(empty)"
    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot whose name is atlas with access to several tools.\n\n"

            f"this is the memories you have about the user: \n{memories}\n\n"

            "Tool usage instructions:\n"
            "- Use `rag_tool` for questions about the uploaded PDF or document. "
            "Always retrieve relevant document content before answering PDF-related questions.\n"
            "- Use `web_search` for current events, recent information, or information "
            "that requires an internet search or if the information retrieved from rag_tool is not enough to answer the use query .\n"
            "- When the user explicitly asks to send an email and provides enough details, "
            "you MUST call `send_gmail_tool`. Do not merely say that an email is prepared, "
            "triggered, or awaiting approval. The tool will pause for the interface to show "
            "the user an approval choice before any email is sent.\n"
            "- Use `read_inbox_emails_tool` when the user asks to read or search their inbox.\n"
            "- Use `yt_video_finder` to find YouTube videos and `yt_video_summarizer` to retrieve a video's transcript.\n"
    
            "Answer general questions directly when no tool is required. "
            "Do not invent information from the uploaded document. "
            "If the user asks about a PDF but no document is available, ask them to upload a PDF. "
            "After receiving a tool result, provide a clear and helpful final answer."
        )
    )
    message=[system_message,*state['messages']]
    result=llm.invoke(message)
    print(result.tool_calls)
    print(result.content)
    return {'messages':[result]}

def memory_decider(state:state):
    existing=store.search(namespace)
    existing_text="\n".join(f"[{m.key}] {m.value['fact']}" for m in existing) or "(empty)"
    user_msg=next(
        (
            message.content
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        ),
        "",
    )
    if not user_msg:
        return {}
    decider_prompt=f"""You manage a user's long-term memory.

Existing memories:
{existing_text}

Check last user message: "{user_msg}"

Choose ONE action:
- ADD: a new lasting fact that does not conflict with existing ones
- UPDATE: the message replaces or corrects an existing memory (give its key and the new fact)
- DELETE: the user wants an existing memory removed (give its key)
- NONE: nothing worth remembering

Return ONLY a JSON object with exactly these keys:
{{"action": "ADD|UPDATE|DELETE|NONE", "fact": string or null, "key": string or null}}"""
    try:
        result=decider_llm.invoke(decider_prompt)
    except Exception:
        # Memory is optional: never fail the user-facing conversation if it cannot be saved.
        return {}
    now=datetime.now(timezone.utc).isoformat()
    if result.action=='ADD':
        store.put(namespace,str(uuid.uuid4()),{'fact':result.fact,'saved_at':now})
    elif result.action=='UPDATE':
        old=store.get(namespace,result.key)
        store.put(namespace,result.key,{
            'fact':result.fact,
            'saved_at':now,
            'previous': old['fact'] if old else None
        })
    elif result.action=='DELETE':
        store.delete(namespace,result.key)
    else:
        pass
    return {}

graph=StateGraph(state)
graph.add_node('chat_node',chat_node)
graph.add_node('memory_decider_node',memory_decider)
graph.add_node('tools',tool_node)
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node','memory_decider_node')
graph.add_conditional_edges('memory_decider_node',tools_condition)
graph.add_edge('tools','chat_node')
workflow=graph.compile(checkpointer=checkpoint,store=store)

def get_all_threads():
    """Return unique saved conversation IDs, newest checkpoint first."""
    threads = []
    seen = set()
    for ckpt in checkpoint.list(None):
        thread_id = ckpt.config["configurable"].get("thread_id")
        if thread_id and thread_id not in seen:
            seen.add(thread_id)
            threads.append(thread_id)
    return threads
