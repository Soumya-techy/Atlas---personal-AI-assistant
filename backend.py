import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from typing import TypedDict, Annotated
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever,PineconeHybridSearchRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
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


llm=ChatGroq(model='openai/gpt-oss-120b',api_key=os.getenv('GROQ_API_KEY'))
embeding_model=HuggingFaceEmbeddings(model='BAAI/bge-large-en-v1.5')
reranking_model=HuggingFaceCrossEncoder(model_name='BAAI/bge-reranker-v2-m3')
splitter=RecursiveCharacterTextSplitter(chunk_size=900,chunk_overlap=120)


hybrid=PineconeHybridSearchRetriever(embeddings=embeding_model,index=index,sparse_encoder=BM25Encoder.default(),top_k=10,alpha=0.5)
compressor=CrossEncoderReranker(model=reranking_model,top_n=5)
mid_retreiver=MultiQueryRetriever.from_llm(
      retriever=hybrid,
      llm=llm
)
main_retreiver=ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=mid_retreiver)


def ingest_document(file_path):
    chunks=splitter.split_documents(PyPDFLoader(file_path).load())
    hybrid.add_texts(
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
        answers=main_retreiver.invoke(query)
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
    Sends an email using the Gmail API.
    
    Args:
        to (str): Recipient email address
        subject (str): Subject of the email
        body (str): Plain text body of the email
    """
    try:
        service = get_gmail_service()
        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject
        
        # Encode to base64url format required by Gmail API
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        body_payload = {'raw': raw_message}
        
        result = service.users().messages().send(userId="me", body=body_payload).execute()
        return f"Email sent successfully. Message ID: {result['id']}"
    except Exception as e:
        return f"Failed to send email: {str(e)}"



def read_inbox_emails_tool(max_results: int = 5, query: str = "is:unread") -> list:
    """
    AI Agent Tool: Fetches emails matching specific filters (like a sender's email) from the Inbox.
    
    Args:
        max_results (int): Maximum number of emails to fetch (default is 5).
        query (str): Gmail search query string. 
                     Examples: 
                     - "is:unread" (default)
                     - "from:boss@company.com" (to read from a specific email)
                     - "from:boss@company.com is:unread" (combination)
    """
    try:
        service = get_gmail_service()
        
        # Combined: We lock the target to the INBOX but use the dynamic query 'q' 
        # so your agent can search for specific email addresses.
        results = service.users().messages().list(
            userId='me', 
            labelIds=['INBOX'], 
            q=query,
            maxResults=max_results
        ).execute()
        
        messages = results.get('messages', [])
        
        if not messages:
            return f"No messages found matching the criteria: '{query}'"
            
        parsed_emails = []
        
        for msg in messages:
            msg_id = msg['id']
            message = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
            
            payload = message.get('payload', {})
            headers = payload.get('headers', [])
            
            subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), '(No Subject)')
            sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), '(Unknown Sender)')
            snippet = message.get('snippet', '')
            
            def extract_plain_text(parts_list):
                for part in parts_list:
                    mime_type = part.get('mimeType', '')
                    body_obj = part.get('body', {})
                    
                    if mime_type == 'text/plain' and 'data' in body_obj:
                        return base64.urlsafe_b64decode(body_obj['data']).decode('utf-8', errors='ignore')
                    
                    if 'parts' in part:
                        nested_text = extract_plain_text(part['parts'])
                        if nested_text:
                            return nested_text
                return None

            body = ""
            if 'parts' in payload:
                body = extract_plain_text(payload['parts'])
            elif 'body' in payload and 'data' in payload['body']:
                body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')

            parsed_emails.append({
                "id": msg_id,
                "from": sender,
                "subject": subject,
                "snippet": snippet,
                "body": body.strip() if body else snippet
            })
            
        return parsed_emails
        
    except Exception as e:
        return f"Failed to retrieve emails: {str(e)}"

print(read_inbox_emails_tool(max_results=1,query='from:s39010281@gmail.com'))