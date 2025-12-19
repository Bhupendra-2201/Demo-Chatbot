"""
Query Lambda - AWS Bedrock RAG with S3 Vector Engine
Uses Orchestrator → Retriever Agent pattern for intelligent RAG
"""

import boto3
from os import getenv
import logging
import json
from decimal import Decimal
import base64
import re


# ===== Logging =====
from typing import List, Dict, Any
import time

LOG = logging.getLogger()
LOG.setLevel(logging.INFO)

# ===== Configuration =====
llm_model_id = getenv("LLM_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
CHAT_HISTORY_TABLE = getenv("CHAT_HISTORY_TABLE", "chatHistory_table")
CHAT_HISTORY_WINDOW = int(getenv("CHAT_HISTORY_WINDOW", "10"))
TOP_K = int(getenv("TOP_K", "2"))

aws_region = getenv("AWS_REGION", "us-east-1")
dynamodb = boto3.resource("dynamodb", region_name=aws_region)
table = dynamodb.Table(CHAT_HISTORY_TABLE)
s3_client = boto3.client("s3", region_name=aws_region)
s3_bucket_name = getenv("S3_BUCKET_NAME", "aws-bedrock-demo-01")
bedrock_client = boto3.client("bedrock-runtime", region_name=aws_region)

from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key, Attr

# ===== Agentic Components =====
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from agents.core import Agent, tool
from agents.retriever import RetrieverAgent, similarity_search
from conditional_orchestrator import ConditionalOrchestrator

# ===== Chat History Functions =====
def get_chat_history(chat_id: str, limit: int = 10) -> List[Dict]:
    """Retrieve chat history from DynamoDB"""
    try:
        response = table.query(
            KeyConditionExpression=Key("chat_id").eq(chat_id),
            ScanIndexForward=False, # Get latest first
            Limit=limit
        )
        items = response.get("Items", [])
        # Reverse to get chronological order (oldest first)
        return items[::-1]
    except Exception as e:
        LOG.error(f"Failed to get chat history: {e}")
        return []

def store_message(chat_id: str, role: str, content: str, user_id: str = "unknown"):
    """Store message in DynamoDB"""
    try:
        # LOG.debug(f"Storing message in DynamoDB table '{CHAT_HISTORY_TABLE}' for chat '{chat_id}'...")
        table.put_item(
            Item={
                "chat_id": chat_id,
                "timestamp": str(time.time()), # Sort Key
                "role": role,
                "content": content,
                "user_id": user_id 
            }
        )
        # LOG.debug("Message stored successfully.")
    except Exception as e:
        LOG.error(f"Failed to store message: {e}")

def delete_chat_history(chat_id: str) -> bool:
    """Delete all messages for a specific chat ID"""
    try:
        # 1. Query all items for this chat
        response = table.query(
            KeyConditionExpression=Key("chat_id").eq(chat_id)
        )
        items = response.get('Items', [])
        
        # 2. Delete using batch writer for efficiency
        with table.batch_writer() as batch:
            for item in items:
                batch.delete_item(
                    Key={
                        'chat_id': item['chat_id'],
                        'timestamp': item['timestamp']
                    }
                )
        LOG.info(f"Deleted chat {chat_id} history ({len(items)} messages)")
        return True
    except Exception as e:
        LOG.error(f"Failed to delete chat history: {e}")
        return False

def list_conversations(user_id: str) -> List[Dict]:
    """
    List all unique chat sessions for a user.
    Since table key is chat_id, we must SCAN with FilterExpression on user_id.
    This is expensive at scale but fine for demo.
    """
    try:
        # Scan for all items with this user_id
        response = table.scan(
            FilterExpression=Attr("user_id").eq(user_id),
            ProjectionExpression="chat_id, #t, content",
            ExpressionAttributeNames={"#t": "timestamp"} 
        )
        items = response.get("Items", [])
        
        # Aggregate by chat_id
        chats = {}
        for item in items:
            cid = item['chat_id']
            ts = item['timestamp']
            
            # Keep the latest timestamp for sorting
            if cid not in chats:
                chats[cid] = {"id": cid, "timestamp": ts, "title": "New Chat"}
            
            if ts > chats[cid]["timestamp"]:
                chats[cid]["timestamp"] = ts
                
            # Heuristic for title: Use the first user message? 
            # For now, just keep "Chat <Date>" or rely on frontend to title it.
            # Or we can try to find the earliest message content.
        
        # Sort by timestamp desc
        sorted_chats = sorted(chats.values(), key=lambda x: x["timestamp"], reverse=True)
        return sorted_chats
    except Exception as e:
        LOG.error(f"Failed to list conversations: {e}")
        return []

def get_full_chat_history(chat_id: str) -> List[Dict]:
    """Get full history for UI loading"""
    return get_chat_history(chat_id, limit=50)

# System Prompt
ORCHESTRATOR_PROMPT = """
You are the Main Orchestrator. 
Analyze the user's request and route it to the appropriate specialized agent or handle it yourself.

ROUTING RULES:
1. If the user asks for information (policies, documents, facts) -> Call Retriever Agent.
   - Example: "What is the sick leave policy?"
   - Example: "How do I apply for a refund?"
2. If the user says "Hi", "Thanks", or engages in small talk -> Answer directly (Casual).
   - Be concise and friendly.
3. If the user asks to clarify previous context -> Analysis required. If it's about facts, use Retriever.

You have access to the following tools:
- call_retriever: To search the Knowledge Base. Use this for ANY factual question.
"""

class Orchestrator(Agent):
    def __init__(self, user_id: str, chat_id: str, history: List[Dict], model_id: str):
        self.user_id = user_id
        self.chat_id = chat_id
        self.history = history
        
        super().__init__(
            name="Orchestrator",
            model_id=model_id, # Use dynamic model_id
            instructions=ORCHESTRATOR_PROMPT,
            tools=[self.call_retriever]
        )

    @tool
    def call_retriever(self, query: str):
        """
        Delegates the query to the Retriever Agent.
        Args:
            query: The user's question to be answered.
        """
        # Instantiate Retriever with context
        agent = RetrieverAgent(self.user_id, self.chat_id, self.model_id)
        
        # Run the sub-agent
        response = agent.run(query, history=[])
        return response

# ===== Query =====
def query_rag(user_id, chat_id, message, model_id=llm_model_id, features=None):
    """
    Process RAG query with feature-based routing
    """
    if model_id is None:
        model_id = llm_model_id
    
    # Default features
    if features is None:
        features = {
            'smart_mode': True,
            'use_doc_agent': True,
            'use_kb': True,
            'use_websearch': False,
            'doc_chat_only': True
        }
    
    LOG.info(f"RAG query: user={user_id}, chat={chat_id}, features={features}")
    
    # Get chat history
    history = get_chat_history(chat_id, limit=CHAT_HISTORY_WINDOW * 2)
    
    # Store user message
    store_message(chat_id, "user", message, user_id)
    
    # Route based on features
    # Route based on features
    # Smart Mode = Agentic Orchestrator (LLM decides)
    # Manual Mode = Direct Query (Hardcoded logic based on flags)
    if features.get('smart_mode', False):
        response = agentic_query(user_id, chat_id, message, history, model_id, features)
    else:
        response = direct_query(user_id, chat_id, message, history, model_id, features)
    
    # Store assistant response
    store_message(chat_id, "assistant", response, user_id)
    
    return response


def agentic_query(user_id, chat_id, message, history, model_id, features):
    """Agentic workflow with conditional tools (Smart Mode)"""
    try:
        # Build tools list based on features and permissions
        tools = []
        
        # 1. Retrieval Tool (Docs / KB)
        # In Smart Mode, RAG (Retrieval) is the core capability. 
        tools.append('call_retriever')
        
        # 2. Web Search
        # In Smart Mode, the Agent decides if it needs the web.
        # We enable it by default.
        if features.get('smart_mode', False) or features.get('use_websearch', False):
            tools.append('search_web')
            LOG.info("Smart Mode/Web Flag: Enabling 'search_web' tool.")
        
        if not tools:
            # Fallback
            LOG.info("Smart Mode but no tools enabled. Fallback to direct.")
            return direct_query(user_id, chat_id, message, history, model_id, features)
        
        # Check model compatibility
        if "amazon.titan" in model_id or "nova" in model_id:
            raise ValueError("Model doesn't support tools")
        
        # Create orchestrator
        orchestrator = ConditionalOrchestrator(
            user_id, chat_id, history, model_id,
            enabled_tools=tools,
            doc_chat_only=features.get('doc_chat_only', True)
        )
        
        LOG.info(f"Agentic mode (Smart) initialized. Tools: {tools}")
        
        # Run Orchestrator
        # Reinforce tool use behavior with a message injection
        reinforced_message = message
        if features.get('smart_mode', True): # Default to true here
             reinforced_message = f"{message}\n\n[System Note: If this query references a person, project, document, or specific fact, you MUST query your 'call_retriever' tool first. Do not answer 'I don't know' without searching.]"

        response = orchestrator.run(reinforced_message, history=history)
        return response
        
    except Exception as e:
        LOG.warning(f"Agentic failed ({e}). Falling back to direct.")
        return direct_query(user_id, chat_id, message, history, model_id, features)


def direct_query(user_id, chat_id, message, history, model_id, features):
    """Direct LLM with optional RAG/WebSearch (non-agentic)"""
    LOG.info("Direct query mode")
    
    context_sections = []
    
    # Add chat document context (only if Document Agent is enabled)
    if features.get('use_doc_agent', False):
        try:
            results = similarity_search(
                message, user_id,
                chat_id,  # Always chat-only for Document Agent
                k=TOP_K
            )
            if results:
                kb_text = "\n\n".join([f"[Chat Doc {i+1}]\n{r['chunk']}" for i, r in enumerate(results)])
                context_sections.append(f"<chat_documents>\n{kb_text}\n</chat_documents>")
                LOG.info(f"Added {len(results)} chat document results")
        except Exception as e:
            LOG.error(f"Chat document search failed: {e}")
    
    # Add global KB context
    if features.get('use_kb', False):
        try:
            results = similarity_search(
                message, user_id,
                None,  # Global search (all documents)
                k=TOP_K
            )
            if results:
                kb_text = "\n\n".join([f"[KB Doc {i+1}]\n{r['chunk']}" for i, r in enumerate(results)])
                context_sections.append(f"<knowledge_base>\n{kb_text}\n</knowledge_base>")
                LOG.info(f"Added {len(results)} KB results")
        except Exception as e:
            LOG.error(f"KB search failed: {e}")
    
    # Add web search
    if features.get('use_websearch', False):
        try:
            web_text = perform_web_search(message, num_results=3)
            if web_text and "error" not in web_text.lower():
                context_sections.append(f"<web_search>\n{web_text}\n</web_search>")
                LOG.info("Added web search")
        except Exception as e:
            LOG.error(f"Web search failed: {e}")
    
    # Build prompt
    if context_sections:
        system_prompt = f"""{rag_chat_bot_prompt}

{chr(10).join(context_sections)}

Cite sources when using context. If question can't be answered from context, say so."""
    else:
        system_prompt = casual_prompt
    
    # Build messages
    messages = []
    for msg in history[-CHAT_HISTORY_WINDOW:]:
        messages.append({"role": msg["role"], "content": [{"text": msg["content"]}]})
    messages.append({"role": "user", "content": [{"text": message}]})
    
    # Call Bedrock
    converse_kwargs = {
        "modelId": model_id,
        "messages": messages,
        "inferenceConfig": {"maxTokens": 2000, "temperature": 0.7}
    }
    
    if "amazon.titan" not in model_id:
        converse_kwargs["system"] = [{"text": system_prompt}]
    
    try:
        response = bedrock_client.converse(**converse_kwargs)
        output = response.get("output", {}).get("message", {}).get("content", [])
        return "".join(b.get("text", "") for b in output)
    except Exception as e:
        LOG.error(f"Bedrock failed: {e}")
        return f"Error: {str(e)}"

def fallback_rag(user_id, chat_id, message, history, model_id):
    """
    Fallback RAG implementation (direct retrieval + LLM)
    Used when agentic framework is not available
    """
    LOG.info("Using fallback RAG (non-agentic)")
    
    # Search for relevant context
    search_results = similarity_search(message, user_id, chat_id, k=TOP_K)
    print("Search results:", search_results)
    if search_results:
        context = "\n\n".join([r["chunk"] for r in search_results])
        print("=== RETRIEVED CHUNKS ===\n", context, "\n========================")
        system_prompt = rag_chat_bot_prompt + f"\n\n<context>\n{context}\n</context>"
    else:
        system_prompt = rag_chat_bot_prompt + casual_prompt
    
    # Format history for Converse API
    messages = []
    for msg in history[-CHAT_HISTORY_WINDOW:]:
        messages.append({
            "role": msg["role"],
            "content": [{"text": msg["content"]}]
        })
    
    # Add current message
    messages.append({
        "role": "user",
        "content": [{"text": message}]
    })
    
    # Prepare Converse API arguments
    converse_kwargs = {
        "modelId": model_id,
        "messages": messages,
        "inferenceConfig": {"maxTokens": 2000, "temperature": 0.7}
    }

    # Handle models that don't support 'system' parameter (like Amazon Titan)
    if "amazon.titan" in model_id:
        if messages and messages[-1]["role"] == "user":
            original_text = messages[-1]["content"][0]["text"]
            messages[-1]["content"][0]["text"] = f"System Instructions:\n{system_prompt}\n\nUser Query:\n{original_text}"
            
    else:
        converse_kwargs["system"] = [{"text": system_prompt}]

    # Call Bedrock Converse API
    try:
        response = bedrock_client.converse(**converse_kwargs)
        
        output_message = response.get("output", {}).get("message", {})
        content_blocks = output_message.get("content", [])
        return "".join(block.get("text", "") for block in content_blocks)
        
    except Exception as e:
        LOG.error(f"Bedrock Converse failed: {e}")
        return f"Error calling LLM: {str(e)}"

# ===== Presigned URL (for file uploads from query UI) =====
def create_presigned_post(event):
    """Generate presigned S3 POST URL for file upload"""
    query_params = event.get('queryStringParameters', {})
    email_id = "empty_email_id"
    
    if 'requestContext' in event and 'authorizer' in event['requestContext']:
        if 'claims' in event['requestContext']['authorizer']:
            email_id = event['requestContext']['authorizer']['claims']['email']
    
    if 'file_extension' in query_params and 'file_name' in query_params:
        extension = query_params['file_extension']
        file_name = query_params['file_name']
        usecase_type = query_params.get('type', 'bedrock')
        
        # Sanitize file name
        file_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '', file_name).replace(' ', '_')
        s3_key = f"{usecase_type}/data/{file_name}.{extension}"
        
        response = s3_client.generate_presigned_post(
            Bucket=s3_bucket_name,
            Key=s3_key,
            Fields={'x-amz-meta-email_id': email_id},
            Conditions=[{'x-amz-meta-email_id': email_id}]
        )
        
        return http_success_response(response)
    else:
        return http_failure_response('Missing file_extension field cannot generate signed url')

# ===== Lambda Handler =====
def handler(event, context):
    LOG.info("--- S3 Vector Engine RAG with Bedrock (RESTful) ---")
    LOG.info(f"Event: {event}")
    
    if 'httpMethod' in event:
        api_map = {
            'POST/query': lambda x: rag_query(x),
            'POST/delete-chat': lambda x: delete_chat(x),
            'GET/list-chats': lambda x: list_chats(x),
            'GET/get-chat-history': lambda x: get_chat_history(x),
            'POST/file_data': lambda x: create_presigned_post(x),
        }
        
        http_method = event.get('httpMethod', '')
        api_path = http_method + event.get('resource', '')
        
        try:
            if api_path in api_map:
                LOG.info(f"Handling API: {api_path}")
                return respond(None, api_map[api_path](event))
            else:
                LOG.info(f"API not found: {api_path}")
                return respond(http_failure_response('api_not_supported'), None)
        except Exception as e:
            LOG.exception(f"Error processing API: {api_path}")
            return respond(http_failure_response('system_exception'), None)
    
    return {'statusCode': '200', 'body': 'Bedrock RAG says hello'}

def rag_query(event):
    """
    Handle RAG query from HTTP POST
    Expected body: {"user_id": "...", "chat_id": "...", "message": "...", "features": {...}}
    """
    try:
        body = json.loads(event.get('body', '{}'))
        user_id = body.get('user_id')
        chat_id = body.get('chat_id')
        message = body.get('message')
        model_id = body.get('model_id')
        features = body.get('features', {
            'smart_mode': True,
            'use_doc_agent': True,
            'use_kb': True,
            'use_websearch': False,
            'doc_chat_only': True
        })
        
        LOG.info(f"Query request: features={features}")
        
        if not all([user_id, chat_id, message]):
            return http_failure_response('Missing required parameters: user_id, chat_id, message')
        
        response = query_rag(user_id, chat_id, message, model_id, features)
        return http_success_response({"response": response})
        
    except Exception as e:
        LOG.error(f"RAG query failed: {e}")
        return http_failure_response(str(e))

def delete_chat(event):
    """
    Handle Chat Deletion
    Expected query string: chat_id
    """
    try:
        query_params = event.get('queryStringParameters', {})
        chat_id = query_params.get('chat_id')
        if not chat_id:
             # Try body
             body = json.loads(event.get('body', '{}'))
             chat_id = body.get('chat_id')
             
        if not chat_id:
            return http_failure_response("Missing chat_id")
            
        success = delete_chat_history(chat_id)
        if success:
             return http_success_response(f"Chat {chat_id} deleted")
        else:
             return http_failure_response("Failed to delete chat")
             
    except Exception as e:
         LOG.error(f"Delete chat failed: {e}")
         return http_failure_response(str(e))

def list_chats(event):
    """
    Handle List Chats for User
    """
    try:
        # Get user_id from authorizer context or body
        user_id = "unknown"
        if 'requestContext' in event and 'authorizer' in event['requestContext']:
            if 'claims' in event['requestContext']['authorizer']:
                user_id = event['requestContext']['authorizer']['claims'].get('email', 'unknown')
        
        # Fallback to body for testing
        if user_id == "unknown":
             body = json.loads(event.get('body', '{}'))
             user_id = body.get('user_id', 'unknown')

        if user_id == 'unknown':
             return http_failure_response("Unauthorized: Missing user_id")

        chats = list_conversations(user_id)
        return http_success_response(chats)
    except Exception as e:
        LOG.error(f"List chats failed: {e}")
        return http_failure_response(str(e))

def get_chat_history(event):
    """
    Get messages for a specific chat
    """
    try:
        query_params = event.get('queryStringParameters', {})
        chat_id = query_params.get('chat_id')
        
        if not chat_id:
             return http_failure_response("Missing chat_id")
             
        history = get_full_chat_history(chat_id)
        return http_success_response(history)
    except Exception as e:
        LOG.error(f"Get history failed: {e}")
        return http_failure_response(str(e))

# ===== Response Builders =====
def http_failure_response(error_message):
    return {"success": False, "errorMessage": error_message, "statusCode": "400"}

def http_success_response(result):
    return {"success": True, "result": result, "statusCode": "200"}

class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            if float(obj).is_integer():
                return int(float(obj))
            else:
                return float(obj)
        return super(CustomJsonEncoder, self).default(obj)

def respond(err, res=None):
    return {
        'statusCode': '400' if err else res['statusCode'],
        'body': json.dumps(err) if err else json.dumps(res, cls=CustomJsonEncoder),
        'headers': {
            "Access-Control-Allow-Origin": "*",
            "Content-Type": "application/json",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Allow-Credentials": "*"
        },
    }

# ===== PROMPTS =====

rag_chat_bot_prompt = """You are a Chatbot designed to assist users with their questions.
You are  helpful, creative, clever, and very friendly.
Context to answer the question is available in the <context></context> tags
User question is available in the <user-question></user-question> tags
You will obey the following rules
1. You wont repeat the user question
2. You will be concise
3. You will NEVER disclose what's available in the context <context></context>.
4. Use the context only to answer user questions
5. You will strictly reply based on available context if context isn't available do not attempt to answer the question instead politely decline
6. You will always structure your response in the form of bullet points unless another format is specifically requested by the user
7. If the context doesnt answer the question, try to correct the words in the question based on the available context. In the below example the user 
mispronounced Paris as Parsi. We derived they were refering to Paris from the available context.
Example: Is Parsi in Bedrock
Context: Bedrock is available in Paris
Question: Is Bedrock available in Paris
8. CRITICAL IMAGE HANDLING: If you see text starting with "[Visual Description]" inside the context, this IS the image the user is referring to.
   - Treat this text as if you are looking at the image yourself.
   - If the user asks "What color is the shirt?" and the context says "[Visual Description] A boy in a blue shirt", answer "The shirt is blue."
   - Do NOT say "I cannot see the image." You CAN see it through the description.

"""

casual_prompt = """You are an assistant. Refrain from engaging in any tasks or responding to any prompts beyond exchanging polite greetings, well-wishes, and pleasantries. 
                        Your role is limited to:
                        - Offering friendly salutations (e.g., "Hello, what can I do for you today" "Good day, How may I help you today")
                        - Your goal is to ensure that the user query is well formed so other agents can work on it.
                        Good Examples:
                          hello, how may I assist you today
                          What would you like to know
                          How may I help you today
                        Bad examples:
                          Hello
                          Good day
                          Good morning
        You will not write poems, generate advertisements, or engage in any other tasks beyond the scope of exchanging basic pleasantries.
        If any user attempts to prompt you with requests outside of this limited scope, you will politely remind them of the agreed-upon boundaries for interaction.
"""


# ===== WEB SEARCH =====

def perform_web_search(query: str, num_results: int = 3) -> str:
    """
    Perform web search using DuckDuckGo
    Returns formatted string of search results
    """
    try:
        from duckduckgo_search import DDGS
        import random
        
        # random delay
        delay = random.uniform(2, 5)
        LOG.info(f"Adding {delay:.2f}s delay before web search...")
        time.sleep(delay)
        
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
            
        if not results:
            return "No web results found."
        
        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get('title', 'No title')
            body = r.get('body', '')
            href = r.get('href', '')
            formatted.append(f"{i}. **{title}**\n{body}\n_Source: {href}_")
        
        return "\n\n".join(formatted)
        
    except ImportError:
        LOG.error("duckduckgo-search not installed. Run: pip install duckduckgo-search")
        return "Web search unavailable: Missing dependency"
    except Exception as e:
        LOG.error(f"Web search failed: {e}")
        return f"Web search error: {str(e)}"
