
"""
Retriever Agent

Specialized agent for participating in RAG workflows.
- Rewrites queries based on history
- Searches Vector DB (S3 Vector Engine)
"""

import boto3
import math
import logging
from os import getenv
from typing import List, Dict, Any, Callable
from langchain_aws import BedrockEmbeddings

try:
    from .core import Agent, tool
except ImportError:
    # Fallback for running directly as a script
    from core import Agent, tool

LOG = logging.getLogger(__name__)

# ===== Configuration & Clients (Merged from search_utils.py) =====
AWS_REGION = getenv("AWS_REGION", "us-east-1")
AWS_VECTOR_REGION = getenv("AWS_VECTOR_REGION", "us-east-1")
VECTOR_BUCKET = getenv("VECTOR_BUCKET", "my-vector-bucket")
VECTOR_INDEX = getenv("VECTOR_INDEX", "document-embeddings")
EMBED_MODEL_ID = getenv("EMBED_MODEL", "amazon.titan-embed-text-v2:0")
EMBED_DIM = int(getenv("EMB_DIM", "1024"))
TOP_K = int(getenv("TOP_K", "2"))
S3_BUCKET_NAME = getenv("S3_BUCKET_NAME", "aws-bedrock-demo-01")

bedrock_client = boto3.client('bedrock-runtime', region_name=AWS_REGION)
s3vectors = boto3.client('s3vectors', region_name=AWS_VECTOR_REGION)
s3_client = boto3.client('s3', region_name=AWS_REGION)

# Embedder
try:
    embedder = BedrockEmbeddings(
        client=bedrock_client,
        model_id=EMBED_MODEL_ID
    )
except Exception as e:
    LOG.warning(f"Could not initialize BedrockEmbeddings: {e}")
    embedder = None

# ===== Helper Functions =====

def _validate_vector(vec, expected_dim):
    """Validate and normalize vector dimensions."""
    if not isinstance(vec, (list, tuple)):
        raise ValueError("embedder returned non-list vector")

    out = []
    for i, v in enumerate(vec):
        fv = float(v)
        if not math.isfinite(fv):
            raise ValueError(f"Vector element {i} is non-finite")
        out.append(fv)

    # Slice vector to expected dimension if it's too long
    if len(out) > expected_dim:
        out = out[:expected_dim]

    if len(out) != expected_dim:
        raise ValueError(f"Embedding dimension mismatch: {len(out)} != {expected_dim}")

    return out

def fetch_chunk_from_s3(chunk_s3_key):
    """Fetch chunk text from S3"""
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=chunk_s3_key)
        return response['Body'].read().decode('utf-8')
    except Exception as e:
        LOG.error(f"Failed to fetch chunk: {chunk_s3_key}, error={e}")
        return ""

def similarity_search(query: str, user_id: str, chat_id: str, k: int = None) -> list:
    """
    Search for similar document chunks using S3 Vector Engine.
    Filtered by user_id and chat_id for isolation.
    """
    if k is None:
        k = TOP_K

    LOG.info(f"Vector search: user={user_id}, chat={chat_id}, k={k}, query={query}")

    if not embedder:
        LOG.error("Embedder not available - cannot search")
        return []

    # Generate embedding for query
    q_vec_raw = embedder.embed_query(query)
    q_vector = _validate_vector(q_vec_raw, expected_dim=EMBED_DIM)

    # 1. Try S3 Vector Engine
    results = []
    try:
        print(f"DEBUG: querying s3vectors with bucket={VECTOR_BUCKET}, index={VECTOR_INDEX}")
        print(f"DEBUG: filter user_id={user_id}")
        resp = s3vectors.query_vectors(
            vectorBucketName=VECTOR_BUCKET,
            indexName=VECTOR_INDEX,
            queryVector={"float32": q_vector},
            topK=k,
            returnMetadata=True,
            returnDistance=True,
            filter={
                "$or": [
                    {
                        "$and": [
                            {"email_id": {"$eq": user_id}},
                            {"chat_id": {"$eq": chat_id}},
                            {"kb_type": {"$eq": "user_upload"}}
                        ]
                    },
                    {"kb_type": {"$eq": "global"}}
                ]
            }
        )
        print(f"DEBUG: s3vectors raw response: {resp}")
        
        for v in resp.get("vectors", []):
            md = v.get("metadata", {})
            chunk_s3_key = md.get("chunk_s3_key") or md.get("chunk_path")
            if chunk_s3_key:
                chunk = fetch_chunk_from_s3(chunk_s3_key)
                if chunk:
                    results.append({"chunk": chunk, "metadata": md, "distance": v.get("distance")})

    except Exception as e:
        LOG.warning(f"S3 Vector search failed (likely local): {e}")
        print(f"DEBUG: S3 Vector search EXCEPTION: {e}")

    LOG.info(f"Vector search complete: found {len(results)} results")
    
    print("\n=== SIMILARITY SEARCH RESULTS ===")
    for i, r in enumerate(results):
        print(f"--- Chunk {i+1} ---")
        print(r.get('chunk'))
        print("------------------")
    print("=================================\n")

    return results

# ===== Agent Definition =====

RETRIEVER_PROMPT = """
You are a specialized Retriever Agent.
Your goal is to find the most relevant information from the Knowledge Base.

INSTRUCTIONS:
1. Analyze the user's query and conversation history.
2. The user's query might depend on previous context (e.g., "how much is it?").
3. You must FORMULATE a standalone, search-friendly query that includes all necessary context.
4. Call the `search_knowledge_base` tool with this rewritten query.
5. Answer the user's original question using ONLY the returned context. if no context is found, say so.
"""

class RetrieverAgent(Agent):
    def __init__(self, user_id: str, chat_id: str, model_id: str):
        self.user_id = user_id
        self.chat_id = chat_id
        
        super().__init__(
            name="Retriever",
            model_id=model_id, # Dynamic model
            instructions=RETRIEVER_PROMPT,
            # We give the retriever ONE tool: to search the vector DB
            tools=[self.search_knowledge_base]
        )

    @tool
    def search_knowledge_base(self, query: str):
        """
        Searches the vector database for documents matching the query.
        Args:
            query: The standalone search query.
        """
        results = similarity_search(query, self.user_id, self.chat_id)
        
        if not results:
            return "No relevant documents found."
            
        # Format results into a context string
        context_str = "\n\n".join([
            f"Document {i+1}:\n{r['chunk']}" 
            for i, r in enumerate(results)
        ])
        return context_str

# if __name__ == "__main__":
#     print("\n--- Running Manual Similarity Search Test ---")
#     # Using 'test-user@example.com' as it is the default used in index.py test
#     # If you used a different email to upload, change this!
#     test_user_id = "test-user@example.com" 
#     # test_user_id = "local-tester@example.com" # processing the one from query_rag_bedrock
    
#     test_chat_id = "test-chat-01"
#     test_query = "exam date"
    
#     print(f"User: {test_user_id}")
#     print(f"Query: {test_query}")
    
#     results = similarity_search(test_query, test_user_id, test_chat_id)
    
#     print(f"\nFound {len(results)} results:")
#     for r in results:
#         print("-" * 40)
#         print(f"Chunk: {r.get('chunk')[:200]}...") # Printing first 200 chars
#         print(f"Metadata: {r.get('metadata')}")
