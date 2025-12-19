# Demo Chatbot

A Serverless RAG (Retrieval-Augmented Generation) application built with AWS Bedrock, LangChain, and the S3 Vector Engine.

## Overview

This project demonstrates an agentic approach to building RAG applications. It utilizes the S3 Vector Engine for vector storage, significantly reducing costs while maintaining performance.

The system features a Streamlit frontend and a serverless AWS Lambda backend that supports:
- Smart Routing: An Orchestrator Agent decides if a query needs RAG or just a chat response.
- Multi-Chat Isolation: Users can have multiple independent chat sessions.
- Document Indexing: Upload and index PDFs, Images (OCR), and Text files.

## Key Features

*   **S3 Vector Engine**: Serverless vector storage directly in S3.
*   **Agentic Workflow**: Uses an Orchestrator and specialized Retriever agents.
*   **LangChain Integration**: Built on standard LangChain components for Bedrock.
*   **Multi-Modal Support**: Processes PDFs and Images (using Bedrock OCR).
*   **Per-User Security**: Data isolation between users and chats.
*   **Cost Effective**: Low running cost due to serverless architecture.

## Project Structure

*   frontend/ - Streamlit Chat Interface.
*   backend/lambdas/ - Serverless functions.
    *   index_lambda/ - Handles file uploads and embedding generation.
    *   query_lambda/ - Handles RAG queries and agentic logic.
*   deployment/ - Ready-to-deploy zip artifacts.

## Deployment

### Prerequisites
*   AWS Account with Bedrock Models enabled (Claude 3 Sonnet, Titan Embeddings v2).

### Deploying High-Level

1.  **Backend**: Deploy the Zips from the deployment folder to AWS Lambda.
    *   index_lambda.zip -> index-lambda
    *   query_lambda.zip -> query-lambda
2.  **Infrastructure**:
    *   Create S3 Buckets (1 for files, 1 for vectors).
    *   Create DynamoDB Tables (ChatHistory, IndexAudit).
    *   (Optional) Setup API Gateway to trigger Lambdas.
3.  **Frontend**:
    *   Update the Streamlit app configuration with your API/Cognito details.
    *   Run the Streamlit app.
