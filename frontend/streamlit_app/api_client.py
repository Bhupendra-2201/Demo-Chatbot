import requests
import json
import logging

BASE_URL = "https://h74g6f3fs8.execute-api.us-east-1.amazonaws.com/stage"

class APIClient:
    def __init__(self, user_id="local-tester@example.com"):
        self.headers = {
            "Content-Type": "application/json",
            "x-user-id": user_id
        }
        self.user_id = user_id

    def set_user(self, user_id):
        self.user_id = user_id
        self.headers["x-user-id"] = user_id

    def set_token(self, id_token):
        """Set the Authorization token for all requests"""
        if id_token:
            self.headers["Authorization"] = f"Bearer {id_token}"

    def list_files(self):
        try:
            # Change to POST as per index.py handler
            resp = requests.post(f"{BASE_URL}/rag/get-indexed-files-by-user", headers=self.headers)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    return data.get("result", [])
            return []
        except Exception as e:
            logging.error(f"Error listing files: {e}")
            return []

    def get_presigned_url(self, filename, ext, chat_id, doc_title="Untitled", type="index"):
        params = {
            "file_name": filename, 
            "file_extension": ext, 
            "chat_id": chat_id, # <--- Added chat_id
            "type": type,
            "doc_title": doc_title
        }
        try:
            # Added trailing slash before ? to match user's working Postman URL
            resp = requests.get(f"{BASE_URL}/rag/get-presigned-url/", params=params, headers=self.headers)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    return data.get("result")
            return None
        except Exception as e:
            logging.error(f"Error getting presigned URL: {e}")
            return None

    def upload_file(self, file_bytes, filename, chat_id, doc_title="Untitled"):
        # 1. Get Presigned URL
        ext = filename.split('.')[-1]
        url_data = self.get_presigned_url(filename, ext, chat_id, doc_title)
        
        if not url_data:
            return False, "Failed to get upload URL"

        # 2. Upload to S3
        try:
            url = url_data['url']
            fields = url_data['fields']
            files = {'file': file_bytes}
            
            s3_resp = requests.post(url, data=fields, files=files)
            
            if s3_resp.status_code != 204:
                return False, f"S3 Upload Failed: {s3_resp.text}"
                
            # 3. Success (Async Processing by S3 Trigger)
            # The backend Lambda is triggered automatically by S3. We don't need to manually trigger it.
            return True, "File uploaded successfully! Indexing has started in the background."

        except Exception as e:
            return False, str(e)

    def trigger_s3_event(self, bucket, key):
        payload = {"bucket": bucket, "key": key}
        try:
            resp = requests.post(f"{BASE_URL}/rag/trigger-s3-event", json=payload, headers=self.headers)
            return resp.json()
        except Exception as e:
            return {"success": False, "errorMessage": str(e)}

    def delete_file(self, s3_key):
        try:
            resp = requests.post(f"{BASE_URL}/rag/del-file", json={"s3_key": s3_key}, headers=self.headers)
            return resp.json()
        except Exception as e:
            return {"success": False, "errorMessage": str(e)}

    def delete_chat(self, chat_id):
        try:
            resp = requests.post(f"{BASE_URL}/rag/delete-chat", params={"chat_id": chat_id}, headers=self.headers)
            if resp.status_code == 200:
                return resp.json()
            return {"success": False, "errorMessage": resp.text}
        except Exception as e:
            return {"success": False, "errorMessage": str(e)}

    def list_chats(self):
        try:
            resp = requests.get(f"{BASE_URL}/rag/list-chats", headers=self.headers)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                     return data.get("result", [])
            return []
        except Exception as e:
            logging.error(f"Error listing chats: {e}")
            return []

    def get_chat_history(self, chat_id):
        try:
            resp = requests.get(f"{BASE_URL}/rag/get-chat-history", params={"chat_id": chat_id}, headers=self.headers)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                     return data.get("result", [])
            return []
        except Exception as e:
            logging.error(f"Error getting history: {e}")
            return []

    def query_rag(self, chat_id, message, features=None):
        payload = {
            "user_id": self.user_id,
            "chat_id": chat_id,
            "message": message,
            "features": features or {
                "use_agent": True,
                "use_kb": True,
                "use_websearch": False,
                "doc_chat_only": True
            }
        }
        try:
            resp = requests.post(f"{BASE_URL}/rag/query", json=payload, headers=self.headers)
            if resp.status_code == 200:
                return resp.json()
            # Return actual error details for debugging
            return {"body": json.dumps(f"Error {resp.status_code}: {resp.text}")}
        except Exception as e:
             return {"body": json.dumps(f"System Error: {e}")}
