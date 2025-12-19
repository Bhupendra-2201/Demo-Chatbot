import boto3
import sys

# --- Configuration ---
VECTOR_BUCKET_NAME = 'my-vector-bucket'
INDEX_NAME = 'document-embeddings'
AWS_REGION = 'us-east-1'

# Maximum number of keys the DeleteVectors API can accept in one call
BATCH_SIZE = 500

def delete_all_vectors_in_index(bucket_name, index_name, region):
    """
    Lists all vectors in a specified S3 vector index and deletes them in batches.
    """
    print(f"Attempting to delete all vectors from index '{index_name}' in bucket '{bucket_name}' in region {region}...")
    
    try:
        # Create an s3vectors client
        s3vectors_client = boto3.client('s3vectors', region_name=region)
        
        # Use a paginator to handle potentially millions of vectors
        paginator = s3vectors_client.get_paginator('list_vectors')
        
        # The returnData and returnMetadata parameters can be set to False to optimize the listing process
        # as we only need the keys for deletion.
        pagination_config = {'PageSize': 500}
        pages = paginator.paginate(
            vectorBucketName=bucket_name,
            indexName=index_name,
            returnData=False,
            returnMetadata=False,
            PaginationConfig=pagination_config
        )
        
        keys_to_delete = []
        total_deleted_count = 0

        for page in pages:
            if 'vectors' in page:
                for vector in page['vectors']:
                    keys_to_delete.append(vector['key'])
            
            # If we reach the batch size limit, delete the collected keys
            if len(keys_to_delete) >= BATCH_SIZE:
                delete_batch(s3vectors_client, bucket_name, index_name, keys_to_delete)
                total_deleted_count += len(keys_to_delete)
                keys_to_delete = []
                print(f"Total vectors deleted so far: {total_deleted_count}")

        # Delete any remaining keys in the last batch
        if keys_to_delete:
            delete_batch(s3vectors_client, bucket_name, index_name, keys_to_delete)
            total_deleted_count += len(keys_to_delete)

        print(f"\nSuccessfully deleted all vectors from index '{index_name}'. Total vectors removed: {total_deleted_count}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Deletion process may be incomplete. Please check permissions and configuration.")
        sys.exit(1)

def delete_batch(client, bucket_name, index_name, keys):
    """Helper function to perform the batch delete operation."""
    try:
        client.delete_vectors(
            vectorBucketName=bucket_name,
            indexName=index_name,
            keys=keys
        )
        print(f"Deleted batch of {len(keys)} vectors.")
    except Exception as e:
        # In a production script, you might want more robust error handling and retries
        print(f"Error deleting batch: {e}")
        raise

if __name__ == "__main__":
    # !!! IMPORTANT: Replace placeholder values below with your actual details !!!
    # Example: 
    # VECTOR_BUCKET_NAME = 'my-rag-vector-bucket'
    # INDEX_NAME = 'document-index-v1'
    # AWS_REGION = 'us-east-1'

    if VECTOR_BUCKET_NAME == '<YOUR_VECTOR_BUCKET_NAME>' or INDEX_NAME == '<YOUR_INDEX_NAME>' or AWS_REGION == '<YOUR_AWS_REGION>':
        print("Error: Please update the script configuration variables (VECTOR_BUCKET_NAME, INDEX_NAME, AWS_REGION) with your actual values.")
        sys.exit(1)
        
    delete_all_vectors_in_index(VECTOR_BUCKET_NAME, INDEX_NAME, AWS_REGION)
