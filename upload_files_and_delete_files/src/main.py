import base64
import boto3
import uuid
import json
import os
import mimetypes

s3 = boto3.client('s3')
MAX_TOTAL_SIZE_MB = 15  
MAX_TOTAL_SIZE_BYTES = MAX_TOTAL_SIZE_MB * 1024 * 1024  

def lambda_handler(event, context):
    try:
        if 'body' in event and event['body']:
            body = json.loads(event['body'])
        else:
            body = event

        bucket_name = body.get('bucket_name')
        user_id = body.get('user_id')
        context_id = body.get('context_id')
        name = body.get('name', f"default-{uuid.uuid4()}")

        if not all([bucket_name, user_id, context_id, name]):
            return {
                "statusCode": 400,
                "body": json.dumps({"message": "Bucket name, user ID, context ID, and name are required."})
            }

        action = body.get('action', 'upload')  

        collection_prefix = os.path.join(user_id, context_id, name, "")

        if action == 'delete':
            return delete_collection(bucket_name, user_id, context_id)

        if 'files' in body and isinstance(body['files'], list):
            files = body['files']
        else:
            return {
                "statusCode": 400,
                "body": json.dumps({"message": "No files found to upload."})
            }

        total_size = sum(len(base64.b64decode(file_data['file_content'])) for file_data in files)

        if total_size > MAX_TOTAL_SIZE_BYTES:
            return {
                "statusCode": 400,
                "body": json.dumps({"message": f"Total file content exceeds {MAX_TOTAL_SIZE_MB} MB limit."})
            }

        uploaded_files = []
        for file_data in files:
            file_name = file_data['file_name']
            file_content = file_data['file_content']
            file_bytes = base64.b64decode(file_content)

            file_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"

            file_key = os.path.join(collection_prefix, file_name)

            s3.put_object(
                Bucket=bucket_name,
                Key=file_key,
                Body=file_bytes,
                ContentType=file_type
            )

            uploaded_files.append({
                "file_name": file_name,
                "file_key": file_key,
                "status": "uploaded"
            })

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Files uploaded successfully",
                "name": name,
                "files": uploaded_files
            })
        }

    except Exception as e:
        print(f"Error: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "message": str(e)
            })
        }

def delete_collection(bucket_name, user_id, context_id):
    try:
        # Define the prefix for the entire context, ignoring 'name' level
        collection_prefix = os.path.join(user_id, context_id, "")

        # List all objects under the specified prefix
        objects_to_delete = s3.list_objects_v2(Bucket=bucket_name, Prefix=collection_prefix)

        if 'Contents' in objects_to_delete:
            # Prepare list of objects to delete
            delete_keys = [{'Key': obj['Key']} for obj in objects_to_delete['Contents']]
            
            # Delete all objects under the prefix
            s3.delete_objects(Bucket=bucket_name, Delete={'Objects': delete_keys})

            return {
                "statusCode": 200,
                "body": json.dumps({"message": "All files in the context deleted successfully"})
            }
        else:
            return {
                "statusCode": 404,
                "body": json.dumps({"message": "No files found in the specified context"})
            }
        
    except Exception as e:
        print(f"Error during deletion: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"message": str(e)})
        }


