from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from google.cloud import bigquery

# Initialize the FastAPI app
app = FastAPI()

class QARetrieval:
    def __init__(self, qa_file_path):
        # Initialize the SentenceTransformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load the Q&A pairs from an Excel file
        self.qa_df = pd.read_excel(qa_file_path)
        
        # Generate embeddings for all questions
        self.question_embeddings = self.model.encode(self.qa_df['Question'].tolist())
        
        # Set up FAISS index for similarity search
        self.dimension = self.question_embeddings.shape[1]  # Embedding dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(self.question_embeddings).astype('float32'))  # Add embeddings to FAISS index
        
        # Initialize BigQuery client
        self.bq_client = bigquery.Client()

    def get_answer(self, query, k=1, similarity_threshold=0.99):
        # Generate embedding for the user query
        query_embedding = self.model.encode([query])
        
        # Search for similar questions in FAISS index
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        # Normalize distances to calculate similarity
        similarities = 1 - distances[0] / 100
        
        # Filter results by similarity threshold
        results = []
        for idx, similarity in zip(indices[0], similarities):
            if similarity >= similarity_threshold:
                row = self.qa_df.iloc[idx]
                results.append({
                    'question': row['Question'],
                    'answer': row['Answer'],
                    'table_query': row.get('Table', None),
                    'similarity': similarity
                })
        return results

    def run_sql_query(self, sql_query):
        # Run the SQL query using BigQuery
        try:
            query_job = self.bq_client.query(sql_query)
            df = query_job.to_dataframe()
            # Clean the DataFrame to replace NaN, inf, and -inf with None
            df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
            return df
        except Exception as e:
            print(f"Error executing SQL query: {e}")
            return None

    def process_query(self, query, k=1, similarity_threshold=0.99):
        # Perform similarity search
        results = self.get_answer(query, k, similarity_threshold)
        if not results:
            return {"error": "I couldn't locate any details for your request. Is there something else I can help with?"}
        
        responses = []
        for result in results:
            if result['answer'].strip().upper().startswith("SELECT"):
                # Execute the SQL query in the "Answer" column
                sql_query = result['answer']
                result_df = self.run_sql_query(sql_query)
                if result_df is not None:
                    responses.append({"query": query, "result": result_df.to_dict(orient="records")})
                else:
                    responses.append({"query": query, "error": "Failed to execute the SQL query."})
            elif "table" in query.lower():
                # Handle "table" requests by using the "Table" column
                sql_query = result['table_query']
                if sql_query:
                    result_df = self.run_sql_query(sql_query)
                    if result_df is not None:
                        responses.append({"query": query, "result": result_df.to_dict(orient="records")})
                    else:
                        responses.append({"query": query, "error": "Failed to execute the SQL query."})
                else:
                    responses.append({"query": query, "error": "No SQL query provided in the 'Table' column."})
            else:
                # Handle other cases where the answer is plain text
                responses.append({"query": query, "answer": result['answer']})
        
        return responses


# Initialize the QA system with your Excel file
qa_system = QARetrieval('Q&A filec.xlsx')


class QueryRequest(BaseModel):
    query: str
    k: int = 1
    similarity_threshold: float = 0.99


@app.post("/process_query/")
def process_query(request: QueryRequest):
    try:
        results = qa_system.process_query(request.query, request.k, request.similarity_threshold)
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
