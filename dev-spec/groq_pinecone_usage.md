Integrating RAG with Groq in a couple lines of code.
Integrating proprietary data with the Groq API is very straightforward. The instructions below outline the steps to connect your own database to the Groq API via Python. To follow the steps in this post you will need the following:

Data stored in a Vector Database (this demo utilizes Pinecone).
Set up a free account with Pinecone and create an index on a free tier and follow this guide to download sample data and index it into Pinecone.
A Groq API key – get yours for free today at console.groq.com
1. Connect to your database

 
import pinecone

pinecone.init(
    api_key='xxxx',
    environment='xxxx'
)
pinecone_index = pinecone.Index('name-of-index')
2. Convert questions into a vector representation

Use an embedding model to convert questions into a vector representation. This blog does not focus on what an embedding model is. For more information, please check out this link.

from transformers import AutoModel

embedding_model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
user_query = "user query"
query_embeddings = embedding_model.encode(user_query).tolist()
3. Query your database

result = pinecone_index.query(
vector=query_embeddings,
top_k=5, #this is the number of results that are returned
include_values=False,
include_metadata=True
)
4. Add the retrieved information to the LLM system prompt. 

This provides information to the LLM about how to act and respond. 

The exact json fields will depend on how you structured your index. 

matched_info = ' '.join(item['metadata']['text'] for item in result['matches'])
sources = [item['metadata']['source'] for item in result['matches']]
context = f"Information: {matched_info} and the sources: {sources}"
sys_prompt = f"""
Instructions:
- Be helpful and answer questions concisely. If you don't know the answer, say 'I don't know'
- Utilize the context provided for accurate and specific information.
- Incorporate your preexisting knowledge to enhance the depth and relevance of your response.
- Cite your sources
Context: {context}
"""
5. Ask GroqAPI to answer your question

Export your Groq API key in your terminal i.g. export GROQ_SECRET_ACCESS_KEY=””
from groq.cloud.core import  Completion
with Completion() as completion:
        response, id, stats = completion.send_prompt("llama2-70b-4096", user_prompt=user_query, system_prompt=sys_prompt)