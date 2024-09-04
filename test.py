from RAG.VectorBase import VectorStore
from RAG.utils import ReadFiles
from RAG.LLM import OpenAIChat, InternLMChat
from RAG.Reranker import BgeReranker
from RAG.Embeddings import BgeEmbedding
embedding = BgeEmbedding(path='models/bge-m3')
reranker=BgeReranker(path='models/bge-reranker-base')


files = ReadFiles(path='./documents')
vector = VectorStore(files.get_content())
#print(vector.get_vector(EmbeddingModel=embedding))
vector.get_vector(EmbeddingModel=embedding)
vector.persist('storages')
vector.load_vector('./storages') # 加载本地的数据库
question = ' How Does the Reasoning Program Help?'
content = vector.query(question, EmbeddingModel=embedding, k=20)
content=reranker.rerank(question, content, 10)
content=' '.join(content)
chat = OpenAIChat(model='gpt-3.5-turbo-1106')
print(chat.chat(question, [], content))