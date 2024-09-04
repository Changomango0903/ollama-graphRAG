from langchain_core.runnables import  RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
import os
from os import listdir
from os.path import isfile, join
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

from dotenv import load_dotenv

load_dotenv(override=True)
# def combine_texts():
#     data_path = "../docs"
#     files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
#     for name in files:
#         print(name)
#         with open(data_path + "/" + name, 'r') as firstfile, open("./info.txt", 'a') as secondfile:
#             for line in firstfile:
#                 secondfile.write(line)
#             secondfile.write('\n')
graph = Neo4jGraph()

loader = TextLoader(file_path="./info.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=24)
documents = text_splitter.split_documents(documents=docs)

llm = ChatOllama(model="llama3.1", temperature=0)
llm_transformer = LLMGraphTransformer(llm=llm)

graph_documents = llm_transformer.convert_to_graph_documents(documents)