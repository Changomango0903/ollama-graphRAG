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
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
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

loader = TextLoader(file_path="./dummytext.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=48)
documents = text_splitter.split_documents(documents=docs)

# llm = ChatOllama(model="llama3.1", temperature=0)
# llm_transformer = LLMGraphTransformer(llm=llm)

# graph_documents = llm_transformer.convert_to_graph_documents(documents)
# graph.add_graph_documents(
#     graph_documents,
#     baseEntityLabel=True,
#     include_source=True
# )
def showGraph():
    driver = GraphDatabase.driver(
        uri = os.environ["NEO4J_URI"],
        auth = (os.environ["NEO4J_USERNAME"],
                os.environ["NEO4J_PASSWORD"]))
    session = driver.session()
    widget = GraphWidget(graph = session.run("MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t").graph())
    widget.node_label_mapping = 'id'
    return widget

vector_index = Neo4jVector.from_existing_graph(
    OllamaEmbeddings(model='llama3.1:8b'),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)
vector_retriever = vector_index.as_retriever()

class Entities(BaseModel):
    """Identifying information about entities."""

    names: list[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting organizations, persons, entities, names, and significant terms from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)


llm = OllamaFunctions(model="llama3.1:8b", temperature=0)
entity_chain = prompt | llm.with_structured_output(Entities)
entity_chain.invoke({"question": "What is the USA?"}).names

def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    if not words:
        return ""
    full_text_query = " AND ".join([f"{word}~2" for word in words])
    print(f"Generated Query: {full_text_query}")
    return full_text_query.strip()


# Fulltext index query
def graph_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result

print(graph_retriever("What is the USA?"))

def full_retriever(question: str):
    graph_data = graph_retriever(question)
    vector_data = [el.page_content for el in vector_retriever.invoke(question)]
    final_data = f"""Graph data:
{graph_data}
vector data:
{"#Document ". join(vector_data)}
    """
    return final_data

template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
        {
            "context": full_retriever,
            "question": RunnablePassthrough(),
        }
    | prompt
    | llm
    | StrOutputParser()
)

print(chain.invoke(input="Tell me more about slavery"))