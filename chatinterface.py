import random
import gradio as gr
import os
import shutil

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.document_loaders import TextLoader
from langchain_community.chat_models import ChatOllama
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_experimental.llms.ollama_functions import OllamaFunctions

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings

from dotenv import load_dotenv



load_dotenv(override=True)

graph = Neo4jGraph()
llm = ChatOllama(model = "llama3.1:8b", temperature=0)
llm_transformer = LLMGraphTransformer(llm=llm)
llm_functions = OllamaFunctions(model="llama3.1:8b", temperature=0)
GRAPH_DOCUMENTS = None
VECTOR_RETRIEVER = None
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

class Entities(BaseModel):
    """Identifying information about entities."""

    names: list[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )


entity_chain = prompt | llm_functions.with_structured_output(Entities)


files = []

def process_file(fileobj):
    path = "./docs/" + os.path.basename(fileobj)
    shutil.copyfile(fileobj.name, path)
    #File processing here
    loader = TextLoader(file_path = path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(separators = ["\n\n", "\n"], chunk_size=250, chunk_overlap=96)
    documents = text_splitter.split_documents(documents=docs)
    GRAPH_DOCUMENTS = llm_transformer.convert_to_graph_documents(documents)
    print(GRAPH_DOCUMENTS[0])
    graph.add_graph_documents(
        GRAPH_DOCUMENTS,
        baseEntityLabel=True,
        include_source=True
    )
    vector_index = Neo4jVector.from_existing_graph(
        OllamaEmbeddings(model="llama3.1:8b"),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    global VECTOR_RETRIEVER
    VECTOR_RETRIEVER = vector_index.as_retriever()

def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    if not words:
        return ""
    full_text_query = " AND ".join([f"{word}~2" for word in words])
    print(f"Generated Query: {full_text_query}")
    return full_text_query.strip()

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

def full_retriever(question: str):
    graph_data = graph_retriever(question)
    vector_data = [el.page_content for el in VECTOR_RETRIEVER.invoke(question)]
    final_data = f"""Graph data:
{graph_data}
vector data:
{"#Document ". join(vector_data)}
    """
    return final_data

def random_response(message, history):
    #Chat here
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
    return chain.invoke(input=message)

with gr.Blocks() as demo:
    chatInterface = gr.ChatInterface(random_response)
    gr.Interface(
        fn=process_file,
        inputs = [
            "file",
        ],
        outputs="text"
    )
demo.launch(server_name='0.0.0.0')

#Control Flow for GraphRAG LLM:
# Load into Gradio -> Detect File Insertion -> Generate Query
# -> Initialize LLM with Neo4j -> Ollama with GraphRAG into random_response