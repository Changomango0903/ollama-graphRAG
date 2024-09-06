RAG vs GraphRAG
    The main difference between traditional RAGs (Vector Databases) and GraphRAGs are the representation of data and how the LLM understands each approach. In RAG, a similarity score is set to identify content that may or may not be related to the query being asked. On the other hand, GraphRAGs preserve relations which allows for more complex usecases which may involve structured knowledge along with intricate relationships.
What makes a GraphRAG work?
    - Quality of Data
        Data that is given to the Generation Model needs to be organized in a way that is easy for the LLM to understand. Convoluded and unformatted documents may cause the LLM to have difficulties in parsing and creating Node Schemas for insertion into the Graph model.
    - Generation Model
        This is the most important aspect of GraphRAGs. Utilizing a LLM to convert documents into Nodes to insert into GraphRAG models. There are a few aspects to consider when generating Nodes.
            1. Are the Relations + Types representative of the data?
                - Oftentimes, LLMs generate queries and nodes that are considered "trash" as they either contribute little to the overall dataset (loosely connected) or don't make the connection to other nodes that you'd expect to have connected.
            2. How are you chunking + overlapping the document?
                - In order to not use up all the tokens, it's necessary for you to chunk the text up so the LLM can provide you with the node schemas. However by chunking the document, you expose the GraphRAG to potential issues such as Ambiguiation where 2 nodes are created off of the same Concept/Entity. Ways to solve this ambiguity includes a second pass of the documents by an LLM, larger chunks, LLM with larger parameters, etc.
            3. LLM Restrictions
Remedies to the GraphRAG deficiencies
    There are some proposed ways to remedy the defects that GraphRAG presents. The most notable one being a hybrid search combining both RAG and GraphRAG methodologies. By either starting off with a Vector DB or a Graph DB and extending to the other, a multi-faceted approach to RAG could be achieved with varying performance. The most common combination of HybridRAG would be Vector + Graphs. 