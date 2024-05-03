from langchain_community.vectorstores import Pinecone
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from pinecone import Pinecone as PineconeClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
import os
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.chains import LLMChain
import config

def data_loader(path):
    doc = PyPDFDirectoryLoader(path)
    data = doc.load()
    return data

dir_path = 'docs/'

#data = data_loader(dir_path)


os.environ['PINECONE_API_KEY'] = config.PINECONE_API_KEY
os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY

def data_splitter(docs, chunk_size = 1000, chunk_overlap = 100):
    split = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap=chunk_overlap)
    doc_chunks = split.split_documents(docs)
    return doc_chunks

#doc_chunks = data_splitter(data)

embeddings = OpenAIEmbeddings()

PC_key = os.getenv('PINECONE_API_KEY')
PineconeClient(api_key = PC_key, index_api='portfolio' )
index_name = 'portfolio'
#index = Pinecone.from_documents(doc_chunks, embeddings, index_name=index_name)
index = PineconeVectorStore(index_name=index_name, embedding=embeddings)

def get_similar_docs(query, k=2):
    similar_docs = index.similarity_search(query, k = k)
    return similar_docs

llm =ChatOpenAI(model_name = 'gpt-3.5-turbo-0125', temperature=0.9)
chain = load_qa_chain(llm, chain_type='stuff')

def get_answer(query):
    relevant_docs = get_similar_docs(query)
    #print(relevant_docs)
    rag_response = chain.run(input_documents = relevant_docs,  question = query)
    print(rag_response)
    examples = [
    {
        "query": "Hi, What is your name",
        "answer": "Hi there, I am Subot, personal assistant of Subramanyam, how can I help you?" 
    }, {
        "query": "Hi, Good Morning",
        "answer": "Hello, Very Good Morning, How can I help you?"
    }, {
        "query": "What can you do for me?",
        "answer": "I am a support assistant designed to help you with queries regarding Subramanyam"
    }, {
        "query": "Tell me about his experience in PL/SQL",
        "answer": "I have worked for LiquidHub in December 2018, gained immense knowledgw on ETL, handiling large data, ACID properties, triggers, etc "
    }, {
        "query": "How many years he worked for capgemini?",
        "answer": "I worked for capgemini from 2019 until Sepemer 2022 which is almost 3.5 years and has acheieved awards and immense grip on AI."
    }, {
        "query": "I am sleeping goodnight!",
        "answer": "I hope you have got the information you were looking for, bye good night and take care."
    }
    ]


    example_template = '''
    Questions = {query},
    Answer = {answer}
    '''

    example_prompt = PromptTemplate(
    input_variables=['query','answer'],
    template = example_template
    )
    example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=200
    )
    prefix = """Always remember You are personal assistant of sri phani subramanyam, tell his experience like you are his digital assistant
    """
    suffix = """
    Question: {userInput}
    Response: """

    new_prompt_template = FewShotPromptTemplate(
    example_selector = example_selector,
    example_prompt = example_prompt,
    prefix = prefix,
    suffix = suffix,
    input_variables = ['userInput'],
    example_separator = '\n\n'
    )
    llm_gen = llm = ChatOpenAI(temperature=0.9)
    
    output_res = LLMChain(llm = llm_gen, prompt = new_prompt_template)
    final_res = output_res.invoke(rag_response)
    print(new_prompt_template)
    return final_res['text']


