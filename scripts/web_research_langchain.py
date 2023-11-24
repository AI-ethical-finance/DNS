!pip install langchain
!pip install openai
!pip install chromadb
!pip install tiktoken

import inspect
import re

from getpass import getpass
from langchain import OpenAI, PromptTemplate
from langchain.chains import LLMChain, LLMMathChain, TransformChain, SequentialChain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI

OPENAI_API_KEY = getpass()

#llm = OpenAI(
#    temperature=0, model_name='gpt-4',
#    openai_api_key=OPENAI_API_KEY
#    )

chat_model = ChatOpenAI(
    temperature=0, model_name='gpt-4',
    openai_api_key=OPENAI_API_KEY
)

chat_model.predict("hi!")

from langchain.retrievers.web_research import WebResearchRetriever

import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper

# Vectorstore
vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), persist_directory="./chroma_db_oai"
)

# LLM
llm = ChatOpenAI(model_name='gpt-4', temperature=0, openai_api_key=OPENAI_API_KEY)

# Search
os.environ["GOOGLE_CSE_ID"] = ""
os.environ["GOOGLE_API_KEY"] = ""
search = GoogleSearchAPIWrapper()

# Initialize
web_research_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore,
    llm=llm,
    search=search,
)

!pip install html2text
from langchain.chains import RetrievalQAWithSourcesChain

user_input = "Successful debt-for-nature swaps in Mexico"
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm, retriever=web_research_retriever
)
result = qa_chain({"question": user_input})
result

# Run
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)
user_input = "Successful debt-for-nature swaps in Colombia"
docs = web_research_retriever.get_relevant_documents(user_input)

from langchain.chains.question_answering import load_qa_chain

chain = load_qa_chain(llm, chain_type="refine")
output = chain(
    {"input_documents": docs, "question": user_input}, return_only_outputs=True
)
output["output_text"]

import os
import re
from typing import List
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers.pydantic import PydanticOutputParser

# LLMChain
search_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are an assistant tasked with improving Google search
    results. Generate FIVE Google search queries that are similar to
    this question. The output should be a numbered list of questions and each
    should have a question mark at the end: {question}""",
)


class LineList(BaseModel):
    """List of questions."""

    lines: List[str] = Field(description="Questions")


class QuestionListOutputParser(PydanticOutputParser):
    """Output parser for a list of numbered questions."""

    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = re.findall(r"\d+\..*?\n", text)
        return LineList(lines=lines)


llm_chain = LLMChain(
    llm=llm,
    prompt=search_prompt,
    output_parser=QuestionListOutputParser(),
)

# Initialize
web_research_retriever_llm_chain = WebResearchRetriever(
    vectorstore=vectorstore,
    llm_chain=llm_chain,
    search=search,
)

# Run
docs = web_research_retriever_llm_chain.get_relevant_documents(user_input)

docs









