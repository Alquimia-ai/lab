from langchain.chat_models import ChatOllama
import streamlit as st
from alquimia import ModelManager
from alquimia.aimodels.intent import OpenVINOIntentModel
from alquimia.aimodels.ner import OpenVINONERModel
from alquimia.connectors import HTTPClient
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents import AgentExecutor
from langchain.tools.render import render_text_description
from langchain.schema import StrOutputParser
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from typing import Optional, Type
import json
from typing import Type
import os
INTENT_API = os.environ.get("INTENT_API")
INTENT_API_TOKEN = os.environ.get("INTENT_API_TOKEN")
NER_API = os.environ.get("NER_API")
NER_API_TOKEN = os.environ.get("NER_API_TOKEN")


LABEL_MAP = {
    "LABEL_0": "inventory",
    "LABEL_1": "checkout",
    "LABEL_2": "irrelevant",
    "LABEL_3": "conversational",
    "LABEL_4": "feedback",
    "LABEL_5": "postSale"
}

NER_LABEL_MAP = [
    "O",
    "PRODUCT_QUANTITY",
    "PRODUCT_QUANTITY",
    "PRODUCT_TYPE",
    "PRODUCT_TYPE",
    "PRODUCT_CHARACTERISTICS",
    "PRODUCT_CHARACTERISTICS",
    "PRODUCT_BRAND",
    "PRODUCT_BRAND",
    "PRODUCT_GENDER",
    "PRODUCT_GENDER",
    "PRODUCT_SIZE",
    "PRODUCT_SIZE"
]


AI_MODELS = {
    'intent': {
        'class': OpenVINOIntentModel,
        'actions': [{
            'name': 'infer',
            'method': 'post',
        }],
        'connector': {
            'class': HTTPClient,
            'config': {
                'base_url': INTENT_API,
                'token': INTENT_API_TOKEN,
            }
        }
    },
    'ner': {
        'class': OpenVINONERModel,
        'actions': [{
            'name': 'infer',
            'method': 'post'
        }],
        'connector': {
            'class': HTTPClient,
            'config': {
                'base_url': NER_API,
                'token': NER_API_TOKEN
            },
        },
    },
}
alquimia = ModelManager.fromConfig(AI_MODELS)


class IntentInput(BaseModel):
    text: str = Field()


class IntentTool(BaseTool):
    name = "Intent Recognition tool"
    description = "Intent Model API.Use this tool every time you need to recognize the intent of the user"
    args_schema: Type[BaseModel] = IntentInput

    def _run(self, text: str, run_manager: Optional[CallbackManagerForToolRun] = None):
        """Use the tool."""
        print("Running intent tool")
        print(text)
        return "Inventory"

    def _arun(self, text: int, run_manager: Optional[AsyncCallbackManagerForToolRun] = None):
        print("Running intent tool")
        print(text)
        return "Inventory"


class NERTool(BaseTool):
    name = "NER tool"
    description = "use this tool every time you need to "
    args_schema: Type[BaseModel] = IntentInput

    def _run(self, text: str, run_manager: Optional[CallbackManagerForToolRun] = None):
        """Use the tool."""
        print("Running intent tool")
        print(text)
        return "Inventory"

    def _arun(self, text: int, run_manager: Optional[AsyncCallbackManagerForToolRun] = None):
        print("Running intent tool")
        print(text)
        return "Inventory"


tools = [IntentTool()]


with st.sidebar:
    option = st.sidebar.selectbox(
        "Configuration for leviathan",
        ("Chat", "LLMs", "Tools")
    )


def prompting_page():
    st.title("Prompts section")


def tools_page():
    st.title("Define here all your customs tools")


llm = ChatOllama(model="mistral:instruct")


def llm_page():
    st.title("LLMs section")


def chat():
    sentiment_prompt = PromptTemplate.from_template(
        """
        I am an AI Agent that has to analyze the sentiment of a statement. 
        The question is: {input}
        The schema json to answer has properties:
            language: spanish, english, french, german, italian

            aggressiveness: describes how aggressive the statement is, the higher the number the more aggressive.  Must be in the range [1,5]

            sentiment: describes the sentiment of the statement

            answer: the  question by the user in english
        """
    )
    prompt = PromptTemplate.from_template("""
    You are an AI Agent tasked with receiving analyzed data from a previous sentiment analysis stage. Your role is to reason on this data and interact with the available tools to determine the appropriate action type. You must construct a JSON response based on this analysis. Here's the schema for the response:
                                          
        language: the detected language (spanish, english, french, german, italian)
        aggressiveness: describes how aggressive the statement is, the higher the number the more aggressive.  Must be in the range [1,5]
        sentiment: describes the sentiment of the statement
        action_type: the action type/intent identified by the IntentTool
                                          
    These are some of the previous task completed regarding this question:
    {sentiment}
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
                                          
    Your approach should be:

    1. Receive the output from the sentiment analysis stage.
    2. Understand and reason about the language, sentiment, and aggressiveness of the statement.
    3. Use the 'IntentTool' to determine the action type/intent based on the analyzed data.
    4. Construct the JSON response with all the gathered information.

    Use the following format:

    Thought: Consider the sentiment analysis output: language, sentiment, and aggressiveness.
    Action: Determine the most suitable action using 'IntentTool'.
    Action Input: Use the sentiment analysis output and the question.
    Observation: Analyze the output from 'IntentTool'.
    Final Thought: With all the information, construct the final JSON response.
    Final Answer: the final question in the mentioned json format

    Proceed with the given sentiment data and tools at your disposal
                                          
    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """
                                          )
    prompt = prompt.partial(
        tools=render_text_description(tools),
    )
    llm_with_stop = llm.bind(stop=["\nObservation"])
    agent = (
        {
            "input": lambda x: print((json.loads(x["sentiment"])).get("answer")) or (json.loads(x["sentiment"])).get("answer"),
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
            "sentiment": lambda x: json.loads(x["sentiment"])
        }
        | prompt
        | llm_with_stop
        | ReActSingleInputOutputParser()
    )
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    chain = (
        {"sentiment": sentiment_prompt | llm | StrOutputParser()}
        | agent_executor
    )
    result = chain.invoke(
        {"input": "Estoy interesado en comprar una remera color rojo"})
    print(result.get("output"))
    st.title("Chat section")


with st.container():
    if option == "LLMs":
        llm_page()
    elif option == "Tools":
        tools_page()
    elif option == "Chat":
        chat()
