from langchain.chat_models import ChatOllama
import streamlit as st
from alquimia import ModelManager
from alquimia.aimodels.intent import OpenVINOIntentModel
from alquimia.aimodels.ner import OpenVINONERModel
from alquimia.connectors import HTTPClient
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
import json
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


def leviathan():
    st.title("Chat")
    prompt = st.chat_input("Write something here")
    with st.chat_message(name="Leviathan", avatar="🐙"):
        st.write("Hello, I'm Leviathan, alquimia methodologist thinker")
    if (prompt):
        with st.chat_message(name="You", avatar="👤"):
            st.write(prompt)
        answer = execute_leviathan(prompt)
        if (answer):
            with st.chat_message(name="Leviathan", avatar="🐙"):
                st.write(answer)


def execute_leviathan(query: str) -> str:
    sentiment_prompt = PromptTemplate.from_template(
        """
        You are an  AI Agent that has to analyze the sentiment of a statement. 
        The question is: {input}
        The schema json to answer has properties:
            language: spanish, english, french, german, italian
            aggressiveness: describes how aggressive the statement is, the higher the number the more aggressive.  Must be in the range [1,5]
            sentiment: describes the sentiment of the statement
            translation: the  question by the user in {language}
        """
    )
    sentiment_chain = sentiment_prompt | llm | StrOutputParser()
    output = sentiment_chain.invoke(
        {"input": query, "language": "english"})
    sentiment_output = json.loads(output)
    with st.chat_message(name="Thinkings", avatar="💭"):
        st.write(
            f"Detected language :red[{sentiment_output.get('language')}] with a sentiment of :green[{sentiment_output.get('sentiment')}] and an aggressiveness of {sentiment_output.get('aggressiveness')} scale")
    with st.chat_message(name="Openshift AI", avatar="https://www.svgrepo.com/show/354273/redhat-icon.svg"):
        st.write("Calling Intent Tool")
        # Service is down
        output_intent = alquimia.model("intent").infer(
            sentiment_output.get("translation"))
        intent = LABEL_MAP.get(output_intent[0].label)
        st.write(f"Intent: {intent}")
    decision_matrix = """
    intent|knowdlege_base| entity_extraction
    inventory | graph, true
    checkout |null, false
    irrelevant | null,false
    conversational | embedding,false
    postSale | graph,true
    """
    thinking_prompt = PromptTemplate.from_template(
        """
        You are an AI Agent tasked with receiving analyzed data from a previous sentiment analysis stage, also a decision matrix is provided. Your role is to reason on this data,take the intent type,reason the action_type, the knowdledge base and if entity extraction must be executed.
        The intent type is: {intent_type}
        User input: {input}
        Decision Matrix: {decision_matrix}

        The final answer must be in this JSON schema:
            knowledge_base: describes if the knowledge base must be used
                type: string (graph, embedding, null)
                shouldUse: boolean
            action_type: describes the action to be taken
                intent: string
                action: string (verb in infinitive)
            entity_extraction: True or false
        """
    )
    thinking_prompt = thinking_prompt.partial(
        intent_type=intent, decision_matrix=decision_matrix)
    thinking_chain = thinking_prompt | llm | StrOutputParser()
    output = thinking_chain.invoke(
        {"input": sentiment_output.get("translation")})
    output = json.loads(output)
    print(output)
    if output.get("entity_extraction") is True:
        with st.chat_message(name="Openshift AI", avatar="https://www.svgrepo.com/show/354273/redhat-icon.svg"):
            st.write("Calling NER Tool")
            # Service is down
            entities = alquimia.model(
                'ner').infer(query, label_map=NER_LABEL_MAP)
            st.write(
                {'entities': f"{'  '.join([e.toHuman() for e in entities])}"})
    final_answer = {**output, **sentiment_output}
    return final_answer


if option == "LLMs":
    llm_page()
elif option == "Tools":
    tools_page()
elif option == "Chat":
    leviathan()
