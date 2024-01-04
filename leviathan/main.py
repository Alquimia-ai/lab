from langchain.chat_models import ChatOllama
import streamlit as st
from alquimia import ModelManager
from alquimia.aimodels.intent import OpenVINOIntentModel
from alquimia.aimodels.ner import OpenVINONERModel
from alquimia.connectors import HTTPClient
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from uuid import uuid4
import json
import os
import pandas as pd

INTENT_API = os.environ.get("INTENT_API")
INTENT_API_TOKEN = os.environ.get("INTENT_API_TOKEN")
NER_API = os.environ.get("NER_API")
NER_API_TOKEN = os.environ.get("NER_API_TOKEN")
prompt_file = open("./prompt.json", "r")
prompts = json.loads(prompt_file.read())
leviathan_prompts = prompts.get("leviathan")
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
        ("Chat", "Config")
    )


def config():
    st.title("Configuration for Leviathan")
    uploaded_files = st.file_uploader(
        "Decision matrix", accept_multiple_files=True, type=["csv"])
    if len(uploaded_files) > 1:
        st.write("Only one file is allowed")
        return
    if uploaded_files:
        # Read the CSV file
        df = pd.read_csv(uploaded_files[0], delimiter=";")

        # Get the column headers and join them with '|'
        header_string = '|'.join(df.columns.astype(str))

        # Initialize a list to hold each row as a string, starting with the header string
        string_rows = [header_string]

        # Iterate over DataFrame rows
        for index, row in df.iterrows():
            # Join the row values with '|', convert to string, and add to the list
            string_rows.append('|'.join(row.astype(str)))

        # Join all row strings with '\n' to form the final string
        final_string = '\n'.join(string_rows)

        def save():
            new_prompts = {"leviathan": {
                **prompts.get("leviathan"),
                "decision_matrix": final_string
            }}
            open("./prompt.json", "w").write(json.dumps(new_prompts))
            st.write(":green[Saved successfully!!] 🎉")

        st.button("Save", type="primary", on_click=save)
    st.subheader("Custom Prompts")
    sentiment_prompt = st.text_area(
        label="Sentiment Prompt",
        value=leviathan_prompts.get("sentiment_prompt")
    )
    thinking_prompt = st.text_area(
        label="Thinking Prompt",
        value=leviathan_prompts.get("thinking_prompt")
    )


llm = ChatOllama(model="mistral:instruct")


def execute_leviathan(query: str) -> str:
    st.write("Using session id: ", session_id)
    message_history = RedisChatMessageHistory(session_id=session_id)
    print(message_history.messages)
    sentiment_prompt = PromptTemplate.from_template(
        leviathan_prompts.get("sentiment_prompt")
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
    thinking_prompt = PromptTemplate.from_template(
        leviathan_prompts.get("thinking_prompt")
    )
    thinking_prompt = thinking_prompt.partial(
        intent_type=intent, decision_matrix=leviathan_prompts.get("decision_matrix"))
    thinking_chain = thinking_prompt | llm | StrOutputParser()
    output = thinking_chain.invoke(
        {"input": sentiment_output.get("translation")})
    output = json.loads(output)
    if output.get("entity_extraction") is True:
        with st.chat_message(name="Openshift AI", avatar="https://www.svgrepo.com/show/354273/redhat-icon.svg"):
            st.write("Calling NER Tool")
            # Service is down
            entities = alquimia.model(
                'ner').infer(sentiment_output.get("translation"), label_map=NER_LABEL_MAP)
            st.write(entities)
            # We now have the entities, I must now search for coincidences in the context
            entities_prompt = PromptTemplate.from_template(
                """
               You are an AI Agent tasked with accurately tracking and relating entities in a conversation. Use the entities identified by the NER model to check against our conversation history. Your role is to analyze the user's current question to determine if it refers to previously mentioned entities or introduces new ones. Pay special attention to shifts in conversation topics or focus.
                Conversation History:
                {conversation_history}
                User Input:
                {question}
                Entities:
                {entities}
                In your response, format the answer as a JSON object. The JSON should include:
                    entities: [
                            type: [type of entity],
                            value: [value of entity],
                            relevance: [indicate if the entity is 'current', 'previous', or 'irrelevant' based on the user's current question],
                            context: [description of how this entity is related to the current user input and the overall conversation history]
                    ]
                Focus on identifying the entities that are directly relevant to the current user question. Include 'previous' entities only if they are contextually linked to the current question. Avoid including entities that have become irrelevant to the current line of inquiry.
            """
            )
            entities_prompt = entities_prompt.partial(
                entities=f"{'  '.join([e.toHuman() for e in entities])}")
            entities_chain = entities_prompt | llm | StrOutputParser()
            entities_output = entities_chain.invoke(
                {"question": sentiment_output.get("translation"), "conversation_history": message_history.messages})
            entities_output = entities_output.replace("\\", "")
            # Temporally add the entities to the history (should pass to the consultant)
            message_history.add_user_message(
                sentiment_output.get("translation"))
            message_history.add_ai_message(entities_output)
            entities_output = json.loads(entities_output)
            final_answer = {**output, **entities_output}
            return final_answer

    final_answer = {**output, **sentiment_output}
    return final_answer


if option == "Config":
    config()
elif option == "Chat":
    session_id = uuid4()
    st.title("Chat")
    col1, col2 = st.columns([5, 1])
    with col1:
        st.text_input(
            "Session id:",
            value=session_id,
            key="placeholder",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        result = st.button("New", type="primary", use_container_width=True)
        if result:
            session_id = uuid4()

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
