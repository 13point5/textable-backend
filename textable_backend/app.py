from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import instructor
from openai import OpenAI
from langchain.prompts import (
    load_prompt,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field
from enum import Enum
from typing import List
from dotenv import load_dotenv
from google.cloud import translate_v2 as translate

load_dotenv()

translate_client = translate.Client()

instructor_client = instructor.from_openai(OpenAI())

app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


system_prompt = SystemMessagePromptTemplate(
    prompt=load_prompt("prompts/system_prompt.yaml")
)


class Message(BaseModel):
    id: str
    role: str
    content: str


class FeedbackGrade(str, Enum):
    none = "none"
    good = "good"
    great = "great"


class Feedback(BaseModel):
    grade: FeedbackGrade = Field(
        ...,
        description="The grade of the feedback. none means that there is room for improvement but we want to be nice, so if there is no french at all or the french used is really bad then the grade is none. good means not a lot of problems. great means that it was perfect.",
    )
    content: str = Field(
        ...,
        description="The corrected version of what the user tried to say so that they can learn.",
    )


class ChatRequest(BaseModel):
    messages: List[Message]


class ChatResponse(BaseModel):
    response: str
    feedback: Feedback


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    messages = [
        (
            HumanMessage(content=message.content)
            if message.role == "human"
            else AIMessage(content=message.content)
        )
        for message in body.messages
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            *messages,
        ]
    )

    llm = ChatOpenAI()

    chain = prompt | llm

    last_human_message = next(
        message for message in reversed(messages) if message.type == "human"
    )

    feedback = instructor_client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=Feedback,
        messages=[
            {
                "role": "system",
                "content": "You are an expert Bilingual French Tutor for English speakers. You need to analyse a learner who is trying to speak French but mostly speaks English. Provide a grade and the corrected version of what the learner tried to say.",
            },
            {"role": "user", "content": last_human_message.content},
        ],
    )

    ai_response = chain.invoke(input={})

    return ChatResponse(response=ai_response.content, feedback=feedback)


class TranslateTextRequest(BaseModel):
    text: str


@app.post("/translate")
def translate_text(body: TranslateTextRequest):
    result = translate_client.translate(body.text, target_language="en")

    return result
