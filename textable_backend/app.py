import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import instructor
from openai import OpenAI
from langchain.prompts import (
    load_prompt,
    SystemMessagePromptTemplate,
)

from pydantic import BaseModel, Field
from enum import Enum
from typing import List
from dotenv import load_dotenv
from google.cloud import translate_v2 as translate

load_dotenv()

translate_client = translate.Client()

openai_client = OpenAI()

instructor_client = instructor.from_openai(openai_client)

app = FastAPI()

origins = ["http://localhost:5173", "https://textable.vercel.app"]

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


class AuthRequest(BaseModel):
    password: str


@app.post("/auth")
def check_auth(body: AuthRequest):
    if body.password == os.getenv("PASSWORD"):
        return {"status": "success"}
    else:
        return {"status": "error", "message": "Incorrect password."}


def load_bot_system_prompt(bot_name: str):
    return SystemMessagePromptTemplate(
        prompt=load_prompt(
            os.path.join(os.path.dirname(__file__), f"prompts/{bot_name}_prompt.yaml")
        )
    )


redbot_prompt = load_bot_system_prompt("redbot")
greenbot_prompt = load_bot_system_prompt("greenbot")
purplebot_prompt = load_bot_system_prompt("purplebot")


def get_bot_system_prompt(bot_name: str):
    if bot_name == "redbot":
        return redbot_prompt
    elif bot_name == "greenbot":
        return greenbot_prompt
    elif bot_name == "purplebot":
        return purplebot_prompt
    else:
        raise ValueError(f"Unknown bot name: {bot_name}")


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
    perfect: bool = Field(
        ..., description="Whether the user's response was perfect or not."
    )


class MessageContent(BaseModel):
    text: str
    images: List[str]


class Message(BaseModel):
    id: str
    role: str
    content: MessageContent


class ChatRequest(BaseModel):
    roomId: str
    messages: List[Message]


class ChatResponse(BaseModel):
    response: str
    feedback: Feedback


def format_message_content(content: MessageContent):
    formatted_content = [{"type": "text", "text": content.text}]

    for img in content.images:
        formatted_content.append({"type": "image_url", "image_url": img})

    return formatted_content


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):

    bot_system_prompt = get_bot_system_prompt(body.roomId)

    messages = [{"role": "system", "content": bot_system_prompt.prompt.template}]

    for message in body.messages:
        messages.append(
            {
                "role": message.role,
                "content": format_message_content(message.content),
            }
        )

    bot_response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
    )

    last_human_message = next(
        message for message in reversed(messages) if message["role"] == "user"
    )

    feedback = instructor_client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=Feedback,
        messages=[
            {
                "role": "system",
                "content": "You are an expert Bilingual French Tutor for English speakers. You need to analyse a learner who is trying to speak French but mostly speaks English. Provide a grade and the corrected version of what the learner tried to say.",
            },
            {"role": "user", "content": last_human_message["content"][0]["text"]},
        ],
    )

    return ChatResponse(
        response=bot_response.choices[0].message.content, feedback=feedback
    )


class TranslateTextRequest(BaseModel):
    text: str


@app.post("/translate")
def translate_text(body: TranslateTextRequest):
    result = translate_client.translate(body.text, target_language="en")

    return result
