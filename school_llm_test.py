import os
from openai import OpenAI

api_key = os.getenv("SCHOOL_LLM_API_KEY")

if not api_key:
    raise RuntimeError("SCHOOL_LLM_API_KEY 환경변수가 없습니다.")

client = OpenAI(
    api_key=api_key,
    base_url="https://factchat-cloud.mindlogic.ai/v1/gateway",
)

response = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[
        {"role": "user", "content": "컵 찾아줘를 JSON으로 바꿔줘. 형식: {\"action\":\"find\",\"object\":\"cup\"}"}
    ],
)

print(response.choices[0].message.content)