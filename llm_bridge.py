import json
from voice_find_school_llm import get_llm_response

def clean(text):
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return text


def parse_command(user_input):

    prompt = f"""
사용자 명령을 JSON으로 변환해라.

형식:
{{"action":"find","object":"cup"}}

입력:
{user_input}

JSON만 출력
"""

    result = get_llm_response(prompt)

    print("[LLM]", result)

    try:
        return json.loads(clean(result))
    except:
        return None
