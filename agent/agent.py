"""
Agent 메인 루프 - Ollama 로컬 LLM + Tool Use (완전 무료)
- OpenAI-compatible API로 Ollama 호출 (LLM)
- Role A의 call_tool()로 실제 통계 모델 실행 (Python 함수, API 없음)
- 다중 턴 대화 + Tool Use 루프
"""

import json
import re
import uuid
from openai import OpenAI
from agent.tools import TOOLS, call_tool
from agent.prompts import get_system_prompt

# Ollama OpenAI-compatible 엔드포인트 (로컬, 무료)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

MODEL = "qwen2.5:7b"


def run_agent(user_message: str, conversation_history: list, age: int = 65) -> tuple[str, list]:
    """
    사용자 메시지를 받아 Agent 응답 반환.
    Tool Use가 필요하면 자동으로 통계 모듈 호출 후 해석.

    Returns:
        (응답 텍스트, 업데이트된 대화 기록)
    """
    system_prompt = get_system_prompt(age)
    history = conversation_history.copy()
    history.append({"role": "user", "content": user_message + "\n\n(반드시 한국어로만 답변해주세요.)"})

    openai_tools = _to_openai_tools(TOOLS)

    for _ in range(5):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_prompt}] + history,
            tools=openai_tools,
            tool_choice="auto",
        )

        message = response.choices[0].message

        if not message.tool_calls:
            content = message.content or ""
            xml_calls = _parse_xml_tool_calls(content)
            if xml_calls:
                clean_content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL).strip()
                history.append({
                    "role": "assistant",
                    "content": clean_content or None,
                    "tool_calls": [
                        {"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                         "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments", c.get("parameters", {})))}}
                        for c in xml_calls
                    ],
                })
                for c in xml_calls:
                    args = c.get("arguments", c.get("parameters", {}))
                    result = handle_tool_call(c["name"], args if isinstance(args, dict) else json.loads(args))
                    history.append({
                        "role": "tool",
                        "tool_call_id": f"call_{uuid.uuid4().hex[:8]}",
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                continue

            reply = _force_korean(_strip_tool_artifacts(content))
            history.append({"role": "assistant", "content": reply})
            return reply, history

        history.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ],
        })
        for tool_call in message.tool_calls:
            result = handle_tool_call(
                tool_call.function.name,
                json.loads(tool_call.function.arguments),
            )
            history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

    return "계산을 완료했습니다. 결과를 확인해 주세요.", history


def handle_tool_call(tool_name: str, tool_input: dict) -> dict:
    """Tool 이름 → Role A의 실제 통계 모델 실행 → 결과 dict 반환"""
    try:
        return call_tool(tool_name, tool_input)
    except Exception as e:
        return {"error": str(e)}


def _parse_xml_tool_calls(text: str) -> list[dict] | None:
    """<tool_call>...</tool_call> 형식 파싱 (대소문자·공백 허용)"""
    matches = re.findall(r'<\s*tool_call\s*>(.*?)<\s*/\s*tool_call\s*>', text, re.DOTALL | re.IGNORECASE)
    if not matches:
        orphan = re.findall(r'(\{.*?\})\s*<\s*/\s*tool_call\s*>', text, re.DOTALL)
        matches = orphan if orphan else []
    result = []
    for m in matches:
        try:
            result.append(json.loads(m.strip()))
        except Exception:
            pass
    return result if result else None


def _strip_tool_artifacts(text: str) -> str:
    """최종 reply에서 <tool_call> 잔재 제거"""
    text = re.sub(r'<\s*tool_call\s*>.*?<\s*/\s*tool_call\s*>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<\s*/?tool_call\s*>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\{[^{}]*"name"\s*:\s*"[^"]+?".*?\}\s*<\s*/\s*tool_call\s*>', '', text, flags=re.DOTALL)
    return text.strip()


def _has_chinese(text: str) -> bool:
    return any("一" <= ch <= "鿿" or "㐀" <= ch <= "䶿" for ch in text)


def _force_korean(text: str) -> str:
    """중국어 문자 포함 시 한국어로 재번역 (최대 3회)"""
    if not _has_chinese(text):
        return text
    for _ in range(3):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a Korean translator. Output ONLY Korean. 반드시 한국어로만 번역하세요. 중국어 절대 금지."},
                {"role": "user", "content": text},
            ],
        )
        result = response.choices[0].message.content or text
        if not _has_chinese(result):
            return result
        text = result
    return re.sub(r'[一-鿿㐀-䶿]', '', text).strip()


def _to_openai_tools(tools: list) -> list:
    """Claude Tool Use 형식 → OpenAI/Ollama 형식 변환"""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]
