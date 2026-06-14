"""
Agent 메인 루프 - Groq API (llama-3.3-70b-versatile)
- Tool calling + 한국어 응답 생성
- Role A의 call_tool()로 실제 통계 모델 실행 (Python 함수, API 없음)
"""

import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv
from agent.tools import TOOLS, call_tool
from agent.prompts import get_system_prompt

load_dotenv()

def _get_api_key() -> str:
    # Streamlit Cloud secrets 우선, 없으면 .env 사용
    try:
        import streamlit as st
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY", "")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=_get_api_key(),
)

MODEL = "llama-3.1-8b-instant"

_OUT_OF_SCOPE_KEYWORDS = [
    "대출", "보험", "세금", "세무", "부동산", "주식 추천", "코인", "암호화폐",
    "의료", "법률", "이혼", "상속", "증여", "취업", "창업",
]
_OUT_OF_SCOPE_REPLY = (
    "대출·보험·세무 등은 이 서비스에서 제공하지 않습니다. "
    "자산 수명 시뮬레이션, 연금 전략, 또래 비교, 이상 거래 탐지 중 궁금한 게 있으면 말씀해 주세요."
)

_TOOL_LABEL = {
    "run_survival":      "생존 분석",
    "run_monte_carlo":   "자산 수명 시뮬레이션",
    "run_pension":       "연금 최적화",
    "run_portfolio":     "포트폴리오 최적화",
    "run_clustering":    "또래 비교 분석",
    "run_anomaly_score": "이상 거래 탐지",
}


def run_agent(user_message: str, conversation_history: list, age: int = 65) -> tuple[str, list]:
    """사용자 메시지를 받아 Agent 응답 반환."""
    clean_msg = user_message.split("]")[-1] if "]" in user_message else user_message
    if any(kw in clean_msg for kw in _OUT_OF_SCOPE_KEYWORDS):
        history = conversation_history.copy()
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": _OUT_OF_SCOPE_REPLY})
        return _OUT_OF_SCOPE_REPLY, history

    system_prompt = get_system_prompt(age)
    history = conversation_history.copy()
    history.append({"role": "user", "content": user_message})

    openai_tools = _to_openai_tools(TOOLS)
    tool_results: list[tuple[str, dict]] = []

    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system_prompt}] + history,
                tools=openai_tools,
                tool_choice="auto",
            )
        except Exception as e:
            err = str(e)
            print(f"[Agent API error] {type(e).__name__}: {err}")
            if "rate_limit" in err.lower() or "429" in err or "too many" in err.lower():
                import re as _re
                wait = _re.search(r'in (\d+m\d+)', err)
                wait_str = wait.group(1) if wait else "잠시"
                reply = f"요청이 너무 많아 잠시 후 다시 시도해 주세요. ({wait_str} 후 이용 가능합니다)"
            elif "401" in err or "authentication" in err.lower() or "invalid api key" in err.lower():
                reply = "API 키 오류가 발생했습니다. 서비스 관리자에게 문의해 주세요."
            elif "404" in err or "model" in err.lower() and "not found" in err.lower():
                reply = "AI 모델을 불러오지 못했습니다. 잠시 후 다시 시도해 주세요."
            else:
                reply = f"일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
            history.append({"role": "assistant", "content": reply})
            return reply, history

        message = response.choices[0].message

        if not message.tool_calls:
            reply = _korean_reply(user_message, tool_results, age) if tool_results else _korean_short_reply(user_message, age)
            history.append({"role": "assistant", "content": reply})
            return reply, history

        history.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in message.tool_calls
            ],
        })
        for tc in message.tool_calls:
            result = handle_tool_call(tc.function.name, json.loads(tc.function.arguments))
            tool_results.append((tc.function.name, result))
            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

    reply = _korean_reply(user_message, tool_results, age) if tool_results else _korean_short_reply(user_message, age)
    history.append({"role": "assistant", "content": reply})
    return reply, history


def _korean_short_reply(user_message: str, age: int) -> str:
    """Tool 호출 없이 범위 안내 또는 간단 응답"""
    system_prompt = get_system_prompt(age)
    prompt = (
        f"사용자 질문: {user_message}\n\n"
        "이 서비스는 자산 수명 시뮬레이션, 연금 전략, 또래 비교를 제공합니다.\n"
        "이상 거래 탐지 관련 질문이면 '버튼 모드의 이상 거래 탐지 폼을 이용해 주세요'라고 안내하세요.\n"
        "서비스 범위 밖이면 한 문장으로 안내하세요. 범위 안이면 어떤 기능을 쓰면 되는지 안내하세요.\n"
        "반드시 순수 한국어로만 작성하세요. 영어·한자·힌디어 등 다른 언어 문자는 절대 사용하지 마세요.\n"
        "1~2문장으로만 작성하세요."
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return _clean_tool_names(response.choices[0].message.content or "")


def _korean_reply(user_message: str, tool_results: list[tuple[str, dict]], age: int) -> str:
    """Tool 결과를 자연스러운 한국어로 설명"""
    system_prompt = get_system_prompt(age)
    results_text = ""
    for name, result in tool_results:
        label = _TOOL_LABEL.get(name, name)
        results_text += f"\n[{label} 결과]\n{json.dumps(result, ensure_ascii=False, indent=2)}\n"

    prompt = (
        f"사용자 질문: {user_message}\n\n"
        f"통계 모델이 계산한 결과:{results_text}\n"
        "위 결과를 바탕으로 자연스럽고 따뜻한 한국어로 설명해 주세요.\n"
        "전문 용어는 쉬운 말로 바꾸고, 가장 중요한 수치를 첫 문장에 제시하세요.\n"
        "3~5문장으로 간결하게 작성하세요.\n"
        "절대로 'run_portfolio', 'run_monte_carlo' 같은 영문 함수명을 쓰지 마세요.\n"
        "반드시 순수 한국어로만 작성하세요. 영어·한자·힌디어 등 다른 언어 문자는 절대 사용하지 마세요."
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return _clean_tool_names(response.choices[0].message.content or "")


_FOREIGN_RE = re.compile(
    r'[一-鿿'   # 한자 (중국어/일본어)
    r'぀-ゟ'    # 히라가나
    r'゠-ヿ'    # 가타카나
    r'ऀ-ॿ'    # 힌디어 (데바나가리)
    r'؀-ۿ'    # 아랍어
    r'Ѐ-ӿ]'   # 키릴 문자
)

def _clean_tool_names(text: str) -> str:
    for en, ko in _TOOL_LABEL.items():
        text = re.sub(rf'\b{en}\b', f'"{ko}"', text, flags=re.IGNORECASE)
    text = _FOREIGN_RE.sub('', text)
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def handle_tool_call(tool_name: str, tool_input: dict) -> dict:
    try:
        return call_tool(tool_name, tool_input)
    except Exception as e:
        return {"error": str(e)}


def _to_openai_tools(tools: list) -> list:
    """Claude Tool Use 형식 → OpenAI/Groq 형식 변환"""
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
