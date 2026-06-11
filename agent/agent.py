"""
Agent 메인 루프 - 두 모델 역할 분담 (완전 무료 로컬)
- llama3.1:8b  : tool calling 담당 (어떤 도구를 쓸지 결정 + 실행)
- exaone3.5:7.8b: 한국어 응답 생성 담당 (자연스러운 한국어 설명)
- Role A의 call_tool()로 실제 통계 모델 실행 (Python 함수, API 없음)
"""

import json
import re
import uuid
from openai import OpenAI
from agent.tools import TOOLS, call_tool
from agent.prompts import get_system_prompt

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

MODEL_TOOL  = "llama3.1:8b"     # tool calling 담당
MODEL_REPLY = "exaone3.5:7.8b"  # 한국어 응답 생성 담당


_OUT_OF_SCOPE_KEYWORDS = [
    "대출", "보험", "세금", "세무", "부동산", "주식 추천", "코인", "암호화폐",
    "의료", "법률", "이혼", "상속", "증여", "취업", "창업",
]
_OUT_OF_SCOPE_REPLY = (
    "대출·보험·세무 등은 이 서비스에서 제공하지 않습니다. "
    "자산 수명 시뮬레이션, 연금 전략, 또래 비교, 이상 거래 탐지 중 궁금한 게 있으면 말씀해 주세요."
)


def run_agent(user_message: str, conversation_history: list, age: int = 65) -> tuple[str, list]:
    """
    사용자 메시지를 받아 Agent 응답 반환.
    1) llama3.1 이 tool 을 결정·호출
    2) tool 결과가 있으면 exaone3.5 가 자연스러운 한국어로 설명
    """
    # 범위 밖 키워드 사전 필터
    clean_msg = user_message.split("]")[-1] if "]" in user_message else user_message
    if any(kw in clean_msg for kw in _OUT_OF_SCOPE_KEYWORDS):
        history = conversation_history.copy()
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": _OUT_OF_SCOPE_REPLY})
        return _OUT_OF_SCOPE_REPLY, history

    # 범위 밖 키워드 사전 필터
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
    tool_results: list[tuple[str, dict]] = []  # (tool_name, result)

    for _ in range(5):
        response = client.chat.completions.create(
            model=MODEL_TOOL,
            messages=[{"role": "system", "content": system_prompt}] + history,
            tools=openai_tools,
            tool_choice="auto",
        )

        message = response.choices[0].message

        if not message.tool_calls:
            content = message.content or ""

            # fallback 파싱: XML 태그 또는 bare JSON 형식 tool call 감지
            xml_calls = _parse_xml_tool_calls(content) or _parse_bare_json_tool_calls(content)
            valid_xml = [c for c in xml_calls if "name" in c] if xml_calls else []
            if valid_xml:
                clean_content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL).strip()
                history.append({
                    "role": "assistant",
                    "content": clean_content or None,
                    "tool_calls": [
                        {"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                         "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments", c.get("parameters", {})))}}
                        for c in valid_xml
                    ],
                })
                for c in valid_xml:
                    args = c.get("arguments", c.get("parameters", {}))
                    result = handle_tool_call(c["name"], args if isinstance(args, dict) else json.loads(args))
                    tool_results.append((c["name"], result))
                    history.append({
                        "role": "tool",
                        "tool_call_id": f"call_{uuid.uuid4().hex[:8]}",
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                continue

            # tool 호출이 있었으면 exaone 으로 한국어 설명 생성
            if tool_results:
                reply = _korean_reply(user_message, tool_results, age)
            else:
                reply = "대출, 보험, 세무 등 이 서비스 범위 밖의 질문입니다. 자산 수명 시뮬레이션, 연금 전략, 또래 비교, 이상 거래 탐지 중 궁금한 게 있으면 말씀해 주세요."

            history.append({"role": "assistant", "content": reply})
            return reply, history

        # tool_calls 처리
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

    # 최대 루프 초과 — 결과가 있으면 exaone 으로 마무리
    reply = _korean_reply(user_message, tool_results, age) if tool_results else _korean_short_reply(user_message, age)
    history.append({"role": "assistant", "content": reply})
    return reply, history


_TOOL_LABEL = {
    "run_survival":      "생존 분석",
    "run_monte_carlo":   "자산 수명 시뮬레이션",
    "run_pension":       "연금 최적화",
    "run_portfolio":     "포트폴리오 최적화",
    "run_clustering":    "또래 비교 분석",
    "run_anomaly_score": "이상 거래 탐지",
}

_TOOL_NAMES_RE = re.compile(
    r'\b(run_survival|run_monte_carlo|run_pension|run_portfolio'
    r'|run_clustering|run_anomaly_score)\b',
    re.IGNORECASE,
)

def _korean_short_reply(user_message: str, age: int) -> str:
    """도구 호출 없이 답변할 때 — 1~2문장으로만"""
    system_prompt = get_system_prompt(age)
    prompt = (
        f"사용자 질문: {user_message}\n\n"
        "이 서비스는 자산 수명 시뮬레이션, 연금 전략, 또래 비교, 이상 거래 탐지를 제공합니다.\n"
        "질문이 서비스 범위 밖이면 한 문장으로 간단히 안내하고 끝내세요.\n"
        "절대 재무 분석·조언을 추가하지 마세요. 1~2문장만 작성하세요."
    )
    response = client.chat.completions.create(
        model=MODEL_REPLY,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    for en, ko in _TOOL_LABEL.items():
        raw = re.sub(rf'\b{en}\b', f'"{ko}"', raw, flags=re.IGNORECASE)
    return raw


def _korean_reply(user_message: str, tool_results: list[tuple[str, dict]], age: int) -> str:
    """exaone3.5 로 tool 결과를 자연스러운 한국어 설명으로 변환"""
    system_prompt = get_system_prompt(age)

    results_text = ""
    for name, result in tool_results:
        label = _TOOL_LABEL.get(name, name)
        results_text += f"\n[{label} 결과]\n{json.dumps(result, ensure_ascii=False, indent=2)}\n"

    prompt = (
        f"사용자 질문: {user_message}\n\n"
        f"통계 모델이 계산한 결과:{results_text}\n"
        "위 계산 결과를 바탕으로 사용자에게 자연스럽고 따뜻한 한국어로 설명해 주세요.\n"
        "전문 용어는 쉬운 말로 바꾸고, 가장 중요한 수치를 첫 문장에 제시하세요.\n"
        "전체 3~5문장으로 간결하게 작성하세요.\n"
        "절대로 'run_portfolio', 'run_monte_carlo' 같은 영문 함수명을 응답에 쓰지 마세요."
    )

    response = client.chat.completions.create(
        model=MODEL_REPLY,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    # 혹시 남아 있는 tool 이름 한국어로 대체
    for en, ko in _TOOL_LABEL.items():
        raw = re.sub(rf'\b{en}\b', f'"{ko}"', raw, flags=re.IGNORECASE)
    return raw


def handle_tool_call(tool_name: str, tool_input: dict) -> dict:
    """Tool 이름 → Role A의 실제 통계 모델 실행 → 결과 dict 반환"""
    try:
        return call_tool(tool_name, tool_input)
    except Exception as e:
        return {"error": str(e)}


def _parse_xml_tool_calls(text: str) -> list[dict] | None:
    """<tool_call>...</tool_call> 형식 파싱"""
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


def _parse_bare_json_tool_calls(text: str) -> list[dict] | None:
    """태그 없이 bare JSON {"name": ..., "parameters": ...} 형식 파싱
    llama3.1이 Python 불리언(True/False/None)을 그대로 출력하는 경우도 처리"""
    def _normalize(s: str) -> str:
        s = re.sub(r'\bTrue\b',  'true',  s)
        s = re.sub(r'\bFalse\b', 'false', s)
        s = re.sub(r'\bNone\b',  'null',  s)
        return s

    def _try_parse(s: str):
        try:
            return json.loads(s)
        except Exception:
            try:
                return json.loads(_normalize(s))
            except Exception:
                return None

    # 전체가 JSON 한 덩어리인 경우
    obj = _try_parse(text.strip())
    if isinstance(obj, dict) and "name" in obj:
        return [obj]

    # 텍스트 중간에 섞여 있는 경우 — 중첩 중괄호까지 포함해서 찾기
    result = []
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                chunk = text[start:i+1]
                obj = _try_parse(chunk)
                if isinstance(obj, dict) and "name" in obj and ("parameters" in obj or "arguments" in obj):
                    result.append(obj)
                start = None
    return result if result else None


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
