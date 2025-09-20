import os
import streamlit as st
from dotenv import load_dotenv

# ❶ .env から環境変数を読み込み
load_dotenv()

# ❷ LangChain（OpenAI 連携 & プロンプト）
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ❸ 選択した「専門家の役割」に応じてシステムメッセージを返す
def system_message_for(expert_role: str) -> str:
    presets = {
        "データサイエンティスト": (
            "あなたは一流のデータサイエンティストです。"
            "前提・仮説・検証手法・評価指標・限界を順序立てて説明し、"
            "必要なら簡潔な数式や疑似コードも示してください。"
            "専門用語は中高生にも伝わるよう短く補足してください。"
        ),
        "プロダクトマネージャー": (
            "あなたは熟練のプロダクトマネージャーです。"
            "ユーザ課題→解決案→成功指標(KPI)→リスク/代替案→次アクションの順で提案してください。"
            "箇条書きを基本に、過度に長文にしないでください。"
        ),
    }
    return presets.get(
        expert_role, "あなたは有能なアシスタントです。簡潔かつ正確に答えてください。"
    )


# ❹ ユーザー入力と役割を受け取り、LLM 応答を返す関数（※要件の関数）
def generate_response(user_text: str, expert_role: str) -> str:
    # モデルはコスト/速度のバランスが良い 4o-mini を既定に
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # 必要に応じて変更可
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message_for(expert_role)),
            ("human", "{input_text}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"input_text": user_text})


# ❺ Streamlit UI
st.set_page_config(
    page_title="Streamlit × LangChain LLMアプリ", page_icon=":robot_face:", layout="centered"
)

st.title(":robot_face: Streamlit × LangChain LLMアプリ")
st.caption(
    "1つの入力フォームとラジオボタンで役割を切り替え、LangChain を経由して OpenAI の LLM に質問します。"
)

with st.expander(":information_source: このアプリの概要と使い方", expanded=True):
    st.markdown(
        """
**概要**  
- 入力テキストを LangChain 経由で LLM に渡し、回答を表示します。  
- ラジオボタンで **専門家の役割** を選ぶと、回答の観点（システムメッセージ）が切り替わります。  

**使い方**  
1. 役割（データサイエンティスト / プロダクトマネージャー）を選択  
2. テキストを入力して **送信**  
3. 画面下に回答が表示されます  

**注意**  
- ローカルでは `.env` の `OPENAI_API_KEY` が読み込まれます。  
- Streamlit Community Cloud では **Secrets** に `OPENAI_API_KEY` を設定してください。  
        """
    )

# 役割の選択（ラジオ）
role = st.radio(
    "専門家の役割を選択してください：",
    options=["データサイエンティスト", "プロダクトマネージャー"],
    horizontal=True,
)

# 入力フォーム
with st.form(key="llm_form"):
    user_text = st.text_area(
        "入力テキスト",
        placeholder="ここに質問や要件、文章を入力してください（例：このA/Bテスト設計をレビューして）",
        height=150,
    )
    submitted = st.form_submit_button("送信")

# APIキー存在チェック（実行前に軽くガード）
api_key_exists = bool(
    os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
)

if submitted:
    if not user_text.strip():
        st.warning("入力テキストを入力してください。")
    elif not api_key_exists:
        st.error(
            "OpenAI API キーが見つかりませんでした。ローカルは `.env`、デプロイは Secrets に `OPENAI_API_KEY` を設定してください。"
        )
    else:
        with st.spinner("LLM に問い合わせています…"):
            try:
                answer = generate_response(user_text, role)
                st.markdown("### :brain: 回答")
                st.write(answer)
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
                st.info(
                    "API キー、モデル名、ネットワーク、依存関係バージョン（langchain / langchain-openai）をご確認ください。"
                )