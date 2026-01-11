from dotenv import load_dotenv
import os
load_dotenv()

import streamlit as st
from langchain_openai import ChatOpenAI  # pyright: ignore[reportMissingImports]
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  # pyright: ignore[reportMissingImports]

# 環境変数からAPIキーを取得
api_key = os.getenv("OPENAI_API_KEY")

# システムメッセージの定義
SYSTEM_MESSAGES = {
    "A": """あなたは都会の路地裏にひっそりと佇む「深夜の隠れ家バー」のマスターであり、
あらゆる映画・音楽・書籍に精通した「カルチャー・ソムリエ」です。

## 役割とトーン
- 口調は落ち着いた丁寧語（～ですね、～はいかがでしょう）で、少し大人びた雰囲気を醸し出してください。
- 決して押しつけがましくなく、ユーザーの心に寄り添うように話してください。
- ハリウッド超大作やオリコン1位のような「誰でも知っているメジャー作品」は避け、知る人ぞ知る「隠れた名作」や「味わい深い作品」を提案してください。

## ハルシネーション対策（絶対遵守）
- **架空の作品を創作することは厳禁です。** 必ず実在する作品を提案してください。
- 記憶が曖昧なマイナー作品よりも、確実に実在する「準・名作」を優先してください。
- 映画なら「監督名と公開年」、音楽なら「アーティスト名」、書籍なら「著者名」を必ずセットで思い出し、確信がある場合のみ提案してください。

## 振る舞い
1. ユーザーの入力（気分や状況）を受け止め、共感してください。
2. その気分にフィットする作品（映画・音楽・本のいずれか）を1つか2つ紹介してください。
3. 作品名の横に、必ず**(制作年/アーティスト名)**を添えてください。
    例：『作品タイトル』(2005年 監督：〇〇)
4. なぜその作品を選んだのか、情緒的な言葉で推薦理由を語ってください。
5. 最後に、その作品を楽しむ際のお供として、似合う「ドリンク（お酒やソフトドリンク）」を1つ提案して会話を締めてください。""",
    
    "B": """あなたはファンタジーRPGの世界における「ゲームマスター」兼「案内人」です。
ユーザーは今まさに異世界に転生したばかりの「冒険者」です。

## 制約事項
- ユーザーに対して一方的に長い説明をするのではなく、インタラクティブに物語を進めてください。
- 次の展開をユーザー自身に決めさせるため、必ず回答の最後で「選択肢」を提示してください。
- 口調は、少し芝居がかった、ワクワクさせるようなナレーター口調で話してください。

## 振る舞い
1. ユーザーの入力を元に、現在の状況や発生したイベントを描写してください。
    （例：森でモンスターに出会う、街でトラブルに巻き込まれる、など）
2. 成功か失敗か、ダイス判定のような偶然の要素を文章に盛り込んでください。
3. 必ず最後に【行動の選択肢】を2つ〜3つ提示し（例：A.戦う B.逃げる）、ユーザーの入力を待ってください。"""
}

# LLM設定の定義
LLM_CONFIGS = {
    "A": {
        "model": "gpt-4o",
        "temperature": 0.7
    },
    "B": {
        "model": "gpt-4o",
        "temperature": 1.0
    }
}

# LLM応答を取得する関数
def get_llm_response(user_input: str, selected_expert: str, chat_history: list) -> str:
    """
    入力テキストと選択された専門家に基づいてLLMからの回答を取得する関数
    
    Args:
        user_input: ユーザーが入力したテキスト
        selected_expert: ラジオボタンで選択された専門家（"A"または"B"）
        chat_history: チャット履歴のリスト
    
    Returns:
        LLMからの回答テキスト
    """
    # 選択した専門家に応じたLLM設定を取得
    llm_config = LLM_CONFIGS[selected_expert]
    
    # LangChainでLLMを初期化
    llm = ChatOpenAI(
        model=llm_config["model"],
        temperature=llm_config["temperature"],
        api_key=api_key
    )
    
    # メッセージの構築
    messages = []
    
    # システムメッセージを追加
    system_message = SystemMessage(content=SYSTEM_MESSAGES[selected_expert])
    messages.append(system_message)
    
    # チャット履歴を追加
    messages.extend(chat_history)
    
    # ユーザーの入力を追加
    human_message = HumanMessage(content=user_input)
    messages.append(human_message)
    
    # プロンプトをLLMに送信
    response = llm.invoke(messages)
    answer = response.content
    
    return answer

# セッション状態の初期化
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_expert" not in st.session_state:
    st.session_state.selected_expert = "A"

# 専門家の説明の定義
EXPERT_DESCRIPTIONS = {
    "A": """* **🍷 深夜のカルチャー・ソムリエ**
    * 静かな夜に、隠れた名作映画や音楽を語り合いたい時に。
    * あなたにぴったりのドリンクも提案します。""",
    "B": """* **⚔️ 異世界転生の案内人**
    * 退屈な日常を忘れて、スリルある冒険に出かけたい時に。
    * あなたの選択次第で物語の結末が変わります。"""
}

# プレースホルダーの定義
PLACEHOLDERS = {
    "A": "今夜の気分を教えてください",
    "B": "あなたの行動を入力してください"
}

# 入力ラベルの定義
INPUT_LABELS = {
    "A": "今夜の気分を教えてください:",
    "B": "あなたの行動を選択してください:"
}

# 画面に入力フォームを表示
st.markdown("### 🚪 あなたは今日、誰と話をしますか？")
st.markdown("""
ここは、言葉一つで世界が変わる不思議なチャットルームです。
今のあなたの気分に合わせて、パートナーを選んでください。
""")

# 専門家の選択（ラジオボタン）
expert_options = {
    "A": "深夜のカルチャー・ソムリエ",
    "B": "異世界転生の案内人"
}

selected_expert = st.radio(
    "パートナーを選択してください:",
    options=list(expert_options.keys()),
    format_func=lambda x: expert_options[x],
    index=list(expert_options.keys()).index(st.session_state.selected_expert)
)

# 専門家が変更されたら履歴をリセット
if selected_expert != st.session_state.selected_expert:
    st.session_state.chat_history = []
    st.session_state.selected_expert = selected_expert
    st.rerun()

# 選択した専門家の説明を表示
st.markdown(EXPERT_DESCRIPTIONS[selected_expert])

# チャット履歴の表示
if st.session_state.chat_history:
    st.subheader("会話履歴")
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

# 入力フォーム
placeholder_text = PLACEHOLDERS[selected_expert]
input_label = INPUT_LABELS[selected_expert]
user_input = st.text_input(input_label, placeholder=placeholder_text, key="user_input")

# 送信ボタン
if st.button("送信"):
    if not user_input:
        st.warning("プロンプトを入力してください。")
    elif not api_key:
        st.error("OPENAI_API_KEYが環境変数に設定されていません。.envファイルを確認してください。")
    else:
        try:
            # プロンプトをLLMに送信
            with st.spinner("LLMが回答を生成中..."):
                answer = get_llm_response(user_input, selected_expert, st.session_state.chat_history)
            
            # チャット履歴に追加
            human_message = HumanMessage(content=user_input)
            st.session_state.chat_history.append(human_message)
            st.session_state.chat_history.append(AIMessage(content=answer))
            
            # 入力欄をクリア
            if "user_input" in st.session_state:
                del st.session_state.user_input
            
            # ページをリロードして履歴を更新
            st.rerun()
            
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")

# 履歴をクリアするボタン
if st.button("会話履歴をクリア"):
    st.session_state.chat_history = []
    st.rerun()
