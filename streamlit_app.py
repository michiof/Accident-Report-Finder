# For Streamlit online app

import streamlit as st
import pandas as pd
from openai import OpenAI
import pinecone
import tiktoken

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"],)

# OpenAIのモデルの定義
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

# Streamlit setup
st.set_page_config(
   page_title="Accident Report Finder",
   page_icon="🔍",
)

# プロンプトを作成する
def make_message(user_input, user_input_emb, num_of_output):
    related_data = get_relevant_data(user_input_emb)
    messages = [
        {"role": "system", "content": "関連データの内容を読み取って、ユーザーが入力した事故原因と原因が類似している事故を見つけます。"},
        {"role": "user", "content": f"""
            以下のユーザーが入力した事故原因と事故の原因が類似した事故を関連データから{num_of_output}件見つけ出し、以下に指定する出力フォーマットで出力してください。\n\n
            ユーザーが入力した事故原因：\n
            {user_input}\n\n
            関連データ：\n
            {related_data}\n\n
            出力フォーマット（Markdown形式で出力してください）:\n
            類似事故は以下のとおりです。\n\n
            i件目\n\n
            事故の名称：\n<Title>\n\n
            事故の種類: \n<Type_accident>\n\n
            事故の概要：\n<Outline>\n\n
            事故の原因：\n<Cause>\n\n
            報告書のURL: \n<URL>\n\n
        """},
    ]
    return messages

# トークン数を計算する
def num_tokens(text: str, model: str) -> int:
    encoding = tiktoken.encoding_for_model(GPT_MODEL)
    return len(encoding.encode(text))

# Pineconeのmetadataを取得する
def get_metadata(match):
    info = []
    for key, value in match["metadata"].items():
        info.append(f"{key}: {value}\n")
    return "\n".join(info)

# Multiselectionの値からPinecone用のフィルターを作成し、session_stateに保存する
def make_pinecone_filter(filter_selection):
    filter = []
    if '軽微なものを除く' in filter_selection:
        filter.append({"Severity": {"$in": ["2"]}})
    if '小型船舶を除く' in filter_selection:
        filter.append({"Cat_GrossTon": {"$in": ["Cat3"]}})
    
    if len(filter) > 1:
        filter_dic_for_pinecone = {"$and": filter}
    elif filter:
        filter_dic_for_pinecone = filter[0]
    else:
        filter_dic_for_pinecone = {}
    st.session_state['filter_dic'] = filter_dic_for_pinecone

# 類似ベクトルデータの抽出
def get_relevant_data(query_embedding, top_k=20):
    pinecone.init(api_key=st.secrets["PINECONE_API_KEY"], environment=st.secrets["PINECONE_ENVIRONMENT"])
    pinecone_index = pinecone.Index(st.secrets["PINECONE_INDEX"])
    token_budget = 4096 - 1500 #関連データのトークン上限値の設定
    relevant_data = ""
    filter_dic = st.session_state['filter_dic']
    for _ in range(3): # エラー発生時は3回までトライする
        try:
            results = pinecone_index.query(
                            vector=query_embedding,
                            filter=filter_dic,
                            top_k=top_k, 
                            include_metadata=True
                        )
            for i, match in enumerate(results["matches"], start=1):
                metadata = get_metadata(match)
                next_relevant_data = f"\n\nRelevant data {i}:\n{metadata}"
                if (
                    num_tokens(relevant_data + next_relevant_data, model=GPT_MODEL)
                    > token_budget
                ):
                    break
                else:
                    relevant_data += next_relevant_data
            return relevant_data
        except Exception as e:
            print(f"Error: {str(e)}")
            continue
    raise Exception("Error: Failed to retrieve relevant data after 3 attempts")

# Embeddingsの計算
def cal_embedding(user_input, model=EMBEDDING_MODEL):
    for _ in range(3): # エラー発生時は3回までトライする
        try:
            return client.embeddings.create(input=user_input, model=model).data[0].embedding
        except Exception as e:
            print(f"Error: {str(e)}")
            continue
    raise Exception("Failed to calculate embedding after 3 attempts")

# 検索画面での処理
def chat_page(num_of_output):
    new_msg = st.text_area("検索したい事故原因を入力してください:", placeholder="潮流が強く舵が効かなくなった。")
    
    if st.button("検索"):
        if new_msg:
            try:
                with st.spinner("検索中..."):
                    user_input = f"検索する原因: {new_msg}"
                    user_input_emb = cal_embedding(new_msg)
                    CHAT_INPUT_MESSAGES = make_message(user_input, user_input_emb, num_of_output)
                with st.spinner("文章生成中..."):
                    response_all = ""
                    temp_placeholder = st.empty()
                    stream = client.chat.completions.create(model=GPT_MODEL,messages=CHAT_INPUT_MESSAGES, temperature=0.0, stream=True)
                    for part in stream:
                        response_delta = (part.choices[0].delta.content or "")
                        response_all += response_delta
                        temp_placeholder.write(response_all)
                st.session_state.messages.append("---")
                st.session_state.messages.append(response_all)
                st.session_state.messages.append(user_input)
                temp_placeholder.empty() #Stream部分の非表示
                st.empty()
            except Exception as e:
                print(str(e))
                st.error(f"再度検索してください。エラーが発生しました。")

    st.write("---")
    st.write("検索履歴:")
    output_messages = ""
    for message in reversed(st.session_state.messages):
        st.write(message)
        output_messages += message + "\n\n"

    # 履歴のクリアボタン
    if st.button("履歴を消去"):
        st.session_state.messages = []
        st.empty()
    
    st.download_button("検索結果を保存", output_messages)

def main():
    st.title("🔍 Accident Report Finder")
    st.caption("入力した事故原因と類似する過去の船舶事故を検索できます。")
    st.caption("データ出典: [運輸安全委員会](https://jtsb.mlit.go.jp/jtsb/ship/index.php)が公開している15,334件(2023年12月1日時点)の船舶事故報告書データを本プログラム用に加工して利用しています。")
    st.write("---")
    st.sidebar.title("Accident Report Finder")
    with st.sidebar:
        st.write("Version: 1.1.0")
        st.write("Made by [Michio Fujii](https://github.com/michiof)")
        st.write("---")
        
        num_of_output = st.slider("最大検索件数", 1, 5, 3)

        # filter設定
        filter_selection = st.multiselect("フィルター",
                                ['軽微なものを除く', '小型船舶を除く'],
                                ['軽微なものを除く', '小型船舶を除く']
                            )
        make_pinecone_filter(filter_selection)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    chat_page(num_of_output)

if __name__ == "__main__":
    main()
