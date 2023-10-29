# For Streamlit online app
import streamlit as st
import pandas as pd
import openai
import pinecone
import tiktoken

openai.api_key = st.secrets["OPENAI_API_KEY"]
pinecone.init(api_key=st.secrets["PINECONE_API_KEY"], environment=st.secrets["PINECONE_ENVIRONMENT"])
pinecone_index = pinecone.Index(st.secrets["PINECONE_INDEX"])

# OpenAIのモデルの定義
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

# プロンプトを作成する
def make_message(user_input, user_input_emb):
    related_data = get_relevant_data(user_input_emb)
    messages = [
        {"role": "system", "content": '関連データの内容を読み取って、ユーザーが入力した事故原因と原因が類似している事故を見つけます。'},
        {"role": "user", "content": f'''
            以下のユーザーが入力した事故原因と事故の原因が類似した事故を関連データから3つ見つけ出し、以下に指定する出力フォーマットで出力してください。\n\n
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
        '''},
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
        info.append(f'{key}: {value}\n')
    return '\n'.join(info)

# 類似ベクトルデータの抽出
def get_relevant_data(query_embedding, top_k=10):
    token_budget = 4096 - 1500 #関連データのトークン上限値の設定
    relevant_data = ""
    try:
        results = pinecone_index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        for i, match in enumerate(results["matches"], start=1):
            metadata = get_metadata(match)
            next_relevant_data = f'\n\nRelevant data {i}:\n{metadata}'
            if (
                num_tokens(relevant_data + next_relevant_data, model=GPT_MODEL)
                > token_budget
            ):
                break
            else:
                relevant_data += next_relevant_data
        return relevant_data
    except Exception as e:
        st.warning(f"データベース検索中にエラーが発生しました。もう一度お試しください: {e}")
        return False

# Embeddingsの計算
def cal_embedding(user_input, model=EMBEDDING_MODEL):
    try:
        embedding = openai.Embedding.create(input=user_input, model=model)["data"][0]["embedding"]
        return embedding
    except Exception as e:
        st.warning(f"Embeddings計算中にエラーが発生しました。もう一度お試しください: {e}")
        return False

# 検索画面での処理
def chat_page():
    new_msg = st.text_input('検索したい事故原因を入力してください:（例）潮流が強く舵が効かなくなった。')
    
    if st.button('検索'):
        if new_msg:
            with st.spinner('検索中...'):
                user_input = f'検索する原因: {new_msg}'
                user_input_emb = cal_embedding(new_msg)
                CHAT_INPUT_MESSAGES = make_message(user_input, user_input_emb)

            # 文章生成中はtemp_placeholderにストリーミングで表示
            try:
                with st.spinner('文章生成中...'):
                    response_all = ""
                    temp_placeholder = st.empty()
                    response = openai.ChatCompletion.create(model=GPT_MODEL,messages=CHAT_INPUT_MESSAGES, temperature=0.0, stream=True)
                    for chunk in response:
                        response_delta = chunk['choices'][0]['delta'].get('content', '')
                        response_all += response_delta
                        temp_placeholder.write(response_all)
                st.session_state.messages.append('---')
                st.session_state.messages.append(response_all)
                st.session_state.messages.append(user_input)
                temp_placeholder.empty() #Stream部分の非表示
                st.empty()
            except Exception as e:
                st.error(f"再度検索してください。エラーが発生しました: {str(e)}")

    st.write('---')
    st.write('検索履歴:')
    for message in reversed(st.session_state.messages):
        st.write(message)

    # 履歴のクリアボタン
    if st.button('履歴のクリア'):
        st.session_state.messages = []
        st.empty()

def main():
    st.title('🔍 Accident Report Finder =beta=')
    st.write('入力した文章と原因が類似する過去の船舶事故を検索できます。')
    st.write('運輸安全委員会が公開している14,875件の船舶事故報告書(2023年6月時点)を本プログラムの検索用に加工して利用しています。')
    st.write('出典：[運輸安全委員会ホームページ: 報告書検索](https://jtsb.mlit.go.jp/jtsb/ship/index.php)')
    st.write('---')

    # メッセージのリストを維持する
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    chat_page()

if __name__ == '__main__':
    main()
