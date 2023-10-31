#!/usr/bin/env python3
import streamlit as st
import pandas as pd
from scipy import spatial
import chardet
import os
from dotenv import load_dotenv
import openai
import ast
import tiktoken
import datetime

# .envからAPIキーの読み込み
load_dotenv('.env')
openai.api_key = os.environ.get("OPENAI_API_KEY")

# OpenAIのモデルの定義
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

# データPATH
filepath_emb = './data/emb.csv'
dir_log = './chatlog'

# プロンプトを作成する
def make_message(hiyari_hatto, hiyari_hatto_emb, related_data_df):
    related_data = get_relevant_data(hiyari_hatto_emb, related_data_df)
    messages = [
        {"role": "system", "content": '関連データの内容を読み取って、ヒヤリハットと類似している事故を見つけます。'},
        {"role": "user", "content": f'''
            以下のヒヤリハットと類似した事故を関連データから3つ見つけ出し、以下に指定する出力フォーマットで出力してください。\n\n
            {hiyari_hatto}\n\n
            関連データ：\n
            {related_data}\n\n
            出力フォーマット（マークダウンで出力してください）:\n
            類似事故は以下のとおりです。\n\n
            <i>件目\n\n
            事故の概要：\n<概要>\n\n
            事故の原因：\n<原因>\n\n
            報告書のURL: \n<URL>\n\n
        '''},
    ]
    return messages

# トークン数を計算する
def num_tokens(text: str, model: str) -> int:
    encoding = tiktoken.encoding_for_model(GPT_MODEL)
    return len(encoding.encode(text))

# 類似ベクトルの検索（コサイン類似度）：dfで返す <- ここから
def get_relevant_data(query_embedding, related_data_df, top_k=10):
    token_budget = 4096 - 1500 #関連データのトークン上限値の設定
    relevant_data = ""

    related_data_df["Embedding"] = related_data_df["Embedding"].apply(ast.literal_eval) #リストに変換
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y)
    strings_and_relatednesses = [
        (row, relatedness_fn(query_embedding, row["Embedding"]))
        for i, row in related_data_df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    results = [row.drop("Embedding") for row, _ in strings_and_relatednesses][:top_k] # "row.drop("Embedding"), _" : Embedding列を除外, relatednessesを除外
    
    counter = 1
    for item in results:
        next_relevant_data = (
            f'\n\n関連データ（{counter}）:\n'
            f'事故の名称: {item["事故等名"]}\n'
            f'URL: {item["報告書（PDF）公表"]}\n'
            f'概要: {item["概要"]}\n'
            f'原因: {item["原因"]}\n'
        )
        if num_tokens(relevant_data + next_relevant_data, model=GPT_MODEL) > token_budget:
            break
        else:
            relevant_data += next_relevant_data
            counter += 1
    
    return relevant_data

# Embeddingsの計算
def cal_embedding(hiyari_hatto, model=EMBEDDING_MODEL):
    try:
        embedding = openai.Embedding.create(input=hiyari_hatto, model=model)["data"][0]["embedding"]
        return embedding
    except Exception as e:
        st.warning(f"Embeddings計算中にエラーが発生しました。もう一度お試しください: {e}")
        return False

# 検索画面での処理
def chat_page(related_data_df):
    new_msg = st.text_input('ヒヤリハットを入力してください。')
    
    if st.button('検索'):
        if not os.path.exists(filepath_emb):
            st.warning('参照データがありません。データ準備画面で設定してください。')
            return
        
        if new_msg:
            with st.spinner('検索中...'):
                hiyari_hatto = f'ヒヤリハット: {new_msg}'
                hiyari_hatto_emb = cal_embedding(new_msg)
                CHAT_INPUT_MESSAGES = make_message(hiyari_hatto, hiyari_hatto_emb, related_data_df)
            
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
                st.session_state.messages.append(hiyari_hatto)
                temp_placeholder.empty() #Stream部分の非表示
                st.empty()
            except Exception as e:
                st.error(f"再度検索してください。エラーが発生しました: {str(e)}")

    st.write('---')
    st.write('検索履歴:')
    for message in reversed(st.session_state.messages):
        st.write(message)

    # チャットの履歴をテキストファイルに保存
    if st.button('検索結果を保存'):
        current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"{dir_log}/chat_history_{current_time}.txt"
        with open(file_name, 'w', encoding='utf-8') as file:
            for message in st.session_state.messages:
                file.write(message + '\n')
        
        st.success(f'検索結果が {file_name} に保存されました。')

    # 履歴のクリアボタン
    if st.button('履歴のクリア'):
        st.session_state.messages = []
        st.empty()

# ファイルのエンコーディングを調べる関数
def detect_file_encoding(file_content):
    result = chardet.detect(file_content)
    return result['encoding']

# データ準備画面
def csv_import_page(reload=False):
    uploaded_file = st.file_uploader("CSV/TSVファイルをインポート", type=["csv", "tsv"])
    if uploaded_file:
        # ファイルのエンコーディングを自動検出
        detected_encoding = detect_file_encoding(uploaded_file.read())
        # 一度ストリームを読むと、再度読むためにはポインタを先頭に戻す必要がある
        uploaded_file.seek(0)
        
        try:
            df = pd.read_csv(uploaded_file, encoding=detected_encoding, delimiter='\t', on_bad_lines='skip', nrows=10)
            st.write(f'読み込んだデータの行数: {len(df)} (上限1,000)')
            emb_row = None
            
            # ラジオボタンを使用して検索方法を選択
            search_method = st.radio("検索方法を選択してください:", ("原因の類似性を検索", "概要の類似性を検索"))

            if search_method == "原因の類似性を検索":
                emb_row = '原因'
                if '原因' in df.columns:
                    emb_row = '原因'
                else:
                    st.warning("読み込んだファイルに '概要' というカラムが存在しません。")
            elif search_method == "概要の類似性を検索":
                if '概要' in df.columns:
                    emb_row = '概要'
                else:
                    st.warning("読み込んだファイルに '概要' というカラムが存在しません。")
            
            if emb_row and st.button("ベクトル計算を実行"):
                st.write('Embeddingsの計算中です。しばらくお待ちください...')

                # 進行状況バーの初期化
                progress_bar = st.progress(0)
                total_rows = len(df)

                # Embeddingsを計算して保存
                embeddings = []
                for idx, row in df.iterrows():
                    embedding = cal_embedding(row[emb_row])
                    embeddings.append(embedding)
                    
                    # 進行状況の更新
                    progress_bar.progress((idx + 1) / total_rows)

                df['Embedding'] = embeddings

                # Embeddingsが含まれるデータをoutput.csvに出力
                df.to_csv(filepath_emb, index=False)
                st.success('Embeddingを含むCSVファイルを保存しました。検索が可能です。')
                
                if reload:
                    if st.button('ページを再読み込み'):
                        st.experimental_rerun()
                        
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            st.error(f"データの読み込みに失敗しました: {str(e)}")

def main():
    st.title('🔍 Accident Report Finder')
    st.write('入力したヒヤリハットと類似する過去の事故を検索できます。')

    # メッセージのリストを維持する
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # ベクトルデータファイルが保存されているかの確認
    if not os.path.exists(filepath_emb):
        dir_name = os.path.dirname(filepath_emb)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        st.warning('参照データがありません。データを読み込んでください。')
        csv_import_page(reload=True)
    else:
        related_data_df = pd.read_csv(filepath_emb)
        os.makedirs(dir_log, exist_ok=True) #Logディレクトリがない場合は作成
        tab_chat, tab_import_file = st.tabs(["検索画面", "データ準備画面"])
        with tab_chat:
            chat_page(related_data_df)
        with tab_import_file:
            csv_import_page()

if __name__ == '__main__':
    main()
