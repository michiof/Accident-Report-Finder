from flask import Flask, request, render_template
import os
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
import pinecone
import tiktoken

app = Flask(__name__)

load_dotenv('.env')
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
os.environ.get("OPENAI_API_KEY")
os.environ.get("OPENAI_API_KEY")

# OpenAIのモデルの定義
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

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
    return filter_dic_for_pinecone

# 類似ベクトルデータの抽出
def get_relevant_data(query_embedding, top_k=20):
    pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment=os.environ.get("PINECONE_ENVIRONMENT"))
    pinecone_index = pinecone.Index(os.environ.get("PINECONE_INDEX"))
    token_budget = 4096 - 1500 #関連データのトークン上限値の設定
    relevant_data = ""
    filter_dic = make_pinecone_filter('軽微なものを除く')
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

@app.route("/", methods=["GET", "POST"])
def chat_page():
    response_message = ""
    if request.method == "POST":
        new_msg = request.form["message"]
        if new_msg:
            try:
                user_input = f"検索する原因: {new_msg}"
                user_input_emb = cal_embedding(new_msg)
                CHAT_INPUT_MESSAGES = make_message(user_input, user_input_emb, 3)

                response_message = client.chat.completions.create(model=GPT_MODEL, messages=CHAT_INPUT_MESSAGES, temperature=0.0).choices[0].message
            except Exception as e:
                response_message = f"エラーが発生しました: {str(e)}"

    return render_template("chat_page.html", response=response_message)

if __name__ == "__main__":
    app.run()
