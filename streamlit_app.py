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
   menu_items={
        'About': "**Accident Report Finder** v1.2.0 made by [Michio Fujii](https://github.com/michiof)",
    }
)

# プロンプトを作成する
def make_message(user_input, user_input_emb, num_of_output, language):
    related_data = get_relevant_data(user_input_emb)
    if language == "ja":
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
    else:
        messages = [
            {"role": "system", "content": "Read the content of the relevant data and find accidents that are similar to the cause of the accident entered by the user."},
            {"role": "user", "content": f"""
                Find {num_of_output} accidents from the related data that are similar cause to the following user input, and output them in the specified output format below.\n
                If related data is written in Japanese, please trnslate them to English and output.\n\n
                Cause of the accident entered by the user:\n
                {user_input}\n\n
                Related data:\n
                {related_data}\n\n
                Output format (Please output in Markdown format):\n
                The similar accidents are as follows.\n\n
                **No. i:**\n\n
                **Name of accident:**  <Title> (translate in English and output)\n\n
                **Type of accident:**  <Type_accident> (translate in English and output)\n\n
                **Outline of accident:**  <Outline> (translate in English and output)\n\n
                **Cuase of accident:**  <Cause> (translate in English and output)\n\n
                **Report URL:**\n<URL>\n\n
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
    if "Severity_2" in filter_selection:
        filter.append({"Severity": {"$in": ["2"]}})
    if "Cat3" in filter_selection:
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
def chat_page(num_of_output, language):
    if language == "ja":
        label_msg_text_area = "検索したい事故原因を入力してください:"
        placeholder_text_area = "潮流が強く舵が効かなくなった。"
        lable_load_sample = "サンプル質問を挿入"
        label_results = "### 検索履歴: "
        label_search_botton = "検索"
        label_reset_button = "リセット"
        label_save_button = "検索結果をダウンロード"
        msg_header_search_text = "検索する原因: "
        msg_while_searching = "検索中..."
        msg_gen_result = "検索結果を出力中..."
        error_message = "再度検索してください。エラーが発生しました。"
    else:
        label_msg_text_area = "Please enter the cause of the accident you want to search for:"
        placeholder_text_area = "The tidal current was strong, and the rudder became ineffective."
        lable_load_sample = "Input a sample"
        label_results = "### Results: "
        label_search_botton = "Search"
        label_reset_button = "Reset"
        label_save_button = "Download search results."
        msg_header_search_text = "Cause to search for: "
        msg_while_searching = "Searching..."
        msg_gen_result = "Outputting search results..."
        error_message = "Please search again. An error has occurred."

    new_msg = st.text_input(label_msg_text_area, value=st.session_state.sample_question, placeholder=placeholder_text_area)
    if st.button(lable_load_sample):
        st.session_state.sample_question = placeholder_text_area # load a sample question
    if st.button(label_search_botton):
        if new_msg:
            try:
                with st.spinner(msg_while_searching):
                    user_input = f"{msg_header_search_text}{new_msg}"
                    user_input_emb = cal_embedding(new_msg)
                    CHAT_INPUT_MESSAGES = make_message(user_input, user_input_emb, num_of_output, language)
                with st.spinner(msg_gen_result):
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
                st.error(error_message)

    st.write("---")
    st.write(label_results)
    output_messages = ""
    for message in reversed(st.session_state.messages):
        st.write(message)
        output_messages += message + "\n\n"

    # 履歴のクリアボタン
    if st.button(label_reset_button):
        st.session_state.messages = []
        st.empty()
    
    st.download_button(label_save_button, output_messages)

def main():
    st.title("🔍 Accident Report Finder")
    language_selection = st.radio("Language", ("English", "日本語"), horizontal=True)

    if language_selection == "日本語":
        language = "ja"
        caption_1 = "原因やヒヤリハットを入力すると、過去の類似船舶事故を検索できます。"
        caption_2 = "データ出典: [運輸安全委員会](https://jtsb.mlit.go.jp/jtsb/ship/index.php)が公開している15,334件(2023年12月1日時点)の船舶事故報告書データを本プログラム用に加工して利用しています。"
        label_num_of_output = "最大検索件数"
        label_filter_title = "フィルター"
        label_filter_1 = "軽微なものを除く"
        label_filter_2 = "小型船舶を除く"
    else:
        language = "en"
        caption_1 = "You can search for past similar ship accidents by entering the cause or near-miss incidents."
        caption_2 = "Data source: The program uses processed data from 15,334 ship accident reports (as of December 1, 2023) publicly available from the [Japan Transport Safety Board](https://jtsb.mlit.go.jp/jtsb/ship/index.php)."
        label_num_of_output = "Max. search results"
        label_filter_title = "Eexclude:"
        label_filter_1 = "Minor accident"
        label_filter_2 = "Small craft"
    st.caption(caption_1)
    st.caption(caption_2)
    st.write("---")
    st.sidebar.title("Accident Report Finder")
    with st.sidebar:
        st.write("Version: 1.2.0")
        st.write("Made by [Michio Fujii](https://github.com/michiof)")
        st.write("---")
        
        num_of_output = st.slider(label_num_of_output, 1, 5, 3)
        filter_selection = st.multiselect(label_filter_title,
                                [label_filter_1, label_filter_2],
                                [label_filter_1, label_filter_2]
                            )
        converted_filter_selection = [text.replace(label_filter_1,"Severity_2").replace(label_filter_2, "Cat3") for text in filter_selection]
        make_pinecone_filter(converted_filter_selection)

    if "messages" not in st.session_state:
        st.session_state['messages']= []
    if 'sample_question' not in st.session_state:
        st.session_state['sample_question'] = ""

    chat_page(num_of_output, language)

if __name__ == "__main__":
    main()
