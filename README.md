# Accident Report Finder

This is a web application using the GPT API and Streamlit to search for past maritime accidents similar to near-miss reports. Although this program is developed with a Japanese interface, there are plans for multilingual support. (The timeline for this is uncertain.) The below details are provided in Japanese. If you are interested, please get in touch. I hope you find it intriguing.　The license for this project is MIT.

このWEBアプリケーションは、[船学](https://fune-gaku.com)と日本船長協会の月報のタイアップ企画のデモ・プログラムです。
以下の手順でセットアップして利用できます。

## セットアップ
1. OpenAIのAPIキーを取得する。
2. .env.exmpleを.envにファイル名を変更して保存します。
3. 名前変更した.env開き、OPENAI_API_KEY= に続けてAPIキーを入力し保存します。
4. `pip install -r requirements.txt`または`pip3 install -r requirements.txt`で必要なモジュールをインストールします。
5. ターミナル(Mac)またはコマンドプロンプト(Windows)を開き、次のコマンドを実行します。`streamlit run main.py`

## 利用方法
セットアップ完了後、
1. 過去の事故データを[運輸安全委員会の船舶事故報告書検索](https://jtsb.mlit.go.jp/jtsb/ship/index.php)からcsv形式で取得します。
2. Webブラウザを開き、http://localhost:8501にアクセスします。
3. 画面表示に沿って、取得したデータをインポートします。初回は原因をベクトル化することをお勧めします。
4. ベクトル化が完了したら、検索画面に移動し、ニアミスを入力すると過去の類似事故が検索できます。
