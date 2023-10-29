# Accident Report Finder

This is a web application using the GPT API and Streamlit to search for past maritime accidents similar to near-miss reports. Although this program is developed with a Japanese interface, there are plans for multilingual support. (The timeline for this is uncertain.) The below details are provided in Japanese. If you are interested, please get in touch. I hope you find it intriguing.

[船学](https://fune-gaku.com)と日本船長協会の月報のタイアップ企画のデモ・プログラムです。
以下の手順でセットアップして利用できます。

## セットアップ
1. OpenAIのAPIキーを.envに保存する。
2. requirements_for_main.txtに記載のモジュールをインストールします。
3. ターミナル(Mac)またはコマンドプロンプト(Windows)を開き、次のコマンドを実行します。`streamlit run main.py`

## 利用方法
セットアップ完了後、
1. csv形式で保存した過去の事故データを準備します。csvの列には"原因"または"概要"のいずれか、または両方の列を用意します。プログラムはいずれかの列をベクトル化します。
2. Webブラウザを開き、http://localhost:8501にアクセスします。
3. 画面表示に沿って、取得したデータをインポートします。初回は原因をベクトル化することをお勧めします。
4. ベクトル化が完了したら、検索画面に移動し、事故原因を入力すると過去の類似事故が検索できます。

セットアップが難しい場合はブラウザから利用できる[WEB版](https://accident-report-finder.streamlit.app)をご利用いただけます。ただし、事前の予告なく公開を終了する場合があります。