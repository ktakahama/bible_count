import streamlit as st
from bible_wordcount import create_analysis, tokenize_japanese, analyze_sentence_lengths, analyze_pos_distribution
import tempfile
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

st.title('聖書テキスト分析ツール')

# テキスト入力エリア
text_input = st.text_area("テキストを入力してください", height=300)

# 分析ボタン（常に表示）
if st.button('分析を開始'):
    if text_input:
        # 一時ファイルを作成
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(text_input.encode('utf-8'))
            tmp_file_path = tmp_file.name

        with st.spinner('分析中...'):
            # 分析を実行
            output_file = create_analysis(tmp_file_path)
            
            # 結果を表示
            st.success('分析が完了しました！')
            
            # 結果ファイルを読み込んで表示
            if os.path.exists(output_file):
                with open(output_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    # HTMLコンテンツをiframeで表示（高さを十分に確保）
                    st.components.v1.html(html_content, height=6000, scrolling=True)
            else:
                st.error('分析結果ファイルが見つかりませんでした。')

        # 一時ファイルを削除
        os.unlink(tmp_file_path)
    else:
        st.warning('テキストを入力してください。')
