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
                    # HTMLコンテンツをiframeで表示（スクロールなし）
                    st.components.v1.html(html_content, height=None, scrolling=False)
                
                # 追加の分析とグラフ
                st.markdown("## 追加分析")
                
                # 文の長さの分析
                sentence_stats = analyze_sentence_lengths(text_input)
                df_sentence = pd.DataFrame([sentence_stats])
                st.markdown("### 文の長さの分布")
                fig = go.Figure(data=[
                    go.Bar(name='文の長さ', x=list(sentence_stats.keys()), y=list(sentence_stats.values()))
                ])
                fig.update_layout(barmode='group')
                st.plotly_chart(fig)
                
                # 品詞の分布
                pos_dist = analyze_pos_distribution(text_input)
                df_pos = pd.DataFrame(list(pos_dist.items()), columns=['品詞', '出現回数'])
                st.markdown("### 品詞の分布")
                fig = px.pie(df_pos, values='出現回数', names='品詞', title='品詞の分布')
                st.plotly_chart(fig)
                
                # 単語の出現頻度
                tokens = tokenize_japanese(text_input)
                word_freq = Counter(tokens)
                df_word = pd.DataFrame(word_freq.most_common(20), columns=['単語', '出現回数'])
                st.markdown("### 単語の出現頻度（上位20）")
                fig = px.bar(df_word, x='単語', y='出現回数', title='単語の出現頻度')
                st.plotly_chart(fig)
                
                # 単語の長さの分布
                word_lengths = [len(word) for word in tokens]
                df_length = pd.DataFrame({'長さ': word_lengths})
                st.markdown("### 単語の長さの分布")
                fig = px.histogram(df_length, x='長さ', title='単語の長さの分布')
                st.plotly_chart(fig)
            else:
                st.error('分析結果ファイルが見つかりませんでした。')

        # 一時ファイルを削除
        os.unlink(tmp_file_path)
    else:
        st.warning('テキストを入力してください。')
