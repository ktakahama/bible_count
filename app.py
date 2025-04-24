import streamlit as st
from bible_wordcount import create_analysis, tokenize_japanese, analyze_sentence_lengths, analyze_pos_distribution
import tempfile
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import plotly.io as pio

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
                
                # 追加の分析を実行
                sentence_stats = analyze_sentence_lengths(text_input)
                pos_dist = analyze_pos_distribution(text_input)
                tokens = tokenize_japanese(text_input)
                word_freq = Counter(tokens)
                word_lengths = [len(word) for word in tokens]
                
                # グラフをHTMLに変換
                fig_sentence = go.Figure(data=[
                    go.Bar(name='文の長さ', x=list(sentence_stats.keys()), y=list(sentence_stats.values()))
                ])
                fig_sentence.update_layout(barmode='group')
                sentence_html = pio.to_html(fig_sentence, full_html=False)
                
                df_pos = pd.DataFrame(list(pos_dist.items()), columns=['品詞', '出現回数'])
                fig_pos = px.pie(df_pos, values='出現回数', names='品詞', title='品詞の分布')
                pos_html = pio.to_html(fig_pos, full_html=False)
                
                df_word = pd.DataFrame(word_freq.most_common(20), columns=['単語', '出現回数'])
                fig_word = px.bar(df_word, x='単語', y='出現回数', title='単語の出現頻度')
                word_html = pio.to_html(fig_word, full_html=False)
                
                df_length = pd.DataFrame({'長さ': word_lengths})
                fig_length = px.histogram(df_length, x='長さ', title='単語の長さの分布')
                length_html = pio.to_html(fig_length, full_html=False)
                
                # 追加分析のHTMLを生成
                additional_html = f"""
                <div class="section">
                    <h2>追加分析</h2>
                    
                    <div class="section">
                        <h3>文の長さの分布</h3>
                        {sentence_html}
                    </div>
                    
                    <div class="section">
                        <h3>品詞の分布</h3>
                        {pos_html}
                    </div>
                    
                    <div class="section">
                        <h3>単語の出現頻度（上位20）</h3>
                        {word_html}
                    </div>
                    
                    <div class="section">
                        <h3>単語の長さの分布</h3>
                        {length_html}
                    </div>
                </div>
                """
                
                # 元のHTMLに追加分析を挿入
                html_content = html_content.replace('</div>\n</body>', f'{additional_html}\n</div>\n</body>')
                
                # HTMLを直接表示
                st.markdown(html_content, unsafe_allow_html=True)
            else:
                st.error('分析結果ファイルが見つかりませんでした。')

        # 一時ファイルを削除
        os.unlink(tmp_file_path)
    else:
        st.warning('テキストを入力してください。')
