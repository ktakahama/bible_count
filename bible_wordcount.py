import pandas as pd
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.charfilter import *
from janome.tokenfilter import *
from janome import sysdic
from janome.dic import UserDictionary
import tempfile
from collections import defaultdict
from datetime import datetime
import argparse
import re
from statistics import mean, median, stdev
import openai
from typing import Dict, List
from dotenv import load_dotenv

# NLTKのデータをダウンロード
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# .envファイルを読み込む
load_dotenv()

def analyze_sentence_patterns(text):
    # 文末表現を分析
    sentences = text.split('。')
    endings = defaultdict(int)
    for s in sentences:
        if s:
            # 最後の文字や表現を取得
            if s.endswith('か'):
                endings['疑問'] += 1
            elif s.endswith('だ') or s.endswith('である'):
                endings['断定'] += 1
            elif s.endswith('う') or s.endswith('よう'):
                endings['意志・勧誘'] += 1
            elif s.endswith('ない'):
                endings['否定'] += 1
            else:
                endings['その他'] += 1
    
    return endings

def analyze_word_pairs(tokens):
    # 共起関係の分析
    pairs = defaultdict(int)
    for i in range(len(tokens)-1):
        pair = (tokens[i], tokens[i+1])
        pairs[pair] += 1
    return pairs

def analyze_sentence_lengths(text):
    """
    文の長さを分析する
    """
    # 文を分割（句点、感嘆符、疑問符で区切る）
    sentences = re.split(r'[。！？]', text)
    sentence_lengths = [len(tokenize_japanese(sentence)) for sentence in sentences if sentence.strip()]
    return dict(Counter(sentence_lengths))

def analyze_pos_distribution(text):
    """
    品詞の分布を分析する
    """
    t = Tokenizer()
    pos_counts = Counter()
    
    for token in t.tokenize(text):
        pos = token.part_of_speech.split(',')[0]
        pos_counts[pos] += 1
    
    return pos_counts

def analyze_emotion_words(text):
    # 感情語のリスト
    emotion_words = {
        '喜び': {'喜ぶ', '楽しむ', '笑う', '幸せ', '喜び', '歓喜', '喜び'},
        '悲しみ': {'悲しい', '泣く', '嘆く', '哀れむ', '悲しみ', '嘆き'},
        '怒り': {'怒る', '憤る', '憤慨', '怒り', '憤り'},
        '恐れ': {'恐れる', '怖い', '恐れ', '怖れ', '恐ろしい'},
        '愛': {'愛する', '愛', '慈しむ', '慈愛', '愛情'},
        '憎しみ': {'憎む', '憎しみ', '嫌う', '嫌悪', '憎悪'}
    }
    
    emotion_counts = {}
    for emotion, words in emotion_words.items():
        count = sum(text.count(word) for word in words)
        if count > 0:
            emotion_counts[emotion] = count
    
    return emotion_counts

def extract_proper_nouns(text):
    # ユーザー辞書を作成
    user_dict_file = create_user_dict()
    
    # ユーザー辞書を使用してTokenizerを初期化
    t = Tokenizer(udic=user_dict_file, udic_enc="utf8", udic_type="csv")
    
    # 固有名詞を抽出
    proper_nouns = []
    for token in t.tokenize(text):
        pos = token.part_of_speech.split(',')
        if pos[0] == '名詞' and pos[1] == '固有名詞':
            proper_nouns.append(token.surface)
    
    # 一時ファイルを削除
    os.remove(user_dict_file)
    
    return Counter(proper_nouns)

def analyze_context(text, target_word, window_size=50):
    # 特定の単語の文脈を分析
    contexts = []
    for match in re.finditer(target_word, text):
        start = max(0, match.start() - window_size)
        end = min(len(text), match.end() + window_size)
        context = text[start:end]
        # 対象の単語を【】で囲む
        context = context.replace(target_word, f"【{target_word}】")
        contexts.append(context)
    
    return contexts

def get_color_for_word(word, color_map, index):
    if word not in color_map:
        # より明確に区別できる色のリスト（異なる色相）
        colors = [
            '#FF0000',  # 真っ赤
            '#0000FF',  # 真っ青
            '#FF00FF',  # マゼンタ
            '#00FFFF',  # シアン
            '#FF8000',  # オレンジ
            '#8000FF',  # 紫
            '#00FF00',  # 真っ緑
            '#FF1493',  # ディープピンク
            '#00CED1',  # ダークターコイズ
            '#FF4500'   # オレンジレッド
        ]
        # 単語の出現順に色を選択
        color_map[word] = colors[index % len(colors)]
    return color_map[word]

def highlight_frequent_words(text, frequent_words):
    # 単語ごとの色を管理する辞書
    color_map = {}
    
    # 頻出単語を強調表示
    for i, (word, count) in enumerate(frequent_words):
        if count >= 3:  # 3回以上出現する単語のみ
            color = get_color_for_word(word, color_map, i)
            # HTMLのspanタグで背景色でハイライト（白文字）
            text = text.replace(word, f'<span style="background-color: {color}; color: white; padding: 2px 2px; border-radius: 2px;">{word}</span>')
    return text

def get_word_explanations(text: str) -> Dict[str, str]:
    """
    ChatGPTを使用して、本文から重要な単語を抽出し、その説明を取得する関数
    """
    # APIキーを.envファイルから取得
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("警告: .envファイルにOPENAI_API_KEYが設定されていません。")
        return {"エラー": "APIキーが設定されていません。.envファイルにOPENAI_API_KEYを設定してください。"}
    
    client = openai.OpenAI(api_key=api_key)
    
    explanations = {}
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """あなたは聖書の専門家です。与えられた聖書のテキストから、重要な単語を抽出し、その聖書での意味を説明してください。
以下の形式で出力してください：
単語1: 説明1
単語2: 説明2
...
説明は1行で、聖書の文脈での意味を簡潔に述べてください。"""},
                {"role": "user", "content": f"以下の聖書のテキストから重要な単語を抽出し、その意味を説明してください。また解釈の助けになるようなコメントを加えてください。\n\n{text}"}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        # レスポンスを解析
        result = response.choices[0].message.content.strip()
        for line in result.split('\n'):
            if ':' in line:
                word, explanation = line.split(':', 1)
                word = word.strip()
                explanations[word] = explanation.strip()
    except Exception as e:
        print(f"Error getting explanations: {str(e)}")
        explanations = {"エラー": "説明を取得できませんでした。"}
    
    return explanations

def create_analysis(input_file):
    """
    テキストファイルを分析し、結果をHTMLファイルとして出力する
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 分析を実行
    tokens = tokenize_japanese(text)
    sentence_stats = analyze_sentence_lengths(text)
    pos_dist = analyze_pos_distribution(text)
    
    # 結果をHTMLとして出力
    output_file = input_file.replace('.txt', '_analysis.html')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>テキスト分析結果</title>
            <style>
                body {{
                    font-family: 'Helvetica Neue', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    background-color: white;
                    border-radius: 8px;
                    padding: 30px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                    margin-top: 0;
                }}
                .section {{
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    padding: 20px;
                    margin-bottom: 30px;
                }}
                .section h2 {{
                    color: #2c3e50;
                    margin-top: 0;
                }}
                .stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-top: 15px;
                }}
                .stat-item {{
                    background-color: white;
                    padding: 15px;
                    border-radius: 6px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }}
                .stat-item p {{
                    margin: 0;
                    font-size: 1.1em;
                }}
                .stat-item .label {{
                    color: #7f8c8d;
                    font-size: 0.9em;
                }}
                .stat-item .value {{
                    color: #2c3e50;
                    font-weight: bold;
                    font-size: 1.2em;
                }}
                ul {{
                    list-style-type: none;
                    padding: 0;
                    margin: 0;
                }}
                li {{
                    background-color: white;
                    padding: 10px 15px;
                    margin-bottom: 8px;
                    border-radius: 4px;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                }}
                @media (max-width: 600px) {{
                    .stats {{
                        grid-template-columns: 1fr;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>テキスト分析結果</h1>
                
                <div class="section">
                    <h2>基本統計</h2>
                    <div class="stats">
                        <div class="stat-item">
                            <p class="label">総文字数</p>
                            <p class="value">{len(text)}</p>
                        </div>
                        <div class="stat-item">
                            <p class="label">総単語数</p>
                            <p class="value">{len(tokens)}</p>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>文の長さの分布</h2>
                    <ul>
                        {''.join(f'<li>{length}単語の文: {count}個</li>' for length, count in sorted(sentence_stats.items()))}
                    </ul>
                </div>
                
                <div class="section">
                    <h2>品詞の分布</h2>
                    <ul>
                        {''.join(f'<li>{pos}: {count}回</li>' for pos, count in sorted(pos_dist.items(), key=lambda x: x[1], reverse=True))}
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """)
    
    return output_file

def create_user_dict():
    # 一時ファイルに辞書定義を書き込む
    user_dict_data = """モーセ,モーセ,モーセ,固有名詞,*,*,*,モーセ,モーセ,モーセ,外国,*,*
アブラハム,アブラハム,アブラハム,固有名詞
イサク,イサク,イサク,固有名詞
ヤコブ,ヤコブ,ヤコブ,固有名詞
アダム,アダム,アダム,固有名詞
エバ,エバ,エバ,固有名詞
ノア,ノア,ノア,固有名詞
ヨセフ,ヨセフ,ヨセフ,固有名詞
ダビデ,ダビデ,ダビデ,固有名詞
ソロモン,ソロモン,ソロモン,固有名詞
イエス,イエス,イエス,固有名詞
マリア,マリア,マリア,固有名詞
ペテロ,ペテロ,ペテロ,固有名詞
パウロ,パウロ,パウロ,固有名詞
カイン,カイン,カイン,固有名詞
アベル,アベル,アベル,固有名詞
サウル,サウル,サウル,固有名詞
サムエル,サムエル,サムエル,固有名詞
エリヤ,エリヤ,エリヤ,固有名詞
エリシャ,エリシャ,エリシャ,固有名詞
ヨハネ,ヨハネ,ヨハネ,固有名詞
ルカ,ルカ,ルカ,固有名詞
マタイ,マタイ,マタイ,固有名詞
マルコ,マルコ,マルコ,固有名詞
ユダ,ユダ,ユダ,固有名詞
エデン,エデン,エデン,固有名詞
ピソン,ピソン,ピソン,固有名詞
ギホン,ギホン,ギホン,固有名詞
ヒデケル,ヒデケル,ヒデケル,固有名詞
ユフラテ,ユフラテ,ユフラテ,固有名詞
アッスリヤ,アッスリヤ,アッスリヤ,固有名詞
クシ,クシ,クシ,固有名詞
ハビラ,ハビラ,ハビラ,固有名詞
"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
        f.write(user_dict_data)
    return f.name

def tokenize_japanese(text):
    """
    日本語テキストをトークン化する
    """
    t = Tokenizer()
    tokens = []
    for token in t.tokenize(text):
        # 品詞が名詞、動詞、形容詞、副詞の場合のみトークンとして追加
        pos = token.part_of_speech.split(',')[0]
        if pos in ['名詞', '動詞', '形容詞', '副詞']:
            tokens.append(token.surface)
    return tokens

if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='聖書のテキスト分析')
    parser.add_argument('--suffix', type=str, help='出力ファイル名の末尾に追加する文字列')
    args = parser.parse_args()
    
    create_analysis('input/target.txt', args.suffix)
