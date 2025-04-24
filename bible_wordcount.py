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
    # 文を分割
    sentences = re.split(r'[。！？]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 文の長さを計算
    lengths = [len(s) for s in sentences]
    
    return {
        '平均文長': mean(lengths),
        '中央値文長': median(lengths),
        '標準偏差': stdev(lengths) if len(lengths) > 1 else 0,
        '最短文長': min(lengths),
        '最長文長': max(lengths),
        '文の総数': len(lengths)
    }

def analyze_pos_distribution(text):
    # Tokenizerを初期化
    t = Tokenizer()
    
    # 品詞の分布を分析
    pos_counter = Counter()
    for token in t.tokenize(text):
        pos = token.part_of_speech.split(',')[0]
        pos_counter[pos] += 1
    
    return pos_counter

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
    # Tokenizerを初期化
    t = Tokenizer()
    
    # 固有名詞を抽出
    proper_nouns = []
    for token in t.tokenize(text):
        pos = token.part_of_speech.split(',')
        if pos[0] == '名詞' and pos[1] == '固有名詞':
            proper_nouns.append(token.surface)
    
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
    
    # Tokenizerを初期化
    t = Tokenizer()
    
    # 日本語のストップワード（名詞用に調整）
    stop_words = {'それ', 'これ', 'あれ', 'どれ', 'ここ', 'そこ', 'あそこ', 'どこ',
                 'だれ', 'なに', '方々', 'ため', 'とき', 'もの', 'こと', 'ところ',
                 'よう', 'ほう', 'まま', 'みな', 'かた', 'われわれ', 'あなた',
                 'たち','うち', 'がた', '彼ら', '人々','これら', 'わたし', 'すべて',
                 'もろもろ', 'ゆえ', '今や', '間', '上', '中', '下', '前', '後',
                 '内', '外', '他', '私', '我々', '彼', '此', '其', '何', '時',
                 '者', '人', '方', '物', '事', '所', '場合', '部分', '問題', '状態',
                 '結果', '関係', '相手', '程度', '目的', '理由', '原因', '結論',
                 '意味', '意見', '考え', '気持ち', '感じ', '言葉', '話', '声',
                 '顔', '目', '手', '足', '体', '頭', '心', '気', '空気', '天気',
                 '今日', '明日', '昨日', '今', '時間', '場所', '一つ', '二つ',
                 '三つ', '四つ', '五つ', '一人', '二人', '三人', '四人', '五人',
                 'する', 'れる', 'られる', 'せる', 'させる', 'なる', 'なら', 'ならば',
                 'です', 'ます', 'でした', 'ました', 'でしょう', 'ましょう', 'だろう',
                 'かもしれない', 'かもしれぬ', 'かもしれん', 'かもしれず', 'かもしれ',
                 'だろうか', 'でしょうか', 'だろうね', 'でしょうね', 'だろうよ',
                 'でしょうよ', 'だろうな', 'でしょうな', 'だろうに', 'でしょうに',
                 'だろうが', 'でしょうが', 'だろうと', 'でしょうと', 'だろうから',
                 'でしょうから', 'だろうし', 'でしょうし', 'だろうけど', 'でしょうけど',
                 'だろうけれど', 'でしょうけれど', 'だろうが', 'でしょうが', 'だろうと',
                 'でしょうと', 'だろうから', 'でしょうから', 'だろうし', 'でしょうし',
                 'だろうけど', 'でしょうけど', 'だろうけれど', 'でしょうけれど'}
    
    # 分析を実行
    tokens = tokenize_japanese(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    word_freq = Counter(tokens)
    frequent_words = [(word, count) for word, count in word_freq.most_common(10) if count >= 3]
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
                    width: 100%;
                    margin: 0;
                    padding: 0;
                    background-color: white;
                }}
                .container {{
                    background-color: white;
                    border-radius: 8px;
                    padding: 30px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    width: 100%;
                    max-width: 100%;
                    box-sizing: border-box;
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
                    width: 100%;
                    box-sizing: border-box;
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
                .highlight {{
                    background-color: #fff3cd;
                    padding: 2px 4px;
                    border-radius: 3px;
                }}
                .text-container {{
                    margin-bottom: 30px;
                    font-size: 16px;
                    line-height: 1.8;
                    padding: 20px;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    width: 100%;
                    box-sizing: border-box;
                }}
                .frequent-words {{
                    margin: 20px 0;
                    padding: 20px;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    width: 100%;
                    box-sizing: border-box;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    background-color: white;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    box-sizing: border-box;
                }}
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                @media (max-width: 768px) {{
                    .container {{
                        padding: 15px;
                    }}
                    .section {{
                        padding: 15px;
                    }}
                    .text-container {{
                        padding: 15px;
                        font-size: 14px;
                    }}
                    .frequent-words {{
                        padding: 15px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>テキスト分析結果</h1>
                
                <div class="section">
                    <h2>分析対象テキスト</h2>
                    <div class="highlight">注: 色付きで表示されている単語は、上位10個の頻出単語です。同じ単語は同じ色で表示されています。</div>
                    <div class="text-container">
                        {highlight_frequent_words(text, frequent_words)}
                    </div>
                </div>
                
                <div class="section">
                    <h2>頻出単語トップ10</h2>
                    <div class="frequent-words">
                        <table>
                            <tr>
                                <th>単語</th>
                                <th>回数</th>
                            </tr>
                            {''.join(f'<tr><td><span style="background-color: {get_color_for_word(word, {}, i)}; color: white; padding: 2px 4px; border-radius: 3px;">{word}</span></td><td>{count}回</td></tr>' for i, (word, count) in enumerate(frequent_words))}
                        </table>
                    </div>
                </div>
                
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
                    <div class="stats">
                        {''.join(f'''
                        <div class="stat-item">
                            <p class="label">{pos}</p>
                            <p class="value">{count}回</p>
                            <div class="samples">
                                {', '.join([word for word, freq in sorted(Counter([token.surface for token in t.tokenize(text) if token.part_of_speech.split(',')[0] == pos]).items(), key=lambda x: x[1], reverse=True)[:10]])}
                            </div>
                        </div>
                        ''' for pos, count in sorted(pos_dist.items(), key=lambda x: x[1], reverse=True))}
                    </div>
                </div>
            </div>
        </body>
        </html>
        """)
    
    return output_file

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
