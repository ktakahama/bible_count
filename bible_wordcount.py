import pandas as pd
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import MeCab
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

# .envファイルを読み込む
load_dotenv()

# 必要なNLTKデータをダウンロード
nltk.download('punkt')
nltk.download('stopwords')

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
    # ユーザー辞書を作成
    user_dict_file = create_user_dict()
    
    # ユーザー辞書を使用してTokenizerを初期化
    t = Tokenizer(udic=user_dict_file, udic_enc="utf8", udic_type="csv")
    
    # 品詞の分布を分析
    pos_counter = Counter()
    for token in t.tokenize(text):
        pos = token.part_of_speech.split(',')[0]
        pos_counter[pos] += 1
    
    # 一時ファイルを削除
    os.remove(user_dict_file)
    
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

def create_analysis(text_file, suffix=None):
    # テキストファイルを読み込む
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
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
                 '三つ', '四つ', '五つ', '一人', '二人', '三人', '四人', '五人'}
    
    # 日本語テキスト用にトークン化
    tokens = tokenize_japanese(text)
    
    # ストップワードを除去
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
    # 単語の頻度をカウント
    word_freq = Counter(tokens)
    
    # 上位10個の単語を取得（3回以上出現するもののみ）
    frequent_words = [(word, count) for word, count in word_freq.most_common(10) if count >= 3]
    
    # 出力ファイル名を日付付きで作成
    current_date = datetime.now().strftime('%Y%m%d')
    if suffix:
        output_file = f'output/analysis_{current_date}_{suffix}.md'
    else:
        output_file = f'output/analysis_{current_date}.md'
    
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs('output', exist_ok=True)
    
    # 結果をファイルに書き込む
    with open(output_file, 'w', encoding='utf-8') as f:
        # CSSスタイルの追加
        f.write("""<style>
            body {
                font-family: 'Helvetica Neue', Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                margin: 0;
                padding: 20px;
            }
            .container {
                max-width: 100%;
                margin: 0 auto;
                padding: 0 20px;
            }
            .section {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .section-title {
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
                margin-top: 0;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background-color: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            th, td {
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #3498db;
                color: white;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .highlight {
                background-color: #fff3cd;
                padding: 2px 4px;
                border-radius: 3px;
            }
            .text-container {
                margin-bottom: 30px;
                font-size: 16px;
                line-height: 1.8;
                padding: 20px;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .frequent-words {
                margin: 20px 0;
                padding: 20px;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .analysis-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
            }
            @media (max-width: 768px) {
                .container {
                    padding: 0 10px;
                }
                .section {
                    padding: 15px;
                }
                th, td {
                    padding: 8px 10px;
                }
            }
        </style>\n\n""")
        
        f.write('<div class="container">\n')
        f.write("# 聖書テキスト分析レポート\n\n")
        
        # 分析対象テキストセクション
        f.write('<div class="section">\n')
        f.write('<h2 class="section-title">分析対象テキスト</h2>\n')
        f.write('<div class="highlight">注: 色付きで表示されている単語は、上位10個の頻出単語です。同じ単語は同じ色で表示されています。</div>\n\n')
        
        # テキストコンテナ
        f.write('<div class="text-container">\n')
        f.write(highlight_frequent_words(text, frequent_words) + "\n")
        f.write('</div>\n\n')
        
        # 頻出単語セクション
        f.write('<div class="frequent-words">\n')
        f.write('  <h3>頻出単語トップ10</h3>\n')
        f.write('  <table>\n')
        f.write('    <tr>\n')
        f.write('      <th>単語</th>\n')
        f.write('      <th>回数</th>\n')
        f.write('    </tr>\n')
        for i, (word, count) in enumerate(frequent_words):
            color = get_color_for_word(word, {}, i)
            f.write(f'    <tr>\n')
            f.write(f'      <td><span style="background-color: {color}; color: white; padding: 2px 4px; border-radius: 3px;">{word}</span></td>\n')
            f.write(f'      <td>{count}回</td>\n')
            f.write(f'    </tr>\n')
        f.write('  </table>\n')
        f.write('</div>\n')
        f.write('</div>\n\n')
        
        # 分析結果セクション
        f.write('<div class="section">\n')
        f.write('<h2 class="section-title">分析結果</h2>\n\n')
        
        # 分析結果をグリッドで表示
        f.write('<div class="analysis-container">\n')
        
        # テキスト統計
        f.write('<div>\n')
        f.write('<h3>1. テキスト統計</h3>\n')
        f.write('<table>\n')
        f.write('  <tr>\n')
        f.write('    <th>項目</th>\n')
        f.write('    <th>値</th>\n')
        f.write('  </tr>\n')
        f.write(f'  <tr><td>総単語数</td><td>{len(tokens)}</td></tr>\n')
        f.write(f'  <tr><td>ユニークな単語数</td><td>{len(set(tokens))}</td></tr>\n')
        f.write(f'  <tr><td>語彙の豊富さ</td><td>{len(set(tokens))/len(tokens):.3f}</td></tr>\n')
        f.write('</table>\n')
        f.write('</div>\n')
        
        # 文の長さの分析
        f.write('<div>\n')
        f.write('<h3>2. 文の長さの分析</h3>\n')
        sentence_stats = analyze_sentence_lengths(text)
        f.write('<table>\n')
        f.write('  <tr>\n')
        f.write('    <th>統計量</th>\n')
        f.write('    <th>値</th>\n')
        f.write('  </tr>\n')
        for stat, value in sentence_stats.items():
            f.write(f'  <tr><td>{stat}</td><td>{value}</td></tr>\n')
        f.write('</table>\n')
        f.write('</div>\n')
        
        # 品詞の分布
        f.write('<div>\n')
        f.write('<h3>3. 品詞の分布</h3>\n')
        pos_dist = analyze_pos_distribution(text)
        f.write('<table>\n')
        f.write('  <tr>\n')
        f.write('    <th>品詞</th>\n')
        f.write('    <th>出現回数</th>\n')
        f.write('  </tr>\n')
        for pos, count in pos_dist.most_common():
            f.write(f'  <tr><td>{pos}</td><td>{count}回</td></tr>\n')
        f.write('</table>\n')
        f.write('</div>\n')
        
        f.write('</div>\n')  # analysis-containerの終了
        f.write('</div>\n')  # sectionの終了
        f.write('</div>\n')  # containerの終了
    
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
    # ユーザー辞書を作成
    user_dict_file = create_user_dict()
    
    # ユーザー辞書を使用してTokenizerを初期化
    t = Tokenizer(udic=user_dict_file, udic_enc="utf8", udic_type="csv")
    
    # トークン化とフィルタリング
    tokens = []
    for token in t.tokenize(text):
        pos = token.part_of_speech.split(',')
        pos_type = pos[0]
        
        # 名詞（固有名詞、一般名詞、代名詞など）
        if pos_type == '名詞':
            # 1文字の名詞は除外
            if len(token.surface) > 1:
                tokens.append(token.surface)  # 表層形を使用
        
        # 動詞（基本形ではなく表層形を使用）
        elif pos_type == '動詞':
            tokens.append(token.surface)  # 表層形を使用
        
        # 形容詞
        elif pos_type == '形容詞':
            tokens.append(token.surface)  # 表層形を使用
        
        # 副詞
        elif pos_type == '副詞':
            tokens.append(token.surface)  # 表層形を使用
    
    # 「モー」を「モーセ」に置換、「テロ」を「ペテロ」に置換
    tokens = ['モーセ' if token == 'モー' else 'ペテロ' if token == 'テロ' else token for token in tokens]
    
    # 一時ファイルを削除
    os.remove(user_dict_file)
    
    return tokens

if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='聖書のテキスト分析')
    parser.add_argument('--suffix', type=str, help='出力ファイル名の末尾に追加する文字列')
    args = parser.parse_args()
    
    create_analysis('input/target.txt', args.suffix)
