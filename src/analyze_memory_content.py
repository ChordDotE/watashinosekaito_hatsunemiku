import json
import re
from pathlib import Path
import sys
from collections import OrderedDict
import ast

# カスタム例外クラス
class ConfigError(Exception):
    """設定関連のエラーを表す例外クラス"""
    pass

def analyze_episodic_memories(content):
    """
    エピソード記憶の内部要素を分析し、各要素の比率を計算する
    
    Args:
        content: contentフィールドの文字列
        
    Returns:
        dict: エピソード記憶の分析結果を含む辞書
    """
    # エピソード記憶部分を抽出
    episodic_start = content.find('episodic_memories=[')
    if episodic_start == -1:
        return None
    
    # 次のセクションの開始位置を見つける
    semantic_start = content.find('semantic_memories=')
    if semantic_start == -1:
        episodic_content = content[episodic_start:]
    else:
        episodic_content = content[episodic_start:semantic_start]
    
    # エピソード記憶の総文字数
    total_episodic_length = len(episodic_content)
    
    # 文字列をバイト配列に変換し、各文字がどの要素に属するかを記録
    char_attribution = [None] * total_episodic_length
    
    # EpisodicMemoryクラスの構造に基づいて要素を定義
    elements = {
        'header': 0,  # episodic_memories=[ の部分
        'memory_id': 0,
        'timestamp_start': 0,
        'timestamp_end': 0,
        'duration_minutes': 0,
        'location': 0,
        'participants': 0,
        'summary': 0,
        'activities': 0,
        'insights': 0,
        'future_improvements': 0,
        'emotion': 0,
        'importance': 0,
        'recall_count': 0,
        'last_recalled': 0,
        'retrieval_count': 0,
        'associated_episodic_ids': 0,
        'related_memories': 0,
        'extensions': 0,
        'structure': 0,  # 構造要素（括弧、カンマなど）
        'other': 0  # 上記以外の要素
    }
    
    # ヘッダー部分を記録
    header_match = re.search(r'episodic_memories=\[', episodic_content)
    if header_match:
        start, end = header_match.span()
        for i in range(start, end):
            char_attribution[i] = 'header'
        elements['header'] = end - start
    
    # 各要素のパターンを定義
    patterns = {
        'memory_id': r'memory_id=\'[^\']*\'',
        'timestamp_start': r'timestamp_start=\'[^\']*\'',
        'timestamp_end': r'timestamp_end=\'[^\']*\'',
        'duration_minutes': r'duration_minutes=\d+',
        'location': r'location=\'[^\']*\'',
        'participants': r'participants=\[[^\]]*\]',
        'summary': r'summary=\'[^\']*\'',
        'activities': r'activities=\[.*?\]|activities=None',
        'insights': r'insights=\[[^\]]*\]|insights=None',
        'future_improvements': r'future_improvements=\[[^\]]*\]|future_improvements=None',
        'emotion': r'emotion=\'[^\']*\'',
        'importance': r'importance=\d+\.\d+',
        'recall_count': r'recall_count=\d+',
        'last_recalled': r'last_recalled=\'[^\']*\'',
        'retrieval_count': r'retrieval_count=\d+',
        'associated_episodic_ids': r'associated_episodic_ids=\[[^\]]*\]',
        'related_memories': r'related_memories=\[[^\]]*\]',
        'extensions': r'extensions=\{[^\}]*\}'
    }
    
    # Activityクラスのフィールドのパターン
    activity_patterns = {
        'activity_time': r'time=\'[^\']*\'',
        'activity_description': r'description=\'[^\']*\'',
        'activity_participants': r'participants=\[[^\]]*\]',
        'activity_details': r'details=\'[^\']*\'|details=None'
    }
    
    # 要素辞書にActivityフィールドを追加
    for key in activity_patterns.keys():
        elements[key] = 0
    
    # 構造要素のパターン
    structure_pattern = r'EpisodicMemory\(|\), |Activity\(|\), |, |\[|\]|\{|\}'
    
    # 各要素の出現位置を記録
    for element, pattern in patterns.items():
        for match in re.finditer(pattern, episodic_content):
            start, end = match.span()
            # この範囲がまだ属性付けされていない場合のみ記録
            if all(attr is None for attr in char_attribution[start:end]):
                for i in range(start, end):
                    char_attribution[i] = element
                elements[element] += end - start
    
    # Activityフィールドの出現位置を記録
    for element, pattern in activity_patterns.items():
        for match in re.finditer(pattern, episodic_content):
            start, end = match.span()
            # この範囲がまだ属性付けされていない場合のみ記録
            if all(attr is None for attr in char_attribution[start:end]):
                for i in range(start, end):
                    char_attribution[i] = element
                elements[element] += end - start
    
    # 構造要素の出現位置を記録
    for match in re.finditer(structure_pattern, episodic_content):
        start, end = match.span()
        # この範囲がまだ属性付けされていない場合のみ記録
        if all(attr is None for attr in char_attribution[start:end]):
            for i in range(start, end):
                char_attribution[i] = 'structure'
            elements['structure'] += end - start
    
    # 未属性の文字数を計算
    elements['other'] = char_attribution.count(None)
    
    # 結果を辞書にまとめる
    results = {}
    for element, length in elements.items():
        percentage = (length / total_episodic_length) * 100
        results[element] = {
            'length': length,
            'percentage': percentage
        }
    
    return {
        'total_length': total_episodic_length,
        'elements': results
    }

def analyze_content_sections(file_path):
    """
    メモリファイルのcontentフィールドを分析し、各セクションの比率を計算する
    
    Args:
        file_path: 分析対象のJSONファイルのパス
        
    Returns:
        dict: 分析結果を含む辞書
        
    Raises:
        ConfigError: ファイルが見つからない、またはJSONとして解析できない場合
    """
    try:
        # ファイルを読み込む
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # contentフィールドを取得
        content = data.get('content', '')
        if not content:
            raise ConfigError("contentフィールドが見つかりません")
            
        total_length = len(content)
        
        # 主要セクションのキーワード
        section_keywords = [
            'participants=',
            'episodic_memories=',
            'semantic_memories=',
            'procedural_memories=',
            'working_memory=',
            'associative_memory=',
            'user_experience='
        ]
        
        # 各セクションの開始位置を特定
        section_positions = []
        for keyword in section_keywords:
            pos = content.find(keyword)
            if pos != -1:
                section_positions.append((keyword.rstrip('='), pos))
        
        # 開始位置でソート
        section_positions.sort(key=lambda x: x[1])
        
        # 各セクションの長さと比率を計算
        results = OrderedDict()
        for i, (section_name, start_pos) in enumerate(section_positions):
            # 次のセクションの開始位置または文字列の終わりまでを現在のセクションとする
            end_pos = section_positions[i+1][1] if i < len(section_positions) - 1 else total_length
            section_length = end_pos - start_pos
            percentage = (section_length / total_length) * 100
            results[section_name] = {
                'length': section_length,
                'percentage': percentage
            }
        
        return {
            'total_length': total_length,
            'sections': results
        }
    except FileNotFoundError:
        raise ConfigError(f"ファイル {file_path} が見つかりません")
    except json.JSONDecodeError:
        raise ConfigError(f"ファイル {file_path} の形式が不正です")

def main():
    """メイン関数"""
    try:
        # コマンドライン引数からファイルパスを取得するか、デフォルトパスを使用
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
        else:
            file_path = "miku/src/memory/langmem_db/memory_20250505_102028.json"
        
        # ファイルパスが存在するか確認
        if not Path(file_path).exists():
            raise ConfigError(f"ファイル {file_path} が見つかりません")
        
        # ファイルを読み込む
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # contentフィールドを取得
        content = data.get('content', '')
        
        # 分析を実行
        results = analyze_content_sections(file_path)
        
        # 結果を表示
        print(f"総文字数: {results['total_length']}")
        print("各セクションの比率:")
        
        # 比率の降順でソート
        sorted_sections = sorted(
            results['sections'].items(), 
            key=lambda x: x[1]['percentage'], 
            reverse=True
        )
        
        for section_name, data in sorted_sections:
            print(f"{section_name}: {data['length']} 文字 ({data['percentage']:.2f}%)")
        
        # エピソード記憶の詳細分析
        print("\nエピソード記憶の内部要素の分析:")
        episodic_results = analyze_episodic_memories(content)
        
        if episodic_results:
            print(f"エピソード記憶の総文字数: {episodic_results['total_length']}")
            
            # 比率の降順でソート
            sorted_elements = sorted(
                episodic_results['elements'].items(), 
                key=lambda x: x[1]['percentage'], 
                reverse=True
            )
            
            for element_name, data in sorted_elements:
                print(f"{element_name}: {data['length']} 文字 ({data['percentage']:.2f}%)")
        else:
            print("エピソード記憶の分析に失敗しました")
            
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
