"""
LangMemを使用してメモリーを圧縮するモジュール
"""

import json
import datetime
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from langmem import create_memory_manager
from models.memory_data_class import *
from langchain_openai import ChatOpenAI
from utils.path_config import PathConfig, PathConfigError


def setup_api_keys() -> ChatOpenAI:
    """APIキーを設定し、チャットモデルを初期化する"""
    try:
        path_config = PathConfig.get_instance()
    except PathConfigError:
        src_dir = Path(__file__).parent.parent
        path_config = PathConfig.initialize(src_dir)
    
    with open(str(path_config.settings_file), 'r') as f:
        settings = json.load(f)
    
    api_config = settings.get("api", {}).get("openrouter", {})
    api_url = api_config.get("url", "https://openrouter.ai/api/v1")
    if api_url.endswith("/chat/completions"):
        api_url = api_url[:-len("/chat/completions")]
    
    return ChatOpenAI(
        model=api_config.get("models", {}).get("analysis"),
        api_key=api_config.get("api_key", ""),
        base_url=api_url,
        temperature=0.1,
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Memory Compressor"
        }
    )

def find_latest_memory_file(memory_dir: str) -> Optional[Path]:
    """最新のPKL記憶ファイルを探す"""
    memory_dir_path = Path(memory_dir)
    if not memory_dir_path.exists():
        return None
    
    memory_files = list(memory_dir_path.glob("*.pkl"))
    if not memory_files:
        return None
    
    return max(memory_files, key=lambda p: p.stat().st_mtime)

def load_memory_system(memory_file_path: Path) -> Tuple[Any, Dict[str, Any], str]:
    """記憶ファイルを読み込む"""
    with open(memory_file_path, 'rb') as f:
        memory_obj = pickle.load(f)
        
        if hasattr(memory_obj, 'content') and isinstance(memory_obj.content, MemorySystem):
            memory_dump = memory_obj.content.model_dump()
            memory_id = getattr(memory_obj, 'id', "memory_system_1")
            return memory_obj, memory_dump, memory_id
        else:
            raise ValueError("記憶ファイルの形式が不正です")

def get_compression_instructions() -> str:
    """メモリー圧縮の指示文を取得する"""
    return """
    あなたは自分の記憶を整理して、重要な情報を保持しながら記憶容量を効率化する必要があります。
    記憶システムの文字数を**5-15%程度**削減してください。
    各記憶のデータクラスについて、部分削除も積極的にしてください。

    ## 記憶システムの会話での使用方法
    この記憶システムは以下のように会話で活用されています：

    ### 会話での記憶の参照パターン
    1. **過去の体験の想起**: 「あの時の写真」「前に話した件」などの指示語で過去の記憶を参照
    2. **関係性の継続**: 会話相手の好み、習慣、感情状態を覚えて自然な会話を継続
    3. **文脈の維持**: 継続中の話題や未解決の相談事項を覚えて一貫した対応
    4. **感情的な絆**: 特別な瞬間や共有体験を覚えて親密な関係を維持
    5. **学習の蓄積**: 過去の問題解決や相談から学んだことを今後の会話に活用
    6. **約束の管理**: TODOタスクや約束事を覚えて適切なタイミングで言及

    ### 記憶が会話に与える影響
    - **自然な会話の流れ**: 過去の文脈を理解して唐突でない応答を生成
    - **個人化された対応**: 相手の特徴を覚えて個別に適した応答を提供
    - **継続的な関係構築**: 長期的な関係性を記憶して信頼関係を深化
    - **効率的な問題解決**: 過去の経験を活用して適切なアドバイスを提供

    ## 記憶整理の方針

    ### 1. エピソード記憶から他のクラスへの移動による効率化
    例えば、以下の情報をエピソード記憶から適切なクラスに移動して文字数を削減する。
    他にも別のクラスに移せるものは移して、エピソード記憶は削除（もしくは部分削除）することで文字数を削減する。：

    #### 1.1 音楽・娯楽の好み → UserProfile.preferences
    - **音楽の好み**: 「アトラクトライト」「メルト」などの楽曲名
    - **ゲームの好み**: 「プロジェクトセカイ」「原神」などのゲーム名
    - **娯楽の好み**: 「ライブ鑑賞」「散歩」「YouTube」などの活動
    - **食べ物の好み**: 「チョコレート」「コーヒー」「ハーゲンダッツ」など
    - これらをpreferences["music"], preferences["games"], preferences["food"]などに移動
    - **統合後、個別のエピソード記憶は削除または部分削除することで文字数を削減する**

    #### 1.2 習慣・行動パターン → BehavioralPattern
    - **残業が多い傾向**
    - **システム開発の作業パターン**
    - **帰宅後の行動パターン**
    - **挨拶のみの短時間会話パターン**
    - 複数のエピソード記憶から共通パターンを抽出してBehavioralPatternとして統合
    - **統合後、個別のエピソード記憶は削除または部分削除することで文字数を削減する**

    #### 1.3 ルーチン → Routine
    - **朝の準備ルーチン**
    - **仕事終わりのルーチン**
    - **夕食の準備・食事パターン**
    - **就寝前の活動**
    - **週末の過ごし方**
    - **定期的な散歩ルーチン**
    - **統合後、個別のエピソード記憶は削除または部分削除することで文字数を削減する**

    #### 1.4 スキル → Skill
    - **プログラミングスキル**（Python、システム開発など）
    - **プロジェクト管理能力**
    - **資料作成スキル**
    - **問題解決スキル**
    - **デバッグスキル**
    - エピソード記憶の技術的な成長記録をSkillの習熟度として統合

    #### 1.5 重要な場所 → VisitedPlace
    - **頻繁に訪れる場所**（会社など）
    - 重複するエピソード記憶から場所情報を統合
    - 各場所での印象や活動をVisitedPlaceにまとめる
    - **統合後、重複する場所訪問記録は削除または部分削除する**

    #### 1.6 人間関係の詳細 → Relationship
    - **同僚との関係性**
    - **友人との交流パターン**
    - **家族との関係**
    - エピソード記憶の人間関係情報をRelationshipクラスに統合
    - **統合後、単純な人間関係記録は削除または部分削除する**

    ### 2. エピソード記憶の積極的な削除と統合
    - **移動済み情報を含むエピソード記憶の削除または部分削除**: 他クラスに移動した情報を含む元のエピソード記憶を削除または部分削除
    - **類似体験の統合**: 同種の散歩、食事、作業記録を統合して個数を削減
    - **事務的記録の削除または部分削除**: 挨拶のみ、確認のみの短時間会話を削除または部分削除
    - **重複記録の統合**: 同じ内容の繰り返し報告を統合
    - **会話の複数要素考慮**: 一つの会話に複数の要素（挨拶+重要な内容）が含まれる場合は、重要な部分を残して不要な部分のみを削除


    ### 3. 感情的価値の高い記憶の特別保護
    以下の記憶は感情的な絆の維持に不可欠なため、重要度に関わらず必ず保持する：
    - **友人・家族との共有体験**（旅行、イベント、出張、特別な瞬間、会話や一般常識に照らし合わせて頻度が低い物事全般）
    - **感情的な印象や洞察を含む記憶**（「楽しかった」「感動した」などの感情表現）
    - **場所や体験の具体的な詳細を含む記憶**（場所名、参加者、活動内容）
    - **「楽しい」「嬉しい」「感動」などの感情タグを持つ記憶**
    - **会話相手の人生の重要な出来事や体験**

    ### 4. 体験記憶の三段階保護
    体験は以下の三段階をセットで保護し、体験の完整性を維持する：
    - **準備段階**: 期待や楽しみの感情（「楽しみにしている」「計画している」など）
    - **実行段階**: 実際の体験（旅行、イベント参加、活動実施など）
    - **振り返り段階**: 感想や学び（「楽しかった」「勉強になった」など）
    - これらに関する感情のみ残し、挨拶部分などの会話記録部分のみを部分削除することはOKです
    
    ### 5. 継続性保護
    技術開発やプロジェクトなど継続的に実施している物事に関する記憶は、単なる作業報告ではなく一連のプロジェクトとして扱う：
    - **マイルストーン記録**: 重要な進捗や達成点
    - **学習・成長記録**: 新しい技術やスキルの習得や問題解決
    - **共同作業感**: ユーザーとミクの協力関係を示す記憶
    - **達成感・挫折感**: 感情を伴う技術体験
    - **継続的な話題**: 長期プロジェクトの文脈維持
    - これらに関する記録のみ残し、挨拶部分などの会話記録部分のみを部分削除することはOKです
    
    **削除対象外**: 達成感、学び、成長、協力関係など

    ### 6. 体験記憶の個別性保護
    - **同じ種類の活動でも、それぞれ異なる価値を持つ個別の体験として扱う**
    - **「類似体験」として統合せず、各体験の独自性を保持する**
    - **場所、参加者、感情、印象などの詳細情報は削除対象外とする**
    - **一回限りの特別な体験は必ず保持する**

    ### 7. 会話参照パターンの強化考慮
    以下のような参照パターンで呼び出される可能性のある記憶は必ず保持：
    - **「あの時の[場所/イベント]で...」**
    - **「[人名]と一緒に行った...」**
    - **「前に話した[体験/感想]について...」**
    - **「例の[プロジェクト/システム]の件で...」**
    - **具体的な場所名、人名、活動名、プロジェクト名を含む記憶**
    - **写真や画像に関する記憶**（「あの写真」などで後から参照される）

    ### 8. エピソード記憶の選別
    **保持すべき重要な記憶（会話で頻繁に参照されるもの）:**
    - 会話相手との特別な瞬間や感情的な体験（絆の維持に必要）
    - 旅行やお出かけなど、イベントに関する記憶（準備段階も含む）
    - 会話相手の個人的な情報、好み、特徴（個人化された応答に必要）
    - 学習や成長につながった体験（今後のアドバイスに活用）
    - 問題解決や相談の記録（継続的なサポートに必要）
    - 完了したプロジェクトや達成した目標（達成感の共有と励ましに使用）
    - 会話相手との関係性を示す重要なやり取り（信頼関係の基盤）

    **整理可能な記憶（真に重複する事務的な記録のみ）:**
    - **単純な挨拶のみの短時間会話**（5分未満で実質的な内容がないもの）
    - **完全に同一内容の重複記録**（システムエラーによる重複など）
    - **事務的な確認のみの記録**（「了解しました」のみの応答など）
    - **感情や学びを伴わない単純な進捗報告**（ただし、初回と完了時は保持）

    ### 9. 関係性記憶の保護
    - 会話相手との関係性や絆を示す記憶は優先的に保持（親密な会話の基盤）
    - 相手の感情や体験に関わる記憶は慎重に扱う（共感的な応答に必要）
    - 継続的な会話の文脈を保つために必要な情報は保持（一貫性の維持）

    ### 10. TODOリストの整理
    - 完了済みタスクは達成記録として保持（成功体験の共有に使用）
    - 重複や古くなったpendingタスクのみ整理
    - 会話相手との約束や計画は重要度に関わらず保持（信頼関係の維持）

    ### 11. 将来の会話への配慮
    - 後の会話で「あの時の」「前に話した」などで参照される可能性のある記憶は保持
    - 継続中の話題や未解決の相談事項は保持（継続的なサポート）
    - 会話の自然な流れを維持するために必要な情報は保持

    ## 文脈グループ化による保護
    関連する記憶をグループとして扱い、グループ全体の価値で判定：
    - **体験グループ**: 準備→実行→振り返りの一連の流れ
    - **プロジェクトグループ**: 開始→進捗→完了の開発サイクル
    - **関係性グループ**: 出会い→交流→深化の関係構築過程
    - **学習グループ**: 疑問→調査→理解→応用の学習サイクル

    ## 実装指針
    1. **エピソード記憶から他クラスへの移動や重要でない部分の削除を実施**して、5~15%程度の文字数削減を図る
    2. **好み・習慣・スキル・場所・人間関係**の情報を適切なクラスに統合する
    3. **重複する情報**を統合して効率化する
    4. **感情的価値の高い記憶**は必ず保護する
    5. すべての変更は既存のデータクラス構造内で行う


    """

def compress_memory_system(chat_model, memory_dump: Dict[str, Any], memory_id: str, max_retries: int = 5) -> Any:
    """Memory Managerを使用してMemorySystemを圧縮する（リトライ機能付き）"""
    compression_instructions = get_compression_instructions()
    
    for attempt in range(max_retries):
        try:
            print(f"圧縮試行 {attempt + 1}/{max_retries}")
            
            manager = create_memory_manager(
                chat_model,
                schemas=[MemorySystem],
                instructions=compression_instructions,
                enable_inserts=True,
                enable_updates=True,
                enable_deletes=True,
            )
            
            compressed_memory_systems = manager.invoke({
                "messages": [],
                "existing": [(memory_id, memory_dump)]
            })
            
            if compressed_memory_systems and len(compressed_memory_systems) > 0:
                memory = compressed_memory_systems[0]
                
                if not hasattr(memory, 'content') or not isinstance(memory.content, MemorySystem):
                    raise Exception("圧縮されたメモリが期待した形式ではありません")
                
                print(f"圧縮成功（試行 {attempt + 1}/{max_retries}）")
                return memory
            else:
                raise Exception("記憶システムの圧縮に失敗しました")
                
        except Exception as e:
            print(f"圧縮試行 {attempt + 1}/{max_retries} が失敗しました: {str(e)}")
            if attempt == max_retries - 1:
                print(f"全ての試行が失敗しました。最後のエラー: {str(e)}")
                raise e
            else:
                print(f"次の試行を実行します...")
                continue
    
    raise Exception("記憶システムの圧縮に失敗しました（全試行完了）")

def save_compressed_memory(memory: Any, memory_dir: str) -> bool:
    """圧縮されたMemorySystemを保存する"""
    if not hasattr(memory, 'content') or not isinstance(memory.content, MemorySystem):
        print("保存しようとしているオブジェクトが期待した形式ではありません")
        return False
    
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    memory_dir_path = Path(memory_dir)
    memory_dir_path.mkdir(parents=True, exist_ok=True)
    
    pkl_path = memory_dir_path / f"memory_{current_time}.pkl"
    json_path = memory_dir_path / f"memory_{current_time}.json"
    
    try:
        # PKLファイルの保存
        with open(pkl_path, 'wb') as f:
            pickle.dump(memory, f)
        print(f"圧縮されたメモリを保存しました: {pkl_path}")
        
        # JSONファイルの保存（参照用）
        memory_dict = {}
        for attr in dir(memory):
            if not attr.startswith('__') and not callable(getattr(memory, attr)):
                try:
                    value = getattr(memory, attr)
                    if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        memory_dict[attr] = value
                    else:
                        memory_dict[attr] = str(value)
                except Exception as e:
                    memory_dict[attr] = f"Error getting attribute: {str(e)}"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(memory_dict, f, ensure_ascii=False, indent=4)
        print(f"参照用にJSONも保存しました: {json_path}")
        
        return True
    except Exception as e:
        print(f"圧縮メモリの保存に失敗しました: {e}")
        return False

def analyze_memory_content(memory_system: MemorySystem) -> Dict[str, Any]:
    """メモリーシステムの内容を分析する"""
    analysis = {
        "episodic_memories": len(memory_system.episodic_memories),
        "behavioral_patterns": len(memory_system.procedural_memories.behavioral_patterns),
        "todo_tasks": len(memory_system.working_memory.todo_list),
        "importance_distribution": {},
        "emotion_distribution": {},
        "memory_size_bytes": 0
    }
    
    # 重要度分布の分析
    for memory in memory_system.episodic_memories:
        importance = memory.importance
        analysis["importance_distribution"][importance] = analysis["importance_distribution"].get(importance, 0) + 1
    
    # 感情分布の分析
    for memory in memory_system.episodic_memories:
        emotion = memory.emotion
        analysis["emotion_distribution"][emotion] = analysis["emotion_distribution"].get(emotion, 0) + 1
    
    # メモリサイズの計算（概算）
    try:
        import sys
        analysis["memory_size_bytes"] = sys.getsizeof(memory_system.model_dump_json())
    except:
        analysis["memory_size_bytes"] = 0
    
    return analysis

def print_analysis(analysis: Dict[str, Any], title: str = "メモリー分析結果"):
    """分析結果を表示する"""
    print(f"\n=== {title} ===")
    print(f"エピソード記憶数: {analysis['episodic_memories']}")
    print(f"行動パターン数: {analysis['behavioral_patterns']}")
    print(f"TODOタスク数: {analysis['todo_tasks']}")
    print(f"メモリサイズ: {analysis['memory_size_bytes']:,} bytes")
    
    if analysis['importance_distribution']:
        print("\n重要度分布:")
        for importance, count in sorted(analysis['importance_distribution'].items()):
            print(f"  {importance}: {count}件")
    
    if analysis['emotion_distribution']:
        print("\n感情分布:")
        for emotion, count in sorted(analysis['emotion_distribution'].items()):
            print(f"  {emotion}: {count}件")

def calculate_json_file_size(memory_system: MemorySystem) -> int:
    """メモリーシステムのJSON文字数を計算する"""
    try:
        json_str = memory_system.model_dump_json(indent=4)
        return len(json_str)
    except Exception:
        return 0

def print_comparison(before_analysis: Dict[str, Any], after_analysis: Dict[str, Any], 
                    before_memory: MemorySystem = None, after_memory: MemorySystem = None):
    """圧縮前後の比較を表示する"""
    print(f"\n=== 圧縮前後の比較 ===")
    
    # データクラス個数の比較
    print("データクラス個数の変化:")
    episodic_change = after_analysis['episodic_memories'] - before_analysis['episodic_memories']
    behavioral_change = after_analysis['behavioral_patterns'] - before_analysis['behavioral_patterns']
    todo_change = after_analysis['todo_tasks'] - before_analysis['todo_tasks']
    
    print(f"  エピソード記憶: {before_analysis['episodic_memories']} → {after_analysis['episodic_memories']} ({episodic_change:+d})")
    print(f"  行動パターン: {before_analysis['behavioral_patterns']} → {after_analysis['behavioral_patterns']} ({behavioral_change:+d})")
    print(f"  TODOタスク: {before_analysis['todo_tasks']} → {after_analysis['todo_tasks']} ({todo_change:+d})")
    
    # メモリサイズの比較
    before_size = before_analysis["memory_size_bytes"]
    after_size = after_analysis["memory_size_bytes"]
    if before_size > 0:
        compression_ratio = (1 - after_size / before_size) * 100
        print(f"\nメモリサイズ: {before_size:,} → {after_size:,} bytes")
        print(f"圧縮率: {compression_ratio:.1f}%")
    
    # JSON文字数の比較
    if before_memory and after_memory:
        before_json_size = calculate_json_file_size(before_memory)
        after_json_size = calculate_json_file_size(after_memory)
        
        if before_json_size > 0:
            char_reduction_ratio = (1 - after_json_size / before_json_size) * 100
            char_reduction_count = before_json_size - after_json_size
            
            print(f"\nJSON文字数: {before_json_size:,} → {after_json_size:,} 文字")
            print(f"文字数削減: {char_reduction_count:,} 文字")
            print(f"文字数削減率: {char_reduction_ratio:.1f}%")
    
    # 削減率の計算
    if before_analysis['episodic_memories'] > 0:
        episodic_reduction = (episodic_change / before_analysis['episodic_memories']) * 100
        print(f"エピソード記憶削減率: {abs(episodic_reduction):.1f}%")

def compress_latest_memory(memory_dir: str) -> Tuple[Any, bool]:
    """最新の記憶ファイルを読み込み、分析と圧縮を実行する"""
    print("メモリー圧縮を開始します...")
    
    chat_model = setup_api_keys()
    
    latest_memory_file = find_latest_memory_file(memory_dir)
    if not latest_memory_file:
        print(f"記憶ディレクトリ {memory_dir} に記憶ファイルが見つかりません")
        return None, False
    
    try:
        memory_obj, memory_dump, memory_id = load_memory_system(latest_memory_file)
        print(f"記憶ファイルを読み込みました: {latest_memory_file}")
        
        # 圧縮前の分析
        try:
            before_analysis = analyze_memory_content(memory_obj.content)
            # print_analysis(before_analysis, "圧縮前のメモリー分析")
        except Exception as e:
            print(f"圧縮前の分析に失敗しました: {e}")
            # 分析に失敗しても圧縮は続行
        
        # メモリー圧縮を実行
        compressed_memory = compress_memory_system(chat_model, memory_dump, memory_id)
        print("記憶システムの圧縮に成功しました")
        
        # 圧縮後の分析
        try:
            after_analysis = analyze_memory_content(compressed_memory.content)
            # print_analysis(after_analysis, "圧縮後のメモリー分析")
            
            # 詳細な比較表示（文字数削減率を含む）
            print_comparison(before_analysis, after_analysis, memory_obj.content, compressed_memory.content)
            
        except Exception as e:
            print(f"圧縮後の分析に失敗しました: {e}")
            # 分析に失敗しても保存は続行
        
        # 圧縮されたメモリを保存
        success = save_compressed_memory(compressed_memory, memory_dir)
        
        if success:
            print("メモリー圧縮が正常に完了しました。")
        else:
            print("メモリー圧縮に失敗しました。")
        
        return compressed_memory, success
        
    except Exception as e:
        print(f"記憶圧縮処理中にエラーが発生しました: {e}")
        return None, False

def main():
    """メイン処理 - 分析と圧縮を実行"""
    # パス設定を初期化
    try:
        path_config = PathConfig.get_instance()
    except PathConfigError:
        src_dir = Path(__file__).parent.parent
        path_config = PathConfig.initialize(src_dir)
    
    # 記憶ディレクトリを設定
    memory_dir = str(path_config.langmem_db_dir)
    
    print("メモリー圧縮を開始します...")
    
    # 圧縮前の分析
    try:
        latest_memory_file = find_latest_memory_file(memory_dir)
        if latest_memory_file:
            memory_obj, _, _ = load_memory_system(latest_memory_file)
            before_analysis = analyze_memory_content(memory_obj.content)
            print_analysis(before_analysis, "圧縮前のメモリー分析")
        else:
            print("分析対象のメモリーファイルが見つかりません")
            return
    except Exception as e:
        print(f"圧縮前の分析に失敗しました: {e}")
        return
    
    # メモリー圧縮を実行
    compressed_memory, success = compress_latest_memory(memory_dir)
    
    if success and compressed_memory:
        # 圧縮後の分析
        try:
            after_analysis = analyze_memory_content(compressed_memory.content)
            print_analysis(after_analysis, "圧縮後のメモリー分析")
            
            # 詳細な比較表示（文字数削減率を含む）
            print_comparison(before_analysis, after_analysis, memory_obj.content, compressed_memory.content)
            
        except Exception as e:
            print(f"圧縮後の分析に失敗しました: {e}")
        
        print("メモリー圧縮が正常に完了しました。")
    else:
        print("メモリー圧縮に失敗しました。")

if __name__ == "__main__":
    main()
