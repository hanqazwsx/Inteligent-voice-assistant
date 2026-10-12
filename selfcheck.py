#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能语音备忘录 · 自检脚本
==========================

十组离线自检，**不需要麦克风、不需要 API Key、不需要安装重型依赖**：

  1. 文件完整性      —— requirements.txt / main.py / webui.py / 模板 / .env.example / README.md
  2. 语法与导入校验  —— AST 语法解析 + 未定义全局名静态扫描 + 模块导入 + 依赖可用性
  3. 数据库 Schema   —— 建表逻辑、必备字段、索引、写入/读取往返
  4. 模拟数据流测试  —— 伪造 ASR 文本与 DeepSeek 响应，跑通"识别→整理→解析→落库"
  5. 状态提示与导出  —— 四个状态提示文案、Markdown 追加/导出、DOCX、--replay 回放
  6. 性能与截断      —— CPU 默认档、处理期间暂停采集、丢块自愈、结束词剥离与后台探测
  7. 唤醒词与指令     —— 唤醒/分段/作废/退出四类指令、同音字容错、完整状态流转剧本
  8. 拼音模糊匹配     —— 实测的 ASR 错别字（錄音完畢 / 坐着飞 / 錯誤費）照样触发控制词
  9. 音频预处理与后端 —— 降噪/增益的数值行为、funasr 文本清洗、ASR 后端自动选择
 10. Web 看板与删除   —— 页面/JSON/SSE/导出接口、手动结束/放弃按钮、删卡同步

运行::

    python selfcheck.py

退出码 0 表示全部通过，1 表示存在失败项。
"""

from __future__ import annotations

import ast
import builtins
import importlib
import importlib.util
import inspect
import json
import sqlite3
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_FILES = ("requirements.txt", "main.py", "webui.py", "templates/index.html",
                  ".env.example", "README.md")
CHECKED_SOURCES = ("main.py", "webui.py", "selfcheck.py")
THIRD_PARTY = {
    "numpy": "音频数组运算",
    "sounddevice": "麦克风采集",
    "torch": "silero-vad 运行时",
    "silero_vad": "VAD 免唤醒断句",
    "faster_whisper": "本地语音识别",
    "pypinyin": "控制词拼音匹配（错别字容错）",
    "funasr": "可选的中文专用 ASR 后端（SenseVoiceSmall）",
    "openai": "DeepSeek API 客户端",
}

# --------------------------------------------------------------------------- #
# 报告收集
# --------------------------------------------------------------------------- #

RESULTS: list[tuple[bool, str, str]] = []
SKIPPED: list[tuple[str, str]] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    RESULTS.append((bool(ok), label, detail))
    mark = " OK " if ok else "FAIL"
    line = f"  [{mark}] {label}"
    if detail:
        line += f"\n         {detail}"
    print(line, flush=True)
    return bool(ok)


def skip(label: str, detail: str = "") -> None:
    """环境不具备条件时跳过（不算失败，也不计入通过数）。"""
    SKIPPED.append((label, detail))
    line = f"  [SKIP] {label}"
    if detail:
        line += f"\n         {detail}"
    print(line, flush=True)


SECTION_NO = 0
SECTION_TOTAL = 10


def section(title: str) -> None:
    global SECTION_NO
    SECTION_NO += 1
    print(f"\n[{SECTION_NO}/{SECTION_TOTAL}] {title}\n" + "-" * 72, flush=True)


# --------------------------------------------------------------------------- #
# 1. 文件完整性
# --------------------------------------------------------------------------- #


def check_files() -> None:
    section("文件完整性检查")
    for name in REQUIRED_FILES:
        path = PROJECT_ROOT / name
        if path.is_file():
            size = path.stat().st_size
            lines = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
            check(True, f"{name:<20} 存在", f"{size:,} 字节 / {lines} 行")
        else:
            check(False, f"{name:<20} 缺失", f"预期路径：{path}")
    extra = PROJECT_ROOT / ".gitignore"
    if extra.is_file():
        check(True, ".gitignore            存在（保护 .env 与 memos.db 不入库）")


# --------------------------------------------------------------------------- #
# 2. 语法与导入校验
# --------------------------------------------------------------------------- #


def collect_defined_names(tree: ast.AST) -> set[str]:
    """收集模块中被绑定的所有名字（含函数参数、推导式变量、异常别名）。"""
    names: set[str] = set(dir(builtins))
    names.update({"__name__", "__file__", "__doc__", "__package__", "__spec__", "__loader__", "__builtins__"})
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def find_undefined_names(tree: ast.AST) -> dict[str, int]:
    """保守地找出"读取了但从未被绑定"的全局名——基本等同于变量名拼写错误。"""
    defined = collect_defined_names(tree)
    undefined: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in defined:
            undefined.setdefault(node.id, node.lineno)
    return undefined


def check_syntax_and_imports() -> None:
    section("语法与导入校验")

    for name in CHECKED_SOURCES:
        path = PROJECT_ROOT / name
        if not path.is_file():
            check(False, f"{name:<20} 语法检查", "文件不存在")
            continue
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            check(False, f"{name:<20} 语法检查", f"第 {exc.lineno} 行：{exc.msg}")
            continue
        check(True, f"{name:<20} 语法检查通过", f"{len(source.splitlines())} 行，AST 解析成功")

        undefined = find_undefined_names(tree)
        if undefined:
            detail = "、".join(f"{n}(L{line})" for n, line in sorted(undefined.items(), key=lambda kv: kv[1]))
            check(False, f"{name:<20} 未定义引用扫描", f"发现可疑未绑定名字：{detail}")
        else:
            check(True, f"{name:<20} 未定义引用扫描", "所有被读取的名字均有定义")

    # --- 真正导入 main 模块，触发模块级代码执行 ---
    try:
        module = importlib.import_module("main")
        check(True, "import main         模块导入成功",
              f"__version__ = {getattr(module, '__version__', '未知')}")
    except Exception as exc:
        check(False, "import main         模块导入失败", f"{type(exc).__name__}: {exc}")
        return

    # --- 关键接口是否齐备 ---
    expected_api = (
        "Config", "SCHEMA_SQL", "REQUIRED_COLUMNS", "init_db", "insert_memo", "fetch_memos",
        "extract_json_object", "normalize_memo", "normalize_todos", "call_deepseek", "organize",
        "local_organize", "resolve_cn_time", "strip_control_words", "contains_stop_keyword",
        "cn_numeral_to_int", "build_vad_iterator", "VoiceListener", "Transcriber",
        "MemoPipeline", "Utterance", "parse_args", "main", "load_dotenv", "fallback_title",
        "guess_category", "parse_record_time", "print_memo", "run_doctor", "build_vad",
        # 唤醒词 / 指令 / 状态机
        "match_command", "match_command_detail", "find_keyword", "wake_word_forms",
        "clean_transcript", "_split_keywords", "AppState", "Broadcaster",
        "memo_to_dict", "start_web_server", "STATE_STANDBY", "STATE_RECORDING",
        "STATE_PROCESSING", "STATE_LABELS",
        # 拼音模糊匹配
        "pinyin_available", "char_syllables", "syllables_of", "syllable_coverage",
        "find_pinyin_keyword", "find_pinyin_window", "strip_pinyin_tail",
        "DEFAULT_SHORT_END_KEYWORDS",
        # 音频预处理 / ASR 后端
        "preprocess_audio", "denoise_spectral", "clean_funasr_text",
    )
    missing = [name for name in expected_api if not hasattr(module, name)]
    if missing:
        check(False, "关键接口齐备性", f"缺少：{', '.join(missing)}")
    else:
        check(True, "关键接口齐备性", f"全部 {len(expected_api)} 个公开接口均已定义")

    # --- 延迟导入策略：不装第三方包也能 import ---
    # 只看模块顶层语句；函数体内部的 import 属于"延迟导入"，正是想要的效果
    module_level: list[str] = []
    for node in ast.parse((PROJECT_ROOT / "main.py").read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            module_level += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module_level.append((node.module or "").split(".")[0])
    leaked = sorted({name for name in module_level if name in THIRD_PARTY})
    check(not leaked, "延迟导入策略",
          f"模块顶层直接导入了重型依赖：{'、'.join(leaked)}" if leaked
          else "torch / whisper / 音频库全部延迟到函数内导入，可在裸 Python 环境下自检")

    # --- 第三方依赖可用性（仅提示，不计入失败）---
    print("\n  第三方依赖可用性（缺失不影响本次离线自检）：", flush=True)
    for module_name, purpose in THIRD_PARTY.items():
        found = importlib.util.find_spec(module_name) is not None
        print(f"         [{'已安装' if found else '未安装'}] {module_name:<16} {purpose}", flush=True)


# --------------------------------------------------------------------------- #
# 3. 数据库 Schema 验证
# --------------------------------------------------------------------------- #


def check_schema(tmp_dir: Path) -> sqlite3.Connection | None:
    section("数据库 Schema 验证")
    import main

    db_path = tmp_dir / "schema_check.db"
    try:
        conn = main.init_db(db_path)
    except Exception as exc:
        check(False, "建表逻辑", f"{type(exc).__name__}: {exc}")
        return None
    check(True, "建表逻辑执行成功", f"数据库文件：{db_path.name}")

    columns = {row["name"]: row for row in conn.execute("PRAGMA table_info(memos)")}
    check(bool(columns), "memos 表已创建", f"共 {len(columns)} 列：{' / '.join(columns)}")

    for column in main.REQUIRED_COLUMNS:
        info = columns.get(column)
        check(info is not None, f"字段 {column:<12} 存在",
              f"类型={info['type']} 非空约束={bool(info['notnull'])}" if info else "字段缺失")
    extra = [c for c in columns if c not in main.REQUIRED_COLUMNS]
    check(True, "附加字段", f"{', '.join(extra)}（主键与创建时间，不影响业务字段）")

    pk = [name for name, info in columns.items() if info["pk"]]
    check(pk == ["id"], "主键约束", f"PRIMARY KEY = {pk}")

    indexes = [row["name"] for row in conn.execute("PRAGMA index_list(memos)")]
    check(len(indexes) >= 2, "索引已建立", f"{', '.join(indexes)}")

    # 幂等性：重复初始化不应报错或重复建表
    try:
        conn2 = main.init_db(db_path)
        same = {r["name"] for r in conn2.execute("PRAGMA table_info(memos)")} == set(columns)
        conn2.close()
        check(same, "建表幂等性", "重复 init_db 未破坏既有结构（IF NOT EXISTS 生效）")
    except Exception as exc:
        check(False, "建表幂等性", f"{type(exc).__name__}: {exc}")

    # 跨线程可用：MemoPipeline 可能被放进后台线程执行，连接必须撑得住
    result: dict = {}

    def use_from_another_thread() -> None:
        try:
            result["count"] = conn.execute("SELECT COUNT(*) FROM memos").fetchone()[0]
            result["ok"] = True
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"

    worker = threading.Thread(target=use_from_another_thread, name="schema-probe")
    worker.start()
    worker.join(timeout=5)
    check(result.get("ok") is True, "连接可跨线程使用",
          "check_same_thread=False 生效，MemoPipeline 放进后台线程不会崩"
          if result.get("ok") else result.get("error", "子线程未返回结果"))

    # 读写往返
    probe = {
        "record_time": "2026-10-08 10:00:00",
        "raw_text": "自检探针",
        "title": "schema",
        "category": "其他",
        "summary": "往返测试",
        "todos": [{"task": "验证读写", "due": None}],
    }
    try:
        memo_id = main.insert_memo(conn, probe)
        row = conn.execute("SELECT * FROM memos WHERE id = ?", (memo_id,)).fetchone()
        check(row is not None and row["raw_text"] == "自检探针", "写入/读取往返",
              f"新记录 id={memo_id}，todos 落库为 TEXT：{row['todos']!r}")
        check(json.loads(row["todos"]) == probe["todos"], "todos JSON 序列化",
              "列表 → JSON 字符串 → 列表 无损往返")
        conn.execute("DELETE FROM memos WHERE id = ?", (memo_id,))
        conn.commit()
    except Exception as exc:
        check(False, "写入/读取往返", f"{type(exc).__name__}: {exc}")

    return conn


# --------------------------------------------------------------------------- #
# 4. 模拟数据流测试
# --------------------------------------------------------------------------- #

MOCK_ASR_TEXT = "嗯，那个，明天下午三点要跟客户开会，记得提前准备好方案，别忘了发邮件给张总，就这些，结束"

MOCK_LLM_RESPONSE = """```json
{
  "title": "明天下午与客户开会",
  "category": "工作",
  "summary": "明天下午三点与客户开会，需提前准备方案并通知张总。",
  "todos": [
    {"task": "提前准备好会议方案", "due": "2026-10-09 15:00"},
    {"task": "发邮件给张总", "due": "2026-10-09 15:00"}
  ]
}
```"""

MOCK_BASE_TIME = "2026-10-08 10:00:00"


def fake_deepseek_client(content: str, fail_first: bool = False):
    """构造一个 OpenAI 兼容的假客户端，不产生任何网络请求。"""
    state = {"calls": 0}

    def create(**kwargs):
        state["calls"] += 1
        if fail_first and state["calls"] == 1:
            raise RuntimeError("模拟网关不支持 response_format")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), state


def check_mock_flow(conn: sqlite3.Connection | None) -> None:
    section("模拟数据流测试（伪造 ASR 输入 + DeepSeek 响应）")
    import main

    if conn is None:
        check(False, "模拟数据流", "前置的数据库检查未通过，跳过")
        return

    # --- 4.1 JSON 解析健壮性 ---
    try:
        parsed = main.extract_json_object(MOCK_LLM_RESPONSE)
        check(parsed.get("title") == "明天下午与客户开会", "解析 ```json 代码块包裹的响应",
              f"title={parsed.get('title')!r}, todos={len(parsed.get('todos', []))} 条")
    except Exception as exc:
        check(False, "解析 ```json 代码块包裹的响应", f"{type(exc).__name__}: {exc}")

    variants = {
        "纯 JSON": '{"title": "纯JSON", "category": "生活", "summary": "s", "todos": []}',
        "带前后缀说明": '好的，以下是整理结果：\n{"title": "带前缀", "category": "工作", "summary": "s", "todos": []}\n希望有帮助！',
        "尾逗号": '{"title": "尾逗号", "category": "学习", "summary": "s", "todos": [],}',
    }
    for label, payload in variants.items():
        try:
            ok = main.extract_json_object(payload).get("title") is not None
        except Exception as exc:
            ok = False
            label += f" ({type(exc).__name__})"
        check(ok, f"解析变体 · {label}")

    try:
        main.extract_json_object("这是一段完全没有 JSON 的胡言乱语")
        check(False, "非法输出应被拒绝", "预期抛 ValueError，实际未抛出")
    except ValueError:
        check(True, "非法输出应被拒绝", "无法解析时抛 ValueError，交由调用方降级")

    # --- 4.2 字段归一化 ---
    messy = {"title": "", "category": None, "summary": "  ", "todos": ["直接写字符串", {"content": "对象写 content"}, None, {"task": ""}]}
    normalized = main.normalize_memo(messy, "买牛奶", "2026-10-08 10:00:00")
    # 只要求**核心字段**兜住：SOAP 五段对非问诊内容是空串，拿 REQUIRED_COLUMNS 整体
    # 断言"都非空"会把"日常备忘不该长病历"这条设计判成失败。
    core = ("record_time", "raw_text", "title", "category", "summary", "todos")
    check(all(normalized[k] for k in core), "字段归一化 · 空值兜底",
          f"title={normalized['title']!r} category={normalized['category']!r}")
    check(all(normalized.get(key) == "" for key in main.SOAP_COLUMNS)
          and not main.has_soap(normalized),
          "字段归一化 · SOAP 缺失时是空串而不是 None",
          "没填的段统一成空串，卡片据此退回摘要视图；"
          "None 会在打印和 JSON 里变成 'None' 字样")
    check(len(normalized["todos"]) == 2, "字段归一化 · todos 形态兼容",
          f"字符串/对象/None/空任务 → {json.dumps(normalized['todos'], ensure_ascii=False)}")

    # --- 4.3 控制词过滤 ---
    stripped = main.strip_control_words(MOCK_ASR_TEXT)
    check(
        not any(kw in stripped for kw in ("结束", "完毕", "就这些"))
        and "明天下午三点要跟客户开会" in stripped,
        "控制词过滤",
        f"{MOCK_ASR_TEXT[:24]}… → {stripped}",
    )
    check(main.contains_stop_keyword("帮我记一下，完毕。", ("结束", "完毕")), "结束词命中检测",
          "忽略标点后仍能命中「完毕」")

    # --- 4.4 相对时间解析 ---
    base = datetime.strptime(MOCK_BASE_TIME, "%Y-%m-%d %H:%M:%S")
    cases = [
        ("明天下午三点开会", f"{base + timedelta(days=1):%Y-%m-%d} 15:00"),
        ("后天晚上八点", f"{base + timedelta(days=2):%Y-%m-%d} 20:00"),
        ("今天中午十二点半", f"{base:%Y-%m-%d} 12:30"),
        ("半小时后提醒我", f"{base + timedelta(minutes=30):%Y-%m-%d %H:%M}"),
        ("三小时后", f"{base + timedelta(hours=3):%Y-%m-%d %H:%M}"),
        ("下周三上午十点", f"{base + timedelta(days=6):%Y-%m-%d} 10:00"),
    ]
    for text, expected in cases:
        got = main.resolve_cn_time(text, base)
        check(got == expected, f"时间解析 · {text}", f"{got}（预期 {expected}）")

    check(main.resolve_cn_time("这个想法挺有意思的", base) is None, "时间解析 · 无时间表达返回 None",
          "不含任何时间词时返回 None，调用方据此把 due 置空")
    # 只说日期、没说时刻时，按 09:00 兜底（这一约定写进了 main.resolve_cn_time 的 docstring）
    check(main.resolve_cn_time("明天提交周报", base) == f"{base + timedelta(days=1):%Y-%m-%d} 09:00",
          "时间解析 · 只给日期时默认 09:00", "「明天提交周报」→ 次日 09:00")

    # 下周三 应落在星期三
    resolved = main.resolve_cn_time("下周三上午十点", base)
    check(datetime.strptime(resolved, "%Y-%m-%d %H:%M").weekday() == 2, "「下周三」落在星期三",
          f"{resolved} 是星期{['一', '二', '三', '四', '五', '六', '日'][datetime.strptime(resolved, '%Y-%m-%d %H:%M').weekday()]}")

    # --- 4.5 本地兜底整理 ---
    local = main.local_organize(MOCK_ASR_TEXT, MOCK_BASE_TIME)
    check(local["category"] == "工作", "本地兜底 · 分类", f"category={local['category']!r}")
    check(len(local["todos"]) >= 2, "本地兜底 · 待办抽取",
          f"{json.dumps(local['todos'], ensure_ascii=False)}")
    check(not any(filler in local["title"] for filler in ("嗯", "那个")),
          "本地兜底 · 填充词清理", f"title={local['title']!r}")
    check(all(t["due"] == "2026-10-09 15:00" for t in local["todos"]),
          "本地兜底 · 待办继承全句时间",
          "从句内无时间表达时，回退到整段话解析出的「明天下午三点」")

    # --- 4.6 DeepSeek 调用链路（含 response_format 兼容回退）---
    client, state = fake_deepseek_client(MOCK_LLM_RESPONSE, fail_first=True)
    try:
        data = main.call_deepseek(client, "deepseek-chat", MOCK_ASR_TEXT, MOCK_BASE_TIME)
        check(state["calls"] == 2 and data.get("category") == "工作", "DeepSeek 调用 · 兼容回退",
              f"首次因 response_format 失败后自动重试，共 {state['calls']} 次调用并解析成功")
    except Exception as exc:
        check(False, "DeepSeek 调用 · 兼容回退", f"{type(exc).__name__}: {exc}")

    # --- 4.7 接口异常时降级到本地规则 ---
    broken_client, _ = fake_deepseek_client("（这里不是 JSON）")
    memo, source = main.organize(MOCK_ASR_TEXT, MOCK_BASE_TIME, broken_client, "deepseek-chat")
    check(source == "local" and memo["title"], "异常降级 · LLM 不可用时不丢数据",
          f"source={source!r} title={memo['title']!r}")

    # --- 4.8 端到端：伪造 ASR → 整理 → 落库 ---
    class FakeTranscriber:
        """冒充 faster-whisper，让整条链路在无音频、无模型的情况下跑通。"""

        def __init__(self, text):
            self.text = text
            self.calls = []

        def transcribe(self, audio, beam_size=None, quick=False):
            self.calls.append(quick)
            return self.text

    # 所有落盘位置都指向临时目录：自检绝不能污染真实工作目录
    temp_dir = Path(conn.execute("PRAGMA database_list").fetchone()[2]).parent
    md_path = temp_dir / "e2e_check.md"
    cfg = main.Config(db_path=str(temp_dir / "e2e_check.db"),
                      markdown_path=str(md_path), beep=False)
    transcriber = FakeTranscriber(MOCK_ASR_TEXT)
    client, _ = fake_deepseek_client(MOCK_LLM_RESPONSE)
    pipeline = main.MemoPipeline(cfg, conn, transcriber, client)

    utterance = main.Utterance(
        audio=object(),  # FakeTranscriber 不会读它，无需 numpy
        started_at=base,
        ended_at=base + timedelta(seconds=6),
        duration=6.0,
        reason="静音断句",
    )
    before = conn.execute("SELECT COUNT(*) FROM memos").fetchone()[0]
    print("\n  —— 以下为端到端模拟输出的实际渲染 ——", flush=True)
    memo_id = pipeline.handle_audio(utterance)
    after = conn.execute("SELECT COUNT(*) FROM memos").fetchone()[0]
    print("  —— 模拟输出结束 ——\n", flush=True)

    check(memo_id is not None and after == before + 1, "端到端 · 数据库落库",
          f"新增记录 id={memo_id}（{before} → {after} 条）")

    row = dict(conn.execute("SELECT * FROM memos WHERE id = ?", (memo_id,)).fetchone())
    check(row["record_time"] == MOCK_BASE_TIME, "端到端 · record_time 取自录音起始时刻",
          f"record_time={row['record_time']}（而非处理完成时刻 {datetime.now():%Y-%m-%d %H:%M:%S}）")
    check(row["raw_text"] == main.strip_control_words(MOCK_ASR_TEXT, cfg.stop_keywords),
          "端到端 · raw_text 已剔除控制词", f"raw_text={row['raw_text']!r}")
    check(row["title"] == "明天下午与客户开会" and row["category"] == "工作", "端到端 · LLM 字段落库",
          f"title={row['title']!r} category={row['category']!r}")
    todos = json.loads(row["todos"])
    check(len(todos) == 2 and todos[0]["due"] == "2026-10-09 15:00", "端到端 · todos JSON 完整落库",
          json.dumps(todos, ensure_ascii=False))

    check(transcriber.calls == [False], "端到端 · 转写未被提前截断逻辑误触发",
          f"快速度探测调用记录 = {transcriber.calls}（仅正式转写一次）")
    markdown_text = md_path.read_text(encoding="utf-8") if md_path.is_file() else ""
    check("## 2. 明天下午与客户开会" in markdown_text and "- [ ] 发邮件给张总" in markdown_text,
          "端到端 · 落库的同时追加 Markdown",
          f"{md_path.name} 已写入，条目标题与待办都在文档里")

    # --- 4.9 纯文字模式（--text）---
    # 注意把 db / md 全部指向临时目录：自检不能污染真实工作目录
    temp_dir = Path(conn.execute("PRAGMA database_list").fetchone()[2]).parent
    # 工作目录里可能本来就躺着 my_memos.md（用户真的跑过 app）。这里要验的是
    # "这次 --text **没有写它**"，而不是"这个文件不存在" —— 后者会被真实使用误伤。
    home_md = PROJECT_ROOT / "my_memos.md"

    def _fingerprint(path: Path):
        return (path.stat().st_mtime_ns, path.stat().st_size) if path.is_file() else None

    home_before = _fingerprint(home_md)
    code = 0
    try:
        code = main.main([
            "--text", "提醒我周五下午两点交周报",
            "--db", str(temp_dir / "cli_check.db"),
            "--md-path", str(temp_dir / "cli_check.md"),
            "--no-llm",
        ])
    except SystemExit as exc:  # --version / --help 之类
        code = int(exc.code or 0)
    except Exception as exc:
        check(False, "CLI · --text 纯文字模式", f"{type(exc).__name__}: {exc}")
        code = 1
    check(code == 0, "CLI · --text 纯文字模式（--no-llm）", "参数解析、本地整理、落库、打印全链路正常")
    check((temp_dir / "cli_check.md").is_file() and _fingerprint(home_md) == home_before,
          "CLI · --text 也同步追加 Markdown",
          f"文字模式走同一条 pipeline，写进 --md-path 指定的位置；"
          f"工作目录的 {home_md.name} 一个字节都没动")


# --------------------------------------------------------------------------- #
# 5. 状态提示与文档导出
# --------------------------------------------------------------------------- #


def check_status_and_export(tmp_dir: Path, conn: sqlite3.Connection | None) -> None:
    section("状态提示与文档导出")
    import main

    if conn is None:
        check(False, "状态提示与导出", "前置的数据库检查未通过，跳过")
        return

    # --- 5.1 状态提示：把终端输出抓下来断言措辞 ---
    import contextlib
    import io

    cfg = main.Config(db_path=str(PROJECT_ROOT / "memos.db"), markdown_path="", beep=False)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        main.announce_listening(cfg, first=True)
        main.announce_recording(cfg)
        main.announce_thinking("DeepSeek ")
    output = captured.getvalue()

    expectations = [
        ("🎙️", "[正在倾听中...]", "就绪状态"),
        ("🔴", "[正在录音...]", "录音状态"),
        ("⏳", "[录音结束，DeepSeek 整理中...]", "整理状态"),
    ]
    for icon, text, label in expectations:
        check(icon in output and text in output, f"状态提示 · {label}", f"{icon} {text}")

    check(f"静音 {cfg.silence_ms / 1000:.1f} 秒" in output and "自动保存" in output,
          "状态提示 · 录音中提示了断句条件",
          f"提示里包含「静音 {cfg.silence_ms / 1000:.1f} 秒 或 说「结束/完毕」自动保存」")
    check(output.count("═" * 62) >= 2, "状态提示 · 就绪界线", "倾听状态用整行分隔线框出")

    for name in ("ICON_LISTENING", "ICON_RECORDING", "ICON_THINKING", "ICON_SAVED"):
        if not hasattr(main, name):
            check(False, "状态图标常量", f"缺少 {name}（无法一行替换成纯文本）")
            break
    else:
        check(True, "状态图标可配置", "emoji 集中在 ICON_* 常量里，便于改成纯文本")

    # --- 5.2 Markdown 条目渲染 ---
    memo = {
        "id": 7,
        "record_time": "2026-10-08 10:00:00",
        "raw_text": "明天下午三点开会",
        "title": "客户会议准备",
        "category": "工作",
        "summary": "明天下午三点与客户开会。",
        "todos": [{"task": "准备方案", "due": "2026-10-09 15:00"}, {"task": "发邮件给张总", "due": None}],
    }
    entry = main.format_markdown_entry(memo)
    checks = {
        "二级标题带编号": "## 7. 客户会议准备" in entry,
        "时间与分类": "**时间**：2026-10-08 10:00:00" in entry and "**分类**：工作" in entry,
        "待办渲染成复选框": "- [ ] 准备方案 —— 截止 2026-10-09 15:00" in entry
                            and "- [ ] 发邮件给张总" in entry,
        "保留原始识别原文": "明天下午三点开会" in entry,
        "条目之间用分隔线": entry.rstrip().endswith("---"),
        "不掺 HTML 标签": "<" not in entry,
    }
    for label, ok in checks.items():
        check(ok, f"Markdown 条目 · {label}")

    # --- 5.3 逐条追加（边记边攒）---
    target = tmp_dir / "append_check.md"
    first_ok = main.append_memo_markdown(memo, target)
    memo2 = dict(memo, id=8, title="第二条", record_time="2026-10-09 09:00:00",
                 todos=[], summary="第二条摘要")
    second_ok = main.append_memo_markdown(memo2, target)
    text = target.read_text(encoding="utf-8")
    check(first_ok and second_ok, "Markdown 追加写入", f"文件：{target.name}")
    check(text.count("# 我的语音备忘录") == 1, "标题头只写一次",
          "第二次追加不会重复写标题头")
    check("## 7. 客户会议准备" in text and "## 8. 第二条" in text, "两次追加的内容都在",
          "历史内容未被覆盖（追加而非重写）")
    check(text.index("## 7.") < text.index("## 8."), "追加保持时间顺序", "新条目在文件末尾")
    check("待办：" not in text.split("## 8.")[1], "无待办的条目不渲染待办段",
          "空 todos 时那一节整体省略")

    # --- 5.4 从数据库整体导出 ---
    exported = tmp_dir / "export_check.md"
    count = main.export_markdown(conn, exported)
    body = exported.read_text(encoding="utf-8")
    rows = conn.execute("SELECT COUNT(*) FROM memos").fetchone()[0]
    check(count == rows, "导出条数与数据库一致", f"导出 {count} 条，库中 {rows} 条")
    check("共 %d 条记录" % count in body and "导出时间" in body, "导出头部带统计信息",
          "标题下方注明了记录数与导出时间")
    check(body.count("## ") == count, "每条记录一个二级标题",
          f"共 {body.count('## ')} 个条目；--export-md 是整体重写，可重复执行")

    # --- 5.5 DOCX：缺依赖时必须给出可操作的提示，而不是崩掉 ---
    try:
        from docx import Document

        docx_path = tmp_dir / "export_check.docx"
        docx_count = main.export_docx(conn, docx_path)
        check(docx_count == count and docx_path.is_file() and docx_path.stat().st_size > 0,
              "DOCX 导出", f"python-docx 已安装，导出 {docx_count} 条，"
                           f"{docx_path.stat().st_size:,} 字节")

        # 回读生成的文档，确认标题/摘要/待办真的写进去了（而不是只生成了一个空壳）
        document = Document(str(docx_path))
        texts = [p.text for p in document.paragraphs]
        check(any("明天下午与客户开会" in t for t in texts), "DOCX · 标题写入",
              f"共 {len(document.paragraphs)} 个段落")
        todo_paras = [p for p in document.paragraphs if "发邮件给张总" in p.text]
        check(bool(todo_paras) and todo_paras[0].style.name != "Normal",
              "DOCX · 待办用列表样式",
              f"样式={todo_paras[0].style.name!r}（非 Normal 才算真的成了列表）"
              if todo_paras else "没找到待办段落")
    except ImportError:
        try:
            main.export_docx(conn, tmp_dir / "should_not_exist.docx")
            check(False, "DOCX 缺依赖时的报错", "未安装 python-docx 却未抛 ExportDependencyError")
        except main.ExportDependencyError as exc:
            check("pip install python-docx" in str(exc), "DOCX 缺依赖时的报错",
                  f"抛出 ExportDependencyError 且附带安装命令：{exc}")

    # --- 5.6 回放模式（需要 numpy，缺失时跳过而非失败）---
    try:
        check_replay(tmp_dir)
    except ImportError as exc:
        skip("回放解码与流测试", f"未安装 numpy（{exc}）；pip install -r requirements.txt")


def check_replay(tmp_dir: Path) -> None:
    """造一个 8k 立体声 WAV，验证回放模式的解码、重采样与分块节奏。"""
    import wave

    import numpy as np

    import main

    wav_path = tmp_dir / "replay_check.wav"
    seconds, src_rate = 2.0, 8000
    timeline = np.linspace(0, 2 * np.pi * 440 * seconds, int(src_rate * seconds))
    tone = (np.sin(timeline) * 0.5 * 32767).astype("<i2")
    stereo = np.repeat(tone, 2)                       # 左右声道各一份 → 立体声
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(src_rate)
        handle.writeframes(stereo.tobytes())

    decoded = main.load_wav_16k_mono(wav_path)
    check(decoded.dtype.name == "float32", "回放 · 解码为 float32", f"dtype={decoded.dtype}")
    check(abs(len(decoded) - int(16000 * seconds)) <= 2, "回放 · 重采样到 16k 单声道",
          f"8k 立体声 {int(src_rate * seconds)} 帧 = {seconds:.0f}s → {len(decoded)} 采样点"
          f"（16k × {seconds:.0f}s = {int(16000 * seconds)}）")
    check(0.3 < float(np.abs(decoded).max()) < 0.7, "回放 · 振幅还原正确",
          f"峰值 {float(np.abs(decoded).max()):.3f}（原始正弦波幅度 0.5）")

    cfg = main.Config(silence_ms=2500, markdown_path="", beep=False)
    stream = main.ReplayStream(decoded, cfg, loops=2, realtime=False)
    consumed = 0
    while consumed < len(stream._data):
        block, _overflowed = stream.read(cfg.chunk_size)
        consumed += len(block)
    check(consumed >= len(decoded) * 2, "回放 · 按指定遍数重复",
          f"2 遍音频共 {consumed / 16000:.1f}s；块大小 {cfg.chunk_size} 与 VAD 窗口一致")

    quiet = main.ReplayStream(decoded, cfg, loops=1)
    tail_start = len(quiet._data) - int((cfg.silence_ms / 1000.0 + 0.5) * 16000)
    check(abs(float(np.abs(quiet._data[tail_start:]).max())) < 1e-6,
          "回放 · 每遍结尾留有静音",
          f"尾部静音 ≥ {cfg.silence_ms / 1000:.1f}s，保证最后一遍也能触发静态断句")

    try:
        main.load_wav_16k_mono(tmp_dir / "missing_file.wav")
        check(False, "回放 · 文件缺失时报错", "应当抛异常")
    except Exception as exc:
        check(True, "回放 · 文件缺失时报错", f"{type(exc).__name__}（由 CLI 捕获后提示用户）")


# --------------------------------------------------------------------------- #
# 6. 性能与截断：纯 CPU 默认档 / 采集暂停 / 结束词剥离
# --------------------------------------------------------------------------- #


class _LoopStream:
    """永远灌同一块音频的假麦克风，只用来数"被读了多少次"。"""

    def __init__(self, np, chunk_size: int):
        self._np = np
        self._chunk_size = chunk_size
        self.reads = 0

    def read(self, frames: int) -> tuple:
        self.reads += 1
        time.sleep(0.002)          # 真实设备会阻塞一个块的时间，别让假流把 CPU 空转满
        return self._np.zeros((frames, 1), dtype="float32"), False

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


class _SlowTranscriber:
    """每次转写都磨蹭一会儿，用来验证"扫描不阻塞主循环"。"""

    def __init__(self, text: str, delay: float):
        self.text = text
        self.delay = delay
        self.calls = 0

    def scan(self, audio) -> str:
        return self.transcribe(audio, quick=True)

    def transcribe(self, audio, beam_size=None, quick=False) -> str:
        self.calls += 1
        time.sleep(self.delay)
        return self.text


class _ScriptedTranscriber:
    """按脚本依次吐出识别文本，别的都返回空——用来演"录了哪几段"。"""

    def __init__(self, texts):
        self.texts = list(texts)
        self.seen: list = []

    def scan(self, audio) -> str:
        self.seen.append(len(audio))
        if not self.texts:
            return ""
        return self.texts.pop(0)

    def transcribe(self, audio, beam_size=None, quick=False) -> str:
        return self.scan(audio)


def check_perf_and_truncation(tmp_dir: Path) -> None:
    import main

    import numpy as np

    section("性能与截断（CPU 默认档 / 采集暂停 / 结束词剥离）")

    # --- 6.1 默认档位：不该一上来就把 CPU 压死 ---
    default_cfg = main.Config()
    check(default_cfg.asr_model == "base", "默认 ASR 模型 = base",
          f"asr_model={default_cfg.asr_model!r}"
          f"（small 是纯 CPU 卡顿的主要来源，已降级；更弱的机器可设 tiny）")
    check(default_cfg.early_stop_keyword is False, "结束词提前截断默认关闭",
          "EARLY_STOP_KEYWORD 默认 False —— 断句完全交给 silero-vad，全程不碰 ASR")
    check(default_cfg.silence_ms == 3000, "静音断句阈值 = 3.0s",
          "纯 VAD 计算（只算人声概率），与 ASR 无关，因此极快；"
          "它同时是唤醒模式下的兜底断句（说了话但忘了说分段结束词）")

    # --- 6.2 控制词只在"终止位置"剥离 ---
    strip_cases = [
        ("句中实词必须保留", "会议结束之后给我打电话", "会议结束之后给我打电话"),
        ("无控制词时原文不动", "明天开会。", "明天开会。"),
        ("句尾控制词剥掉", "明天下午三点跟客户开会，结束", "明天下午三点跟客户开会"),
        ("语气词与标点一起吃", "帮我记一下买牛奶，说完了嗯。", "帮我记一下买牛奶"),
        ("连续控制词逐层剥", "别忘了发邮件给张总，就这些，结束", "别忘了发邮件给张总"),
        ("独立小句按控制词处理", "买牛奶，完毕，然后回家", "买牛奶，然后回家"),
    ]
    for label, source, expected in strip_cases:
        actual = main.strip_control_words(source)
        check(actual == expected, f"控制词剥离 · {label}", f"{source!r} → {actual!r}")

    # --- 6.3 处理期间暂停采集 ---
    pause_cfg = main.Config(queue_seconds=0.2, beep=False)
    listener = main.VoiceListener(pause_cfg, None)
    stream = _LoopStream(np, pause_cfg.chunk_size)
    listener._processing.set()          # 模拟"上一条正在转写/整理"
    reader = threading.Thread(target=listener._reader, args=(stream,), daemon=True)
    reader.start()
    time.sleep(0.25)
    drained = stream.reads
    queued_paused = listener._queue.qsize()
    dropped_paused = listener._dropped          # 必须在恢复消费之前读，否则后续必然溢出
    skipped_paused = listener._skipped
    listener._processing.clear()
    time.sleep(0.05)
    queued_resumed = listener._queue.qsize()
    listener.stop()
    reader.join(timeout=2)

    check(drained > 0 and queued_paused == 0,
          "处理期间暂停入队（设备缓冲照读不误）",
          f"暂停期间读走 {drained} 块（防 PortAudio 溢出），入队 {queued_paused} 块 → 队列不积压")
    check(queued_resumed > 0, "处理结束立即恢复采集",
          f"恢复后队列重新有 {queued_resumed} 块，无需重开麦克风设备")
    check(skipped_paused > 0 and dropped_paused == 0,
          "暂停不计入丢弃（dropped 保持 0）",
          f"skipped={skipped_paused}（正常暂停）· dropped={dropped_paused}（真正的积压丢块）")

    # --- 6.4 队列兜底丢块 → 置位断点标记 ---
    hole_cfg = main.Config(queue_seconds=0.2, beep=False)
    hole_listener = main.VoiceListener(hole_cfg, None)
    for _ in range(hole_listener._queue.maxsize):
        hole_listener._queue.put_nowait((0.0, datetime.now(), np.zeros(hole_cfg.chunk_size, "float32")))
    hole_stream = _LoopStream(np, hole_cfg.chunk_size)
    hole_reader = threading.Thread(target=hole_listener._reader, args=(hole_stream,), daemon=True)
    hole_reader.start()
    time.sleep(0.15)
    hole_listener.stop()
    hole_reader.join(timeout=2)
    check(hole_listener._dropped > 0 and hole_listener._hole.is_set(),
          "兜底丢块时置位断点标记",
          f"dropped={hole_listener._dropped} → 主循环据此作废本段，"
          f"避免 VAD 静音计时失真后永远等不到 end（静音断句失效的根因）")

    # --- 6.5 指令扫描跑在后台，不阻塞主循环 ---
    scan_cfg = main.Config(early_stop_keyword=True, beep=False)
    probe = main.VoiceListener(scan_cfg, None, transcriber=_SlowTranscriber("嗯，下一条", 0.4))
    block = np.zeros(scan_cfg.chunk_size, dtype="float32")
    started = time.monotonic()
    probe._start_scan([block], ("exit", "segment_end"))
    elapsed = time.monotonic() - started
    check(elapsed < 0.2, "指令扫描不阻塞主循环",
          f"_start_scan 用时 {elapsed * 1000:.0f}ms 就返回（转写要 400ms），VAD 状态机不受影响")
    time.sleep(0.6)
    hit = probe._take_scan_result()
    check(hit is not None and hit[0] == "segment_end" and hit[1] == "下一条",
          "后台扫描命中分段结束词",
          f"「下一条」在后台被识别到 → {hit!r}，主循环下一轮即可据此截断")
    check(probe._take_scan_result() is None, "扫描结果取走即清",
          "同一次命中只触发一次，不会因为多轮轮询而重复断句")

    stale = main.VoiceListener(scan_cfg, None, transcriber=_SlowTranscriber("下一条", 0.4))
    stale._start_scan([block], ("exit", "segment_end"))
    stale._scan_epoch += 1              # 模拟"这段已结束，用户开始说下一句"
    time.sleep(0.6)
    check(stale._take_scan_result() is None, "迟到的扫描结果被丢弃",
          "epoch 变了就不认旧结果，上一段的结束词不会误伤下一段录音")

    # --- 6.5b 录音途中的扫描窗口只看尾部，不把整段音频反复喂给 Whisper ---
    tail_cfg = main.Config(sample_rate=16000, chunk_size=512, wake_window_s=2.2)
    tail_listener = main.VoiceListener(tail_cfg, None, transcriber=_ScriptedTranscriber([]))
    tail_block = np.zeros(512, dtype="float32")
    long_buffer = [tail_block] * 200        # 约 6.4 秒音频
    tail_listener._start_scan(long_buffer[-max(1, int(2.2 * 16000 / 512)):], ("exit",))
    time.sleep(0.1)
    check(tail_listener.transcriber.seen
          and tail_listener.transcriber.seen[0] <= int(2.2 * 16000) + 512,
          "录音途中的扫描只取尾部窗口",
          f"整段 {len(long_buffer) * 512 / 16000:.1f}s，只喂了 "
          f"{tail_listener.transcriber.seen[0] / 16000:.1f}s 给 Whisper")

    # --- 6.6 高开销开关在 CPU 上会提示 ---
    import contextlib
    import io

    doctor_out = io.StringIO()
    with contextlib.redirect_stdout(doctor_out):
        main.run_doctor(main.Config(early_stop_keyword=True, beep=False,
                                    db_path=str(tmp_dir / "doctor.db")))
    text = doctor_out.getvalue()
    check("注意" in text and "EARLY_STOP_KEYWORD=false" in text, "doctor 提示结束词的开销",
          "开着提前截断时 --doctor 会明确建议关掉它")


# --------------------------------------------------------------------------- #
# 7. 唤醒词指令逻辑与状态流转
# --------------------------------------------------------------------------- #


class _FakeVADIterator:
    """按脚本吐 ``start``/``end`` 的假 VAD，绕开 torch 与真实模型。

    状态机测试必须完全可复现 —— 真实 VAD 依赖音频内容，没法精确控制"第几块开始说话"。
    """

    def __init__(self, events: dict):
        self.events = dict(events)      # {块序号: "start" | "end"}
        self.index = 0
        self.resets = 0

    def __call__(self, _tensor) -> dict:
        kind = self.events.get(self.index)
        self.index += 1
        return {kind: 0.9} if kind else {}

    def reset_states(self) -> None:
        self.resets += 1


class _LevelTranscriber:
    """按音频响度回话的假 ASR：几种响度 = 几句指令，测试因此不依赖 whisper。"""

    LEVELS = ((0.20, "小医小医"), (0.50, "录音完毕"),
              (0.70, "作废"), (0.90, "退出录制"))

    def __init__(self):
        self.peaks: list = []

    def scan(self, audio) -> str:
        import numpy as np

        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        self.peaks.append(peak)
        if peak < 0.05:
            return ""
        level, text = min(self.LEVELS, key=lambda kv: abs(kv[0] - peak))
        return text if abs(level - peak) < 0.1 else ""

    def transcribe(self, audio, beam_size=None, quick=False) -> str:
        return self.scan(audio)


class _SceneStream:
    """按"第几块用多大音量"的剧本出音频的假麦克风。

    音量分段 → 假 ASR 给出对应指令（见 ``_LevelTranscriber``），
    于是"什么时候唤醒、什么时候说下一条"完全由剧本决定，可复现。
    """

    def __init__(self, np, cfg, levels: dict, total_chunks: int, pace: float = 0.0025):
        self._np = np
        self._cfg = cfg
        self._levels = sorted(levels.items())
        self._total = total_chunks
        self._pace = pace
        self._pos = 0

    def _amplitude(self, index: int) -> float:
        amp = 0.0
        for start, value in self._levels:
            if index >= start:
                amp = value
        return amp

    def read(self, frames: int) -> tuple:
        np = self._np
        if self._pos >= self._total:
            time.sleep(0.05)            # 剧本演完就慢下来，别把队列灌爆
            return np.zeros((frames, 1), dtype="float32"), False
        time.sleep(self._pace)          # 模拟设备节奏，让队列保持浅水位
        amp = self._amplitude(self._pos)
        phase = np.arange(frames, dtype="float32") + self._pos * frames
        block = (amp * np.sin(2 * np.pi * 440.0 * phase / 16000.0)).astype("float32")
        self._pos += 1
        return block.reshape(-1, 1), False

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


def _run_scene(cfg, events: dict, levels: dict, total_chunks: int, expected: int,
               stop_when_done: bool = True, timeout: float = 30.0, cancels: list | None = None,
               pokes: list | None = None):
    """跑一遍剧本化的状态机，返回 (收到的句子, listener, 假 VAD)。

    ``cancels`` 非空时把"语音作废"回调收到的 ``(命中的词, 识别原文)`` 记进去。

    ``pokes`` 用来模拟**网友在网页上按按钮**：每个元素是
    ``{"at": 秒, "action": "finish"|"cancel", "accepted": None}``，
    到点后由后台线程调用 ``listener.request_finish()/request_cancel()``，
    并把返回值写回 ``entry["accepted"]``（False = 当时没在录音，按钮会被拒）。
    """
    import main

    import numpy as np

    utterances: list = []
    done = threading.Event()
    fake_vad = _FakeVADIterator(events)
    transcriber = _LevelTranscriber()

    original = main.build_vad_iterator
    main.build_vad_iterator = lambda _vad, _cfg: fake_vad
    try:
        listener = main.VoiceListener(
            cfg, None, transcriber=transcriber,
            on_cancel=(lambda keyword, text: cancels.append((keyword, text)))
            if cancels is not None else None,
        )
        stream = _SceneStream(np, cfg, levels, total_chunks)

        def on_utterance(utterance) -> None:
            utterances.append(utterance)
            if len(utterances) >= expected:
                done.set()

        worker = threading.Thread(target=listener.run, args=(on_utterance, lambda: stream),
                                  name="scene-listener", daemon=True)
        worker.start()

        if pokes:
            def poke_loop() -> None:
                clock = time.monotonic()
                for entry in pokes:
                    time.sleep(max(0.0, clock + float(entry["at"]) - time.monotonic()))
                    entry["accepted"] = bool(
                        listener.request_finish() if entry["action"] == "finish"
                        else listener.request_cancel()
                    )

            threading.Thread(target=poke_loop, name="scene-poke", daemon=True).start()

        done.wait(timeout=timeout)
        if stop_when_done:
            time.sleep(0.4)              # 留出退出指令走完的时间
            listener.stop()
        worker.join(timeout=5)
    finally:
        main.build_vad_iterator = original
    return utterances, listener, fake_vad


def check_commands_and_state_machine(tmp_dir: Path) -> None:
    import main

    section("唤醒词指令逻辑与状态流转")

    # --- 7.1 指令识别（含 ASR 听岔的模糊匹配）---
    cfg = main.Config(beep=False, markdown_path="")
    cases = [
        ("唤醒词原样（叠词）", "小医小医", "wake"),
        ("唤醒词只喊一遍", "小医", "wake"),
        ("唤醒词在句子里", "小医小医 明天下午问诊", "wake"),
        ("唤醒词跟着指令一起说", "小医，下一条", "segment_end"),
        ("ASR 把「医」听成「一」", "小一小一", "wake"),
        ("ASR 把「医」听成「依」", "小依小依", "wake"),
        ("ASR 把「医」听成「姨」", "小姨小姨", "wake"),
        ("ASR 把「小」听成「晓」", "晓医晓医", "wake"),
        ("ASR 把「小」听成「肖」", "肖医肖医", "wake"),
        ("只说一遍且写岔（小一）", "小一，明天下午三点问诊", "wake"),
        ("只说一遍且写岔（小依）", "小依", "wake"),
        ("只说一遍、两个音都写岔", "晓依", "wake"),
        ("分段结束词", "嗯，下一条", "segment_end"),
        ("分段结束词的另一种说法", "这段保存", "segment_end"),
        ("退出词", "好了，退出录制", "exit"),
        ("退出词另一种说法", "彻底结束", "exit"),
        ("作废指令", "作废", "cancel"),
        ("作废指令的另一种说法", "这条不要了", "cancel"),
        ("作废词带语气尾巴", "算了作废吧", "cancel"),
        ("作废词在句尾的小句里", "刚才那段别记录了", "cancel"),
        ("旧唤醒词已失效（换成小医了）", "小记小记", None),
        ("旧唤醒词的错别字写法也不认", "小纪小纪", None),
        ("普通内容不该误触发", "明天下午三点跟客户开会", None),
        ("含「结束」的实词不该误触发", "会议结束之后给我打电话", None),
        ("正文里的「取消」不该误触发", "明天取消会议，改到周四", None),
        ("正文中段的「作废」不该误触发", "把旧方案作废以后重新做", None),
    ]
    for label, text, expected in cases:
        hit = main.match_command_detail(text, cfg)
        actual = hit[0] if hit else None
        check(actual == expected, f"指令识别 · {label}",
              f"{text!r} → {actual!r}" + (f"（命中「{hit[1]}」）" if hit else ""))

    hit = main.match_command_detail("下一条，退出录制", cfg)
    check(hit is not None and hit[0] == "exit", "退出指令优先于分段指令",
          "一句话里同时出现时听最后那道指令，直接结束整个程序")
    hit = main.match_command_detail("这条不要了，退出录制", cfg)
    check(hit is not None and hit[0] == "exit", "退出指令优先于作废指令",
          "作废是丢弃，退出是保存后收工；两者同时出现时以退出为准")
    hit = main.match_command_detail("录音完毕，这条不要了", cfg)
    check(hit is not None and hit[0] == "cancel", "作废指令优先于分段指令",
          "「存下来」和「别存了」同时说时听作废 —— 宁可丢一条，也不能存下用户否定的内容")

    # --- 7.2 入库前的清洗：指令词不能进正文 ---
    clean_cases = [
        ("开头的唤醒词剥掉", "小医小医，明天下午三点开会", "明天下午三点开会"),
        ("只说一遍的唤醒词剥掉", "小医，明天下午三点开会", "明天下午三点开会"),
        ("只说一遍又写岔的唤醒词剥掉", "小一，明天下午三点开会", "明天下午三点开会"),
        ("分段结束词剥掉", "买牛奶，下一条", "买牛奶"),
        ("退出词剥掉", "记一下交电费，退出录制", "记一下交电费"),
        ("作废词剥掉", "记一下买牛奶，作废", "记一下买牛奶"),
        ("句中的实词保留", "会议结束之后给我打电话", "会议结束之后给我打电话"),
    ]
    for label, source, expected in clean_cases:
        actual = main.clean_transcript(source, cfg)
        check(actual == expected, f"入库清洗 · {label}", f"{source!r} → {actual!r}")

    # --- 7.3 状态流转：待机 → 唤醒 → 录音 → 分段保存 → 待机 → 退出 ---
    # 音量剧本：0.20 = "小医小医"，0.50 = "录音完毕"，0.70 = "作废"，0.90 = "退出录制"
    # 整段剧本只跑一两秒，所以把扫描间隔压到 0.02s，让"每 8 个音频块扫一次"
    # 等价于真实运行时"每 wake_scan_interval_s 扫一次"。
    scene_cfg = main.Config(beep=False, markdown_path="", silence_ms=3000,
                            min_segment_ms=800, queue_seconds=8, sample_rate=16000,
                            chunk_size=512, wake_word_enabled=True, wake_word="小医小医",
                            wake_scan_interval_s=0.02)
    events = {20: "start", 179: "start"}
    levels = {0: 0.0, 20: 0.20, 90: 0.50, 111: 0.0, 179: 0.20, 220: 0.90}
    utterances, listener, fake_vad = _run_scene(scene_cfg, events, levels, 320, expected=2)

    check(len(utterances) >= 1, "状态流转 · 唤醒词触发录音并落库",
          f"第 20 块听到「小医小医」→ 进入录音 → 共收到 {len(utterances)} 段")
    if utterances:
        check(utterances[0].reason == "检测到分段结束词", "状态流转 · 分段指令立即截断",
              f"第 90 块说「录音完毕」→ 立刻落库（原因：{utterances[0].reason}），"
              f"而不是干等 {scene_cfg.silence_ms / 1000:.0f} 秒静音")
    check(listener.exit_requested.is_set(), "状态流转 · 退出指令收工",
          "第 220 块说「退出录制」→ 保存后整个程序退出，无需 Ctrl+C")
    check(len(utterances) == 2, "状态流转 · 多段分别入库",
          f"共 {len(utterances)} 段独立录音，开始时间 "
          f"{[u.started_at.strftime('%H:%M:%S') for u in utterances]}")
    if len(utterances) == 2:
        check(utterances[0].started_at <= utterances[1].started_at,
              "状态流转 · 段落时间戳递增", "第二段不早于第一段，顺序不会乱")
        check(utterances[1].reason == "检测到退出指令",
              "状态流转 · 第二段由退出词收尾",
              f"第一段 {utterances[0].reason!r} → 回到待机 → "
              f"再次唤醒后 {utterances[1].reason!r}")
    check(fake_vad.resets >= 1, "状态流转 · 每段结束都重置 VAD",
          f"reset_states 调用 {fake_vad.resets} 次（每段独立计时，不会串段）")
    check(not listener._hole.is_set() and listener._dropped == 0,
          "状态流转 · 全程没有音频积压",
          f"dropped={listener._dropped} · hole={listener._hole.is_set()} "
          f"（指令扫描跑在后台，主循环跟得上采集）")

    # --- 7.4 没有唤醒词就绝不录音（假 ASR 返回空 = 说的都是普通内容）---
    quiet_utts, quiet_listener, _ = _run_scene(
        scene_cfg, {20: "start", 60: "end"}, {0: 0.35}, 160, expected=1,
        stop_when_done=True, timeout=4.0,
    )
    check(not quiet_utts, "状态流转 · 没喊唤醒词就不录音",
          f"全程只是普通说话，结果收到 {len(quiet_utts)} 段 —— 一条都不该有")
    check(quiet_listener.state == main.STATE_STANDBY, "状态流转 · 一直停在待机",
          f"state = {quiet_listener.state!r}")

    # --- 7.5 免唤醒模式：听到人声直接开录 ---
    no_wake_cfg = main.Config(beep=False, markdown_path="", silence_ms=3000,
                              min_segment_ms=800, sample_rate=16000, chunk_size=512,
                              wake_word_enabled=False, wake_scan_interval_s=0.02)
    no_wake_utts, _, _ = _run_scene(
        no_wake_cfg, {20: "start", 60: "end"}, {0: 0.20}, 200, expected=1,
        stop_when_done=True, timeout=10.0,
    )
    check(len(no_wake_utts) == 1, "免唤醒模式 · 听到人声直接开始录",
          f"WAKE_WORD_ENABLED=false 时第 20 块的人声直接触发录音，收到 {len(no_wake_utts)} 段")

    # --- 7.6 语音作废：录音途中说"作废" → 这段彻底消失 ---
    # 剧本：第 20 块喊唤醒词 → 第 90 块说"作废"（0.70）→ 第 179 块重新唤醒 → 第 220 块退出。
    # 结论应该是**只收到 1 段**（退出那段）：作废的那段根本没走到 on_utterance。
    groups = main.VoiceListener(scene_cfg, None)._recording_scan_groups(True)
    check("cancel" in groups, "录音途中会盯着作废词",
          f"唤醒词模式扫描 {groups} —— 作废与退出、分段并存")

    cancels: list = []
    cancel_events = {20: "start", 179: "start"}
    cancel_levels = {0: 0.0, 20: 0.20, 90: 0.70, 130: 0.0, 179: 0.20, 220: 0.90}
    cancel_utts, cancel_listener, _ = _run_scene(
        scene_cfg, cancel_events, cancel_levels, 320, expected=1,
        timeout=30.0, cancels=cancels,
    )
    check(cancels and cancels[0][0], "作废指令被识别并触发丢弃回调",
          f"收到 {cancels[:1]}（命中词 + 识别原文）")
    check(cancel_listener.cancelled == 1, "作废计数 +1",
          f"listener.cancelled = {cancel_listener.cancelled}")
    check(len(cancel_utts) == 1, "作废的那一段没有进入整理链路",
          f"全程只收到 {len(cancel_utts)} 段 —— 被作废的一段既没转写也没入库")
    if len(cancel_utts) == 1:
        check(cancel_utts[0].reason == "检测到退出指令",
              "作废后回到待机，还能重新唤醒继续记",
              f"唯一落库的一段来自退出词（{cancel_utts[0].reason!r}），"
              f"说明作废后状态机干净地回到了待机")

    # --- 7.7 网页手动按钮：说不出指令词时的双保险 ---
    # 剧本：第 10 块喊唤醒词进入录音，之后一直是"认不出来的普通内容"（0.35），
    # 到点由后台线程模拟"网友点了按钮"。真实的收尾必须发生在录音主循环里。
    idle = main.VoiceListener(scene_cfg, None)
    check(idle.request_finish() is False and idle.request_cancel() is False,
          "手动按钮 · 待机时按下去不生效",
          f"待机状态返回 False，接口随后回 409 —— 不会凭空造出一段录音")

    manual_levels = {0: 0.0, 10: 0.20, 60: 0.35}
    finish_poke = [{"at": 1.5, "action": "finish", "accepted": None}]
    finish_utts, finish_listener, _ = _run_scene(
        scene_cfg, {10: "start"}, manual_levels, 1500, expected=1,
        timeout=10.0, pokes=finish_poke,
    )
    check(finish_poke[0]["accepted"] is True, "手动按钮 · 录音中「结束并整理」被接受",
          f"request_finish() → {finish_poke[0]['accepted']}（状态 {main.STATE_RECORDING} 时才受理）")
    check(len(finish_utts) == 1, "手动按钮 · 「结束并整理」立刻截断并交出这一段",
          f"收到 {len(finish_utts)} 段，没有干等静音超时")
    if finish_utts:
        check(finish_utts[0].reason == "网页手动结束", "手动结束的段落带上可追溯的原因",
              f"reason = {finish_utts[0].reason!r}，后续照样送 DeepSeek 整理并落库")
    check(finish_listener.state == main.STATE_STANDBY and finish_listener.cancelled == 0,
          "手动按钮 · 结束之后干净地回到待机",
          f"state = {finish_listener.state!r} · cancelled = {finish_listener.cancelled}")

    drop_poke = [{"at": 1.5, "action": "cancel", "accepted": None}]
    # 作废这段**本该**一句都不产生，所以只能用超时收尾（expected 永远等不到）
    drop_utts, drop_listener, _ = _run_scene(
        scene_cfg, {10: "start"}, manual_levels, 1500, expected=1,
        timeout=3.0, pokes=drop_poke,
    )
    check(drop_poke[0]["accepted"] is True, "手动按钮 · 录音中「放弃本条」被接受",
          f"request_cancel() → {drop_poke[0]['accepted']}")
    check(not drop_utts and drop_listener.cancelled == 1,
          "手动按钮 · 「放弃本条」不转写、不整理、不入库",
          f"收到 {len(drop_utts)} 段 · cancelled = {drop_listener.cancelled}"
          f"（该段音频被直接丢掉，音频缓冲也一起清了）")
    check(drop_listener.state == main.STATE_STANDBY, "放弃之后回到待机继续等唤醒",
          f"state = {drop_listener.state!r}")

    # 两个标志同时被置上时（两台设备几乎同时点），以"结束"为准：宁可留下一段，
    # 也不要因为一次竞态把用户刚说的话整段丢掉。
    tie = main.VoiceListener(scene_cfg, None)
    tie._finish_request.set()
    tie._cancel_request.set()
    check(tie._take_manual() == "finish",
          "手动按钮 · 「结束」优先于「放弃」",
          "同时收到两个请求时先落库，避免竞态把内容吞掉")


# --------------------------------------------------------------------------- #
# 8. 拼音模糊匹配
# --------------------------------------------------------------------------- #


def check_pinyin_matching() -> None:
    """控制词的拼音匹配：ASR 写错字/写繁体，也照样触发。

    表里的"识别原文"不是编出来的，是把 SAPI 合成语音真喂给 faster-whisper base
    得到的实测输出（见 README「为什么错别字也能触发」一节）。
    """
    import main

    section("拼音模糊匹配：ASR 写错字也能认出控制词")

    cfg = main.Config(beep=False, markdown_path="")
    if not main.pinyin_available():
        skip("拼音模糊匹配", "未安装 pypinyin（pip install pypinyin）；本组 20 项全部跳过")
        return

    # --- 逐字转拼音必须与原文下标严格一一对应 ---
    # 控制词不但要"判出来"，还要能按下标从原文里剪掉，所以长度对齐是硬约束
    for text in ("录音完毕", "錄音完畢", "小医小医，下一条", "abc123", "三点半", ""):
        pairs = main.char_syllables(text)
        check(len(pairs) == len(text), f"逐字转拼音 · {text!r} 下标对齐",
              f"{len(text)} 字 → {len(pairs)} 个音节位：{[s for _c, s in pairs]}")

    # --- 音近归一：前后鼻音、平翘舌折叠掉 ---
    folded = [("zhe", "ze"), ("shi", "si"), ("chi", "ci"), ("wang", "wan"),
              ("ying", "yin"), ("shu", "su")]
    wrong = [(raw, main._fold_syllable(raw)) for raw, want in folded
             if main._fold_syllable(raw) != want]
    check(not wrong, "音近归一 · 平翘舌与前后鼻音折叠",
          f"zhe→ze、shi→si、wang→wan… 全部符合预期" if not wrong else f"异常：{wrong}")

    # 只抹 h 是错的：「书(shu)」会变成 hu，跟「会(hui)」「后(hou)」混成一堆
    check(main._fold_syllable("shu") != main._fold_syllable("hui")
          and main._fold_syllable("shu") != main._fold_syllable("hou"),
          "音近归一 · 不会把 shu 错折成 hu",
          "zh/ch/sh 是整组替换而不是删 h —— 否则「会议结束之后」会被判成「退出录制」")

    # --- 实测错别字 → 该触发的控制词一个都不能漏 ---
    asr_cases = [
        ("录音完毕（原样）", "录音完毕", "segment_end"),
        ("录音完毕 被识别成繁体「錄音完畢」", "錄音完畢", "segment_end"),
        ("录音完毕 被听成「入音完畢」", "入音完畢", "segment_end"),
        ("保存这段 被识别成「保存這段」", "保存這段", "segment_end"),
        ("退出录制 被识别成「退出錄製」", "退出錄製", "exit"),
        ("作废 被听成「坐着飞」", "坐着飞", "cancel"),
        ("作废 被听成「錯誤費」", "錯誤費", "cancel"),
        ("这条不要了 被识别成「這條不要了」", "這條不要了", "cancel"),
        ("唤醒词 被听成「小一小一」", "小一小一", "wake"),
        ("唤醒词 被听成「小依小依」", "小依小依", "wake"),
        ("普通句尾带「作废吧」", "算了作废吧", "cancel"),
    ]
    for label, text, expected in asr_cases:
        hit = main.match_command_detail(text, cfg)
        actual = hit[0] if hit else None
        check(actual == expected, f"拼音容错 · {label}",
              f"{text!r} → {actual!r}" + (f"（命中「{hit[1]}」）" if hit else ""))

    # --- 短指令词：单/双字口语词，带语气尾巴也算 ---
    short_cases = [
        ("完毕 被听成「万笔」", "万笔", "segment_end"),
        ("完毕 被听成「玩闭」", "玩闭", "segment_end"),
        ("保存 被听成「宝村」", "宝村", "segment_end"),
        ("保存 被听成「报存」", "报存", "segment_end"),
        ("好了", "好了", "segment_end"),
        ("收工", "收工", "segment_end"),
        ("就这样", "就这样", "segment_end"),
        ("就这些", "就这些", "segment_end"),
        ("说完了", "说完了", "segment_end"),
        ("短词后面带语气字「保存了」", "买牛奶保存了", "segment_end"),
    ]
    for label, text, expected in short_cases:
        hit = main.match_command_detail(text, cfg)
        actual = hit[0] if hit else None
        check(actual == expected, f"短指令词 · {label}",
              f"{text!r} → {actual!r}" + (f"（命中「{hit[1]}」）" if hit else ""))

    # --- 短指令词**只认句尾**：正文里的同一个词不能拦腰截断 ---
    check(main.match_command_detail("文件我已经保存了副本", cfg) is None,
          "短指令词只在句尾生效",
          "「保存」出现在正文中段时不算指令（句尾那一下才算）")

    # 代价也得写清楚：短词+语气字挂在句尾时会被当指令。这是**设计取舍**，
    # 不是 bug —— 关掉 SHORT_END_KEYWORDS 就回到"只有长指令词能收尾"的老行为。
    tail_cost = main.match_command_detail("我这边已经保存了", cfg)
    check(tail_cost is not None and tail_cost[0] == "segment_end",
          "短指令词的已知代价：句尾「保存了」会被当指令",
          "「我这边已经保存了」会截断落库（正文保留「我这边已经」）；"
          "不想承担就把 SHORT_END_KEYWORDS 设成 off，只用长指令词收尾")
    only_long = main.Config(beep=False, markdown_path="", short_end_keywords=())
    check(main.match_command_detail("我这边已经保存了", only_long) is None,
          "关掉 SHORT_END_KEYWORDS 后恢复保守行为",
          "同一个句子不再触发 —— 短词表是可关的开关，不是写死的")
    check(main.match_command_detail("结束", only_long) is None,
          "默认短词表故意不含「结束」",
          "「会议结束了」是最常见的正文，句尾匹配会连语气字一起吃掉，"
          "整条备忘被清空；想说「结束」收尾就把它加进 SEGMENT_END_KEYWORDS")

    # --- 逐字相似度这一层不能比拼音层还松 ---
    # 相似度在四字词上 0.72 等于"四个字里蒙对一个就过"，实测把正文
    # 「把这段录音听完」判成了「录音完毕」（听完 vs 完毕 = 0.75）
    check(main.match_command_detail("把这段录音听完", cfg) is None,
          "长指令词 · 正文里的近形句不误触发",
          "「把这段录音听完」similarity 0.75 却不再是指令：有拼音时逐字层按拼音门槛收紧")

    no_mistake = [
        ("会议结束之后给我打电话", "含「结束」的正文"),
        ("明天取消会议，改到周四", "正文里的动词「取消」"),
        ("把旧方案作废以后重新做", "正文中段的「作废」"),
        ("会议结束了", "句尾的「结束了」——短词表里没有「结束」"),
        ("这个小区的医院在哪儿", "唤醒词被拆散了（「小…医」不相邻，不算唤醒）"),
        ("小记小记", "上一版的唤醒词，换成小医之后必须彻底失效"),
        ("收工时间太晚了", "「收工」不在句尾"),
        ("把这段录音听完", "近形但不该触发"),
    ]
    for text, label in no_mistake:
        hit = main.match_command_detail(text, cfg)
        check(hit is None, f"拼音容错负例 · {label}",
              f"{text!r} → {hit}" if hit else f"{text!r} → 没有触发任何指令")

    # --- 入库清洗：按读音剥掉控制词，且不许把正文一起剪掉 ---
    clean_cases = [
        ("长结束词（繁体）连着正文", "啰嗦半天錄音完畢", "啰嗦半天"),
        ("长结束词被听成入音", "啰嗦半天入音完畢", "啰嗦半天"),
        ("作废（繁体）只剥词", "這條不要了", ""),
        ("作废被听成坐着飞", "记一下买牛奶坐着飞", "记一下买牛奶"),
        ("作废被听成錯誤費", "记一下买牛奶錯誤費", "记一下买牛奶"),
        ("退出词（繁体）", "记一下交电费，退出錄製", "记一下交电费"),
        ("短结束词+语气字", "买牛奶保存了", "买牛奶"),
        ("唤醒词（写法走样）", "小依小依，明天下午三点开会", "明天下午三点开会"),
        ("唤醒词（只说一遍 + 写岔）", "小一，明天下午三点开会", "明天下午三点开会"),
        ("唤醒词（只说一遍且两个音都岔了）", "晓依，明天下午三点开会", "明天下午三点开会"),
        ("多个控制词连用", "买牛奶，就这些，錄音完畢", "买牛奶"),
        ("正文里的实词原样保留", "会议结束之后给我打电话", "会议结束之后给我打电话"),
    ]
    for label, source, expected in clean_cases:
        actual = main.clean_transcript(source, cfg)
        check(actual == expected, f"拼音清洗 · {label}", f"{source!r} → {actual!r}")

    # 候选之间要挑**最长**的：控制词表里既有「完毕」也有「录音完毕」，
    # 短词先命中就只剪掉后半截，正文里会剩一个孤零零的「錄音」
    hit = main.find_pinyin_keyword("錄音完畢", cfg.segment_end_keywords + cfg.short_end_keywords)
    check(hit is not None and hit[0] == "录音完毕", "拼音清洗 · 多个候选命中时取最长",
          f"「錄音完畢」命中 {hit[0]!r} 而不是更短的「完毕」，整段被完整剪掉")

    # 覆盖度而不是相似度：ASR 凭空多塞音节时，关键词的读音仍按顺序全都在
    needle = main.syllables_of("作废")
    haystack = main.syllables_of("坐着飞")
    check(main.syllable_coverage(needle, haystack) >= 0.99,
          "覆盖度算法 · ASR 多塞音节不影响命中",
          f"「作废」zuo-fei 在「坐着飞」zuo-zhe-fei 里覆盖度 "
          f"{main.syllable_coverage(needle, haystack):.2f}（整体相似度只有 0.69，会漏判）")


# --------------------------------------------------------------------------- #
# 9. 音频预处理与 ASR 后端
# --------------------------------------------------------------------------- #


def check_audio_and_backend() -> None:
    import main

    import numpy as np

    section("音频预处理与 ASR 后端选择")

    sample_rate = 16000
    on = main.Config(beep=False, markdown_path="", audio_preprocess=True, audio_denoise=True)
    gain_only = main.Config(beep=False, markdown_path="", audio_preprocess=True, audio_denoise=False)
    off = main.Config(beep=False, markdown_path="", audio_preprocess=False)
    rng = np.random.default_rng(20261008)

    def tone(amp, seconds=2.0, freq=220.0):
        t = np.arange(int(sample_rate * seconds)) / sample_rate
        return (amp * np.sin(2 * np.pi * freq * t)).astype("float32")

    # --- 响度归一：只往上补，绝不往下压 ---
    quiet = tone(0.06)
    out = main.preprocess_audio(quiet, gain_only)
    check(main._rms(out) > main._rms(quiet) * 5, "响度归一 · 声音小就补上来",
          f"rms {main._rms(quiet):.4f} → {main._rms(out):.4f}（上限 "
          f"{gain_only.audio_max_gain_db:.0f} dB，不会无脑放大底噪）")

    loud = (0.8 * np.sign(np.sin(2 * np.pi * 200.0 * np.arange(sample_rate * 2) / sample_rate))
            ).astype("float32")
    out = main.preprocess_audio(loud, gain_only)
    check(abs(main._rms(out) - 0.8) < 0.01, "响度归一 · 已经够响就原样不动",
          f"rms {main._rms(loud):.4f} → {main._rms(out):.4f}"
          "（只补不压：本来录得够大的音频不再被打一遍，少一次改动信号的机会）")

    tiny = tone(0.0008)
    out = main.preprocess_audio(tiny, gain_only)
    gain_db = 20 * np.log10(max(main._rms(out), 1e-12) / main._rms(tiny))
    check(abs(gain_db - gain_only.audio_max_gain_db) < 0.5, "响度归一 · 增益有上限",
          f"极小声音只放大 {gain_db:.2f} dB（上限 {gain_only.audio_max_gain_db:.0f} dB）"
          "，避免把底噪一起抬成噪音")

    # --- 降噪：稳态噪声能量必须降下来 ---
    for sigma in (0.02, 0.1):
        noise = rng.normal(0, sigma, sample_rate * 2).astype("float32")
        out = main.preprocess_audio(noise, on)
        check(main._rms(out) < main._rms(noise), f"降噪 · 纯底噪 σ={sigma} 被压下去",
              f"rms {main._rms(noise):.5f} → {main._rms(out):.5f}")

    # 静音段不能被"归一"抬成一片噪声（真抬起来，ASR 会凭空听出内容）
    mixed = np.concatenate([np.zeros(sample_rate, dtype="float32"), tone(0.2, 1.0)])
    out = main.preprocess_audio(mixed, on)
    check(float(np.abs(out[:sample_rate]).max()) < 1e-3,
          "降噪 · 静音段仍然是静音",
          f"前半段峰值 {float(np.abs(out[:sample_rate]).max()):.2e}"
          "（纯数字静音不参与归一，不会被放大成'有内容'）")

    # --- 削峰保护与极端输入：绝不抛异常、绝不产生 NaN ---
    hot = (3.0 * np.sin(2 * np.pi * 220.0 * np.arange(sample_rate) / sample_rate)).astype("float32")
    out = main.preprocess_audio(hot, on)
    check(float(np.abs(out).max()) <= 0.991, "削峰保护 · 放大后不会削顶破音",
          f"峰值被等比压到 {float(np.abs(out).max()):.4f}")

    extreme = [
        ("零长音频", np.zeros(0, dtype="float32")),
        ("单个采样点", np.array([0.5], dtype="float32")),
        ("全零", np.zeros(sample_rate, dtype="float32")),
        ("常量直流", np.full(sample_rate * 2, 0.3, dtype="float32")),
        ("含 NaN / Inf", np.array([np.nan, np.inf, -np.inf, 0.1, -0.2] * 500, dtype="float32")),
        ("超短（不足一帧）", np.ones(100, dtype="float32")),
    ]
    for label, samples in extreme:
        try:
            out = main.preprocess_audio(samples, on)
            finite = bool(np.all(np.isfinite(out))) and len(out) == len(samples)
        except Exception as exc:                                   # noqa: BLE001
            finite = False
            out = f"{type(exc).__name__}: {exc}"
        check(finite, f"预处理兜底 · {label}",
              f"输入 {len(samples)} 点 → 输出 {len(out) if hasattr(out, '__len__') else out}，"
              "不抛异常、不产生 NaN、长度不变" if finite else f"异常：{out}")

    check(main.preprocess_audio(tone(0.1), off) is not None
          and abs(main._rms(main.preprocess_audio(tone(0.1), off)) - main._rms(tone(0.1))) < 1e-9,
          "AUDIO_PREPROCESS=false 时原样放行",
          "关掉开关就完全不动音频，排查问题时可以一键回到老行为")

    # 预处理在纯 numpy 下完成：不引入 scipy / noisereduce 这类额外依赖。
    # 按 AST 里的真实 import 判断，而不是搜文本 —— 注释里提到这些名字是正常的事。
    imported: set[str] = set()
    for node in ast.walk(ast.parse((PROJECT_ROOT / "main.py").read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    extra_audio = sorted(imported & {"scipy", "noisereduce", "librosa", "soundfile"})
    check(not extra_audio, "预处理只用 numpy 实现",
          "频谱减降噪是手写的（np.fft + sliding_window_view），少一层依赖风险"
          if not extra_audio else f"多引入了：{extra_audio}")

    # --- funasr / SenseVoice 输出清洗 ---
    tag_cases = [
        ("<|zh|><|NEUTRAL|><|Speech|><|woitn|>明天下午三点开会", "明天下午三点开会"),
        ("<|en|><|HAPPY|><|Speech|><|withitn|>Hello", "Hello"),
        ("明天下午三点开会", "明天下午三点开会"),
        ("", ""),
    ]
    # 装了 funasr 就走它官方的 rich_transcription_postprocess，它会把 <|HAPPY|> 之类的
    # 情绪标记**换成表情符号**（Hello → Hello😊）—— 这是官方行为，不是脏数据。
    # 所以这里断言的是"标记剥干净了、正文还在"，而不是逐字相等。
    bad = [(raw, main.clean_funasr_text(raw)) for raw, want in tag_cases
           if "<|" in main.clean_funasr_text(raw) or not main.clean_funasr_text(raw).startswith(want)]
    check(not bad, "SenseVoice 富文本标记清洗",
          "语言/情绪/事件标记全部剥掉（官方后处理还会把情绪转成表情符号），剩下的就是正文"
          if not bad else f"异常：{bad}")
    check(main.clean_funasr_text(None) == "", "清洗函数对 None 也不炸",
          "ASR 返回空结果时得到空串，不会把 None 带进后续链路")

    # --- ASR 后端选择：装了 funasr 才切过去，否则行为与改动前完全一致 ---
    has_funasr = main.Transcriber.funasr_available()
    auto = main.Transcriber(main.Config(beep=False, markdown_path="")).resolve_backend()
    forced = main.Transcriber(
        main.Config(beep=False, markdown_path="", asr_backend="funasr")).resolve_backend()
    pinned = main.Transcriber(
        main.Config(beep=False, markdown_path="", asr_backend="whisper")).resolve_backend()
    check(auto == ("funasr" if has_funasr else "whisper"), "ASR 后端 · auto 按可用性回退",
          f"funasr {'已安装' if has_funasr else '未安装'} → auto 选中 {auto!r}"
          + ("" if has_funasr else "（没装就退回 faster-whisper，老环境行为不变）"))
    check(forced == "funasr" and pinned == "whisper", "ASR 后端 · 显式配置说了算",
          f"ASR_BACKEND=funasr → {forced!r} · ASR_BACKEND=whisper → {pinned!r}")

    transcriber = main.Transcriber(main.Config(beep=False, markdown_path=""))
    check(bool(transcriber.describe()), "ASR 后端 · 有可读的一句话描述",
          f"describe() = {transcriber.describe()!r}（--doctor 与启动日志都用它）")

    check(main.Transcriber.whisper_available() or has_funasr,
          "ASR 后端 · 至少有一个可用",
          f"whisper={'有' if main.Transcriber.whisper_available() else '无'} · "
          f"funasr={'有' if has_funasr else '无'}；两个都缺时才需要看安装提示")

    # --- 依赖缺失时的降级：宁可少一个功能，也别让程序崩在启动那一行 ---
    # 手法是把 sys.modules["openai"] 置成 None：解释器碰到 None 会直接抛
    # ImportError，等价于"这台机器上没装 openai"，不用真的把它卸载掉。
    llm_cfg = main.Config(beep=False, markdown_path="", api_key="sk-selfcheck-dummy")
    saved_openai = sys.modules.get("openai", "∅")
    try:
        sys.modules["openai"] = None
        try:
            degraded = main.build_llm_client(llm_cfg)
            crashed = None
        except Exception as exc:                      # noqa: BLE001 — 就是要抓住崩溃
            degraded, crashed = "崩溃", exc
    finally:
        if saved_openai == "∅":
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = saved_openai
    check(crashed is None and degraded is None,
          "缺 openai 时降级成本地规则而不是崩",
          "没装 openai 只会退回内置规则整理（录音/识别/落库都不受影响），"
          "日志里还会打出当前解释器路径 —— 提示里带路径是因为最常见的翻车方式"
          "不是没装，而是装在了另一个 python 里（比如自动激活的空 .venv）"
          if crashed is None else f"抛出了 {type(crashed).__name__}: {crashed}")
    check("sys.executable" in inspect.getsource(main.build_llm_client),
          "降级日志点名当前解释器",
          "报错时直接告诉你「这条命令该装在哪个 python 里」，省得对着 pip install 发呆")
    check(main.build_llm_client(main.Config(beep=False, markdown_path="")) is None,
          "没有 API Key 时同样安静降级",
          "两种降级（缺包 / 缺 Key）都返回 None，走同一条本地规则通路")

    # --- 配置默认值：短结束词表是可关的开关，默认值要与常量一致 ---
    check(main.Config(beep=False, markdown_path="").short_end_keywords
          == main.DEFAULT_SHORT_END_KEYWORDS,
          "配置默认值 · 短结束词与常量一致",
          f"短结束词默认 {len(main.DEFAULT_SHORT_END_KEYWORDS)} 个："
          f"{' / '.join(main.DEFAULT_SHORT_END_KEYWORDS)}")
    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    missing_env = [key for key in ("ASR_BACKEND", "ASR_FUNASR_MODEL", "AUDIO_PREPROCESS",
                                   "AUDIO_DENOISE", "AUDIO_TARGET_DBFS", "AUDIO_MAX_GAIN_DB",
                                   "SHORT_END_KEYWORDS") if key not in env_example]
    check(not missing_env, ".env.example 收录了新增的配置项",
          f"缺少 {missing_env} —— 用户照模板配不出来" if missing_env
          else "ASR_BACKEND / AUDIO_* / SHORT_END_KEYWORDS 都在模板里，且不含真实 Key")
    check("sk-" not in env_example.replace("sk-xxxx", ""),
          ".env.example 里没有真实 API Key",
          "模板只放 sk-xxx 占位符，真实的 Key 只在 .env（已被 .gitignore 忽略）")


# --------------------------------------------------------------------------- #
# 9.5 医学问诊模式：热词提示词 / 识别后校正 / SOAP 结构 / 老库迁移
# --------------------------------------------------------------------------- #


#: 需求里逐字点名要有的词。它们必须**真的出现在解码提示词里**，
#: 不能只是躺在词表中段等着被截断 —— 这是"医学热词提示"这一条验收的核心。
NAMED_MEDICAL_WORDS = ("主诉", "现病史", "体格检查", "心前区", "听诊", "叩痛",
                       "反跳痛", "湿啰音", "黑便", "血常规", "急诊胃镜", "胃溃疡",
                       "心肌梗死", "血压", "脉搏")


def check_medical_mode(tmp_dir: Path) -> None:
    """请求 G 的验收点：热词提示词、识别后校正、SOAP 五段、老库无痛升级。

    刻意用**独立的数据库文件**，不碰主连接：这一组会插入带 SOAP 内容的病历卡片，
    而 Web 那组是按 ``body.count("## ")`` 统计条目数的 —— SOAP 段渲染成 ``### 主诉``，
    字符串里也含 ``"## "``，混在一起会把那个统计弄歪。
    """
    import main

    section("医学问诊模式（热词 · SOAP · 老库迁移）")

    # --- 9.5.1 热词表本身 ---
    words = main.DEFAULT_MEDICAL_HOTWORDS
    check(len(words) >= 60 and len(set(words)) == len(words),
          "医学热词表 · 规模与去重",
          f"{len(words)} 个词、无重复（症状 / 体征 / 部位 / 检查 / 诊断都覆盖到了）")
    missing_named = [w for w in NAMED_MEDICAL_WORDS if w not in words]
    check(not missing_named, "医学热词表 · 需求点名的词一个不少",
          f"全部 {len(NAMED_MEDICAL_WORDS)} 个点名词都在表里" if not missing_named
          else f"缺：{missing_named}")

    cfg = main.Config(beep=False, markdown_path="")
    prompt = main.build_asr_prompt(cfg)
    check(bool(prompt) and len(prompt) <= main.ASR_PROMPT_MAX_CHARS,
          "解码提示词 · 长度卡在 whisper 的上下文预算内",
          f"{len(prompt)} 字 ≤ {main.ASR_PROMPT_MAX_CHARS} 字"
          "（whisper 只吃得到约 224 token，超了不报错、只默默截断）")
    dropped_names = [w for w in NAMED_MEDICAL_WORDS if w not in prompt]
    check(not dropped_names, "解码提示词 · 点名词真的进了提示词（不是只躺在表里）",
          f"{len(NAMED_MEDICAL_WORDS)} 个点名词全部出现在提示词中" if not dropped_names
          else f"被截断掉了：{dropped_names} —— 词表顺序要按优先级排")
    pairs = [("干啰音", "湿啰音"), ("右上腹", "右下腹"), ("呕血", "咯血")]
    broken_pairs = [(a, b) for a, b in pairs if (a in prompt) != (b in prompt)]
    check(not broken_pairs, "解码提示词 · 反义词对不被拆散",
          "干/湿、左/右、呕/咯这几对要么都在、要么都不在 —— "
          "只喂一半等于让模型失去了区分能力" if not broken_pairs
          else f"只剩一半的对：{broken_pairs}")
    tail_only = [w for w in words if w not in prompt]
    check(len(tail_only) < len(words) // 2, "解码提示词 · 溢出到'仅校正'的词是少数",
          f"{len(words) - len(tail_only)} 个词当解码提示，{len(tail_only)} 个只参与识别后校正")

    check(main.build_asr_prompt(main.Config(beep=False, markdown_path="",
                                            asr_initial_prompt=False)) == "",
          "ASR_INITIAL_PROMPT=false 时不喂提示词", "开关一路管到底")
    check(main.build_asr_prompt(main.Config(beep=False, markdown_path="",
                                            asr_hotwords=())) == "",
          "ASR_HOTWORDS=off 时同样不喂", "热词整组关掉就退回改动前的行为")
    check(main.Config(beep=False, markdown_path="").asr_hotwords == main.DEFAULT_MEDICAL_HOTWORDS
          and main.Config(beep=False, markdown_path="").asr_hotword_correct,
          "热词与校正默认开启",
          "不配任何环境变量也是医学问诊模式；ASR_HOTWORDS=off 才退回改动前的行为")

    # --- 9.5.2 识别后校正：改对哪些、绝不改错哪些 ---
    # 期望值全是**实跑出来的**，不是照着代码猜的。负例比正例重要：医学里左右、
    # 干湿、上下都是相反的体征，改错一个字比不改危害大得多。
    fix_cases = [
        ("双肺闻及湿罗音", "双肺闻及湿啰音", "同音别字「罗」→「啰」被掰回来"),
        ("心电图提示心机梗死", "心电图提示心肌梗死", "「心机」→「心肌」"),
        ("急诊胃境检查", "急诊胃镜检查", "「胃境」→「胃镜」"),
        ("心尖博动弥散", "心尖搏动弥散", "「博动」→「搏动」"),
        ("双下肢水肿，柏油样边", "双下肢水肿，柏油样便", "「柏油样边」→「柏油样便」"),
        # ↓ 以下必须**一个都不改**
        ("双肺闻及干啰音", "双肺闻及干啰音", "干/湿是相反的体征，改了这一条就废了"),
        ("右下腹麦氏点压痛阳性，反跳疼明显", "右下腹麦氏点压痛阳性，反跳疼明显",
         "右下腹不许被改成右上腹（左右相反）"),
        ("患者主诉心前区疼痛三天", "患者主诉心前区疼痛三天", "正确写法不许被二次改写"),
        ("口痛得厉害", "口痛得厉害", "「口痛」不许被塞成「叩击痛」（长度不同，直接排除）"),
        ("主诉：黑便三天", "主诉：黑便三天", "窗口里混进标点就跳过，不跨句改写"),
        ("他的名字叫张伟", "他的名字叫张伟", "日常句子不会被医学术语误伤"),
        ("血长规提示白细胞升高", "血长规提示白细胞升高",
         "多音字「长」在词典里取的是 zhǎng 音，与「常」差了声母 —— 这一层改不动，"
         "正是要靠解码提示词先一步选对字"),
    ]
    for source, expected, why in fix_cases:
        actual = main.correct_medical_terms(source, main.DEFAULT_MEDICAL_HOTWORDS)
        check(actual == expected, f"识别后校正 · {source} → {expected}",
              f"得到 {actual!r}（{why}）")
    check(main.correct_medical_terms("双肺闻及湿罗音", ()) == "双肺闻及湿罗音",
          "识别后校正 · 词表为空时原样放行", "关掉热词就一点都不动文本")
    check(main.Transcriber(main.Config(beep=False, markdown_path="",
                                       asr_hotword_correct=False))._fix_terms("湿罗音") == "湿罗音",
          "ASR_HOTWORD_CORRECT=false 时关掉整层校正",
          "_fix_terms 是这一层的开关包装，关掉就原样返回")

    # --- 9.5.3 SOAP 结构 ---
    check(main.SOAP_COLUMNS == ("chief_complaint", "history", "physical_exam",
                                "assessment", "plan"),
          "SOAP 五段 · 顺序与列名",
          "主诉 → 现病史 → 体格检查 → 诊断思考 → 诊疗计划（卡片/Markdown/DOCX 共用这一份）")
    check([label for _key, label in main.SOAP_SECTIONS] == ["主诉", "现病史", "体格检查",
                                                           "诊断思考", "诊疗计划"],
          "SOAP 五段 · 中文标题", "问诊实训报告的标准栏目名")
    check(set(main.EDITABLE_FIELDS) <= set(main.REGENERATE_FIELDS)
          and {"category", "todos"} <= set(main.REGENERATE_FIELDS),
          "字段白名单 · 可编辑 ⊆ 可重新生成",
          f"手工可改 {len(main.EDITABLE_FIELDS)} 个字段；模型可覆写 "
          f"{len(main.REGENERATE_FIELDS)} 个（多出 category/todos —— 待办是数组、"
          "分类是枚举，做成 contenteditable 一改就坏）")
    check(not (set(main.EDITABLE_FIELDS) & {"category", "todos"}),
          "字段白名单 · category/todos 不给手改", "两个白名单的差集就是这里刻意留的")
    check(set(main.SOAP_COLUMNS) <= set(main.EDITABLE_FIELDS),
          "字段白名单 · SOAP 五段都允许手工修改", "卡片上的 SOAP 区域直接可编辑")

    med_text = ("主诉腹痛三天。现病史三天前吃了火锅，之后上腹持续隐痛，伴恶心呕吐。"
                "体格检查上腹压痛阳性，无反跳痛。初步诊断急性胃炎。"
                "诊疗计划血常规加急诊胃镜。")
    soap = main.split_spoken_soap(med_text)
    check(soap["chief_complaint"] == "腹痛三天" and soap["history"].startswith("三天前吃了火锅")
          and soap["physical_exam"] == "上腹压痛阳性，无反跳痛"
          and soap["assessment"] == "急性胃炎" and soap["plan"] == "血常规加急诊胃镜",
          "本地兜底 · 按口语栏目名切出 SOAP 五段",
          f"主诉={soap['chief_complaint']!r} · 现病史={soap['history'][:12]!r}… · "
          f"体查={soap['physical_exam']!r}")
    check(not any(main.split_spoken_soap("明天下午三点跟客户开会").values()),
          "本地兜底 · 非问诊内容不产出病历",
          "一个栏目名都没提到就返回全空 —— 日常备忘不会长出假病历")

    empty_memo = main.normalize_memo({}, "随便说点什么", "2026-10-09 09:00:00")
    check(all(empty_memo.get(key) == "" for key in main.SOAP_COLUMNS),
          "归一化 · SOAP 字段缺失时填空串", "空串 = 卡片退回摘要视图，而不是显示「None」")
    check(not main.has_soap(empty_memo), "has_soap · 全空判为不是病历")
    check(main.has_soap({**empty_memo, "history": "三天前吃了火锅"}),
          "has_soap · 只填一段也算病历", "只要有一段就按病历卡片渲染")

    regen_content = main.build_regenerate_content(
        {"title": "问诊记录", "raw_text": "原始语音：主诉肚子疼", "chief_complaint": "腹痛 3 天",
         "history": "", "physical_exam": "", "assessment": "", "plan": "", "summary": ""},
        "2026-10-09 09:00:00")
    check(regen_content.find("腹痛 3 天") < regen_content.find("原始语音：主诉肚子疼")
          and "以此为准" in regen_content and "仅作参考" in regen_content,
          "重新润色 · 手工修订版排在原始语音前面",
          "模型对'前文优先'很敏感：顺序反了就会拿原始语音盖掉学生刚改对的地方")

    # --- 9.5.4 独立数据库：落库、白名单 UPDATE、老库迁移 ---
    db = main.init_db(tmp_dir / "medical_check.db")
    try:
        memo_id = main.insert_memo(db, {
            "record_time": "2026-10-09 09:00:00",
            "raw_text": "主诉腹痛三天，现病史三天前吃了火锅",
            "title": "问诊记录 · 腹痛", "category": "问诊记录",
            "summary": "腹痛三天", "todos": [],
            "chief_complaint": "腹痛 3 天", "history": "三天前吃火锅后上腹隐痛",
            "physical_exam": "上腹压痛", "assessment": "急性胃炎", "plan": "血常规 + 急诊胃镜",
        })
        stored = main.fetch_memo(db, memo_id)
        check(all(stored[key] for key in main.SOAP_COLUMNS),
              "落库 · SOAP 五段原样往返", "写入 → 读回，五段一字不差")
        payload = main.memo_to_dict(stored)
        check(payload.get("has_soap") is True
              and [s["key"] for s in payload.get("soap_sections", [])] == list(main.SOAP_COLUMNS),
              "memo_to_dict · 下发 SOAP 字段与渲染顺序",
              "前端拿到 soap_sections 才知道按什么顺序、用什么标题渲染病历卡片")

        updated = main.update_memo(db, memo_id, {
            "chief_complaint": "腹痛 4 天（学生手改）",
            "category": "工作",                     # 不在 EDITABLE_FIELDS 里
            "title; DROP TABLE memos": "注入了",     # 列名注入
            "id": 99999,                            # 主键也不许改
        })
        check(updated["chief_complaint"] == "腹痛 4 天（学生手改）",
              "手工修改 · 白名单字段写进去了", "contenteditable 回写生效")
        check(updated["category"] == "问诊记录" and updated["id"] == memo_id,
              "手工修改 · 白名单外的字段一律丢弃",
              f"category 仍是 {updated['category']!r}、id 仍是 {memo_id} —— "
              "字段名既是校验白名单，也是拼 UPDATE 语句时唯一的列名来源")
        injected = db.execute("SELECT COUNT(*) FROM memos").fetchone()[0]
        check(injected == 1, "手工修改 · 注入式字段名没拼进 SQL",
              "「title; DROP TABLE memos」不在白名单里，连拼串的机会都没有；"
              f"表还在、还有 {injected} 行")

        regenerated = main.update_memo(db, memo_id,
                                       {"category": "工作", "todos": [{"task": "复查", "due": None}],
                                        "assessment": "模型重写的诊断思考"},
                                       allowed=main.REGENERATE_FIELDS)
        check(regenerated["category"] == "工作"
              and regenerated["todos"][0]["task"] == "复查",
              "重新润色 · 走 REGENERATE_FIELDS 能写 category 与 todos",
              "两个白名单的差别就是「谁能写」的差别：模型能刷新待办，学生不能手改数组")
        long_text = "很" * (main.EDIT_FIELD_MAX_CHARS + 500)
        truncated = main.update_memo(db, memo_id, {"summary": long_text})
        check(len(truncated["summary"]) == main.EDIT_FIELD_MAX_CHARS,
              "手工修改 · 单字段长度封顶",
              f"{len(long_text)} 字被截到 {main.EDIT_FIELD_MAX_CHARS} 字 —— "
              "contenteditable 一次误粘贴不该把几 MB 文本塞进库里和每张卡片里")
        check(main.update_memo(db, memo_id, {"title": "改不存在的"}) is not None
              and main.update_memo(db, 999999, {"title": "x"}) is None,
              "手工修改 · 不存在的 id 返回 None", "接口据此回 404，而不是假装写成功")

        # 老库（SOAP 列还不存在时的表结构）升级：加列必须不动数据
        legacy_path = tmp_dir / "legacy_memos.db"
        raw = sqlite3.connect(str(legacy_path))
        raw.executescript("""
            CREATE TABLE memos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_time TEXT NOT NULL, raw_text TEXT NOT NULL,
                title TEXT, category TEXT, summary TEXT, todos TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
            );
        """)
        raw.execute("INSERT INTO memos (record_time, raw_text, title, category, summary, todos)"
                    " VALUES (?,?,?,?,?,?)",
                    ("2026-01-01 09:00:00", "升级前的老记录", "老卡片", "其他",
                     "升级前就存在的内容", "[]"))
        raw.commit()
        raw.close()
        upgraded = main.init_db(legacy_path)
        try:
            columns = {row["name"] for row in upgraded.execute("PRAGMA table_info(memos)")}
            check(set(main.SOAP_COLUMNS) <= columns, "老库升级 · SOAP 五列自动补齐",
                  "缺的列用 ALTER TABLE ADD COLUMN 补上（O(1) 元数据操作，不重建表）"
                  if set(main.SOAP_COLUMNS) <= columns
                  else f"升级后仍然缺：{sorted(set(main.SOAP_COLUMNS) - columns)}")
            row = upgraded.execute("SELECT title, summary, chief_complaint FROM memos").fetchone()
            check(row["title"] == "老卡片" and row["summary"] == "升级前就存在的内容"
                  and (row["chief_complaint"] or "") == "",
                  "老库升级 · 老记录原样保留",
                  "升级只加列、不重建表：老卡片读得出来，新的 SOAP 段是空串（卡片自动退回摘要视图）")
            check(main._migrate_schema(upgraded) == [], "老库升级 · 再迁一次是空操作",
                  "幂等：每次启动都跑一遍也不会有副作用")
        finally:
            upgraded.close()
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# 10. Web 接口与导出
# --------------------------------------------------------------------------- #


def check_web_interface(tmp_dir: Path, conn) -> None:
    section("Web 看板接口与导出")

    if conn is None:
        check(False, "Web 接口", "前置的数据库检查未通过，跳过")
        return
    try:
        import flask  # noqa: F401
    except ImportError as exc:
        skip("Web 看板接口测试", f"未安装 flask（{exc}）；pip install flask")
        return

    import webui

    import main

    # markdown_path 特意打开：删除卡片时"数据库 + my_memos.md 同步"这条链路
    # 只有真的有个文件可写才验得出来
    cfg = main.Config(beep=False, markdown_path=str(tmp_dir / "web_memos.md"),
                      db_path=str(tmp_dir / "web.db"))
    state = main.AppState(wake_mode=True, wake_word=cfg.wake_word)
    pipeline = main.MemoPipeline(cfg, conn, main.Transcriber(cfg), client=None,
                                 on_memo=state.publish_memo)
    app = webui.create_app(cfg, conn, state, pipeline=pipeline)
    client = app.test_client()

    # --- 10.1 页面 ---
    page = client.get("/")
    html = page.get_data(as_text=True)
    check(page.status_code == 200, "GET / 返回看板页面", f"HTTP {page.status_code}")
    check("正在" in html or "待机" in html, "页面显示当前状态",
          "状态胶囊会随 SSE 推送实时变化，无需手动刷新")
    check(cfg.wake_word in html, "页面提示唤醒词", f"页面上能看到「{cfg.wake_word}」用法说明")
    check("/api/export?fmt=md" in html and "/api/export?fmt=docx" in html,
          "页面提供一键导出按钮", "Markdown / Word 各一个下载入口")
    check("EventSource" in html, "页面通过 SSE 自动接收推送",
          "用 EventSource 订阅 /api/stream，新卡片自动浮现")
    check('class="del"' in html and "confirm(" in html and "dataset.removing" in html,
          "页面带着删除按钮与防重复的前端逻辑",
          "点删除先二次确认；同一张卡即便被本地点击和 SSE 广播各触发一次，也只减一次计数")

    # --- 10.2 JSON 接口 ---
    state_json = client.get("/api/state").get_json()
    check(state_json.get("state") == main.STATE_STANDBY
          and "standby" == state_json.get("state"),
          "GET /api/state 返回状态快照",
          f"state={state_json.get('state')!r} label={state_json.get('label')!r}")
    memos_json = client.get("/api/memos?limit=5").get_json()
    check(isinstance(memos_json.get("memos"), list), "GET /api/memos 返回卡片列表",
          f"共 {memos_json.get('count')} 条（按时间倒序）")

    # --- 10.3 SSE：新连接立刻拿到当前状态 ---
    stream_resp = client.get("/api/stream", buffered=False)
    check(stream_resp.mimetype == "text/event-stream", "GET /api/stream 是 SSE 流",
          f"Content-Type = {stream_resp.mimetype}")
    first_line = b""
    for piece in stream_resp.response:
        first_line += piece if isinstance(piece, bytes) else piece.encode()
        if b"\n\n" in first_line:
            break
    stream_resp.close()
    check(b"data:" in first_line and b"state" in first_line,
          "SSE · 刚连上就补一份当前状态",
          f"第一帧：{first_line.decode('utf-8', 'replace').strip()[:90]}")

    # --- 10.4 Broadcaster：状态与卡片真的会推出去 ---
    box = state.broadcaster.subscribe()
    state.set_state(main.STATE_RECORDING, "测试中")
    event = box.get(timeout=2)
    check(event.get("type") == "state" and event.get("state") == main.STATE_RECORDING,
          "Broadcaster · 状态变化即时推送",
          f"收到 {event.get('type')} 事件，label={event.get('label')!r}")
    state.set_state(main.STATE_STANDBY)
    box.get(timeout=2)

    count_before = state.saved_count
    memo_id = pipeline.handle_text("明天下午三点跟客户开会")
    pushed = box.get(timeout=3)
    check(pushed.get("type") == "memo" and pushed["memo"]["id"] == memo_id,
          "Broadcaster · 新卡片即时推送",
          f"落库 #<{memo_id}> 后立刻推出卡片：{pushed['memo']['title']!r}")
    check(state.saved_count == count_before + 1 and state.last_memo_id == memo_id,
          "Broadcaster · 运行统计同步更新", f"本次已记录 {state.saved_count} 条")
    state.broadcaster.unsubscribe(box)

    # --- 10.5 POST /api/text：网页上打字也能生成卡片 ---
    resp = client.post("/api/text", json={"text": "小医小医，记得买牛奶，下一条"})
    payload = resp.get_json()
    check(resp.status_code == 200 and payload.get("ok"), "POST /api/text 生成卡片",
          f"返回 {payload}")
    check(client.post("/api/text", json={"text": "   "}).status_code == 400,
          "POST /api/text 拒绝空内容", "空文本返回 400，不会写出空卡片")

    # --- 10.6 导出下载 ---
    md = client.get("/api/export?fmt=md")
    body = md.get_data(as_text=True)
    headers = {k.lower(): v for k, v in md.headers.items()}
    rows = conn.execute("SELECT COUNT(*) FROM memos").fetchone()[0]
    check(md.status_code == 200 and "attachment" in headers.get("content-disposition", ""),
          "GET /api/export?fmt=md 是下载而不是内嵌",
          f"Content-Disposition = {headers.get('content-disposition', '')}")
    check(body.count("## ") == rows and "共 %d 条记录" % rows in body,
          "Markdown 导出内容完整",
          f"{rows} 条记录各一个二级标题，头部带统计信息")
    check("<" not in body and body.startswith("# "), "Markdown 导出干净",
          "没有混入 HTML 标签，以一级标题开头")

    try:
        import docx  # noqa: F401

        word = client.get("/api/export?fmt=docx")
        check(word.status_code == 200 and len(word.get_data()) > 0,
              "GET /api/export?fmt=docx 生成 Word",
              f"{len(word.get_data()):,} 字节，"
              f"Content-Disposition = {word.headers.get('Content-Disposition', '')}")
    except ImportError:
        word = client.get("/api/export?fmt=docx")
        check(word.status_code == 501 and "python-docx" in str(word.get_json()),
              "缺 python-docx 时给出安装提示", "返回 501 并说明装什么，而不是崩掉")

    bad = client.get("/api/export?fmt=pdf")
    check(bad.status_code == 400, "非法导出格式被拒绝", f"HTTP {bad.status_code}")

    # --- 10.7 删除卡片：数据库 + my_memos.md 同步，并且广播给所有页面 ---
    import queue as _queue

    md_path = Path(cfg.markdown_path)
    victim = pipeline.handle_text("删除测试专用条目7788")
    md_with_victim = md_path.read_text(encoding="utf-8")
    check(victim is not None and "7788" in md_with_victim,
          "准备 · 新卡片同时写进了数据库与 Markdown",
          f"#{victim} 已落库，{md_path.name} 里能找到它")
    rows_before = conn.execute("SELECT COUNT(*) FROM memos").fetchone()[0]
    saved_before = state.saved_count

    box = state.broadcaster.subscribe()
    resp = client.delete(f"/api/delete/{victim}")
    payload = resp.get_json()
    check(resp.status_code == 200 and payload.get("ok"),
          "DELETE /api/delete/<id> 删除卡片", f"返回 {payload}")
    left = conn.execute("SELECT COUNT(*) FROM memos WHERE id = ?", (victim,)).fetchone()[0]
    rows_after = conn.execute("SELECT COUNT(*) FROM memos").fetchone()[0]
    check(left == 0 and rows_after == rows_before - 1, "删除后数据库里没有这条记录",
          f"id={victim} 命中 {left} 行，总行数 {rows_before} → {rows_after}")
    md_after = md_path.read_text(encoding="utf-8")
    check("7788" not in md_after and md_after.count("## ") == rows_after,
          "删除后 my_memos.md 与数据库重新对齐",
          f"被删的条目（含原始文本）已从文档里消失，"
          f"文档 {md_after.count('## ')} 条 = 数据库 {rows_after} 行")

    # 广播：删除事件 + 计数减一的 state 事件，页面据此让卡片消失、数字减 1
    events_seen: list = []
    while True:
        try:
            events_seen.append(box.get(timeout=1))
        except _queue.Empty:
            break
    state.broadcaster.unsubscribe(box)
    kinds = [event.get("type") for event in events_seen]
    check("delete" in kinds and "state" in kinds,
          "删除后广播 delete 事件 + 计数刷新",
          f"推给页面的事件序列：{kinds}（其它标签页也能同步消失）")
    latest = [event for event in events_seen if event.get("type") == "state"]
    check(latest and latest[-1].get("saved_count") == max(0, saved_before - 1),
          "删除后「本次已记录」减一",
          f"saved_count {saved_before} → {latest[-1].get('saved_count') if latest else None}")

    # POST 与 DELETE 等价；删不存在的 id 必须 404 而不是假装成功
    spare = pipeline.handle_text("再删一条试试5566")
    posted = client.post(f"/api/delete/{spare}")
    check(posted.status_code == 200 and posted.get_json().get("ok"),
          "POST /api/delete/<id> 与 DELETE 等价",
          f"网页按钮用哪个方法都能删（HTTP {posted.status_code}）")
    missing = client.delete("/api/delete/999999")
    check(missing.status_code == 404, "删除不存在的卡片返回 404",
          f"HTTP {missing.status_code}：{missing.get_json().get('error')!r}")
    check(main.remove_memo(conn, 999999, "") is None,
          "remove_memo 对不存在的 id 返回 None", "命令行/接口两条路都不会静默成功")

    # --- 10.8 写入串行化：语音线程在存，Web 线程可能在删 ---
    # 插入是"写库 + 追加文档"，删除是"删库 + 整份重刷文档"。不互斥的话，重刷可能
    # 刚好落在"已提交但还没追加"的几微秒空档里，于是同一条在文档里出现两次。
    # 这个窗口靠并发跑很难稳定复现（实测把锁换成空操作也常常蒙混过关），
    # 所以这里直接白盒验证四条写路径确实都在锁里，后面再补一次并发冒烟。
    class _SpyLock:
        """只统计被进入几次，不做真互斥 —— 这一段是单线程调用，够用了。"""

        def __init__(self) -> None:
            self.entered = 0

        def __enter__(self) -> "_SpyLock":
            self.entered += 1
            return self

        def __exit__(self, *exc_info) -> bool:
            return False

    real_lock = main._STORE_LOCK
    first = real_lock.acquire(blocking=False)
    nested = real_lock.acquire(blocking=False) if first else False
    if nested:
        real_lock.release()
    if first:
        real_lock.release()
    check(first and nested, "写入锁是可重入的（RLock）",
          "remove_memo 持锁期间还要进 delete_memo / export_markdown，普通 Lock 会自锁死")

    spy_conn = main.init_db(tmp_dir / "lock.db")
    spy_md = tmp_dir / "lock.md"
    spy = _SpyLock()
    marks: dict = {}
    main._STORE_LOCK = spy                      # type: ignore[assignment]
    try:
        sample = {"record_time": "2026-10-08 12:00:00", "raw_text": "锁检查",
                  "title": "锁检查", "category": "其他", "summary": "锁检查", "todos": []}
        for label, call in (
            ("insert_memo", lambda: sample.update(id=main.insert_memo(spy_conn, sample))),
            ("append_memo_markdown", lambda: main.append_memo_markdown(sample, str(spy_md))),
            ("export_markdown", lambda: main.export_markdown(spy_conn, str(spy_md))),
            ("remove_memo", lambda: main.remove_memo(spy_conn, sample["id"], str(spy_md))),
        ):
            before = spy.entered
            call()
            marks[label] = spy.entered - before
    finally:
        main._STORE_LOCK = real_lock            # type: ignore[assignment]
        spy_conn.close()
    check(all(count >= 1 for count in marks.values()), "写库 + 写文档的四条路径都在锁内",
          f"各函数进入锁的次数：{marks}")

    # 真实并发冒烟：两个写线程 + 一个删线程，不变式是"文档条目数 = 数据库行数"
    race_conn = main.init_db(tmp_dir / "race.db")
    race_md = tmp_dir / "race.md"
    race_cfg = main.Config(beep=False, markdown_path=str(race_md))
    race_pipeline = main.MemoPipeline(race_cfg, race_conn, main.Transcriber(race_cfg),
                                      client=None)
    errors: list = []
    pool: list = []
    pool_lock = threading.Lock()
    stop = threading.Event()

    def saver(tag: str) -> None:
        try:
            for index in range(5):
                memo_id = race_pipeline.handle_text(f"并发写入{tag}第{index}条")
                if memo_id:
                    with pool_lock:
                        pool.append(memo_id)
        except Exception as exc:                      # noqa: BLE001 - 记下来由断言报出
            errors.append(exc)

    def remover() -> None:
        while not stop.is_set():
            with pool_lock:
                memo_id = pool.pop(0) if pool else None
            if memo_id is None:
                time.sleep(0.002)
                continue
            try:
                main.remove_memo(race_conn, memo_id, str(race_md))
            except Exception as exc:                  # noqa: BLE001
                errors.append(exc)

    threads = [threading.Thread(target=remover, name="race-remover"),
               threading.Thread(target=saver, args=("甲",), name="race-save-a"),
               threading.Thread(target=saver, args=("乙",), name="race-save-b")]
    for thread in threads:
        thread.start()
    for thread in threads[1:]:
        thread.join(timeout=30)
    stop.set()
    threads[0].join(timeout=10)

    rows_left = race_conn.execute("SELECT COUNT(*) FROM memos").fetchone()[0]
    entries_left = race_md.read_text(encoding="utf-8").count("## ")
    check(not errors, "并发存/删不抛异常（冒烟）",
          f"2 个写线程 + 1 个删线程同时跑了 10 条 —— 异常：{errors[:2]}")
    check(entries_left == rows_left, "并发存/删后文档与数据库仍然一致（冒烟）",
          f"{race_md.name} 里 {entries_left} 个条目 = 数据库 {rows_left} 行")
    race_conn.close()

    # --- 10.9 手动结束 / 放弃按钮：说不出指令词时的双保险 ---
    page_html = client.get("/").get_data(as_text=True)
    for label, needle in (
        ("状态栏里有「结束并整理」按钮", 'id="btn-finish"'),
        ("状态栏里有「放弃本条」按钮", 'id="btn-drop"'),
        ("按钮藏在只在录音时显示的分组里", 'id="live-controls"'),
        ("按钮文案是中文大字", "结束并整理"),
        ("放弃按钮文案", "放弃本条"),
        ("前端把状态变化接到按钮显隐上", "applyLiveControls"),
        ("前端真的去调控制接口", "/api/control/"),
        ("放弃是不可逆操作，先二次确认", "confirm("),
    ):
        check(needle in page_html, f"网页手动按钮 · {label}", f"页面里能找到 {needle!r}")

    # 没有麦克风的进程（--no-mic）没有可控制的录音，必须明说而不是假装成功
    bare = webui.create_app(cfg, conn, state, pipeline=pipeline)
    resp = bare.test_client().post("/api/control/finish")
    check(resp.status_code == 503, "网页手动按钮 · 无麦克风时返回 503",
          f"HTTP {resp.status_code}：{(resp.get_json() or {}).get('error')!r}")

    check(client.post("/api/control/whatever").status_code in (400, 404),
          "网页手动按钮 · 未知动作被拒绝",
          "action 只认 finish / cancel，别的一律 400")
    check(client.get("/api/control/finish").status_code == 405,
          "网页手动按钮 · 只能 POST",
          "GET 会 405 —— 浏览器预取/爬虫不会误触录音收尾")

    fake = main.VoiceListener(cfg, None)
    webui.set_listener(app, fake)
    resp = client.post("/api/control/finish")
    check(resp.status_code == 409, "网页手动按钮 · 待机时点「结束并整理」返回 409",
          f"HTTP {resp.status_code}：{(resp.get_json() or {}).get('error')!r}"
          "（不在录音就别造一段空录音出来）")

    fake.state = main.STATE_RECORDING
    resp = client.post("/api/control/finish")
    check(resp.status_code == 200 and (resp.get_json() or {}).get("ok"),
          "网页手动按钮 · 录音中「结束并整理」被受理",
          f"返回 {(resp.get_json() or {})}")
    check(fake._finish_request.is_set() and fake.cancelled == 0,
          "网页手动按钮 · 只置标志位，不碰音频缓冲",
          "Web 线程只 set 一个 Event；截断与转写由录音主循环下一轮完成，"
          "不会和 VAD 状态机抢同一个缓冲")
    check(fake._take_manual() == "finish", "录音主循环取到「结束」请求",
          "主循环每轮开头先看有没有网页请求，拿到就立刻收尾")

    resp = client.post("/api/control/cancel")
    check(resp.status_code == 200 and fake._cancel_request.is_set(),
          "网页手动按钮 · 录音中「放弃本条」被受理", f"返回 {(resp.get_json() or {})}")
    check(fake._take_manual() == "cancel", "录音主循环取到「放弃」请求",
          "放弃请求同样在主循环里被消费，音频直接丢弃")
    fake.state = main.STATE_STANDBY
    webui.set_listener(app, None)

    # --- 10.10 卡片上的实时修改与「重新润色」 ---
    # 先造一张带 SOAP 的病历卡片 —— 这一组跑在最后，前面按
    # ``body.count("## ")`` 统计条目数的检查都已经跑完了，多一条不影响它们。
    soap_id = pipeline.handle_text("主诉腹痛三天。现病史三天前吃火锅后上腹隐痛。"
                                   "体格检查上腹压痛。初步诊断急性胃炎。"
                                   "诊疗计划血常规加急诊胃镜。")
    soap_memo = main.fetch_memo(conn, soap_id)
    check(main.has_soap(soap_memo), "病历卡片 · 语音内容被整理成 SOAP 五段",
          f"#{soap_id} 主诉={soap_memo['chief_complaint']!r} "
          f"诊疗计划={soap_memo['plan']!r}")

    card_html = client.get("/").get_data(as_text=True)
    for label, needle in (
        ("SOAP 区域可直接编辑", 'contenteditable="true"'),
        ("可编辑区域按字段名标注", 'data-field='),
        ("五个 SOAP 段都渲染出来了", 'class="soap-block"'),
        ("占位提示走 CSS，不会写进记录", ':empty::before'),
        ("有「💾 保存修改」按钮", "💾 保存修改"),
        ("有「🔄 智能润色/推演」按钮", "智能润色/推演"),
        ("前端把修改回写到保存接口", "/api/memos/update/"),
        ("前端把推演请求发到 re-summarize 接口", "/api/memos/re-summarize/"),
        ("推演期间卡片上有加载提示", "正在调用 DeepSeek 进行医学推理"),
        ("前端上报「学生亲手改过哪些字段」", "touched"),
        ("推演完成后给被保留的段落加标记", "kept"),
        ("手改未保存时离开页面会拦一下", "beforeunload"),
    ):
        check(needle in card_html, f"实时交互 · {label}", f"页面里能找到 {needle!r}")

    # 旧路径还在（老书签、老脚本、浏览器里缓存的旧页面都还在用），新路径也认
    rules = sorted(str(rule) for rule in app.url_map.iter_rules())
    for wanted in ("/api/memos/update/<int:memo_id>",
                   "/api/memos/re-summarize/<int:memo_id>",
                   "/api/memo/<int:memo_id>/update",
                   "/api/memo/<int:memo_id>/regenerate"):
        check(wanted in rules, f"路由已注册 · {wanted}",
              "新路径是现行的，旧路径保留为别名（同一个视图函数挂两条规则）")

    category_before = main.fetch_memo(conn, soap_id)["category"]
    edited = client.post(f"/api/memos/update/{soap_id}",
                         json={"fields": {"physical_exam": "上腹压痛阳性，无反跳痛",
                                          "category": "工作"},
                               "touched": ["physical_exam"]})
    payload = edited.get_json() or {}
    check(edited.status_code == 200 and payload.get("ok"), "POST /api/memos/update 保存手工修改",
          f"返回 {payload.get('memo', {}).get('physical_exam')!r}，"
          f"被忽略的字段 {payload.get('ignored')}")
    check(main.fetch_memo(conn, soap_id)["physical_exam"] == "上腹压痛阳性，无反跳痛",
          "POST /api/memos/update 真的写进了数据库",
          "刷新页面/其它标签页拿到的都是改过的版本")
    check(payload.get("memo", {}).get("category") == category_before
          and "category" in (payload.get("ignored") or []),
          "POST /api/memos/update 拒绝白名单外的字段并如实汇报",
          f"category 是枚举，不给手工改（仍是 {category_before!r}）；接口回 200 但把"
          "被忽略的字段列出来，而不是静默丢弃")
    check(payload.get("markdown_synced") is True
          and payload.get("saved") == ["physical_exam"],
          "保存接口如实汇报「存了哪些字段」与「文档同步了没」",
          f"saved={payload.get('saved')} · markdown_synced={payload.get('markdown_synced')}")

    # 旧路径照样能用（前端已切走，但老页面缓存还在）
    legacy = client.post(f"/api/memo/{soap_id}/update",
                         json={"fields": {"summary": "走旧路径存的"}})
    check(legacy.status_code == 200
          and main.fetch_memo(conn, soap_id)["summary"] == "走旧路径存的",
          "旧路径 /api/memo/<id>/update 仍然可用",
          "别名不是摆设：浏览器里缓存的旧页面还在打这个地址")

    bad_field = client.post(f"/api/memos/update/{soap_id}", json={"fields": {"没这个字段": "x"}})
    check(bad_field.status_code == 400, "POST /api/memos/update 全是非法字段时返回 400",
          f"HTTP {bad_field.status_code}：{(bad_field.get_json() or {}).get('error')!r}")
    check(client.post("/api/memos/update/999999",
                      json={"fields": {"title": "x"}}).status_code == 404,
          "POST /api/memos/update 对不存在的卡片返回 404", "不会凭空造一条记录出来")

    # 没配 API Key（client=None）时"智能推演"必须明说，而不是假装推演成功
    no_llm = client.post(f"/api/memos/re-summarize/{soap_id}", json={"fields": {}})
    check(no_llm.status_code == 503, "POST /api/memos/re-summarize 无 LLM 时返回 503",
          f"HTTP {no_llm.status_code}：{(no_llm.get_json() or {}).get('error')!r}")
    check(client.post("/api/memos/re-summarize/999999",
                      json={"fields": {}}).status_code == 404,
          "POST /api/memos/re-summarize 对不存在的卡片返回 404", "先查 id，再谈推演")

    # --- 临床推演提示词本身 ---
    for label, needle in (
        ("角色是带教医师 / OSCE 考官", "OSCE"),
        ("要求推初步诊断", "最可能的初步诊断"),
        ("要求写鉴别诊断的支持点与不支持点", "支持点与不支持点"),
        ("信息不足时走「待查」而不是猜病名", "待查"),
        ("要求说明每项检查是为了排除什么", "排除或证实"),
        ("要求写护理观察与危险信号", "危险信号"),
        ("不许凭空创造体征", "不许凭空创造临床事实"),
        ("诊断只许写「待排/待查」口径", "待排"),
        ("忠于学生的修改", "忠于学生的修改"),
        ("声明仅用于教学演练", "教学演练"),
    ):
        check(needle in main.CLINICAL_PROMPT, f"临床推演提示词 · {label}",
              f"提示词里能找到 {needle!r}")
    check(main.CLINICAL_PROMPT != main.SYSTEM_PROMPT
          and "整理成结构化 JSON" not in main.CLINICAL_PROMPT,
          "推演用的是另一份提示词，不是首次整理那一份",
          "首次整理是「书记员」（只许如实记录），推演是「带教老师」（要做临床推理）——"
          "合成一份必然两头不讨好")
    check(main.CLINICAL_MAX_TOKENS >= 1500,
          "推演的输出上限比首次整理宽",
          f"max_tokens={main.CLINICAL_MAX_TOKENS}：三层鉴别 + 三层计划写全了超过 1000 token，"
          "沿用 800 会在「诊疗计划」写到一半被截断，而截断的 JSON 直接解析失败")

    # --- 推演输入：全量 JSON + 空段提示 ---
    full = main.build_regenerate_content(
        {"title": "问诊记录", "category": "问诊记录", "summary": "摘要",
         "todos": [{"task": "复查", "due": None}], "raw_text": "原始语音：主诉肚子疼",
         "chief_complaint": "腹痛 3 天", "history": "", "physical_exam": "",
         "assessment": "", "plan": ""}, "2026-10-09 09:00:00")
    check('"chief_complaint": "腹痛 3 天"' in full and '"history": ""' in full,
          "推演输入带完整 JSON，空段以空串出现",
          "空段要是直接省掉，模型只会以为这一茬没提；写成 \"\" 它才知道该走「待查」")
    check("现病史" in full and "字段对照" in full,
          "推演输入附字段中文对照", "JSON 用英文列名，附一行对照省得模型猜")
    check(full.find("腹痛 3 天") < full.find("原始语音：主诉肚子疼"),
          "推演输入 · 学生修订版排在原始语音前面",
          "模型对「前文优先」很敏感，顺序反了就会拿原始语音盖掉学生刚修正的地方")
    check("不要替他补上" in full and "问诊得来的客观事实" not in full,
          "空白的问诊内容旁明确写「不要替他补上」",
          "只标出哪段是空的还不够 —— 得同时说清空的原因是他没问到，不是让你脑补")
    check("这正是要你推演的部分" in full,
          "空白的诊断思考/诊疗计划旁说的是反话：请你推演",
          "两种空段必须分开说。混成一句（「缺的是他还没问到的内容，不要替他补上」）"
          "会让模型连推演也一起省掉 —— 实测里它真的会把这两段留空，"
          "于是整个按钮点下去什么也没发生")
    no_conclusion = main.build_regenerate_content(
        {"chief_complaint": "腹痛 3 天", "history": "1. 起病 3 天", "physical_exam": "上腹压痛",
         "assessment": "已写好的诊断思考", "plan": "已写好的诊疗计划",
         "raw_text": "原始语音"}, "2026-10-09 09:00:00")
    check("这正是要你推演的部分" not in no_conclusion,
          "诊断思考与诊疗计划都写了时，不提「请你推演」那句",
          "提示词里只该出现与当前卡片状态相符的话，no-op 的叮嘱只会稀释注意力")

    # --- 忠于修改：提示词是请求，代码里另有一层保证 ---
    fresh = {"chief_complaint": "腹痛 3 天", "physical_exam": "模型改写的体查",
             "assessment": "模型推演的诊断思考"}
    restore_from = {"chief_complaint": "腹痛 3 天",                 # 跟模型一样，没什么可还原
                    "physical_exam": "上腹压痛阳性（学生核对）",     # 被模型改掉了 → 还原
                    "assessment": "学生自己写的诊断思考草稿",        # 不在保护名单里 → 不动
                    }
    restored = main.restore_edited_fields(
        fresh, restore_from, ["chief_complaint", "physical_exam", "assessment", "todos"])
    check(restored == ["physical_exam"],
          "学生亲手写过的段落，模型改不掉",
          f"还原了 {restored}：主诉两边一模一样就不用动它；诊断思考**故意不还原** ——"
          "那正是按按钮要推演的两段；todos 不在保护名单里，压根没进循环")
    check(fresh["physical_exam"] == "上腹压痛阳性（学生核对）"
          and fresh["assessment"] == "模型推演的诊断思考",
          "「忠于修改」只保护他写过的那几段，不保护诊断思考与诊疗计划",
          "学生自己写了诊断思考还按推演，说明他想看老师的思路，覆盖掉才是他要的")
    annotated = {"chief_complaint": "腹痛 3 天"}
    check(main.restore_edited_fields(
              annotated, {"chief_complaint": "腹痛 3 天（患者自述，已核对）"},
              ["chief_complaint"]) == ["chief_complaint"]
          and annotated["chief_complaint"] == "腹痛 3 天（患者自述，已核对）",
          "学生比模型多写的批注不会被抹掉",
          "模型习惯把描述「精简」掉，可「（患者自述，已核对）」正是学生要留的痕迹")
    emptied = {"title": "模型给的标题"}
    check(main.restore_edited_fields(emptied, {"title": "   "}, ["title"]) == []
          and emptied["title"] == "模型给的标题",
          "学生把某段清空了，就不再拦着模型填",
          "清空是一种表态（「这段我不要了」），拿空串去覆盖模型结果反而更糟")
    check(set(main.PROTECTED_ON_REGENERATE) ==
          set(main.EDITABLE_FIELDS) - {"assessment", "plan"},
          "保护名单 = 可编辑字段 - 诊断思考 - 诊疗计划",
          f"{len(main.PROTECTED_ON_REGENERATE)} 个字段受保护")

    # 接上一个假的 DeepSeek：验证"手改过的版本会被带进提示词，并且改动被写回"
    seeing: dict = {}

    polished = json.dumps({
        "title": "问诊记录 · 腹痛（已推演）", "category": "问诊记录",
        "summary": "学生修订后重写的摘要", "todos": [{"task": "两周后复查胃镜", "due": None}],
        "chief_complaint": "腹痛 3 天",
        "history": "1. 三天前吃火锅后出现上腹隐痛\n2. 伴恶心呕吐",
        "physical_exam": "上腹压痛，反跳痛可疑",
        "assessment": "考虑急性胃炎可能；需与消化性溃疡鉴别：支持——餐后上腹痛；不支持——无反跳痛",
        "plan": "1. 血常规 + 粪隐血：评估有无消化道出血\n2. 急诊胃镜：明确黏膜病变\n3. 两周后复查",
    }, ensure_ascii=False)

    def _capture(**kwargs):
        seeing["system"] = kwargs["messages"][0]["content"]
        seeing["content"] = kwargs["messages"][-1]["content"]
        seeing["max_tokens"] = kwargs.get("max_tokens")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=polished))]
        )

    pipeline.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_capture)))
    student_exam = "上腹压痛阳性，无反跳痛（学生核对过）"
    regenerated = client.post(
        f"/api/memos/re-summarize/{soap_id}",
        json={"fields": {"physical_exam": student_exam,
                         "assessment": "学生在页面上手改的诊断思考"},
              "touched": ["physical_exam", "assessment"]})
    body = regenerated.get_json() or {}
    check(regenerated.status_code == 200 and body.get("ok"),
          "POST /api/memos/re-summarize 调模型并写回",
          f"返回 {body.get('error') or body.get('memo', {}).get('title')!r}")
    check(seeing.get("system") == main.CLINICAL_PROMPT,
          "推演走的是临床带教提示词",
          "不是首次整理那份书记员提示词 —— 后者只许如实记录，推不出鉴别诊断")
    check(seeing.get("max_tokens") == main.CLINICAL_MAX_TOKENS,
          "推演这次调用放宽了输出上限",
          f"max_tokens={seeing.get('max_tokens')}")
    check("学生在页面上手改的诊断思考" in seeing.get("content", "")
          and "上腹压痛阳性，无反跳痛（学生核对过）" in seeing.get("content", ""),
          "推演带上页面上未保存的手改内容",
          "推演要基于「学生改过的那一版」，而不是库里的旧版 —— 这里不写库地把改动合进内存副本，"
          "模型调用失败时它仍显示为未保存，学生不会误以为已经存进去了")
    saved = main.fetch_memo(conn, soap_id)
    check(saved["physical_exam"] == student_exam,
          "体查是学生写的 —— 模型改不掉",
          f"模型想写「上腹压痛，反跳痛可疑」，被还原成 {saved['physical_exam']!r}")
    check(body.get("restored") == ["physical_exam"],
          "接口如实汇报「哪几段是原样保留的」",
          f"restored={body.get('restored')}；诊断思考不在里面 —— 那两段本来就是推演的产物")
    check(saved["assessment"].startswith("考虑急性胃炎可能")
          and saved["history"].startswith("1. 三天前吃火锅")
          and saved["plan"].startswith("1. 血常规"),
          "推演结果覆盖到诊断思考与诊疗计划上",
          f"诊断思考={saved['assessment'][:16]!r}… · 诊疗计划={saved['plan'][:12]!r}…")
    check(saved["raw_text"] == soap_memo["raw_text"]
          and saved["record_time"] == soap_memo["record_time"],
          "POST /api/memos/re-summarize 不动原始识别原文与录制时间",
          "推演改的是「推出什么」，不该改「什么时候说的、原话是什么」")
    check(body.get("markdown_synced") is True
          and isinstance(body.get("elapsed_ms"), int),
          "推演接口汇报文档同步状态与耗时",
          f"markdown_synced={body.get('markdown_synced')} · {body.get('elapsed_ms')}ms")
    check(markdown_path_contains(Path(cfg.markdown_path), "急诊胃镜"),
          "推演后 Markdown 自动重刷",
          f"{Path(cfg.markdown_path).name} 里已经有推演后的诊疗计划")

    # 旧页面缓存里没有 touched 这一项 —— 那时宁可"全都当改过"，
    # 少润色几段只是效果打折，拿模型盖掉学生核对过的描述却是真丢东西。
    stale = client.post(f"/api/memos/re-summarize/{soap_id}",
                        json={"fields": {"physical_exam": student_exam}})   # 注意：没有 touched
    stale_body = stale.get_json() or {}
    check(student_exam in (stale_body.get("memo", {}).get("physical_exam") or ""),
          "请求里没有 touched 时保守处理：当作「全都改过」",
          f"体查仍是 {stale_body.get('memo', {}).get('physical_exam')!r}；"
          "旧页面报上来的 fields 分不出「改过」和「只是有值」，那就一律保护")

    def _explode(**kwargs):
        raise RuntimeError("模拟网关 500：connection reset by peer")

    before_failure = main.fetch_memo(conn, soap_id)
    pipeline.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_explode)))
    failing = client.post(f"/api/memos/re-summarize/{soap_id}",
                          json={"fields": {"plan": "这条手改不该被写进去"}})
    check(failing.status_code == 502,
          "POST /api/memos/re-summarize 调用失败时返回 502 而不是 500",
          f"HTTP {failing.status_code}：{(failing.get_json() or {}).get('error')!r}")
    after_failure = main.fetch_memo(conn, soap_id)
    check(after_failure["plan"] == before_failure["plan"],
          "POST /api/memos/re-summarize 失败时页面上的改动不被写库",
          "模型没返回就不落库：手改内容仍显示为「未保存」，学生不会误以为已经存进去了")
    pipeline.client = None

    # --- 10.11 多选卡片合并 ---
    # 先备两块"碎片"：时间上前后分开，用来验"必须按时间先后拼接"。
    early_id = pipeline.handle_text("主诉腹痛三天。三天前吃火锅后上腹隐痛。",
                                    datetime.now() - timedelta(days=2))
    late_id = pipeline.handle_text("追问补充：伴恶心呕吐，无黑便。上腹压痛阳性，无反跳痛。",
                                   datetime.now() - timedelta(days=1))
    early = main.fetch_memo(conn, early_id)
    late = main.fetch_memo(conn, late_id)
    check(early["record_time"] < late["record_time"],
          "合并测试 · 两块碎片的时间分得开",
          f"#{early_id} {early['record_time']} vs #{late_id} {late['record_time']}")

    for label, needle in (
        ("每张卡片有勾选框", 'class="memo-select"'),
        ("勾选框是 checkbox 类型", 'type="checkbox"'),
        ("工具栏有「合并选中项」按钮", "🔗 合并选中项并重新整理"),
        ("合并按钮默认就是灰的", 'id="merge" disabled'),
        ("页面上有「已选 N 条」计数", 'id="picked"'),
        ("合并期间显示加载动画", "正在调用 DeepSeek 融会贯通合并病历"),
        ("前端把合并请求发给新接口", "/api/memos/merge"),
        ("成功提示按需求原文", "多次记录已成功合并为一张新病例卡片"),
        ("勾选数决定按钮可用性", "applySelection"),
        ("勾选状态挂在页面级、不随重绘丢失", "selected.has(key)"),
        ("新卡片按服务端同款排序插入", "insertCard"),
        ("合并结果不重复弹提示（认 id 不认时间窗）", "justMerged"),
        ("二次确认里说明原卡片会保留", "都会保留"),
    ):
        check(needle in card_html, f"多选合并 · {label}", f"页面里能找到 {needle!r}")
    check("/api/memos/merge" in rules, "路由已注册 · /api/memos/merge",
          "合并走独立接口，不是挂在某张卡片的路径下 —— 它没有「那一张」卡片")

    # 参数校验：宁可在门口拦下来，也别让模型去熔一份缺了一段或者只有一段的病历
    for label, payload, why in (
        ("缺 memo_ids", {}, "没有要合并的对象"),
        ("memo_ids 不是数组", {"memo_ids": 12}, "只认数组，字符串/数字一律挡掉"),
        ("空数组", {"memo_ids": []}, "空数组什么都不选，等于没点这个按钮"),
        ("只选了一条", {"memo_ids": [early_id]},
         "一条的话「合并」就等于「重新润色」，该点的其实是那张卡上的 🔄"),
        ("同一个 id 报两遍", {"memo_ids": [early_id, early_id]},
         "去重后只剩一条 —— 重复 id 会让模型以为有两段病程，凭空多出一个时间节点"),
        ("id 不是数字", {"memo_ids": [early_id, "abc"]}, "得说得清是哪个 id 不合法"),
        ("一次选太多", {"memo_ids": list(range(1, webui.MERGE_MAX_SOURCES + 2))},
         "挡的是「全选 50 条」这种手滑：上下文撑爆、账单也涨"),
    ):
        resp = client.post("/api/memos/merge", json=payload)
        check(resp.status_code == 400, f"POST /api/memos/merge 拒绝 · {label}",
              f"HTTP {resp.status_code}：{(resp.get_json() or {}).get('error')!r} —— {why}")

    resp = client.post("/api/memos/merge", json={"memo_ids": [early_id, 999999]})
    check(resp.status_code == 404 and (resp.get_json() or {}).get("missing") == [999999],
          "POST /api/memos/merge 有一条不在就整个 404，不「能查到几条算几条」",
          f"HTTP {resp.status_code}：{(resp.get_json() or {}).get('error')!r} —— "
          "少熔一段的病历看着一样完整，学生根本发现不了")
    check(client.post("/api/memos/merge",
                      json={"memo_ids": [early_id, late_id]}).status_code == 503,
          "POST /api/memos/merge 无 LLM 时返回 503", "合并必须有模型，本地规则熔不出来")

    # --- 合并提示词 ---
    for label, needle in (
        ("角色是资深临床带教导师", "资深临床带教导师"),
        ("要求去粗取精", "去粗取精"),
        ("要求融会贯通", "融会贯通"),
        ("产出标准 SOAP 病历", "标准 SOAP 临床病历"),
        ("写明含现病史 1.2.3.4", "1.2.3.4"),
        ("不许凭空创造体征", "不许凭空创造临床事实"),
        ("诊断只许写「待排/待查」口径", "待排"),
        ("前后冲突以更晚那段为准", "更晚"),
        ("声明仅用于教学演练", "教学演练"),
    ):
        check(needle in main.MERGE_PROMPT, f"合并提示词 · {label}",
              f"提示词里能找到 {needle!r}")
    check("忠于学生的修改" not in main.MERGE_PROMPT
          and "忠于学生的修改" in main.CLINICAL_PROMPT,
          "合并与润色是两份角色不同的提示词",
          "润色的纪律是「别动学生写下的字」（见 restore_edited_fields 那层代码兜底）；"
          "合并的纪律恰恰是「该动就动」—— 不去重合成就没有合并的意义，"
          "所以这一条**绝不能**出现在合并提示词里")
    for label, spec in (("诊断思考三层要求", main._DIAGNOSIS_SPEC),
                        ("诊疗计划三层要求", main._PLAN_SPEC),
                        ("9 字段输出格式", main._OUTPUT_SPEC)):
        check(spec in main.CLINICAL_PROMPT and spec in main.MERGE_PROMPT,
              f"两份提示词逐字共用同一段{label}",
              "抽成常量复用而不是各抄一份：这三段是医学安全的护栏，"
              "抄成两份迟早改一漏一，而漏掉的那一份，模型就照着漏掉的那版做")
    check(main.MERGE_MAX_TOKENS > main.CLINICAL_MAX_TOKENS,
          "合并的输出上限比推演还宽",
          f"merge={main.MERGE_MAX_TOKENS} vs clinical={main.CLINICAL_MAX_TOKENS}："
          "推演只重写两段，合并要把五段全写一遍；上限给窄了不是「少写点」，"
          "而是 JSON 从中间断掉、整个合并白做")

    # --- 按时间先后拼接 ---
    check(main.memo_sort_key({"record_time": "2026-10-01 08:00:00", "id": 9})
          < main.memo_sort_key({"record_time": "2026-10-01 09:00:00", "id": 2}),
          "排序口径 · 先比录制时间",
          "时间线是主诉里「3 天前」这类表述的锚点，顺序错了模型会把后发生的写成诱因")
    check(main.memo_sort_key({"record_time": "2026-10-01 08:00:00", "id": 2})
          < main.memo_sort_key({"record_time": "2026-10-01 08:00:00", "id": 9}),
          "排序口径 · 同一秒再按 id 破平", "同一秒录的两段按入库先后排才稳定")

    merge_input = main.build_merge_content([late, early])       # 故意倒着传
    check(merge_input.find(early["record_time"]) < merge_input.find(late["record_time"]),
          "合并输入按时间先后排列（传进来是乱的也一样）",
          "函数自己排序，不指望调用方记得先排好")
    check(f"【第 1 段】" in merge_input and f"（记录 #{early_id}）" in merge_input
          and f"（记录 #{late_id}）" in merge_input,
          "合并输入逐段标出录制时间与 id",
          "模型要能引用「哪一段」的信息，才谈得上「融会贯通」")
    check((early["raw_text"] or "")[:12] in merge_input
          and (late["raw_text"] or "")[:12] in merge_input,
          "合并输入带上了每段的原始语音",
          "SOAP 是学生整理过的结构，原始语音才是他没加工过的原话 —— 两样都给，"
          "模型才既有骨架又有细节可以熔")

    # 空段提示沿用「两种空段分开说」的教训（见 build_regenerate_content）
    thin = main.build_merge_content([
        {"id": 1, "record_time": "2026-10-01 08:00:00", "raw_text": "",
         "chief_complaint": "腹痛 3 天", "history": "", "physical_exam": "",
         "assessment": "", "plan": ""}])
    check("不要替他补上" in thin and "这正是要你推演的部分" in thin,
          "合并输入把「没问到的」与「要你推演的」分开说",
          "混成一句会让模型连推演一起省掉 —— 实测里它真的会把诊断思考与诊疗计划留空")
    dense = main.build_merge_content([
        {"id": 1, "record_time": "2026-10-01 08:00:00", "raw_text": "",
         "chief_complaint": "腹痛 3 天", "history": "1. 起病 3 天", "physical_exam": "上腹压痛",
         "assessment": "已写好的诊断思考", "plan": "已写好的诊疗计划"}])
    check("不要替他补上" not in dense and "这正是要你推演的部分" not in dense,
          "五段都写了时，那两句叮嘱都不出现",
          "提示词里只该出现与当前材料相符的话，no-op 的叮嘱只会稀释注意力")

    # --- 合一次真的：接上假 DeepSeek ---
    merge_seeing: dict = {}
    merged_payload = json.dumps({
        "title": "急性上腹痛待查", "category": "问诊记录",
        "summary": "两次记录合并：主诉腹痛 3 天，追问补充恶心呕吐与上腹压痛。",
        "todos": [{"task": "补做血常规 + 腹部超声", "due": None}],
        "chief_complaint": "腹痛 3 天",
        "history": "1. 3 天前吃火锅后出现上腹隐痛\n2. 伴恶心呕吐，无黑便",
        "physical_exam": "上腹压痛阳性，无反跳痛",
        "assessment": "考虑急性胃炎可能；需与消化性溃疡鉴别：支持——餐后上腹痛；"
                      "不支持——无反跳痛",
        "plan": "1. 血常规 + 粪隐血：评估有无消化道出血\n2. 急诊胃镜：明确黏膜病变\n3. 一周后复诊",
    }, ensure_ascii=False)

    def _capture_merge(**kwargs):
        merge_seeing["system"] = kwargs["messages"][0]["content"]
        merge_seeing["content"] = kwargs["messages"][-1]["content"]
        merge_seeing["max_tokens"] = kwargs.get("max_tokens")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=merged_payload))]
        )

    pipeline.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_capture_merge)))
    count_before = conn.execute("SELECT COUNT(*) FROM memos").fetchone()[0]
    saved_before = state.snapshot()["saved_count"]

    resp = client.post("/api/memos/merge", json={"memo_ids": [late_id, early_id]})
    body = resp.get_json() or {}
    merged_id = body.get("id")
    check(resp.status_code == 200 and body.get("ok") and merged_id,
          "POST /api/memos/merge 合并成功",
          f"新卡片 #{merged_id} · 用时 {body.get('elapsed_ms')}ms · "
          f"错误 {body.get('error')!r}")
    check(body.get("merged") == [early_id, late_id],
          "接口按**时间先后**回报合并了哪几段",
          f"merged={body.get('merged')}：传进去是倒着的，回来的必须是顺的")
    check(merge_seeing.get("system") == main.MERGE_PROMPT,
          "合并走的是合并提示词（不是润色那份）",
          "润色是改一张卡，合并是把几段熔成一份 —— 两份角色提示词")
    check(merge_seeing.get("max_tokens") == main.MERGE_MAX_TOKENS,
          "合并这次调用放宽了输出上限", f"max_tokens={merge_seeing.get('max_tokens')}")
    sent = merge_seeing.get("content", "")
    check(sent.find(early["record_time"]) < sent.find(late["record_time"]),
          "发给模型的材料是按时间先后排的",
          "倒着发的话，模型会把后发生的事写成诱因")

    merged_memo = main.fetch_memo(conn, merged_id)
    check(merged_memo is not None and main.has_soap(merged_memo),
          "合并结果真的落库了", f"#{merged_id} 主诉={merged_memo and merged_memo['chief_complaint']!r}")
    check(merged_memo["record_time"] == late["record_time"],
          "合并后的录制时间取**最晚**那段",
          f"{merged_memo['record_time']}：这一条代表的时间跨度是「最早那段 → 最晚那段」，"
          "落款取最晚一刻 —— 取最早的话，10:00/10:05/10:10 三段熔出来的卡片会写着 10:00 "
          "并插在自己碎片的中间，看着像排错了")
    check(merged_memo["plan"].startswith("1. 血常规")
          and merged_memo["assessment"].startswith("考虑急性胃炎可能"),
          "合并结果覆盖到诊断思考与诊疗计划上",
          f"诊断思考={merged_memo['assessment'][:12]!r}…")
    check(merged_memo["raw_text"] == "",
          "合并后的记录没有伪造的「原始语音」",
          "这次合并没有对应录音，编一句「由 #3、#5 合并而来」填进去只会污染"
          "「原始识别原文」这个字段；来路由源记录本身留着")
    check(conn.execute("SELECT COUNT(*) FROM memos").fetchone()[0] == count_before + 1,
          "合并只新增一条记录", "不是把几条改写成一条，是新起一条")
    check(main.fetch_memo(conn, early_id) is not None
          and main.fetch_memo(conn, late_id) is not None,
          "源记录**不删**",
          "合并不是「整理完就能扔」的操作 —— 学生可能还要回头对照自己当时记的原话；"
          "真要清理，卡片上本来就有 🗑️ 删除，由他决定")
    check(state.snapshot()["saved_count"] == saved_before + 1,
          "合并走 publish_memo：计数 +1 并广播",
          "新记录才 +1。若误用 publish_update，页面上那句「已记录 N 条」就少算了这张新卡")
    check(body.get("markdown_synced") is True
          and markdown_path_contains(Path(cfg.markdown_path), "急性上腹痛待查"),
          "合并后 Markdown 自动重刷",
          f"{Path(cfg.markdown_path).name} 里已经有合并出来的那张病历")

    # 排在最前：合并出来的卡片得压在自己那几段碎片**上面**
    order = [row["id"] for row in conn.execute(
        "SELECT id FROM memos ORDER BY record_time DESC, id DESC")]
    check(order.index(merged_id) < min(order.index(early_id), order.index(late_id)),
          "合并卡片排在自己那几段碎片上面（页面 ORDER BY 实测）",
          f"页面顺序 {order}：时间不比最晚那段早、id 又最大，所以它必然压在碎片上方，"
          "「汇总卡 + 底下几段原记录」自然成一组 —— 学生刚记完就合并时，这就是页面顶部")

    # 失败与"假成功"都不能留下垃圾
    def _explode_merge(**kwargs):
        raise RuntimeError("模拟网关 500：connection reset by peer")

    count_before = conn.execute("SELECT COUNT(*) FROM memos").fetchone()[0]
    pipeline.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_explode_merge)))
    failing = client.post("/api/memos/merge", json={"memo_ids": [early_id, late_id]})
    check(failing.status_code == 502
          and conn.execute("SELECT COUNT(*) FROM memos").fetchone()[0] == count_before,
          "POST /api/memos/merge 调用失败时返回 502 且不落库",
          f"HTTP {failing.status_code}：{(failing.get_json() or {}).get('error')!r}")

    empty_payload = json.dumps({"title": "空", "category": "问诊记录", "summary": "",
                                "todos": [], "chief_complaint": "", "history": "",
                                "physical_exam": "", "assessment": "", "plan": ""},
                               ensure_ascii=False)
    pipeline.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=empty_payload))]))))
    hollow = client.post("/api/memos/merge", json={"memo_ids": [early_id, late_id]})
    check(hollow.status_code == 502
          and conn.execute("SELECT COUNT(*) FROM memos").fetchone()[0] == count_before,
          "模型返回了合法 JSON 但五段全空时，也算失败、不落库",
          "这种「成功」的假象比报错更糟：页面上会多出一张空卡片，而学生以为合并好了")
    pipeline.client = None


def markdown_path_contains(path: Path, needle: str) -> bool:
    """文档重刷是异步收尾的一部分，给文件系统一点余地再读。"""
    for _ in range(5):
        try:
            if needle in path.read_text(encoding="utf-8"):
                return True
        except OSError:
            pass
        time.sleep(0.05)
    return False


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #


def print_tree() -> None:
    print("项目结构\n" + "-" * 72, flush=True)
    skip = {"__pycache__", ".git", ".venv", "venv"}
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if any(part in skip for part in path.parts):
            continue
        rel = path.relative_to(PROJECT_ROOT)
        depth = len(rel.parts) - 1
        icon = "📁" if path.is_dir() else "📄"
        size = ""
        if path.is_file() and path.suffix in (".py", ".txt", ".md", ".example") or path.name == ".env.example":
            size = f"  ({path.stat().st_size:,} B)"
        print(f"  {'    ' * depth}{icon} {rel.name}{size}", flush=True)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    started = time.monotonic()
    print("=" * 72)
    print(" 智能语音备忘录 · 自检报告 (selfcheck)")
    print(f" 项目目录：{PROJECT_ROOT}")
    print(f" Python  ：{sys.version.split()[0]}  ({sys.platform})")
    print(f" 运行时间：{datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 72, flush=True)

    check_files()
    check_syntax_and_imports()

    with tempfile.TemporaryDirectory(prefix="voicememo_check_") as tmp:
        conn = check_schema(Path(tmp))
        try:
            check_mock_flow(conn)
            check_status_and_export(Path(tmp), conn)
            check_perf_and_truncation(Path(tmp))
            check_commands_and_state_machine(Path(tmp))
            check_pinyin_matching()
            check_audio_and_backend()
            check_medical_mode(Path(tmp))
            check_web_interface(Path(tmp), conn)
        finally:
            if conn is not None:
                conn.close()

    passed = sum(1 for ok, _, _ in RESULTS if ok)
    failed = [(label, detail) for ok, label, detail in RESULTS if not ok]

    print("\n" + "=" * 72)
    print(f" 自检汇总：{passed}/{len(RESULTS)} 项通过 · 耗时 {time.monotonic() - started:.2f}s"
          + (f" · {len(SKIPPED)} 项跳过" if SKIPPED else ""))
    print("=" * 72, flush=True)
    for label, detail in SKIPPED:
        print(f"  跳过：{label}\n      {detail}", flush=True)
    if failed:
        print("\n未通过项：", flush=True)
        for label, detail in failed:
            print(f"  ✘ {label}\n      {detail}", flush=True)
        print("\n结论：自检未全部通过，请按上述提示修复。\n", flush=True)
        return 1
    print("\n结论：全部通过。可以执行 `python main.py --doctor` 检查真实设备与 API Key，"
          "随后 `python main.py` 开始语音监听（说「小医」或「小医小医」唤醒），"
          "或 `python main.py --web` 同时打开网页看板。\n", flush=True)
    print_tree()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
