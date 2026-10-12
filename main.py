#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能语音备忘录 · Voice Memo
============================

免唤醒词 · 全本地断句 · 本地语音识别 · DeepSeek 结构化整理 · SQLite 落库

流水线::

    麦克风 ──► silero-vad 实时断句 ──► 录音缓冲 ──► faster-whisper 转写 ──┐
             (静音 2.5s / 结束词截断)                                    │
                                                                         ▼
    memos.db (SQLite) ◄──── 结构化写入 ◄──── DeepSeek 整理 ◄──── 原始文本 + 录音时间

用法::

    python main.py                              # 常驻监听，对着麦克风说话即可
    python main.py --text "明天下午三点提醒我开会"    # 跳过麦克风，直接验证文字链路
    python main.py --list                       # 查看最近 10 条备忘录
    python main.py --doctor                     # 环境自检
    python main.py --init-db                    # 仅初始化数据库

设计说明
--------
* ``torch`` / ``silero-vad`` / ``faster-whisper`` / ``sounddevice`` / ``openai``
  全部**延迟导入**：本模块顶层只依赖标准库，因此即使一个第三方包都没装，
  也能 ``import main`` 并跑通 ``selfcheck.py`` 的离线自检。
* 大模型不可用（未配置 KEY / 网络异常 / 返回不可解析）时，自动降级到内置的
  规则整理 ``local_organize``，保证语音内容不会因为接口问题而丢失。
* 录音块带有采集时的时间戳，即使 ASR/LLM 处理期间队列积压，
  ``record_time`` 记录的仍是"真正开口说话"的时刻。
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import queue
import re
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

__version__ = "1.0.0"

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB_FILENAME = "memos.db"
DEFAULT_MARKDOWN_FILENAME = "my_memos.md"
DEFAULT_EXPORT_MD = "memos.md"
DEFAULT_EXPORT_DOCX = "memos.docx"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_LLM_MODEL = "deepseek-chat"
DEFAULT_WEB_PORT = 5000

# 语音控制词：出现即视为"说完了"的截断信号，且不应进入任何输出字段
DEFAULT_STOP_KEYWORDS = ("结束", "完毕", "就这些", "说完了")

# ---- 唤醒词与语音指令 ----
# 写成叠词而不是单说"小医"：``wake_word_forms`` 会自动把叠词**折半**成候选，
# 于是"小医"和"小医小医"两种说法都能唤醒，反过来配就少一种。
DEFAULT_WAKE_WORD = "小医小医"
DEFAULT_SEGMENT_END_KEYWORDS = ("录音完毕", "下一条", "保存这段", "这段保存")
DEFAULT_EXIT_KEYWORDS = ("退出录制", "停止运行", "彻底结束")
# 作废词：说出口就**丢弃**这一段，不转写、不调用 DeepSeek、不入库、不写 Markdown
DEFAULT_CANCEL_KEYWORDS = ("作废", "取消", "这条不要了", "别记录了", "不要了")

# 短指令词：**只认句尾**的截断信号。
# "保存""好了"这类词在正常内容里出现的频率太高（"文件我已经保存了"），
# 认句中就等于把用户的正文拦腰截断；只在句尾生效才是安全的。
# 这里**故意不收"结束"**：句尾匹配会自动吃掉后面的语气字，于是"会议结束了"
# 会被判成指令、整条备忘被清空 —— 而"结束"恰恰是最常见的正文用词。
# 确实想说"结束"收尾，把它加进 SEGMENT_END_KEYWORDS 或 STOP_KEYWORDS 即可。
DEFAULT_SHORT_END_KEYWORDS = ("完毕", "保存", "好了", "收工", "就这样", "就这些", "说完了")

#: 医学问诊场景的热词表。两个用途，缺一不可：
#:
#: 1. **解码期提示词**（``initial_prompt`` / FunASR 的 ``hotword``）——把字面直接喂给
#:    模型，让它在"心前区"和"心钱去"之间优先选前者。这是真正意义上的"纠正"。
#: 2. **识别后校正**（``correct_medical_terms``）——解码期引导不是万能的（SenseVoice
#:    压根不吃 hotword 参数），所以再过一层拼音近似匹配，把"湿罗音"掰回"湿啰音"。
#:
#: 词表的取舍：**只要 ASR 容易听岔、且写错了会改变医学含义的**。像"病人""医生"这种
#: 日常也常说、不太会错的词就不收 —— 提示词每长一个字都要挤占解码上下文。
#:
#: 两个用途**共用一个词表但吃的不是同一段**：解码提示词只装得下前 46 个左右，
#: 后面的词靠识别后校正兜底。所以下面的排序不是分类，是优先级（见词表内的注释）。
DEFAULT_MEDICAL_HOTWORDS = (
    # ⚠️ **顺序就是优先级**：``build_asr_prompt`` 按词表顺序往提示词里塞，塞满
    # ``ASR_PROMPT_MAX_CHARS`` 就停 —— 排在前面的当解码提示词，排在后面的只参与
    # "识别后校正"。所以词表按"写错了会不会改变医学含义"从重到轻排，而不是按科室
    # 分类排：按分类排的初版整组整组地被截掉，症状与检查那两组一个都没进提示词。
    #
    # 几个硬约束：**反义词对必须成对留下** —— 右上腹/右下腹、干啰音/湿啰音、
    # 呕血/咯血。只留一个的话，模型失去区分能力，宁可两边都不喂。
    #
    # 第一梯队：问诊栏目 + 最高频的体征、部位与诊断（这一梯队要能独立塞满提示词）
    "主诉", "现病史", "体格检查", "心前区", "听诊", "叩痛", "反跳痛", "湿啰音",
    "黑便", "血常规", "急诊胃镜", "胃溃疡", "心肌梗死", "血压", "脉搏",
    # 第二梯队：反义词对与最容易被听岔的体征
    "干啰音", "右上腹", "右下腹", "压痛", "心电图", "柏油样便",
    # 第三梯队：常见症状
    "呕血", "咯血", "黄疸", "水肿", "腹泻", "胸闷", "心悸", "气促", "盗汗",
    # 第四梯队：病史栏目、查体手法与阳性体征
    "既往史", "家族史", "生命体征", "心尖搏动", "哮鸣音", "腹肌紧张",
    "视诊", "触诊", "叩诊", "移动性浊音", "肝脾肿大", "发绀",
    # 第五梯队：诊断名词
    "高血压", "糖尿病", "肺炎", "幽门螺杆菌", "十二指肠溃疡",
    # ---- 以下超出提示词预算，只参与"识别后校正"（``correct_medical_terms``）----
    "个人史", "婚育史", "月经史", "辅助检查", "初步诊断", "鉴别诊断", "诊疗计划",
    "心率", "呼吸", "体温", "血氧饱和度", "胸骨后", "剑突下", "脐周", "麦氏点",
    "肾区", "颈静脉", "腹股沟", "锁骨上", "叩击痛", "病理反射", "浅表淋巴结",
    "恶心呕吐", "消瘦", "尿常规", "便常规", "肝肾功能", "心肌酶", "C反应蛋白",
    "腹部超声", "胸部CT", "心绞痛",
)

#: SOAP 五段在库里的列名与中文标题。**顺序就是问诊记录的顺序**，
#: 卡片、Markdown、DOCX 三处渲染都从这里取，改一处即可全部生效。
SOAP_SECTIONS = (
    ("chief_complaint", "主诉"),
    ("history", "现病史"),
    ("physical_exam", "体格检查"),
    ("assessment", "诊断思考"),
    ("plan", "诊疗计划"),
)

#: SOAP 五段的**数据库列名**。下面几个白名单常量都要用它，所以定义在文件最前面
#: —— 模块级赋值是按书写顺序执行的，放到建表语句旁边会 NameError。
SOAP_COLUMNS = tuple(key for key, _ in SOAP_SECTIONS)

#: 前端 ``contenteditable`` 允许回写的全部字段（Web 层据此做白名单校验）。
EDITABLE_FIELDS = ("title", "summary") + tuple(key for key, _ in SOAP_SECTIONS)

#: 单个可编辑字段的长度上限。contenteditable 是个自由文本框，
#: 不设上限的话一次误粘贴就能把几 MB 文本塞进数据库和每一张卡片。
EDIT_FIELD_MAX_CHARS = 4000

#: 「重新润色」允许覆盖的字段。比 ``EDITABLE_FIELDS`` 多出 category 与 todos：
#: 这两个不给手工改（待办是数组、分类是枚举，做成 contenteditable 一改就坏），
#: 但模型重新生成时理应一并刷新 —— 两个白名单的差别就是"谁能写"的区别。
REGENERATE_FIELDS = ("title", "category", "summary", "todos") + SOAP_COLUMNS

# 监听状态机的三个状态（Web 界面直接显示这三个）
STATE_STANDBY = "standby"
STATE_RECORDING = "recording"
STATE_PROCESSING = "processing"
STATE_LABELS = {
    STATE_STANDBY: "⚪ 待机中（等待唤醒词）",
    STATE_RECORDING: "🔴 正在录音中...",
    STATE_PROCESSING: "⏳ 正在整理并生成卡片...",
}
STATE_STANDBY_NO_WAKE = "🎙️ 正在倾听中...（无需唤醒词）"

# ---- 状态提示外观（不喜欢 emoji 就把这里换成 "[听]""[录]""[等]" 之类的纯文本）----
ICON_LISTENING = "🎙️"
ICON_RECORDING = "🔴"
ICON_THINKING = "⏳"
ICON_SAVED = "💾"
ICON_IDLE = "⚪"
ICON_CANCELLED = "🚫"
RULE = "═" * 62

# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #


def _timestamp(when: Optional[datetime] = None) -> str:
    """统一的卡片/状态时间戳格式：``YYYY-MM-DD HH:MM:SS``。"""
    return (when or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")


def log(message: str, tag: str = "系统") -> None:
    """带时间戳的日志输出，立即刷新，避免管道重定向时看不到进度。"""
    print(f"[{datetime.now():%H:%M:%S}] [{tag}] {message}", flush=True)


def _setup_console() -> None:
    """Windows 控制台默认不是 UTF-8，中文/特殊字符容易抛 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def load_dotenv(path: os.PathLike | str = ".env", override: bool = False) -> int:
    """极简 .env 解析器，省掉一个 python-dotenv 依赖。

    支持 ``KEY=VALUE``、``#`` 注释、行内注释、单双引号包裹的值。
    已存在的环境变量默认不覆盖（真实环境变量优先级更高）。
    返回成功注入的变量条数。
    """
    file_path = Path(path)
    if not file_path.is_file():
        return 0
    injected = 0
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip()
        # 去掉未加引号的行内注释
        if value and value[0] not in "\"'" and " #" in value:
            value = value.split(" #", 1)[0].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            injected += 1
    return injected


def _env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name).lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "y", "on", "是", "开")


def beep(enabled: bool = True) -> None:
    """终端蜂鸣。

    Windows 控制台会响系统提示音；Windows Terminal / 部分终端会忽略 ``\\a``，
    此时以状态提示文字为准。可以用 ``--no-beep`` 或 ``BEEP=false`` 关掉。
    """
    if not enabled:
        return
    try:
        sys.stdout.write("\a")
        sys.stdout.flush()
    except Exception:
        pass


def announce_listening(cfg: "Config", first: bool = False) -> None:
    """就绪界线：模型已加载、随时可以开口（免唤醒模式）。"""
    hint = "请直接说话，无需唤醒词" if first else "可以继续说下一条，无需重启"
    silence = f"静音 {cfg.silence_ms / 1000:.1f} 秒自动保存"
    if cfg.early_stop_keyword:
        stops = "/".join(cfg.stop_keywords[:2])
        rule_line = f"      {silence} 或 说「{stops}」提前截断 · Ctrl+C 退出"
    else:
        rule_line = f"      {silence} · Ctrl+C 退出"
    print(f"\n{RULE}", flush=True)
    print(f" {ICON_LISTENING}  [正在倾听中...]  {hint}", flush=True)
    print(rule_line, flush=True)
    print(f"{RULE}", flush=True)


def announce_standby(cfg: "Config") -> None:
    """唤醒词模式的默认态：麦克风开着，但只在听到唤醒词后才开始记录。"""
    wake = cfg.wake_word or DEFAULT_WAKE_WORD
    ends = "/".join(cfg.segment_end_keywords[:3])
    exits = "/".join(cfg.exit_keywords[:2])
    print(f"\n{RULE}", flush=True)
    print(f" {ICON_IDLE}  [{STATE_LABELS[STATE_STANDBY]}]", flush=True)
    print(f"      说「{wake}」开始记录 · 说「{ends}」保存并回到待机", flush=True)
    print(f"      说「{exits}」保存后退出 · 静音 {cfg.silence_ms / 1000:.1f} 秒自动保存"
          f" · Ctrl+C 退出", flush=True)
    print(f"{RULE}", flush=True)


def announce_recording(cfg: "Config", wake_mode: bool = False) -> str:
    """检测到唤醒词/开口，开始录音。返回要广播的状态说明。"""
    beep(cfg.beep)
    silence = cfg.silence_ms / 1000.0
    if wake_mode:
        ends = "/".join(cfg.segment_end_keywords[:2])
        cancel = cfg.cancel_keywords[0] if cfg.cancel_keywords else ""
        tail = f"说「{ends}」立即保存"
        if cancel:
            tail += f" · 说「{cancel}」丢弃本条"
        tail += f" · 或静音 {silence:.1f} 秒自动保存"
    elif cfg.early_stop_keyword:
        tail = f"静音 {silence:.1f} 秒或说「结束」自动保存"
    else:
        tail = f"静音 {silence:.1f} 秒自动保存"
    print(f"\n {ICON_RECORDING} [正在录音...] ({tail})", flush=True)
    return f"静音 {silence:.1f} 秒自动保存"


def announce_thinking(label: str) -> None:
    """已截断，准备提交整理。"""
    print(f" {ICON_THINKING} [录音结束，{label}整理中...] "
          f"（本条处理期间暂停收音，请稍候）", flush=True)


def _clean_str(value: Any) -> Optional[str]:
    """把任意值规整成"干净的非空字符串或 None"，用于兜住大模型的不稳定输出。"""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip().strip("\"'“”").strip()
    if not value or value.lower() in ("null", "none", "n/a", "无", "未知", "-"):
        return None
    return value


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #


@dataclass
class Config:
    """运行期配置。默认值可用环境变量（.env）覆盖。"""

    # --- 音频采集 ---
    sample_rate: int = 16000          # silero-vad / whisper 均要求 16k
    chunk_size: int = 512             # silero-vad v5 在 16k 下的固定窗口长度
    queue_seconds: float = 8.0        # 采集队列缓冲时长，防止处理期间丢音
    pre_speech_pad_ms: int = 300      # 起始回溯缓冲，避免"第一个字"被吞掉

    # --- VAD 断句 ---
    silence_ms: int = 3000            # 静音 3s 兜底断句（纯 VAD，不吃 CPU）
    min_speech_ms: int = 400          # 短于此长度的片段视为噪声丢弃
    min_segment_ms: int = 800         # 唤醒后不足此长度又静音 → 视为"还没开始说"，继续等
    max_record_s: float = 120.0       # 单条录音上限，到点强制截断
    vad_threshold: float = 0.5

    # --- 唤醒词与语音指令 ---
    wake_word: str = DEFAULT_WAKE_WORD          # 唤醒词；留空或 wake_word_enabled=False 则免唤醒
    wake_word_enabled: bool = True              # True=待机等唤醒词；False=一直听（旧行为）
    wake_model: str = "tiny"                    # 扫描唤醒词的模型，越小越快
    wake_window_s: float = 2.2                  # 每次扫描回看多长的音频
    wake_scan_interval_s: float = 0.6           # 两次扫描之间至少间隔多久
    wake_fuzzy_threshold: float = 0.72          # 模糊匹配阈值（ASR 会把"小医"听成"小一")
    wake_aliases: tuple = ()                    # 额外接受的唤醒词写法（同音字表兜不住的）
    segment_end_keywords: tuple = DEFAULT_SEGMENT_END_KEYWORDS
    # 短指令词：和分段结束词同类，但**只认句尾**（"好了""保存"这类词太常见了）
    short_end_keywords: tuple = DEFAULT_SHORT_END_KEYWORDS
    exit_keywords: tuple = DEFAULT_EXIT_KEYWORDS
    # 作废词：录音途中说到就丢弃这一段（详见 match_command_detail 里的"只认句尾"规则）
    cancel_keywords: tuple = DEFAULT_CANCEL_KEYWORDS
    # 默认关闭"结束词"提前截断：它需要在录音途中反复跑 Whisper，是纯 CPU 环境下
    # 最大的性能负担（见 README「为什么默认关掉结束词」）。断句默认完全交给
    # silero-vad —— 只算人声概率，不碰 ASR，极快。
    early_stop_keyword: bool = False
    early_stop_interval_s: float = 1.5
    stop_keywords: tuple = DEFAULT_STOP_KEYWORDS

    # --- 本地 ASR ---
    # auto = 装了 funasr 就用 SenseVoiceSmall（中文专用、CPU 上极快且准），
    # 否则退回 faster-whisper。也可以强制指定 funasr / whisper。
    asr_backend: str = "auto"         # auto / funasr / whisper
    asr_funasr_model: str = "iic/SenseVoiceSmall"
    # faster-whisper 档位：base 是 CPU 默认（比 small 快 2~3 倍、体积小 3 倍）；
    # 机器很弱用 tiny，有 GPU 再上 small/medium/large-v3-turbo。
    asr_model: str = "base"           # tiny / base / small / medium
    asr_device: str = "auto"          # auto / cpu / cuda
    asr_compute_type: str = "int8"    # int8(cpu) / float16(cuda) / int8_float16
    asr_language: str = "zh"
    asr_beam_size: int = 5
    # 医学热词：既当解码期提示词，也当识别后校正的词表（详见 DEFAULT_MEDICAL_HOTWORDS）
    asr_hotwords: tuple = DEFAULT_MEDICAL_HOTWORDS
    asr_initial_prompt: bool = True   # 是否把热词当 initial_prompt 喂给解码器
    asr_hotword_correct: bool = True  # 是否在识别后做一次热词拼音校正
    # 送进 ASR 之前的音频预处理（去直流/去噪/响度归一）
    audio_preprocess: bool = True
    audio_denoise: bool = True
    audio_target_dbfs: float = -3.0   # 归一化目标响度
    audio_max_gain_db: float = 18.0   # 增益上限，避免把纯底噪也放大成"人声"

    # --- 大模型 ---
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    llm_model: str = DEFAULT_LLM_MODEL

    # --- 存储 ---
    db_path: str = DEFAULT_DB_FILENAME
    markdown_path: str = DEFAULT_MARKDOWN_FILENAME   # 每条备忘录自动追加到这里；空字符串=关闭
    beep: bool = True                                # 录音开始/保存完成时的蜂鸣提示

    # --- Web 界面 ---
    web_host: str = "127.0.0.1"                      # 只监听本机；要局域网访问就改成 0.0.0.0
    web_port: int = DEFAULT_WEB_PORT

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            sample_rate=_env_int("SAMPLE_RATE", 16000),
            chunk_size=_env_int("CHUNK_SIZE", 512),
            queue_seconds=_env_float("QUEUE_SECONDS", 8.0),
            pre_speech_pad_ms=_env_int("PRE_SPEECH_PAD_MS", 300),
            silence_ms=_env_int("VAD_SILENCE_MS", 3000),
            min_speech_ms=_env_int("VAD_MIN_SPEECH_MS", 400),
            min_segment_ms=_env_int("VAD_MIN_SEGMENT_MS", 800),
            max_record_s=_env_float("MAX_RECORD_SECONDS", 120.0),
            vad_threshold=_env_float("VAD_THRESHOLD", 0.5),
            wake_word=_env_str("WAKE_WORD", DEFAULT_WAKE_WORD),
            wake_word_enabled=_env_bool("WAKE_WORD_ENABLED", True),
            wake_model=_env_str("WAKE_MODEL", "tiny"),
            wake_window_s=_env_float("WAKE_WINDOW_SECONDS", 2.2),
            wake_scan_interval_s=_env_float("WAKE_SCAN_INTERVAL_SECONDS", 0.6),
            wake_fuzzy_threshold=_env_float("WAKE_FUZZY_THRESHOLD", 0.72),
            wake_aliases=_split_keywords(_env_str("WAKE_ALIASES", "")),
            segment_end_keywords=_split_keywords(
                _env_str("SEGMENT_END_KEYWORDS", ",".join(DEFAULT_SEGMENT_END_KEYWORDS))
            ) or DEFAULT_SEGMENT_END_KEYWORDS,
            # 想彻底关掉短指令词，把它设成 "off" / "-" / "none" / "无" 即可
            short_end_keywords=(
                () if _env_str("SHORT_END_KEYWORDS", "").lower() in ("off", "-", "none", "无")
                else _split_keywords(_env_str("SHORT_END_KEYWORDS",
                                              ",".join(DEFAULT_SHORT_END_KEYWORDS)))
            ),
            exit_keywords=_split_keywords(
                _env_str("EXIT_KEYWORDS", ",".join(DEFAULT_EXIT_KEYWORDS))
            ) or DEFAULT_EXIT_KEYWORDS,
            cancel_keywords=_split_keywords(
                _env_str("CANCEL_KEYWORDS", ",".join(DEFAULT_CANCEL_KEYWORDS))
            ) or DEFAULT_CANCEL_KEYWORDS,
            early_stop_keyword=_env_bool("EARLY_STOP_KEYWORD", False),
            early_stop_interval_s=_env_float("EARLY_STOP_INTERVAL_SECONDS", 1.5),
            stop_keywords=_split_keywords(
                _env_str("STOP_KEYWORDS", ",".join(DEFAULT_STOP_KEYWORDS))
            ) or DEFAULT_STOP_KEYWORDS,
            asr_backend=_env_str("ASR_BACKEND", "auto"),
            asr_funasr_model=_env_str("ASR_FUNASR_MODEL", "iic/SenseVoiceSmall"),
            asr_model=_env_str("ASR_MODEL", "base"),
            asr_device=_env_str("ASR_DEVICE", "auto"),
            asr_compute_type=_env_str("ASR_COMPUTE_TYPE", "int8"),
            asr_language=_env_str("ASR_LANGUAGE", "zh"),
            asr_beam_size=_env_int("ASR_BEAM_SIZE", 5),
            # ASR_HOTWORDS 留空 = 用内置医学词表；填 off/-/none/无 = 整组关掉
            asr_hotwords=(
                () if _env_str("ASR_HOTWORDS", "").lower() in ("off", "-", "none", "无")
                else (_split_keywords(_env_str("ASR_HOTWORDS", ""))
                      or DEFAULT_MEDICAL_HOTWORDS)
            ),
            asr_initial_prompt=_env_bool("ASR_INITIAL_PROMPT", True),
            asr_hotword_correct=_env_bool("ASR_HOTWORD_CORRECT", True),
            audio_preprocess=_env_bool("AUDIO_PREPROCESS", True),
            audio_denoise=_env_bool("AUDIO_DENOISE", True),
            audio_target_dbfs=_env_float("AUDIO_TARGET_DBFS", -3.0),
            audio_max_gain_db=_env_float("AUDIO_MAX_GAIN_DB", 18.0),
            api_key=_env_str("DEEPSEEK_API_KEY"),
            base_url=_env_str("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
            llm_model=_env_str("DEEPSEEK_MODEL", DEFAULT_LLM_MODEL),
            db_path=_env_str("DB_PATH", DEFAULT_DB_FILENAME),
            markdown_path=_env_str("MARKDOWN_PATH", DEFAULT_MARKDOWN_FILENAME),
            beep=_env_bool("BEEP", True),
            web_host=_env_str("WEB_HOST", "127.0.0.1"),
            web_port=_env_int("WEB_PORT", DEFAULT_WEB_PORT),
        )


# --------------------------------------------------------------------------- #
# SQLite 存储
# --------------------------------------------------------------------------- #

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memos (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    record_time TEXT    NOT NULL,                          -- 录音起始时间 YYYY-MM-DD HH:MM:SS
    raw_text    TEXT    NOT NULL,                          -- ASR 原始文本（已去除控制词）
    title       TEXT,                                      -- 标题（≤20 字）
    category    TEXT,                                      -- 分类
    summary     TEXT,                                      -- 摘要
    todos       TEXT,                                      -- 待办 JSON 数组
    chief_complaint TEXT,                                  -- SOAP · 主诉
    history         TEXT,                                  -- SOAP · 现病史
    physical_exam   TEXT,                                  -- SOAP · 体格检查
    assessment      TEXT,                                  -- SOAP · 诊断思考
    plan            TEXT,                                  -- SOAP · 诊疗计划
    created_at  TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE INDEX IF NOT EXISTS idx_memos_record_time ON memos (record_time DESC);
CREATE INDEX IF NOT EXISTS idx_memos_category    ON memos (category);
"""

#: 业务要求的必备字段，selfcheck.py 会据此断言建表结果
REQUIRED_COLUMNS = ("record_time", "raw_text", "title", "category", "summary", "todos",
                    "chief_complaint", "history", "physical_exam", "assessment", "plan")

#: 老库（SOAP 之前建的）缺 ``SOAP_COLUMNS`` 这几列，靠 ``_migrate_schema`` 补上 ——
#: 用 ``ALTER TABLE ADD COLUMN`` 而不是重建表：SQLite 加列是 O(1) 的元数据操作，
#: 既不动现有数据，也不会因为一次升级就把用户攒的历史记录写坏。

#: 写库 + 写文档的串行化锁。
#:
#: 语音线程在"落库 + 追加 Markdown"，Web 线程可能在"删库 + 整份重刷 Markdown"，
#: 两条路径都必须当成**一整段**原子操作：否则删除的重写可能刚好错过一条正在插入的
#: 记录，文档与数据库就对不上了。RLock 是因为 ``remove_memo`` 持锁期间还会调用
#: ``delete_memo`` / ``export_markdown``。
_STORE_LOCK = threading.RLock()


def init_db(db_path: os.PathLike | str = DEFAULT_DB_FILENAME) -> sqlite3.Connection:
    """创建（或复用）数据库并保证表结构存在，返回连接对象。

    ``check_same_thread=False`` 是刻意为之：写入由单一的监听循环串行化，
    放开线程检查是为了让 ``MemoPipeline`` 也能在后台线程里安全使用，
    否则跨线程调用会抛 "SQLite objects created in a thread ..."。
    """
    path = Path(db_path)
    if path.parent and str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    _migrate_schema(conn)
    conn.commit()
    return conn


def _migrate_schema(conn: sqlite3.Connection) -> list:
    """把缺列的老库补齐，返回这次实际新增的列名（新库返回空列表）。"""
    present = {row["name"] for row in conn.execute("PRAGMA table_info(memos)")}
    added = []
    for column in SOAP_COLUMNS:
        if column not in present:
            conn.execute(f"ALTER TABLE memos ADD COLUMN {column} TEXT")
            added.append(column)
    return added


def insert_memo(conn: sqlite3.Connection, memo: dict) -> int:
    """写入一条备忘录，``todos`` 以 JSON 字符串落库。返回自增主键。"""
    columns = ["record_time", "raw_text", "title", "category", "summary", "todos"]
    columns += list(SOAP_COLUMNS)
    values = [
        memo["record_time"],
        memo["raw_text"],
        memo.get("title"),
        memo.get("category"),
        memo.get("summary"),
        json.dumps(memo.get("todos") or [], ensure_ascii=False),
    ]
    values += [memo.get(key) or "" for key in SOAP_COLUMNS]
    placeholders = ", ".join("?" * len(columns))
    with _STORE_LOCK:
        cursor = conn.execute(
            f"INSERT INTO memos ({', '.join(columns)}) VALUES ({placeholders})", values
        )
        conn.commit()
        return int(cursor.lastrowid)


def update_memo(conn: sqlite3.Connection, memo_id: Any, fields: dict,
                allowed: tuple = EDITABLE_FIELDS) -> Optional[dict]:
    """按白名单局部更新一条记录，返回更新后的完整记录（id 不存在时返回 None）。

    网页上 ``contenteditable`` 改完回写（``allowed=EDITABLE_FIELDS``）和
    「重新润色」写回模型结果（``allowed=REGENERATE_FIELDS``）都走这里。

    字段名**必须**来自白名单：它既是"哪些字段允许写"的声明，也是拼进 UPDATE
    语句的**唯一**列名来源 —— 拼接列名时，能挡住 SQL 注入的就只有这一层。
    """
    try:
        key = int(memo_id)
    except (TypeError, ValueError):
        return None
    if fetch_memo(conn, key) is None:
        return None

    updates = {name: value for name, value in (fields or {}).items() if name in allowed}
    if not updates:
        return fetch_memo(conn, key)

    # 逐字段限长：contenteditable 是个自由文本框，不设上限的话一次误粘贴
    # 就能把几 MB 的文本塞进数据库和每一张卡片里。
    assignments, values = [], []
    for name, value in updates.items():
        assignments.append(f"{name} = ?")
        if name == "todos":
            # 待办是列表，落库前一贯转成 JSON 字符串（与 insert_memo 保持一致）
            values.append(value if isinstance(value, str)
                          else json.dumps(value or [], ensure_ascii=False))
        else:
            text = "" if value is None else str(value)
            values.append(text[:EDIT_FIELD_MAX_CHARS])
    values.append(key)
    with _STORE_LOCK:
        conn.execute(f"UPDATE memos SET {', '.join(assignments)} WHERE id = ?", values)
        conn.commit()
    return fetch_memo(conn, key)


def _row_to_memo(row: sqlite3.Row) -> dict:
    """数据库行 → 备忘录字典（``todos`` 反序列化回列表）。"""
    item = dict(row)
    try:
        item["todos"] = json.loads(item.get("todos") or "[]")
    except (TypeError, ValueError):
        item["todos"] = []
    return item


def fetch_memos(conn: sqlite3.Connection, limit: int = 10) -> list[dict]:
    """按录音时间倒序读取最近若干条。"""
    rows = conn.execute(
        "SELECT * FROM memos ORDER BY record_time DESC, id DESC LIMIT ?", (int(limit),)
    ).fetchall()
    return [_row_to_memo(row) for row in rows]


def fetch_all_memos(conn: sqlite3.Connection) -> list[dict]:
    """按时间正序读取全部记录（导出用）。"""
    rows = conn.execute("SELECT * FROM memos ORDER BY record_time ASC, id ASC").fetchall()
    return [_row_to_memo(row) for row in rows]


def fetch_memo(conn: sqlite3.Connection, memo_id: Any) -> Optional[dict]:
    """按 id 取一条；不存在返回 None。"""
    try:
        key = int(memo_id)
    except (TypeError, ValueError):
        return None
    row = conn.execute("SELECT * FROM memos WHERE id = ?", (key,)).fetchone()
    return _row_to_memo(row) if row is not None else None


def delete_memo(conn: sqlite3.Connection, memo_id: Any) -> bool:
    """从数据库删掉一条。返回是否真的删掉了（id 不存在时是 False）。"""
    try:
        key = int(memo_id)
    except (TypeError, ValueError):
        return False
    with _STORE_LOCK:
        cursor = conn.execute("DELETE FROM memos WHERE id = ?", (key,))
        conn.commit()
        return cursor.rowcount > 0


def remove_memo(conn: sqlite3.Connection, memo_id: Any,
                markdown_path: os.PathLike | str = "") -> Optional[dict]:
    """删除一条记录，并让 Markdown 文档重新与数据库对齐。

    ``my_memos.md`` 平时是**追加**写入的（"边记边攒"，历史内容不受影响），所以直接
    从库里删掉一条并不会让文档里的对应段落消失。这里删完库就**整份重写**该文档
    —— ``export_markdown`` 本来就是"按数据库现状重刷"，天然完成同步。

    真的删掉了就返回那条被删的记录（供调用方显示标题），id 不存在则返回 None。
    """
    with _STORE_LOCK:          # 删 + 重刷文档之间不能被别的写入插进来
        memo = fetch_memo(conn, memo_id)
        if memo is None:
            return None
        delete_memo(conn, memo_id)
        if markdown_path:
            export_markdown(conn, markdown_path)
        return memo


# --------------------------------------------------------------------------- #
# 文档导出（Markdown 自动追加 / 历史导出 / DOCX）
# --------------------------------------------------------------------------- #

MARKDOWN_DOC_TITLE = "我的语音备忘录"


class ExportDependencyError(RuntimeError):
    """导出所需的可选依赖没有安装。"""


def _markdown_header(title: str = MARKDOWN_DOC_TITLE, subtitle: str = "") -> str:
    lines = [f"# {title}", "", "> 由 voice-memo 自动生成，音频与语音识别均在本机完成。"]
    if subtitle:
        lines.append(f"> {subtitle}")
    return "\n".join(lines) + "\n\n"


def format_markdown_entry(memo: dict, index: Optional[int] = None) -> str:
    """把一条备忘录渲染成 Markdown 片段。

    刻意不使用 ``<details>`` 之类 HTML：这样用纯文本编辑器打开也是可读的。
    """
    number = memo.get("id") if index is None else index
    title = memo.get("title") or "未命名备忘录"
    lines = [f"## {number}. {title}" if number else f"## {title}", ""]
    lines.append(f"- **时间**：{memo.get('record_time', '')}")
    lines.append(f"- **分类**：{memo.get('category') or '其他'}")
    if has_soap(memo):
        # 问诊记录：SOAP 五段逐段成节，空段整段省略（病历里"未问及"和"空白"
        # 是两回事，与其印一个空标题，不如让读者一眼看出这段确实没写）
        for key, label in SOAP_SECTIONS:
            body = (memo.get(key) or "").strip()
            if body:
                lines += [f"### {label}", "", body, ""]
    else:
        lines += ["", f"**摘要**：{memo.get('summary') or '（无）'}", ""]

    todos = memo.get("todos") or []
    if todos:
        lines += ["**待办**", ""]
        for todo in todos:
            due = f" —— 截止 {todo['due']}" if todo.get("due") else ""
            lines.append(f"- [ ] {todo['task']}{due}")
        lines.append("")

    raw = " ".join((memo.get("raw_text") or "").split())
    if raw:
        lines += [f"*原始识别原文*：{raw}", ""]
    lines += ["---", ""]
    return "\n".join(lines) + "\n"   # 结尾多留一个空行，与下一条之间才有分隔


def append_memo_markdown(memo: dict, path: os.PathLike | str = DEFAULT_MARKDOWN_FILENAME) -> bool:
    """把一条备忘录追加写入 Markdown 文档；文件不存在时先写标题头。

    这是"边记边攒"的路径，所以是**追加**而非重写，历史内容不受影响。
    """
    target = Path(path)
    if not str(target):
        return False
    try:
        with _STORE_LOCK:      # 与"删除后整份重刷"互斥，避免半截追加被覆盖 / 覆盖被追加
            if str(target.parent) not in ("", "."):
                target.parent.mkdir(parents=True, exist_ok=True)
            is_new = not target.exists() or target.stat().st_size == 0
            with target.open("a", encoding="utf-8", newline="\n") as handle:
                if is_new:
                    handle.write(_markdown_header())
                handle.write(format_markdown_entry(memo))
        return True
    except OSError as exc:
        log(f"写入 Markdown 失败（{target}）：{exc}", "导出")
        return False


def export_markdown(
    conn: sqlite3.Connection,
    path: os.PathLike | str = DEFAULT_EXPORT_MD,
    title: str = MARKDOWN_DOC_TITLE,
) -> int:
    """把数据库里的全部历史记录一次性导出为 Markdown（整体重写）。返回条数。

    删除一条记录后也用这个函数让文档与数据库重新对齐 —— 它本来就是"按现状重刷"。
    """
    with _STORE_LOCK:
        memos = fetch_all_memos(conn)
        target = Path(path)
        if str(target.parent) not in ("", "."):
            target.parent.mkdir(parents=True, exist_ok=True)
        subtitle = f"共 {len(memos)} 条记录 · 导出时间 {datetime.now():%Y-%m-%d %H:%M:%S}"
        body = "".join(format_markdown_entry(memo, index=index)
                       for index, memo in enumerate(memos, 1))
        target.write_text(_markdown_header(title, subtitle) + body, encoding="utf-8",
                          newline="\n")
        return len(memos)


def export_docx(conn: sqlite3.Connection, path: os.PathLike | str = DEFAULT_EXPORT_DOCX) -> int:
    """把全部历史记录导出为 Word 文档。需要 ``pip install python-docx``。

    缺少依赖时抛 ``ExportDependencyError``，由调用方给出安装提示。
    """
    try:
        from docx import Document
    except ImportError as exc:
        raise ExportDependencyError(
            "导出 DOCX 需要 python-docx，请先安装：pip install python-docx"
        ) from exc

    memos = fetch_all_memos(conn)
    document = Document()
    document.add_heading(MARKDOWN_DOC_TITLE, level=0)
    document.add_paragraph(
        f"由 voice-memo 自动生成 · 共 {len(memos)} 条记录 · "
        f"导出时间 {datetime.now():%Y-%m-%d %H:%M:%S}"
    )
    for index, memo in enumerate(memos, 1):
        document.add_heading(f"{index}. {memo.get('title') or '未命名备忘录'}", level=2)
        document.add_paragraph(f"时间：{memo.get('record_time', '')}")
        document.add_paragraph(f"分类：{memo.get('category') or '其他'}")
        if has_soap(memo):
            for key, label in SOAP_SECTIONS:
                body = (memo.get(key) or "").strip()
                if body:
                    document.add_paragraph(label, style="Heading 3")
                    document.add_paragraph(body)
        else:
            document.add_paragraph(f"摘要：{memo.get('summary') or '（无）'}")
        todos = memo.get("todos") or []
        if todos:
            document.add_paragraph("待办：")
            for todo in todos:
                due = f"（截止 {todo['due']}）" if todo.get("due") else ""
                document.add_paragraph(f"☐ {todo['task']}{due}", style="List Bullet")
        raw = " ".join((memo.get("raw_text") or "").split())
        if raw:
            document.add_paragraph(f"原始识别原文：{raw}")

    target = Path(path)
    if str(target.parent) not in ("", "."):
        target.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(target))
    return len(memos)


# --------------------------------------------------------------------------- #
# 大模型整理（DeepSeek / OpenAI 兼容协议）
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """你是一个医学生问诊实训的语音病历整理助手。用户会给你一段由语音识别(ASR)得到的问诊录音原始文本，以及这段录音的时间。请把它整理成结构化 JSON。

**首要任务：识别出这是一段问诊记录，并按 SOAP 格式归档。**

要求：
1. 修正明显的 ASR 错别字、重复词和口语冗余（"嗯""那个""就是说"等），但**绝不编造原文没有的信息**。
   特别注意医学专业词：主诉、现病史、体格检查、心前区、听诊、叩击痛、反跳痛、湿啰音、
   黑便、血常规、急诊胃镜、胃溃疡、心肌梗死、血压、脉搏等，按标准医学术语书写。
2. 过滤语音控制词，例如"结束""完毕""就这些""说完了"，它们不得出现在任何输出字段中。
3. 只输出一个 JSON 对象，不要输出解释、前后缀或 Markdown 代码块。

## 判断标准

- 如果这段内容是**问诊/病史采集**（医生问、患者答，或患者陈述症状、体征、检查、诊断、处理），
  就填写下面五个 SOAP 字段；**没提到的字段一律填空字符串 ""，不要脑补，也不要写"未提及"**。
- 如果这只是一段**日常事务性内容**（"明天下午三点开会""记得买牛奶"），
  五个 SOAP 字段全部留空 ""，把内容放进 summary 与 todos。
- 拿不准时按"不是问诊"处理 —— 把日常琐事硬塞进病历，比漏填一段病史更糟。

## JSON 字段说明

- title:    字符串，不超过 20 字，概括这段内容
- category: 字符串，从 [问诊记录, 工作, 生活, 学习, 购物, 健康, 财务, 社交, 灵感, 其他] 里选最贴切的一个
- summary:  字符串，1-3 句客观摘要
- todos:    数组，每项形如 {"task": "待办事项", "due": "YYYY-MM-DD HH:MM 或 null"}；没有待办时返回 []
  （时间用给定的录音时间推算，相对时间要换算成绝对时间；问诊记录里医生交代的复诊、检查安排算待办）
- chief_complaint: 字符串，**主诉**。一句话，固定句式"<症状>+<持续时间>"，例如"心前区疼痛 3 天"。
- history:         字符串，**现病史**。按时间顺序分条陈述，**每条独立一行、以"1. 2. 3."开头**，
                   写清起病诱因、症状演变、伴随症状、诊疗经过。
- physical_exam:   字符串，**体格检查**。生命体征（血压/脉搏/体温/呼吸）与各系统阳性体征，
                   分号分隔；只有正常描述时如实记录。
- assessment:      字符串，**诊断思考**。初步诊断与鉴别诊断，写清支持与不支持的理由。
- plan:            字符串，**诊疗计划**。进一步检查、治疗方案、健康宣教、随访安排。
"""

_JSON_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def extract_json_object(text: Optional[str]) -> dict:
    """从模型输出里稳健地抠出 JSON 对象。

    依次尝试：直接解析 → 剥离 ```json 代码块 → 截取首尾花括号 → 顺手补掉尾逗号。
    全部失败则抛 ``ValueError``（由调用方决定是否降级）。
    """
    if not text or not text.strip():
        raise ValueError("模型返回内容为空")

    candidates: list[str] = []
    stripped = text.strip()
    candidates.append(stripped)

    fenced = _JSON_FENCE_RE.search(stripped)
    if fenced:
        candidates.append(fenced.group(1).strip())

    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start : end + 1])

    for candidate in candidates:
        for attempt in (candidate, _TRAILING_COMMA_RE.sub(r"\1", candidate)):
            try:
                parsed = json.loads(attempt)
            except (ValueError, TypeError):
                continue
            if isinstance(parsed, dict):
                return parsed

    raise ValueError(f"无法从模型输出中解析出 JSON 对象：{text[:200]!r}")


def normalize_todos(value: Any) -> list[dict]:
    """把模型给的 todos 规整成 ``[{"task": str, "due": str|None}, ...]``。

    容忍字符串数组、对象数组、单个字符串、``null`` 等各种形态。
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = [value] if value.strip() else []
    if not isinstance(value, (list, tuple)):
        return []

    todos: list[dict] = []
    for item in value:
        if isinstance(item, dict):
            task = _clean_str(
                item.get("task") or item.get("content") or item.get("text") or item.get("title")
            )
            due = _clean_str(item.get("due") or item.get("time") or item.get("deadline"))
        else:
            task, due = _clean_str(item), None
        if not task:
            continue
        todos.append({"task": task, "due": due})
    return todos


def normalize_memo(data: Any, raw_text: str, record_time: str) -> dict:
    """把整理结果补齐成字段完整、类型正确的备忘录字典。"""
    data = data if isinstance(data, dict) else {}
    title = _clean_str(data.get("title")) or fallback_title(raw_text)
    memo = {
        "record_time": record_time,
        "raw_text": raw_text,
        "title": title[:40],
        "category": _clean_str(data.get("category")) or "其他",
        "summary": _clean_str(data.get("summary")) or (raw_text[:200] or "（无内容）"),
        "todos": normalize_todos(data.get("todos")),
    }
    # SOAP 五段：模型没给（或判定为非问诊内容）就留空串，卡片会自动退回摘要视图。
    # 不做"缺失就填摘要"的兜底 —— 那会让每条日常备忘都长出一份假病历。
    # 这里刻意把 ``None`` 也折成 ``""``：字典会一路传到 print_memo / 前端 JSON，
    # 让"没这一段"在所有出口只有一个表示，省得每个消费方各写一遍 ``or ""``。
    for key in SOAP_COLUMNS:
        memo[key] = _clean_str(data.get(key)) or ""
    return memo


def has_soap(memo: dict) -> bool:
    """这条记录有没有 SOAP 内容（五个字段至少填了一个就渲染成病历卡片）。"""
    return any((memo.get(key) or "").strip() for key in SOAP_COLUMNS)


def call_llm(
    client: Any,
    model: str,
    system_prompt: str,
    user_content: str,
    timeout: float = 60.0,
) -> dict:
    """调用 DeepSeek（OpenAI 兼容接口）并返回解析后的字典。"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    kwargs: dict = dict(model=model, messages=messages, temperature=0.2, max_tokens=800, timeout=timeout)
    try:
        response = client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
    except Exception as exc:  # 部分兼容网关不支持 response_format，退回纯提示词约束
        log(f"首次调用失败（{exc}），改用兼容模式重试…", "LLM")
        response = client.chat.completions.create(**kwargs)

    content = response.choices[0].message.content
    return extract_json_object(content)


def call_deepseek(
    client: Any,
    model: str,
    raw_text: str,
    record_time: str,
    timeout: float = 60.0,
) -> dict:
    """首次整理：原始语音文本 → 结构化卡片。"""
    return call_llm(client, model, SYSTEM_PROMPT,
                    f"录制时间：{record_time}\n原始文本：{raw_text}", timeout)


#: 重新润色时给模型看的内容。刻意**把学生手改的版本放在最前面并声明为准**：
#: 模型对"前文优先"很敏感，顺序反了就会拿原始语音覆盖掉学生刚修正的地方 ——
#: 那正是他最想保住的东西。
def build_regenerate_content(memo: dict, record_time: str) -> str:
    """拼一段"重新润色"的输入：以手工修订版为准，原始语音仅作参考。"""
    lines = [f"录制时间：{record_time}", "", "以下是我手工修订过的病历，**以此为准**：", ""]
    for key, label in SOAP_SECTIONS:
        body = (memo.get(key) or "").strip()
        if body:
            lines += [f"【{label}】", body, ""]
    if not has_soap(memo):
        lines += ["【摘要】", (memo.get("summary") or "").strip(), ""]
    title = (memo.get("title") or "").strip()
    if title:
        lines += [f"【标题】{title}", ""]
    raw = " ".join((memo.get("raw_text") or "").split())
    if raw:
        lines += ["原始语音文本（仅作参考；与上面的修订版冲突时，以修订版为准）：", raw]
    return "\n".join(lines)


def regenerate_memo(
    client: Any,
    model: str,
    memo: dict,
    record_time: str,
    timeout: float = 60.0,
) -> dict:
    """把一张已有的卡片重新交给 DeepSeek 润色，返回新的字段字典。

    只拿来**覆盖**内容字段，``record_time`` / ``raw_text`` 保持原样 ——
    重新润色改的是"怎么表述"，不该动"什么时候说的、原话是什么"。
    """
    data = call_llm(client, model, SYSTEM_PROMPT,
                    build_regenerate_content(memo, record_time), timeout)
    fresh = normalize_memo(data, memo.get("raw_text") or "", record_time)
    fresh["record_time"] = record_time
    return fresh


def build_llm_client(cfg: Config) -> Optional[Any]:
    """构造 OpenAI 兼容客户端；没有 API Key 时返回 None（走本地规则）。"""
    if not cfg.api_key:
        log("未配置 DEEPSEEK_API_KEY，将使用内置规则整理（功能可用，但结构化能力较弱）", "LLM")
        return None
    try:
        from openai import OpenAI
    except ImportError:
        # 没装 openai 时**退化成本地规则**，而不是让程序崩在启动那一行：
        # 录音、识别、入库、Markdown 都不依赖它，只有"整理成结构化卡片"会降级。
        # 这一步最常见的翻车方式其实不是没装，而是**装在了别的解释器里**
        # （比如终端自动激活了一个空的 .venv），所以把解释器路径一起打出来，
        # 免得对着 pip install 发呆 —— pip 装对了，跑的还是另一个 python。
        log(f"未安装 openai 包，将使用内置规则整理（当前解释器：{sys.executable}）", "LLM")
        log("如需 DeepSeek 结构化整理：pip install openai，且必须装在上面这个解释器里", "LLM")
        return None

    return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)


# --------------------------------------------------------------------------- #
# 本地规则整理（大模型不可用时的兜底）
# --------------------------------------------------------------------------- #

_CN_DIGITS = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}
_WEEKDAY_MAP = {"日": 7, "天": 7, "末": 6, "七": 7}

#: 命中即认为是"待办句"的提示词
TODO_CUES = (
    "记得", "别忘了", "别忘", "提醒我", "帮我", "要买", "去买", "需要", "提交",
    "预约", "安排", "准备", "打电话", "发邮件", "回复", "交", "取", "订",
)

CATEGORY_KEYWORDS = (
    ("工作", ("会议", "开会", "汇报", "客户", "项目", "上线", "需求", "方案", "老板",
             "同事", "周报", "加班", "汇报", "对接", "面试")),
    ("学习", ("学习", "课程", "看书", "复习", "考试", "论文", "单词", "作业", "培训", "读书")),
    ("购物", ("买", "采购", "下单", "快递", "超市", "清单", "购买", "网购")),
    ("健康", ("吃药", "医院", "挂号", "健身", "跑步", "体检", "睡觉", "复查", "锻炼", "疫苗")),
    ("财务", ("报销", "发票", "转账", "付款", "账单", "工资", "预算", "缴费", "还钱", "租金")),
    ("社交", ("生日", "聚会", "请客", "约", "拜访", "婚礼", "送礼", "聚餐", "见面")),
    ("灵感", ("想法", "点子", "灵感", "创意", "突然想到", "构思")),
    ("生活", ("家务", "打扫", "修理", "搬家", "遛狗", "接孩子", "浇花")),
)


def cn_numeral_to_int(text: str) -> Optional[int]:
    """把 ``三`` / ``十五`` / ``二十三`` / ``3`` 这类写法转成整数。"""
    text = (text or "").strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if "十" in text:
        left, _, right = text.partition("十")
        tens = _CN_DIGITS.get(left, 1) if left else 1
        if left and left not in _CN_DIGITS:
            return None
        units = _CN_DIGITS.get(right, 0) if right else 0
        if right and right not in _CN_DIGITS:
            return None
        return tens * 10 + units
    total = 0
    for char in text:
        if char not in _CN_DIGITS:
            return None
        total = total * 10 + _CN_DIGITS[char]
    return total


def resolve_cn_time(text: str, base: datetime) -> Optional[str]:
    """把中文相对时间表达解析为 ``YYYY-MM-DD HH:MM``；解析不出则返回 None。

    覆盖：今天/明天/后天/大后天、周三/下周三/本周三、上午/下午/晚上/中午/凌晨、
    ``N点[半|N分]``、``N分钟后``/``N小时后``/``半小时后``。
    只给日期没给时刻时，默认 09:00。
    """
    if not text:
        return None

    # --- 纯相对偏移：N 分钟后 / N 小时后 / 半小时后 ---
    match = re.search(r"(半|[零一二三四五六七八九十两\d]{1,3})\s*个?\s*(分钟|小时|钟头)\s*(?:之?后|以后|后)", text)
    if match:
        raw_amount, unit = match.group(1), match.group(2)
        if raw_amount == "半":
            minutes = 30  # 半小时 → 30 分钟
        else:
            amount = cn_numeral_to_int(raw_amount)
            minutes = None if amount is None else (amount if unit == "分钟" else amount * 60)
        if minutes is not None:
            return f"{base + timedelta(minutes=minutes):%Y-%m-%d %H:%M}"

    target_date: Optional[datetime] = None

    if "大后天" in text:
        target_date = base + timedelta(days=3)
    elif "后天" in text:
        target_date = base + timedelta(days=2)
    elif "明天" in text or "明日" in text:
        target_date = base + timedelta(days=1)
    elif "今天" in text or "今日" in text:
        target_date = base

    weekday_match = re.search(r"(下|这|本)?\s*(?:周|星期|礼拜)\s*([一二三四五六日天末\d])", text)
    if weekday_match:
        prefix, raw_weekday = weekday_match.group(1), weekday_match.group(2)
        weekday = _WEEKDAY_MAP.get(raw_weekday)
        if weekday is None:
            weekday = int(raw_weekday) if raw_weekday.isdigit() else cn_numeral_to_int(raw_weekday)
        if weekday:
            if prefix == "下":
                days = (((7 - base.weekday()) % 7) or 7) + (weekday - 1)
            elif prefix in ("这", "本"):
                days = (weekday - 1) - base.weekday()
                if days < 0:  # "这周三"说在周五 → 理解为下周三
                    days += 7
            else:
                days = ((weekday - 1) - base.weekday()) % 7
            target_date = base + timedelta(days=days)

    # --- 时刻 ---
    hour = minute = None
    time_match = re.search(
        r"([零一二三四五六七八九十两\d]{1,3})\s*[点時时]\s*(半|[零一二三四五六七八九十\d]{1,3}\s*分?)?",
        text,
    )
    if time_match:
        hour = cn_numeral_to_int(time_match.group(1))
        raw_minute = (time_match.group(2) or "").strip()
        if raw_minute.startswith("半"):
            minute = 30
        elif raw_minute:
            minute = cn_numeral_to_int(raw_minute.rstrip("分").strip())

    # --- 半天修饰词 ---
    if hour is not None:
        if any(word in text for word in ("下午", "傍晚")) and hour < 12:
            hour += 12
        elif "晚上" in text and hour < 12:
            hour += 12
        elif "中午" in text and hour <= 2:
            hour += 12
        elif hour == 12 and any(word in text for word in ("凌晨", "早上", "上午", "清晨")):
            hour = 0
        hour = min(max(hour, 0), 23)
        minute = min(max(minute or 0, 0), 59)
    else:
        for word, default_hour in (("凌晨", 5), ("早上", 8), ("上午", 9), ("中午", 12),
                                   ("下午", 15), ("傍晚", 18), ("晚上", 20)):
            if word in text:
                hour, minute = default_hour, 0
                break

    if target_date is None and hour is None:
        return None
    if target_date is None:
        # 只说了时刻：若今天该时刻已过，则理解为明天
        candidate = base.replace(hour=hour or 0, minute=minute or 0, second=0, microsecond=0)
        if hour is not None and candidate <= base:
            candidate += timedelta(days=1)
        return f"{candidate:%Y-%m-%d %H:%M}"

    resolved = target_date.replace(hour=hour if hour is not None else 9,
                                   minute=minute if hour is not None else 0,
                                   second=0, microsecond=0)
    return f"{resolved:%Y-%m-%d %H:%M}"


_STOP_PARTICLES = "吧啊呀呢哦嗯了地得咯"        # "结束吧""说完了嗯"这类尾巴
_STOP_PUNCT = "，,。、！？!?~～;；:： \t"        # 剥控制词时要一并吃掉的标点


def _split_keywords(raw: str) -> tuple:
    """把 "a,b,c" 拆成去空的元组，中英文逗号都认。"""
    return tuple(part.strip() for part in raw.replace("，", ",").split(",") if part.strip())


# --------------------------------------------------------------------------- #
# 关键词匹配（唤醒词 / 语音指令）
#
# 为什么要模糊匹配：ASR 对同一个唤醒词的转写经常漂移 —— "小医小医" 可能变成
# "小一小衣""小依小一""晓医晓依"。用逐字相似度判定，比要求完全相等稳得多，
# 而且不需要额外依赖（difflib 是标准库）。
# --------------------------------------------------------------------------- #


def normalize_for_match(text: str) -> str:
    """匹配前归一化：去掉空白与标点、统一大小写。"""
    if not text:
        return ""
    return re.sub(r"[\s，。、！？,.!?~～;；:：\"'“”‘’（）()\[\]【】-]", "", text).lower()


def similarity(a: str, b: str) -> float:
    """两个短串的相似度 0~1。等长滑窗比对用，所以先按长度差打折。"""
    if not a or not b:
        return 0.0
    from difflib import SequenceMatcher

    ratio = SequenceMatcher(None, a, b).ratio()
    # "小医" 与 "小医小医" 的 ratio 是 0.67，会误判成不相似；长度差单独惩罚更合理
    length_penalty = min(len(a), len(b)) / max(len(a), len(b))
    return ratio * (0.6 + 0.4 * length_penalty)


# --------------------------------------------------------------------------- #
# 拼音模糊匹配
#
# 为什么光有逐字相似度还不够：ASR 的错字绝大多数是**同音字**。实测 faster-whisper
# base 在中文上把控制词听成（本机真实输出）：
#
#     作废 → 坐着飞      退出录制 → 退出錄製      录音完毕 → 入音完畢
#     这条不要了 → 這條不要了                        保存这段 → 保存這段
#
# 逐字比对几乎全军覆没（汉字全对不上），可读音一模一样。转成拼音再比就全中，
# 顺带连繁体字都不用单独处理 —— 錄/录、畢/毕、這/这 的拼音本来就相同。
#
# 两个细节：
# 1. 先做"音近归一"（zh/z、ang/an…），把"万笔/完毕"、"录音/入音"这类
#    只有一两个音节听岔的情况也算进匹配率，而不是全盘不认。
# 2. 用**覆盖度**而不是相似度：ASR 会凭空多塞音节（"作废"→"坐着飞"），
#    只要关键词的读音按顺序都出现在这一小段里就算命中。
#
# pypinyin 是可选依赖：没装就自动跳过这一层，退回逐字模糊匹配（弱一些，但功能
# 不缺），所以自检在没有 pypinyin 的环境里照样能跑完。
# --------------------------------------------------------------------------- #

_PYTHON_PINYIN: Any = None          # None=还没找过；False=确认没有
_PINYIN_CACHE: dict = {}


def pinyin_available() -> bool:
    """pypinyin 是否可用（可用则控制词判定会自动多一层拼音匹配）。"""
    global _PYTHON_PINYIN
    if _PYTHON_PINYIN is None:
        try:
            import pypinyin
        except ImportError:
            _PYTHON_PINYIN = False
            log("未安装 pypinyin，控制词只做逐字模糊匹配（pip install pypinyin 可显著提升容错）", "指令")
        else:
            _PYTHON_PINYIN = pypinyin
    return bool(_PYTHON_PINYIN)


#: 音近归一：这些差别在 ASR 里几乎不算差别，折叠掉能少漏一半
_SYLLABLE_FOLDS = (
    ("iang", "ian"), ("uang", "uan"), ("ang", "an"), ("eng", "en"), ("ing", "in"),
)


def _fold_syllable(syllable: str) -> str:
    """把一个音节的"前后鼻音 / 翘舌音"差别抹平，得到用于比对的音近键。

    注意是 zh→z、ch→c、sh→s 的**整组替换**，不能只把 h 抹掉 ——
    只抹 h 会让"书(shu)"变成"hu"，跟"会(hui)""后(hou)"混成一堆，
    实测会直接把"会议结束之后给我打电话"误判成"退出录制"。
    """
    syl = (syllable or "").strip().lower().replace("ü", "v").replace("u:", "v")
    if syl.startswith("zh"):
        syl = "z" + syl[2:]
    elif syl.startswith("ch"):
        syl = "c" + syl[2:]
    elif syl.startswith("sh"):
        syl = "s" + syl[2:]
    for tail, folded in _SYLLABLE_FOLDS:
        if syl.endswith(tail):
            return syl[: -len(tail)] + folded
    return syl


#: 匹配时直接忽略的字符（空白与标点）：它们保留下标但不参与音节比对
_SKIP_IN_MATCH = set(" \t\r\n，。、！？,.!?~～;；:：\"'“”‘’（）()[]【】-…")


def char_syllables(text: str) -> list:
    """逐字转拼音，返回 ``[(字符, 音近键), ...]``，长度恒等于 ``len(text)``。

    非汉字（数字、字母）原样保留成自己的"音节"，标点空白记为空串占位 ——
    这样下标与原文严格一一对应，后面才好在原文本里精确定位并剪掉控制词。
    """
    if not text:
        return []
    cached = _PINYIN_CACHE.get(text)
    if cached is not None:
        return cached

    if not pinyin_available():
        pairs = [(char, "" if char in _SKIP_IN_MATCH else _fold_syllable(char))
                 for char in text]
    else:
        from pypinyin import Style

        # errors 回调负责非汉字：按字符逐个返回，保证与 text 一一对应
        groups = _PYTHON_PINYIN.pinyin(text, style=Style.NORMAL,
                                       errors=lambda item: list(item))
        pairs = []
        for char, syllables in zip(text, groups):
            if char in _SKIP_IN_MATCH:
                pairs.append((char, ""))
                continue
            syllable = (syllables[0] if syllables else "") or char
            pairs.append((char, _fold_syllable(syllable)))

    if len(_PINYIN_CACHE) < 4096:           # 控制词就那几个，缓存不会无限涨
        _PINYIN_CACHE[text] = pairs
    return pairs


def syllables_of(text: str) -> list:
    """只要音近键序列（跳过标点空白）。"""
    return [syllable for _char, syllable in char_syllables(text) if syllable]


#: 少量"整体不像、但经常互相听岔"的音节对，逐个列出来。
#: 不用通用的字符串相似度来自动判：音节只有两三个字母，"tui/hui""chu/hu"
#: 这种差一个字母的组合相似度高达 0.67，用阈值判定会满盘误伤
#: （实测"会议结束之后给我打电话"会被判成"退出录制"）。宁可只收这几个已知的。
_SYLLABLE_ALIASES = {
    "lu": ("ru",), "ru": ("lu",),           # 录/入、路/如：n/l/r 不分
    "zou": ("zuo",), "zuo": ("zou", "cuo"),  # 走/作：ou-uo 听岔；作/错：z/c 不分
    "cuo": ("zuo",),                         # 实测「作废」被听成「錯誤費」/「坐着飞」
    "shou": ("sou",), "sou": ("shou",),     # 收/搜
    "wan": ("wang",), "wang": ("wan",),     # 已经由 _SYLLABLE_FOLDS 折叠，列着便于阅读
}


def _syllable_equal(a: str, b: str) -> bool:
    """两个音节算不算"听起来是同一个"：归一后相等，或命中已知的近音对。"""
    if a == b:
        return True
    return b in _SYLLABLE_ALIASES.get(a, ())


def _lcs_length(needle: list, hay: list) -> int:
    """两个音节序列的最长公共子序列长度（带音近判定）。序列都很短，DP 足够。"""
    previous = [0] * (len(hay) + 1)
    for left in needle:
        current = [0]
        for index, right in enumerate(hay):
            if _syllable_equal(left, right):
                current.append(previous[index] + 1)
            else:
                current.append(max(current[index], previous[index + 1]))
        previous = current
    return previous[-1]


def syllable_coverage(needle: list, hay: list) -> float:
    """关键词的音节有多少"按顺序出现"在这一小段里（0~1）。

    用覆盖度而不是整体相似度，是为了扛住 ASR 凭空多塞的音节：说"作废"被识别成
    "坐着飞"（zuo-zhe-fei），关键词 zuo-fei 两个音节都按序出现了，覆盖度 1.0，
    整体相似度却只有 0.69 —— 后者会把这条指令漏掉。
    """
    if not needle or not hay:
        return 0.0
    return _lcs_length(needle, hay) / float(len(needle))


def find_pinyin_window(hay: list, needle: list, threshold: float,
                       tail_only: bool = False) -> Optional[tuple]:
    """在音节序列 ``hay`` 里找 ``needle``，命中返回 ``(起, 止)`` 下标，否则 None。

    只试 ``len(needle)`` 和 ``len(needle)+1`` 两种窗口宽度：多出来的那一个音节
    正好兜住"作废吧""就这些了"这类多一个语气字的说法。
    ``tail_only=True`` 时窗口必须贴着末尾（作废词靠这条规则避开正文里的动词）。
    """
    size = len(needle)
    if size == 0 or len(hay) < size:
        return None
    for width in (size, size + 1):
        if width > len(hay):
            continue
        starts = [len(hay) - width] if tail_only else range(len(hay) - width + 1)
        for start in starts:
            window = hay[start:start + width]
            if syllable_coverage(needle, window) >= threshold:
                return (start, start + width)
    return None


def find_pinyin_keyword(text: str, keywords: tuple, threshold: float = 0.85,
                        tail_only: bool = False,
                        aliases: tuple = ()) -> Optional[tuple]:
    """拼音版的 ``find_keyword``：返回 ``(命中的关键词, 起始字符下标, 结束字符下标)``。

    带上字符下标是为了"不只是判出来，还要剪掉"—— ASR 写成"錄音完畢"时，
    按字面做字符串替换是删不掉的，得靠拼音定位出它到底占了哪几个字。
    ``aliases`` 用于唤醒词：命中任意写法都算命中本体。
    """
    if not text or not pinyin_available():
        return None
    candidates = list(keywords) + list(aliases or ())
    if not candidates:
        return None

    pairs = char_syllables(text)
    positions = [index for index, (_char, syllable) in enumerate(pairs) if syllable]
    hay = [pairs[index][1] for index in positions]
    if not hay:
        return None

    best: Optional[tuple] = None
    for candidate in candidates:
        needle = syllables_of(candidate)
        if not needle:
            continue
        window = find_pinyin_window(hay, needle, threshold, tail_only=tail_only)
        if window is None:
            continue
        start, end = window
        # 候选之间要挑**最长**的那个：控制词表里有"完毕"也有"录音完毕"，
        # 短词先命中就会只剪掉后半截，剩一个"錄音"孤零零留在正文里。
        rank = (-len(needle), start)
        if best is None or rank < best[0]:
            # 窗口多出来的那个音节可能是语气字（"作废吧"的"吧"），但对"坐着飞"
            # 这种塞在中间的音节又必须整段剪掉 —— 所以一律按窗口整段返回。
            best = (rank, candidate, positions[start], positions[end - 1] + 1)
    if best is None:
        return None
    return (best[1], best[2], best[3])


def strip_pinyin_tail(text: str, keywords: tuple, threshold: float = 0.85) -> str:
    """把句尾"读起来像控制词、但写成了别字/繁体"的那一截也剪掉（可反复剥）。"""
    if not text or not keywords or not pinyin_available():
        return text
    cleaned = text
    for _ in range(4):                      # "……就这些，錄音完畢"可能连着好几个
        hit = find_pinyin_keyword(cleaned, keywords, threshold, tail_only=True)
        if hit is None:
            break
        _keyword, start, end = hit
        if start >= end:
            break
        cleaned = (cleaned[:start] + cleaned[end:]).rstrip(f" {_STOP_PUNCT}{_STOP_PARTICLES}")
        if not cleaned.strip():
            return ""
    return cleaned


# --------------------------------------------------------------------------- #
# 医学热词：解码期提示词 + 识别后校正
# --------------------------------------------------------------------------- #

#: 提示词的字数上限。whisper 的 ``initial_prompt`` 只吃得到上下文窗口的一半
#: （约 224 个 token），中文大约 1~1.5 字/token，220 字留了余量。
#: 超了 whisper 不会报错，只会**默默截断**，反而把后面的词全丢掉 —— 所以自己截。
ASR_PROMPT_MAX_CHARS = 220

_ASR_PROMPT_LEAD = "以下是医学生问诊与病史采集的录音，会大量出现医学专业词汇："


def build_asr_prompt(cfg: "Config") -> str:
    """把医学热词拼成 ASR 的提示词（``initial_prompt``）。

    按**词表顺序**逐个往提示词里塞，塞满就停 —— 所以 ``DEFAULT_MEDICAL_HOTWORDS``
    里越靠前的词越"值钱"（病史栏目、常用体征排在前面，冷门诊断排在后面）。

    关掉的方式：``ASR_INITIAL_PROMPT=false``，或 ``ASR_HOTWORDS=off``。
    """
    if not getattr(cfg, "asr_initial_prompt", True):
        return ""
    words = [w for w in (getattr(cfg, "asr_hotwords", ()) or ()) if w]
    if not words:
        return ""
    picked: list = []
    used = len(_ASR_PROMPT_LEAD)
    for word in words:
        cost = len(word) + 1                  # 顿号
        if used + cost > ASR_PROMPT_MAX_CHARS:
            break
        picked.append(word)
        used += cost
    return _ASR_PROMPT_LEAD + "、".join(picked) + "。"


#: 校正窗口里不允许出现的字符：混进标点/空格说明这是跨句的巧合，不是听岔的词
_NON_CJK_RE = re.compile(r"[^一-鿿]")

#: 识别后校正**只处理三个字以上**的词。
#: 两字词的同音碰撞率太高 —— "提问"和"体温"（ti-wen）、"口痛"和"叩痛"（kou-tong）
#: 拼音完全一样，靠读音根本分不出来，改了就是把对的改错。三字以上才安全。
MEDICAL_FIX_MIN_CHARS = 3

#: 音节覆盖率门槛。**必须是 1.0（全音节对上）**，这不是保守，是实测踩出来的：
#: 门槛放到 0.65（"三字里蒙对两个"）的第一版把这些改成了 ——
#:
#:     右下腹 → 右上腹      干啰音 → 湿啰音      患者主诉心前区 → 患者主心前区区
#:
#: 在别的领域"差一个字"只是错别字，在医学里**上下、左右、干湿、阴阳都是相反的体征**，
#: 改错一个字比不改危害大得多。所以只接受**完整同音**的替换：ASR 的错法本来就是
#: "音对字错"，而完整的同音词很少会和另一个常用词组撞车。
#: 代价是「反跳疼」（teng）这种差一点点的改不过来 —— 漏改只是没帮上忙，不算错。
MEDICAL_FIX_THRESHOLD = 1.0


def correct_medical_terms(text: str, hotwords: tuple, threshold: float = MEDICAL_FIX_THRESHOLD,
                          min_chars: int = MEDICAL_FIX_MIN_CHARS) -> str:
    """把识别成同音别字的医学术语掰回正确写法。

    这是**解码期提示词的补丁**，不是替代品：whisper 能用 ``initial_prompt`` 引导，
    SenseVoice 压根没有对应的参数，加一层与后端无关的校正才能两边都受益。

    三道关卡共同把误伤压到最低，缺一不可（都是实测逼出来的）：

    1. **长度一致** —— 窗口和术语必须一样长，避免"口痛得"被塞成"叩击痛"；
    2. **全音节同音** —— 见 ``MEDICAL_FIX_THRESHOLD`` 的注释；
    3. **至少共用一个字** —— ASR 听岔时通常只错一个音、其余字还在（"湿罗音"→"湿啰音"
       留着"湿""音"）。一个字都不沾却读音全同，那是巧合，不是听岔。

    替换是**一次性收集、统一落盘**的：先把所有命中窗口记下来（长词优先占位、互不重叠），
    最后才重建字符串。边扫边改会让刚写进去的正确术语又被下一个词条二次匹配。
    """
    if not text or not hotwords or not pinyin_available():
        return text

    # 长词优先："十二指肠溃疡"必须先于"十二指肠"匹配，否则会被短词抢先切成半截
    terms = sorted({w for w in hotwords if len(w) >= min_chars}, key=len, reverse=True)
    length = len(text)
    occupied = [False] * length
    edits: list = []
    for term in terms:
        needle = syllables_of(term)
        size = len(term)
        if not needle:
            continue
        for start in range(0, length - size + 1):
            stop = start + size
            if any(occupied[start:stop]):
                continue
            window = text[start:stop]
            if window == term or _NON_CJK_RE.search(window):
                continue
            if not set(window) & set(term):          # 一个字都不沾 → 巧合
                continue
            if syllable_coverage(needle, syllables_of(window)) < threshold:
                continue
            edits.append((start, stop, term))
            for position in range(start, stop):
                occupied[position] = True

    if not edits:
        return text
    pieces: list = []
    cursor = 0
    for start, stop, term in sorted(edits):
        pieces.append(text[cursor:start])
        pieces.append(term)
        cursor = stop
    pieces.append(text[cursor:])
    return "".join(pieces)


def wake_word_forms(keyword: str) -> tuple:
    """唤醒词的候选写法。

    叠词型（"小医小医"）额外接受说一半的情况 —— 真人经常只说一遍，
    或 ASR 把两个词合并成一个。单字型不加变体，避免误唤醒。
    """
    keyword = normalize_for_match(keyword)
    if not keyword:
        return ()
    forms = {keyword}
    if len(keyword) >= 4 and len(keyword) % 2 == 0:
        half = keyword[: len(keyword) // 2]
        if half == keyword[len(keyword) // 2:] and len(half) >= 2:
            forms.add(half)
    return tuple(sorted(forms, key=len, reverse=True))


# 常见同音/近音字表：ASR 对同一个唤醒词的转写会**整体漂移**
# （"小医小医" → "小一小一" / "小依小依" / "晓医晓医"），逐字精确匹配太脆。
# 每组是一批读音相近的字，唤醒词里的某个字整组一起被替换 ——
# 这样不需要真做拼音，也就没有任何额外依赖。
_HOMOPHONE_GROUPS = (
    "小晓筱肖校潇笑",
    "记纪计继济寄技忌吉集季",
    "好号浩昊郝",
    "备被倍辈背贝",
    "忘王望亡网往妄",
    "录路陆露鹿鲁碌",
    "开凯慨揩",
    "始使史士市施师是示",
    "停亭庭挺廷",
    "止只纸至志治指支",
    "退推腿褪",
    "出初除楚触",
    "彻撤澈",
    "底低抵地弟帝",
    "结节街接截洁杰",
    "束术数树述竖",
    "段断锻缎",
    "这着者折浙",
    "保宝抱报暴堡",
    "存村寸蹲",
    "下夏吓霞",
    "一衣依医伊",
    "条跳挑调",
    "你拟妮尼",
    "忆意义易艺议益",
    "笔比彼币必",
)

_HOMOPHONES = {char: group for group in _HOMOPHONE_GROUPS for char in group}


def wake_word_aliases(keyword: str, extra: tuple = ()) -> tuple:
    """唤醒词"听起来一样"的所有写法（同音字整组替换 + 用户自定义别名）。

    只做**整组替换**，不做全排列组合：真实 ASR 的漂移是系统性的 —— 它听到的"记"
    会一直写成"纪"，不会一个字一个字乱换。于是候选数从指数级降到线性级
    （"小医小医" 只需十几个候选就覆盖了实测最常见的写法）。
    """
    base = normalize_for_match(keyword)
    if not base:
        return ()
    forms = {base}
    for char in dict.fromkeys(base):        # 同一个字只替换一次
        group = _HOMOPHONES.get(char)
        if not group:
            continue
        forms.update(base.replace(char, alt) for alt in group if alt != char)
    for alias in extra or ():
        normalized = normalize_for_match(alias)
        if normalized:
            forms.add(normalized)
    return tuple(sorted(forms, key=len, reverse=True))


#: 逐字同音"全组合"的候选数上限。两字词全排列最多 7×5=35 个，随便组合；
#: 四字叠词全排列是 7×5×7×5=1225 个，每个候选还要跑一遍拼音扫描，不值得 ——
#: 超过这个数的形式退回"整组替换"（见 ``wake_word_aliases`` 的说明）。
_WAKE_COMBO_LIMIT = 64


def _homophone_combos(text: str) -> set:
    """逐字同音替换的**全组合**（"晓依" 这种两处一起漂写法的来源）。"""
    groups = [tuple(_HOMOPHONES.get(char) or char) for char in text]
    total = 1
    for group in groups:
        total *= len(group)
    if total > _WAKE_COMBO_LIMIT:
        return set()
    combos = {""}
    for group in groups:
        combos = {prefix + char for prefix in combos for char in group}
    return combos


def wake_word_candidates(keyword: str, extra: tuple = ()) -> tuple:
    """唤醒词**全部**可接受的写法：每个长度 × 同音写法 + 用户别名。

    为什么不直接用 ``wake_word_aliases``：那个函数只对**整词**做同音替换。
    唤醒词配的是叠词"小医小医"，可真人经常只说一半，ASR 又会把"医"写成"一/依"，
    "只说一遍 + 写岔"正好落在缝里 —— 判不唤醒，或者判成唤醒、清洗却删不掉。
    判定（``find_keyword``）和清洗（``strip_leading_keyword``）必须用同一套候选。

    短候选（折半后的两个字）额外做逐字**全组合**：两字词上"小"和"医"一起漂成
    "晓依"在真人说话里很常见（两个音节都短、都容易被听成别的），而整词候选不做 ——
    "小医小医"的全组合是 1225 个，两处漂移同时发生又太稀罕，不划算。
    """
    forms = set(wake_word_forms(keyword))
    for form in tuple(forms):
        forms.update(wake_word_aliases(form))
        forms.update(_homophone_combos(form))
    for alias in extra or ():
        normalized = normalize_for_match(alias)
        if normalized:
            forms.add(normalized)
    forms.discard("")
    return tuple(sorted(forms, key=len, reverse=True))


def find_keyword(text: str, keywords: tuple, threshold: float = 0.72,
                 wake_word: str = "", wake_aliases: tuple = (),
                 tail_only: bool = False,
                 pinyin_threshold: float = 0.85) -> Optional[str]:
    """在识别文本里找关键词，命中就返回那个关键词，否则 None。

    三层判定，从快到慢：完全包含 → 逐字模糊 → 拼音比对（后者要求装了 pypinyin）。

    ``wake_word`` 非空时，额外接受它的叠词变体和同音写法
    （见 ``wake_word_forms`` / ``wake_word_aliases``），命中一律返回唤醒词本体。

    ``tail_only=True`` 时只认**句尾**那一下（先削掉"作废吧""不要了"这类语气尾巴）。
    "取消"这种词在正文里也是常用动词（"明天取消会议"），只有在句尾才当指令看，
    才不会把一整段正经内容误判成"丢掉别记"。

    ``pinyin_threshold`` 是音节覆盖度门槛。默认 0.85 而不是 0.75：覆盖度是整数
    比值，0.75 落在四字词上恰好等于"允许少一个音节"，实测正是这一条把正文
    "把这段录音听完"判成了"录音完毕"（lu yin ting wan 少了 bi，覆盖度仍有 0.75）。
    0.85 对两三个字的关键词要求与 0.75 完全一致（1/2、2/3 都低于它），只收紧长词。
    """
    haystack = normalize_for_match(text)
    if not haystack:
        return None
    if tail_only:
        haystack = haystack.rstrip(_STOP_PARTICLES)
        if not haystack:
            return None

    # 1) 完全包含（最快，也最常见）
    for keyword in keywords:
        needle = normalize_for_match(keyword)
        if not needle:
            continue
        if tail_only:
            trimmed = needle.rstrip(_STOP_PARTICLES)
            if haystack.endswith(needle) or (trimmed and haystack.endswith(trimmed)):
                return keyword
        elif needle in haystack:
            return keyword
    if wake_word:
        # 字面层用**全部**候选（含折半写法各自的同音替换）：ASR 把单说一遍的
        # "小医"写成"小一/小依"时，逐字相似度和拼音都对不上，只有这一层认得出来。
        for form in wake_word_candidates(wake_word, wake_aliases):
            if len(form) >= 2 and form in haystack:
                return wake_word

    # 2) 滑窗模糊比对：ASR 把字听岔了也能认出来
    #    装了 pypinyin 时这一层要向拼音门槛看齐：逐字相似度在四字词上 0.72
    #    相当于"四个字里蒙对一个就过"，实测正是它把正文"把这段录音听完"
    #    判成了"录音完毕"（听完 vs 完毕，相似度 0.75）。真正该容忍的同音错字
    #    由下面第 3 层精确处理，这一层没必要再放水。
    char_threshold = threshold
    if pinyin_available():
        char_threshold = max(threshold, pinyin_threshold)
    candidates = list(keywords)
    if wake_word:
        for form in wake_word_forms(wake_word):
            candidates.append(form)
    for candidate in candidates:
        needle = normalize_for_match(candidate)
        width = len(needle)
        if width < 2 or len(haystack) < width:
            continue
        if tail_only:
            windows = (haystack[-width:],)
        else:
            windows = (haystack[start:start + width]
                       for start in range(len(haystack) - width + 1))
        for window in windows:
            if similarity(window, needle) >= char_threshold:
                return candidate if candidate != wake_word else wake_word

    # 3) 拼音比对：ASR 写成"錄音完畢""坐着飞"时逐字相似度几乎为 0，
    #    但读音一模一样（见文件上方"拼音模糊匹配"一节）。
    if pinyin_available():
        if wake_word:
            # 这一层**只喂整词**的候选，不喂折半的那些。原因实测出来的：拼音匹配
            # 用的是"覆盖度"（允许中间夹音节，"作废"→"坐着飞"就得靠这个），
            # 两音节的关键词配上这条规则就太松了 —— 正文"明天取消会议"
            # （qu **xiao** hui **yi**）里 xiao 和 yi 隔着 hui 也被算成齐全，
            # 于是"小医"被误判唤醒。折半写法改走上面第 1 层的字面候选。
            aliases = wake_word_aliases(wake_word, wake_aliases)
            hit = find_pinyin_keyword(text, keywords, pinyin_threshold,
                                      tail_only=tail_only, aliases=aliases)
        else:
            hit = find_pinyin_keyword(text, keywords, pinyin_threshold, tail_only=tail_only)
        if hit is not None:
            return wake_word or hit[0]
    return None


def match_command_detail(text: str, cfg: "Config",
                         groups: Optional[tuple] = None) -> Optional[tuple]:
    """判断识别文本里有没有语音指令，返回 ``(指令类别, 命中的词)``，没有则 None。

    类别是 ``"wake"``（唤醒）/ ``"segment_end"``（分段保存）/ ``"cancel"``（作废本条）
    / ``"exit"``（退出程序）。判定顺序即优先级：退出 > 作废 > 分段 > 唤醒 ——
    一句话里出现多个时，听用户最后那道指令（"录音完毕，这条不要了" = 别存）。

    ``groups`` 用来限定这一次只找哪几类（后台扫描每次只关心一部分，能省不少算力）。
    """
    if not text:
        return None
    wanted = ("exit", "cancel", "segment_end", "wake") if groups is None else tuple(groups)
    if "exit" in wanted and cfg.exit_keywords:
        hit = find_keyword(text, cfg.exit_keywords, cfg.wake_fuzzy_threshold)
        if hit:
            return ("exit", hit)
    if "cancel" in wanted and cfg.cancel_keywords:
        # 作废是**破坏性**指令（判错了这条就没了），所以只认句尾那一小段：
        # "取消"在正文里太常见（"明天取消会议"），宁可漏判也不能误伤整段内容。
        hit = find_keyword(text, cfg.cancel_keywords, cfg.wake_fuzzy_threshold,
                           tail_only=True)
        if hit:
            return ("cancel", hit)
    if "segment_end" in wanted:
        # 长指令词（"录音完毕""下一条"）特征明显，句中说出也算数；
        # 短指令词（"完毕""保存""好了""收工"）只在**句尾**算数 —— 它们太常见了，
        # "文件我已经保存了"这种正常句子不该被拦腰截断。
        hit = find_keyword(text, cfg.segment_end_keywords, cfg.wake_fuzzy_threshold)
        if not hit and cfg.short_end_keywords:
            hit = find_keyword(text, cfg.short_end_keywords, cfg.wake_fuzzy_threshold,
                               tail_only=True)
        if hit:
            return ("segment_end", hit)
    if "wake" in wanted and cfg.wake_word_enabled and cfg.wake_word:
        hit = find_keyword(text, (), cfg.wake_fuzzy_threshold, wake_word=cfg.wake_word,
                           wake_aliases=cfg.wake_aliases, pinyin_threshold=0.85)
        if hit:
            return ("wake", hit)
    return None


def match_command(text: str, cfg: "Config") -> Optional[str]:
    """``match_command_detail`` 的简化版：只要类别，不要命中的词。"""
    hit = match_command_detail(text, cfg)
    return hit[0] if hit else None


def clean_transcript(text: str, cfg: "Config") -> str:
    """把 ASR 原文清成可入库的内容。

    剥掉三类东西：开头的唤醒词（"小医，明天开会"）、句尾或独立小句里的控制词
    （结束/完毕/说完了）、以及语音指令词（下一条/退出录制）。句中同名的实词保留。
    """
    if not text:
        return ""
    text = text.strip()
    if cfg.wake_word_enabled and cfg.wake_word:
        text = strip_leading_keyword(text, cfg.wake_word, cfg.wake_aliases)
    controls = (tuple(cfg.stop_keywords) + tuple(cfg.segment_end_keywords)
                + tuple(cfg.short_end_keywords)
                + tuple(cfg.exit_keywords) + tuple(cfg.cancel_keywords))
    # 顺序很重要：**先按读音剥，再按字面剥**。
    # 反过来的话，"這條不要了"会被字面的短词"不要了"先剪一刀，剩下孤零零一个
    # "這條"；拼音那一遍挑的是最长的候选（"这条不要了"），能一次剪干净。
    return strip_control_words(strip_pinyin_tail(text, controls), controls)


def _cut_after_normalized(text: str, count: int) -> str:
    """消费掉 text 开头的 count 个"有效字符"（空白标点不计），返回剩余部分。"""
    seen = 0
    for index, char in enumerate(text):
        if normalize_for_match(char):
            seen += 1
        if seen >= count:
            return text[index + 1:]
    return ""


def strip_leading_keyword(text: str, keyword: str, aliases: tuple = ()) -> str:
    """剥掉开头的唤醒词，允许它被 ASR 写岔（"小依小医，明天开会"→"明天开会"）。

    候选写法必须把**叠词的两个长度都算上**：唤醒词配的是"小医小医"，可真人经常
    只说一遍，ASR 又会把"医"写成"一/依/姨"。"只说一遍 + 写岔"这种组合
    （"小一，明天下午问诊"）如果只对整词做同音替换就漏了，结果唤醒词跟着正文
    一起入库 —— 判定应该唤醒、清洗却删不掉，两边用的候选集必须一致。
    """
    cleaned = text.lstrip()
    if not cleaned:
        return ""
    # 候选集与判定用的那一套**完全相同**（``wake_word_candidates``）：
    # 判定说「唤醒了」的写法，这里就必须删得掉，否则唤醒词会跟着正文一起入库。
    order = wake_word_candidates(keyword, aliases)   # 长的先试，避免"小医"抢在"小医小医"前面
    normalized = normalize_for_match(cleaned)
    for form in order:                             # 完全命中已知写法
        if normalized.startswith(form):
            return _cut_after_normalized(cleaned, len(form)).lstrip(f" {_STOP_PUNCT}")
    for form in order:                             # 开头这一段长得像唤醒词
        width = len(form)
        if width >= 2 and similarity(normalized[:width], form) >= 0.8:
            return _cut_after_normalized(cleaned, width).lstrip(f" {_STOP_PUNCT}")
    # 读音也像（"小姨小依" → "小医小医"，或 ASR 输出繁体）—— 只削开头这几个音节
    if pinyin_available():
        pairs = char_syllables(cleaned)
        positions = [index for index, (_c, syllable) in enumerate(pairs) if syllable]
        hay = [pairs[index][1] for index in positions]
        for form in order:
            needle = syllables_of(form)
            if not needle:
                continue
            for span in (len(needle), len(needle) + 1):
                if span <= len(hay) and syllable_coverage(needle, hay[:span]) >= 0.85:
                    return cleaned[positions[span - 1] + 1:].lstrip(f" {_STOP_PUNCT}")
    return cleaned



def strip_control_words(text: str, keywords: tuple = DEFAULT_STOP_KEYWORDS) -> str:
    """剔除"结束/完毕"这类语音控制词，并清理因此产生的悬空标点与空白。

    只删真正在起**终止作用**的控制词，两种位置：

    1. 句尾 —— 允许夹带语气助词和标点，"…发邮件给张总，说完了嗯。"整段剥掉；
       剥完若前面又露出一个控制词（"…就这些，结束"），会继续往里剥。
    2. 单独成小句 —— 一个从句除了控制词之外没有别的内容（"买牛奶，完毕，回家"），
       这种也按控制词处理。

    中段出现的同名实词一律**保留**："会议结束之后给我打电话"里的"结束"是内容，
    早期版本用朴素 replace 删掉它，会改变原意。
    """
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text)
    keywords = tuple(k for k in keywords if k)
    if not keywords:
        return cleaned.strip(" \t")

    # 长的排前面，避免"说完了"被"完了"抢先匹配
    pattern = "|".join(re.escape(k) for k in sorted(keywords, key=len, reverse=True))

    # 1) 反复剥句尾：控制词 + 语气词 + 标点，直到剥不动为止
    tail_re = re.compile(rf"(?:{pattern})[{_STOP_PARTICLES}]*[{_STOP_PUNCT}]*$")
    while True:
        shrunk = tail_re.sub("", cleaned)
        if shrunk == cleaned:
            break
        cleaned = shrunk

    # 2) 丢掉"只有控制词"的独立小句（连同它前面的那个分隔符）
    parts = re.split(r"([，,。、！？!?；;：:\n]+)", cleaned)
    kept: list = []
    for index, part in enumerate(parts):
        if index % 2 == 0:                       # 偶数位是内容，奇数位是分隔符
            core = re.sub(rf"[{_STOP_PARTICLES}\s]+$", "", part).strip()
            if core and core in keywords:
                if len(kept) % 2 == 0:           # 末尾挂着分隔符，一起撤掉
                    kept.pop()
                continue
        kept.append(part)
    cleaned = "".join(kept)

    cleaned = re.sub(r"[，、,]{2,}", "，", cleaned)
    # 只收尾清理"悬空的分隔符"；句号问号留着，没剥过控制词的原文不该被动到
    return cleaned.strip(" \t，,、;；:")


def contains_stop_keyword(text: str, keywords: tuple = DEFAULT_STOP_KEYWORDS) -> bool:
    """判断一段识别文本里是否出现了截断控制词（忽略空白与标点）。"""
    if not text:
        return False
    squashed = re.sub(r"[\s，。、！？,.!?~～]", "", text)
    return any(keyword in squashed for keyword in keywords)


_LEADING_FILLER_RE = re.compile(r"^(?:那个|这个|就是说|然后|反正|其实|你知道吧|我跟你讲)[，,、\s]*")


def strip_fillers(text: str) -> str:
    """去掉口语填充词，让本地兜底输出的标题/摘要不那么难读。

    只处理两种情况，避免误伤「那个项目」这类实词：
    语气词（嗯/呃/额）出现在任何位置都视作噪声；其余填充词只在从句开头剥离。
    **raw_text 始终保持原样**，本函数仅用于本地规则整理的输出字段。
    """
    if not text:
        return ""
    cleaned = re.sub(r"[嗯呃额]+[，,、]?", "", text)
    parts = re.split(r"([，,。！？!?；;\n]+)", cleaned)
    for index in range(0, len(parts), 2):        # 奇数位是分隔符，跳过
        previous = None
        while previous != parts[index]:          # 反复剥离，处理「那个那个……」这种叠用
            previous = parts[index]
            parts[index] = _LEADING_FILLER_RE.sub("", parts[index]).strip()
    return "".join(parts).strip(" \t，,、;；")


def guess_category(text: str) -> str:
    for category, keywords in CATEGORY_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return category
    return "其他"


def fallback_title(text: str, limit: int = 20) -> str:
    """没有大模型时兜底造标题：在从句边界收尾，避免把词切断。"""
    clauses = [c.strip() for c in re.split(r"[。！？!?；;，,\n]+", text.strip()) if c.strip()]
    if not clauses:
        return "未命名备忘录"
    title = clauses[0]
    for clause in clauses[1:]:
        if len(title) >= 6 or len(title) + len(clause) + 1 > limit:
            break  # 首个从句已经够长，或再拼一个就超长
        title += "，" + clause
    return title[:limit] or "未命名备忘录"


def local_organize(raw_text: str, record_time: str) -> dict:
    """不依赖网络与大模型的规则整理：分类 + 待办抽取 + 相对时间解析。

    待办的时间优先取所在从句里的表达；从句里没有时间时，回退到整段话里解析出的
    那个时间（例如「明天下午三点开会，记得准备方案」——待办继承"明天下午三点"）。
    """
    text = strip_control_words(raw_text)
    base = parse_record_time(record_time) or datetime.now()
    spoken = strip_fillers(text)
    document_due = resolve_cn_time(text, base)
    clauses = [c.strip() for c in re.split(r"[，,。！？!?；;\n]+", spoken) if c.strip()]

    todos = []
    for clause in clauses:
        if any(cue in clause for cue in TODO_CUES):
            todos.append({"task": clause, "due": resolve_cn_time(clause, base) or document_due})

    memo = {
        "title": fallback_title(spoken),
        "category": guess_category(text),
        "summary": spoken[:200] or "（无内容）",
        "todos": todos,
    }
    memo.update(split_spoken_soap(spoken))
    return memo


#: 口语里说出来的 SOAP 栏目名。医生/学生做问诊实训时通常会把栏目念出来
#: （"主诉……现病史……查体……"），所以按这些词切分是可行的。
#: 每个栏目给多个写法，取**最先出现**的那个当分界点。
SPOKEN_SOAP_MARKERS = (
    ("chief_complaint", ("主诉",)),
    ("history", ("现病史", "既往史")),
    ("physical_exam", ("体格检查", "查体", "体查")),
    ("assessment", ("诊断思考", "初步诊断", "诊断")),
    ("plan", ("诊疗计划", "处理意见", "治疗方案")),
)


def split_spoken_soap(text: str) -> dict:
    """把一段"念着栏目名"的口语问诊记录按栏目切开（本地兜底用）。

    这是**没有大模型时的降级方案**，只按栏目名做位置切分，不理解内容：它能把
    "主诉腹痛三天，现病史三天前……"切成两段，但不会把"三天前吃了火锅"归纳成
    "起病诱因"。真要结构化还得靠 DeepSeek —— 有 Key 时这条路径根本不会走到。

    一个栏目名都没提到时返回全空，调用方就会退回摘要视图。
    """
    # 逐栏目找"最先出现的那个写法"：位置 + 该写法的长度（正文要从它后面开始）
    found = []
    for key, markers in SPOKEN_SOAP_MARKERS:
        hits = [(text.find(marker), marker) for marker in markers]
        hits = [(position, marker) for position, marker in hits if position >= 0]
        if hits:
            position, marker = min(hits)
            found.append((position, position + len(marker), key))
    empty = {key: "" for key in SOAP_COLUMNS}
    if not found:
        return empty

    found.sort()
    sections = dict(empty)
    for index, (_, body_start, key) in enumerate(found):
        stop = found[index + 1][0] if index + 1 < len(found) else len(text)
        sections[key] = text[body_start:stop].strip(" \t，,。、;；:")
    return sections


def parse_record_time(value: str) -> Optional[datetime]:
    """解析 ``YYYY-MM-DD HH:MM:SS`` 等常见格式，失败返回 None。"""
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


def organize(
    raw_text: str,
    record_time: str,
    client: Optional[Any] = None,
    model: str = DEFAULT_LLM_MODEL,
    prefer_llm: bool = True,
) -> tuple[dict, str]:
    """整理主入口。返回 ``(memo, source)``，``source`` ∈ {``deepseek``, ``local``}。

    大模型任一环节出错都不抛出，直接降级到本地规则，保证不丢语音内容。
    """
    if prefer_llm and client is not None:
        try:
            data = call_deepseek(client, model, raw_text, record_time)
            return normalize_memo(data, raw_text, record_time), "deepseek"
        except Exception as exc:
            log(f"整理失败（{type(exc).__name__}: {exc}），降级为本地规则整理", "LLM")
    return normalize_memo(local_organize(raw_text, record_time), raw_text, record_time), "local"


# --------------------------------------------------------------------------- #
# 音频预处理：去直流 / 谱减降噪 / 响度归一
#
# 送进 ASR 之前先把音频整干净一点：麦克风增益偏低时 whisper 会漏字，
# 空调声/风扇声/电流声这类稳态底噪又会被当成"有内容"。这一段全部用 numpy 实现，
# 不引入 scipy / noisereduce 之类的额外依赖（也就不用担心版本冲突）。
# --------------------------------------------------------------------------- #


def _rms(samples: Any) -> float:
    """均方根幅度（0~1 量级），静音时趋近 0。"""
    import numpy as np

    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype="float64"))))


def denoise_spectral(samples: Any, sample_rate: int = 16000, frame_ms: float = 32.0,
                     noise_percentile: float = 20.0, over_subtraction: float = 1.2,
                     floor: float = 0.18) -> Any:
    """谱减降噪：估计稳态噪声谱，从每一帧里减掉它。

    做法是最经典的那一套 —— 短时傅里叶 → 取能量最低的若干帧当中位数当作噪声底
    → 从每帧幅度里减掉它 → 保留原相位、逆变换回去。两个参数保证它不会"过度清洁"：

    * ``over_subtraction`` 稍微多减一点，去得更干净；
    * ``floor`` 保留原始幅度的下限比例，避免把清辅音（s/sh/f）也一起抹平 ——
      这类音被抹掉，ASR 反而更容易听错。
    """
    import numpy as np

    samples = np.asarray(samples, dtype="float32")
    frame = max(64, int(sample_rate * frame_ms / 1000))
    if len(samples) < frame * 3:            # 太短就没有"稳态噪声"可言
        return samples
    hop = max(1, frame // 2)
    window = np.hanning(frame).astype("float32")
    frames = np.lib.stride_tricks.sliding_window_view(samples, frame)[::hop]
    spectrum = np.fft.rfft(frames * window, axis=1)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)

    energy = magnitude.sum(axis=1)
    quiet = magnitude[energy <= np.percentile(energy, noise_percentile)]
    if quiet.shape[0] == 0:
        return samples
    noise = np.median(quiet, axis=0)
    cleaned = np.maximum(magnitude - over_subtraction * noise, floor * magnitude)

    restored = np.fft.irfft(cleaned * np.exp(1j * phase), n=frame, axis=1).astype("float32")
    restored *= window
    out = np.zeros(len(samples) + frame, dtype="float32")
    weight = np.zeros(len(samples) + frame, dtype="float32")
    square = window * window
    for index in range(restored.shape[0]):  # 重叠相加；帧数有限，Python 循环够用
        start = index * hop
        out[start:start + frame] += restored[index]
        weight[start:start + frame] += square
    weight[weight < 1e-6] = 1.0
    return (out / weight)[: len(samples)]


def preprocess_audio(audio: Any, cfg: "Config") -> Any:
    """送进 ASR 前统一整一遍：去直流 → 降噪 → 响度归一 → 削峰保护。

    任何一步出意外都只是"少做一步"，绝不抛异常 —— 音频处理失败不该让整条记录丢掉。
    """
    import numpy as np

    if not cfg.audio_preprocess or audio is None or len(audio) == 0:
        return audio
    try:
        samples = np.asarray(audio, dtype="float32")
        if not np.all(np.isfinite(samples)):
            samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
        samples = samples - float(samples.mean())            # 去直流偏置

        if cfg.audio_denoise:
            samples = denoise_spectral(samples, cfg.sample_rate)

        rms = _rms(samples)
        if rms > 1e-5:                                       # 纯数字静音就别放大了
            target = 10.0 ** (cfg.audio_target_dbfs / 20.0)
            gain = min(target / rms, 10.0 ** (cfg.audio_max_gain_db / 20.0))
            # **只往上补，绝不往下压**：本来就录得够响的音频，压到目标响度纯属白折腾，
            # 反而多一次改动信号的机会。实测 A/B 里，增益对识别结果的影响只在
            # "声音偏小"时才是正的。
            gain = max(gain, 1.0)
            samples = samples * gain
        peak = float(np.abs(samples).max()) if len(samples) else 0.0
        if peak > 0.99:                                      # 放大后可能削顶，等比压回来
            samples = samples * (0.99 / peak)
        return np.ascontiguousarray(samples, dtype="float32")
    except Exception as exc:                                 # noqa: BLE001 - 兜底，绝不中断录音
        log(f"音频预处理跳过（{type(exc).__name__}: {exc}）", "ASR")
        return audio


# --------------------------------------------------------------------------- #
# 本地语音识别：FunASR/SenseVoiceSmall（中文专用）或 faster-whisper
# --------------------------------------------------------------------------- #

_FUNASR_TAG_RE = re.compile(r"<\|[^|]*\|>")


def clean_funasr_text(text: Optional[str]) -> str:
    """清掉 SenseVoice 输出里的富文本标记（``<|zh|><|NEUTRAL|><|Speech|>`` 之类）。

    装了 funasr 就用它官方的 ``rich_transcription_postprocess``（顺带处理表情/事件
    标记）；没装（比如离线自检环境）就退回正则剥标签，效果对本项目够用。
    """
    raw = text or ""
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
    except Exception:
        return _FUNASR_TAG_RE.sub("", raw).strip()
    try:
        return (rich_transcription_postprocess(raw) or "").strip()
    except Exception:
        return _FUNASR_TAG_RE.sub("", raw).strip()


class Transcriber:
    """语音识别封装，支持两个后端，模型都是首次调用时才惰性加载。

    * ``funasr``  —— FunASR 的 SenseVoiceSmall：**中文专用**、非自回归，CPU 上又快又准，
      还能直接输出标点和逆文本规整（"三点"→"3点"）。装了就用它。
    * ``whisper`` —— faster-whisper：通用多语言，纯 CPU 上用 base 档。

    ``ASR_BACKEND=auto``（默认）时：装了 funasr 就优先 SenseVoice，否则退回 whisper，
    所以"没装新依赖"的老环境行为完全不变。
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None
        self._scan_model = None          # 唤醒词/指令词扫描用的轻量模型
        self._funasr = None
        self.backend = ""                # 首次加载后确定："funasr" / "whisper"
        # ASR 同一时刻只跑一次：关键词扫描在后台线程里跑，正式转写在主循环里跑，
        # 两者若并发，底层推理库会互相拖慢甚至抢占内部 worker，反而更容易卡。
        self._lock = threading.Lock()

    # -- 后端选择 ---------------------------------------------------------- #
    @staticmethod
    def funasr_available() -> bool:
        """funasr 是否已安装（装了才能用 SenseVoice）。"""
        import importlib.util

        return importlib.util.find_spec("funasr") is not None

    @staticmethod
    def whisper_available() -> bool:
        """faster-whisper 是否已安装。"""
        import importlib.util

        return importlib.util.find_spec("faster_whisper") is not None

    @classmethod
    def available(cls) -> bool:
        """任一后端可用即可工作。"""
        return cls.funasr_available() or cls.whisper_available()

    def resolve_backend(self) -> str:
        """按配置挑后端：auto 优先中文专用的 SenseVoice，装了才用。"""
        wanted = (self.cfg.asr_backend or "auto").strip().lower()
        if wanted in ("funasr", "sensevoice", "sense_voice"):
            return "funasr"
        if wanted in ("whisper", "faster-whisper", "faster_whisper"):
            return "whisper"
        return "funasr" if self.funasr_available() else "whisper"

    def _backend(self) -> str:
        if not self.backend:
            self.backend = self.resolve_backend()
        return self.backend

    def describe(self) -> str:
        """给 --doctor / 启动日志用的一句话描述。"""
        backend = self._backend()
        if backend == "funasr":
            return f"FunASR {self.cfg.asr_funasr_model}（中文专用 · SenseVoice）"
        return (f"faster-whisper {self.cfg.asr_model}"
                f"（device={self.cfg.asr_device}, compute={self.cfg.asr_compute_type}）")

    def describe_hotwords(self) -> str:
        """热词的两条生效路径各是什么状态（--doctor 逐项显示）。"""
        words = getattr(self.cfg, "asr_hotwords", ()) or ()
        if not words:
            return "医学热词 = 关（ASR_HOTWORDS=off）"
        prompt = "开" if build_asr_prompt(self.cfg) else "关"
        fix = "开" if getattr(self.cfg, "asr_hotword_correct", True) else "关"
        # SenseVoice 没有 initial_prompt 这种参数，提示词这一路对它无效 —— 说清楚，
        # 免得用户以为"配了热词识别就该变准"，然后对着没变化的结果发懵。
        prompt_backend = "（仅 faster-whisper 生效）" if self._backend() == "funasr" else ""
        return (f"医学热词 = {len(words)} 个 · 解码期提示词 {prompt}{prompt_backend}"
                f" · 识别后校正 {fix}")

    def preload(self) -> bool:
        """提前加载模型（首次会触发权重下载），让"已就绪"的提示名副其实。

        失败不抛出：首次说话时会自动重试，避免一次网络抖动就劝退整个会话。
        """
        try:
            self._ensure_model()
            return True
        except Exception as exc:
            log(f"语音识别模型预加载失败：{type(exc).__name__}: {exc}", "ASR")
            return False

    # -- FunASR / SenseVoice ----------------------------------------------- #
    def _ensure_funasr(self):
        if self._funasr is None:
            from funasr import AutoModel

            device, _compute = self._resolve_device()
            log(f"加载 {self.describe()}（device={device}，首次运行会自动下载权重）…", "ASR")
            kwargs = dict(model=self.cfg.asr_funasr_model, trust_remote_code=True,
                          device="cuda:0" if device == "cuda" else "cpu")
            try:
                self._funasr = AutoModel(disable_update=True, disable_pbar=True,
                                         disable_log=True, **kwargs)
            except TypeError:
                # 老版本 funasr 不认这几个开关，去掉重试
                self._funasr = AutoModel(**kwargs)
        return self._funasr

    def _transcribe_funasr(self, audio: Any, quick: bool = False) -> str:
        model = self._ensure_funasr()
        with self._lock:
            result = model.generate(
                input=audio,
                cache={},
                language=self.cfg.asr_language or "auto",
                use_itn=True,               # 逆文本规整："三点"→"3点"、"一百"→"100"
                batch_size_s=60,
                merge_vad=False,            # 上游 VAD 已经切好句子了
            )
        if not result:
            return ""
        first = result[0]
        text = first.get("text") if isinstance(first, dict) else str(first)
        return clean_funasr_text(text)

    # -- faster-whisper ---------------------------------------------------- #
    def _ensure_scan_model(self):
        """关键词扫描专用的小模型（默认 tiny）。

        扫描每 0.6 秒就要跑一次，用 base 会把 CPU 吃满；tiny 的识别质量对这种
        "只要听出有没有那几个词"的任务完全够用。若配置里 wake_model 与 asr_model
        相同，直接复用主模型，省一份内存。
        """
        if self._scan_model is None:
            if self.cfg.wake_model == self.cfg.asr_model:
                self._scan_model = self._ensure_model()
            else:
                from faster_whisper import WhisperModel

                device, compute_type = self._resolve_device()
                log(f"加载关键词扫描模型 {self.cfg.wake_model!r}"
                    f"（device={device}, compute={compute_type}）…", "唤醒")
                self._scan_model = WhisperModel(self.cfg.wake_model,
                                                device=device, compute_type=compute_type)
        return self._scan_model

    def _resolve_device(self) -> tuple:
        device = self.cfg.asr_device
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        compute_type = self.cfg.asr_compute_type
        if device == "cpu" and compute_type in ("float16", "int8_float16"):
            compute_type = "int8"  # CPU 不支持 fp16 推理
        return device, compute_type

    def scan(self, audio: Any) -> str:
        """快速转写一小段音频，只服务于关键词判定。

        SenseVoice 本身就够快，扫描直接复用它（同一个模型，省一份内存）；
        faster-whisper 则换用 tiny，避免每 0.6 秒跑一次 base 把 CPU 吃满。
        """
        audio = preprocess_audio(audio, self.cfg)
        if self._backend() == "funasr":
            return self._transcribe_funasr(audio, quick=True)
        model = self._ensure_scan_model()
        with self._lock:
            segments, _info = model.transcribe(
                audio,
                language=self.cfg.asr_language or None,
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            return "".join(segment.text for segment in segments).strip()

    def _ensure_model(self):
        if self._backend() == "funasr":
            return self._ensure_funasr()
        if self._model is None:
            from faster_whisper import WhisperModel

            device, compute_type = self._resolve_device()
            log(f"加载 faster-whisper 模型 {self.cfg.asr_model!r}"
                f"（device={device}, compute={compute_type}）…", "ASR")
            self._model = WhisperModel(self.cfg.asr_model, device=device, compute_type=compute_type)
        return self._model

    def transcribe(self, audio: Any, beam_size: Optional[int] = None, quick: bool = False) -> str:
        """把 float32 单声道 numpy 波形转成文本。

        进模型之前先过一遍 ``preprocess_audio``（去直流 / 降噪 / 响度归一）——
        麦克风音量偏小时这一步对识别率的帮助最明显。

        ``quick=True`` 用于"提前截断"的快速探测：单束搜索、关闭 VAD 二次过滤，
        以最低延迟换取可用性。

        整个转写过程持有 ``self._lock``，保证后台探测与正式转写不会同时占用模型。
        """
        audio = preprocess_audio(audio, self.cfg)
        if self._backend() == "funasr":
            return self._fix_terms(self._transcribe_funasr(audio, quick=quick))
        model = self._ensure_model()
        with self._lock:
            segments, _info = model.transcribe(
                audio,
                language=self.cfg.asr_language or None,
                beam_size=1 if quick else (beam_size or self.cfg.asr_beam_size),
                vad_filter=False,
                condition_on_previous_text=False,
                # 医学热词当提示词喂给解码器：让它在"心前区/心钱去"之间优先选前者。
                # 传空串等价于不传（faster-whisper 内部按 falsy 处理）。
                initial_prompt=build_asr_prompt(self.cfg),
            )
            text = "".join(segment.text for segment in segments).strip()
        return self._fix_terms(text)

    def _fix_terms(self, text: str) -> str:
        """识别后校正的开关包装：关掉时原样返回，不做任何改动。"""
        if not getattr(self.cfg, "asr_hotword_correct", True):
            return text
        return correct_medical_terms(text, self.cfg.asr_hotwords)


# --------------------------------------------------------------------------- #
# 免唤醒监听（silero-vad）
# --------------------------------------------------------------------------- #


@dataclass
class Utterance:
    """一句话的录音结果。"""

    audio: Any                    # np.ndarray, float32, 单声道
    started_at: datetime          # 真正的开口时刻（用于 record_time）
    ended_at: datetime
    duration: float               # 秒
    reason: str = ""              # 截断原因：静音断句 / 检测到结束词 / 超过最长录音时长


def build_vad() -> Any:
    """加载 silero-vad 模型（首次运行会自动下载权重，约 2MB）。"""
    from silero_vad import load_silero_vad

    log("加载 silero-vad 模型（首次运行会自动下载权重）…", "VAD")
    return load_silero_vad()


def build_vad_iterator(vad: Any, cfg: Config) -> Any:
    """构造流式 VAD 迭代器，兼容新旧版 silero-vad 的参数命名。"""
    from silero_vad import VADIterator

    kwargs = dict(
        threshold=cfg.vad_threshold,
        sampling_rate=cfg.sample_rate,
        min_silence_duration_ms=cfg.silence_ms,
        speech_pad_ms=cfg.pre_speech_pad_ms,
    )
    try:
        return VADIterator(vad, **kwargs)
    except TypeError:
        # 兼容 silero-vad < 5 的命名（silence_duration_ms）
        kwargs.pop("min_silence_duration_ms", None)
        kwargs["silence_duration_ms"] = cfg.silence_ms
        return VADIterator(vad, **kwargs)


def _chunks_duration(chunks: list, sample_rate: int) -> float:
    return sum(len(chunk) for chunk in chunks) / float(sample_rate)


class VoiceListener:
    """常驻麦克风监听：唤醒词待机 → 录音 → 语音指令 → 落库 的状态机。

    采集在独立线程完成并打上时间戳进队列；主循环只在队列上跑 VAD 状态机与
    唤醒/指令判定，因此 ASR/LLM 处理耗时不会污染 ``record_time``。

    ``wake_word_enabled=False`` 时退化为旧行为：不需要唤醒词，听到人声就录，
    静音或结束词断句。两种模式共用同一套状态机和关键词扫描。

    关键设计：**所有 Whisper 调用都离开主循环**。唤醒扫描和指令扫描都在后台线程里
    跑，主循环只做 VAD 和事件判定 —— 否则一次几百毫秒的转写就会让队列积压、
    丢块，最后连静音断句都一起失灵。
    """

    def __init__(
        self,
        cfg: Config,
        vad: Any,
        transcriber: Optional[Transcriber] = None,
        llm_label: str = "DeepSeek ",
        on_status: Optional[Callable[[str, str], None]] = None,
        on_cancel: Optional[Callable[[str, str], None]] = None,
    ):
        self.cfg = cfg
        self.vad = vad
        self.transcriber = transcriber
        self.llm_label = llm_label  # 只影响"整理中"提示的文字
        self.on_status = on_status  # (state, label) → 推给 Web 界面
        self.on_cancel = on_cancel  # (命中的作废词, 识别原文) → 推给 Web 界面
        self.cancelled = 0          # 本次运行作废掉的段数
        maxsize = max(4, int(cfg.queue_seconds * cfg.sample_rate / cfg.chunk_size))
        self._queue: "queue.Queue[tuple[float, datetime, Any]]" = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._dropped = 0
        self._skipped = 0              # 处理期间主动跳过的块（正常现象，不是积压）
        # 处理中标志：置位期间采集线程只把设备缓冲读走、不入队，队列因此永不积压
        self._processing = threading.Event()
        # 音频出现断点（真的丢过块）→ 主循环作废当前这段，避免 VAD 永远等不到静音
        self._hole = threading.Event()
        # 关键词扫描：后台线程 + 结果槽；epoch 用来丢弃"上一段录音迟到的判定"
        self._scan: Optional[threading.Thread] = None
        self._scan_result: Optional[tuple] = None      # (group, 命中的词, 识别文本)
        self._scan_epoch = 0
        # 收到退出指令（"退出录制"）→ 主循环收工
        self.exit_requested = threading.Event()
        # 网页上的两个按钮：主循环每一轮开头取一次，取完即清
        self._finish_request = threading.Event()
        self._cancel_request = threading.Event()
        self.state = STATE_STANDBY
        # 真麦克风需要"读走再丢"防止 PortAudio 溢出；注入流则原地停住
        self._drain_while_paused = True

    # -- 状态广播 ---------------------------------------------------------- #
    def _set_state(self, state: str, detail: str = "") -> None:
        self.state = state
        if self.on_status is not None:
            try:
                self.on_status(state, detail)
            except Exception:
                pass          # Web 界面出问题绝不能拖垮录音主循环

    # -- 采集线程 ---------------------------------------------------------- #
    def _reader(self, stream: Any) -> None:
        while not self._stop.is_set():
            if self._processing.is_set():
                # 上一条还在转写/整理 —— 暂停采集。两种流处理方式不同：
                #   · 真麦克风（_drain_while_paused）：必须继续把设备缓冲读走，
                #     否则 PortAudio 立刻报输入溢出。读到就丢，不入队。
                #   · 注入流（--replay / 自检）：干脆不读，音频原地停住，
                #     处理完从断点接着读，测试因此完全可复现。
                # 相比 stop()/start() 重开麦克风，这样没有重开延迟，也不会在
                # Windows 上偶发打不开设备。
                if not self._drain_while_paused:
                    time.sleep(0.02)
                    continue
            try:
                chunk, overflowed = stream.read(self.cfg.chunk_size)
            except Exception as exc:  # 流被关闭 / 设备拔出
                if not self._stop.is_set():
                    log(f"音频读取中断：{exc}", "采集")
                return
            if overflowed:
                log("输入缓冲区溢出，部分音频可能丢失", "采集")

            if self._processing.is_set():
                self._skipped += 1
                continue

            item = (time.monotonic(), datetime.now(), chunk[:, 0].copy())
            try:
                self._queue.put(item, timeout=0.5)
            except queue.Full:
                # 兜底：真到了这一步说明消费端卡住了。丢最旧的块，
                # 同时置位 _hole —— 音频已经不连续，VAD 的静音计时不可信了。
                self._dropped += 1
                self._hole.set()
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(item)
                except (queue.Empty, queue.Full):
                    pass

    # -- 断句状态机 -------------------------------------------------------- #
    def run(
        self,
        on_utterance: Callable[[Utterance], None],
        stream_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        """阻塞式监听；每识别到一句完整的话就回调一次 ``on_utterance``。

        ``stream_factory`` 是测试注入点：默认打开真实麦克风，也可以传入一个返回
        "带 ``read(n) -> (frames, overflowed)`` 的对象"的工厂函数，
        用离线音频回放来验证整条断句链路。
        """
        import torch

        cfg = self.cfg
        self._drain_while_paused = stream_factory is None   # 真麦克风 vs 注入流
        iterator = build_vad_iterator(self.vad, cfg)
        prebuffer: "collections.deque" = collections.deque(
            maxlen=max(1, cfg.pre_speech_pad_ms * cfg.sample_rate // 1000 // cfg.chunk_size)
        )

        if stream_factory is None:
            import sounddevice as sd

            def stream_factory() -> Any:
                return sd.InputStream(
                    samplerate=cfg.sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=cfg.chunk_size,
                )

        try:
            stream_ctx = stream_factory()
            stream = stream_ctx.__enter__()
        except Exception as exc:
            log(f"无法打开麦克风：{exc}", "采集")
            log("提示：检查系统麦克风权限，或先用 `python main.py --text \"...\"` 验证文字链路", "采集")
            return

        wake_mode = bool(cfg.wake_word_enabled and cfg.wake_word)
        try:
            reader = threading.Thread(target=self._reader, args=(stream,), name="audio-reader", daemon=True)
            reader.start()

            # 待机时滚动保留最近一段音频，用来找唤醒词；免唤醒模式只需回溯缓冲那么长
            window_s = cfg.wake_window_s if wake_mode else cfg.pre_speech_pad_ms / 1000.0
            rolling: "collections.deque" = collections.deque(
                maxlen=max(1, int(window_s * cfg.sample_rate / cfg.chunk_size))
            )
            tail_chunks = max(1, int(cfg.wake_window_s * cfg.sample_rate / cfg.chunk_size))
            buffer: list = []
            started_at: Optional[datetime] = None
            pending_speech_start: Optional[datetime] = None
            last_scan_at = 0.0
            speech_active = False

            self._set_state(STATE_STANDBY)
            if wake_mode:
                announce_standby(cfg)
            else:
                announce_listening(cfg, first=True)

            def back_to_idle() -> None:
                """一段（或一次尝试）结束：清缓冲与 VAD 状态，回到待机等下一个唤醒词。"""
                nonlocal buffer, started_at, pending_speech_start, speech_active, last_scan_at
                self._scan_epoch += 1            # 让还在飞的判定结果作废
                self._scan_result = None
                buffer = []
                started_at = None
                pending_speech_start = None
                speech_active = False
                last_scan_at = 0.0
                rolling.clear()
                iterator.reset_states()
                self._set_state(STATE_STANDBY)
                if wake_mode:
                    announce_standby(cfg)
                else:
                    announce_listening(cfg)

            def begin_recording(at: datetime) -> None:
                """唤醒成功 / 免唤醒模式听到人声 → 进入录音。"""
                nonlocal buffer, started_at, last_scan_at
                started_at = at
                buffer = list(rolling)           # 唤醒词那段也在里面，稍后从文本里剥掉
                self._scan_epoch += 1
                self._scan_result = None
                last_scan_at = 0.0               # 立刻允许一次指令扫描
                self._set_state(STATE_RECORDING, announce_recording(cfg, wake_mode=wake_mode))

            while not self._stop.is_set():
                if self._hole.is_set():
                    # 音频真的断过（丢过块）→ VAD 内部的"静音累计"已经失真，再等下去
                    # 永远等不到 end，表现就是"说了半天不保存"。直接作废本段重来。
                    self._hole.clear()
                    if started_at is not None:
                        log("处理延迟导致音频出现断点，本段作废，重新等待", "采集")
                    back_to_idle()

                # ---- 取一次后台扫描的结果（唤醒词 / 语音指令）----
                hit = self._take_scan_result()
                if hit is not None:
                    group, keyword, text = hit
                    if group == "exit":
                        log(f"听到退出指令「{keyword}」→ {text!r}，保存后退出", "指令")
                        if started_at is not None:
                            self._emit(buffer, started_at, datetime.now(), on_utterance,
                                       "检测到退出指令")
                        self.exit_requested.set()
                        self._stop.set()
                        break
                    if group == "cancel":
                        if started_at is not None:
                            # 丢弃：不 _emit → 不转写、不调 DeepSeek、不入库、不写 Markdown
                            self._cancel_recording(keyword, text)
                            back_to_idle()
                    elif group == "wake":
                        if started_at is None:
                            log(f"唤醒词「{keyword}」已识别 → {text!r}", "唤醒")
                            begin_recording(pending_speech_start or datetime.now())
                    elif group == "segment_end" and started_at is not None:
                        log(f"听到分段结束词「{keyword}」→ {text!r}", "指令")
                        self._emit(buffer, started_at, datetime.now(), on_utterance,
                                   "检测到分段结束词")
                        back_to_idle()
                    elif group == "stop" and started_at is not None:
                        # 免唤醒模式下的老式"结束词"提前截断
                        self._emit(buffer, started_at, datetime.now(), on_utterance,
                                   "检测到结束词")
                        back_to_idle()

                # ---- 网页按钮：手动结束 / 放弃本条（双保险，说错话时不用靠嘴救场）----
                manual = self._take_manual()
                if manual == "finish" and started_at is not None:
                    log("收到网页「结束并整理」→ 立即截断这段录音", "指令")
                    self._emit(buffer, started_at, datetime.now(), on_utterance,
                               "网页手动结束")
                    back_to_idle()
                elif manual == "cancel" and started_at is not None:
                    log("收到网页「放弃本条」→ 直接丢弃这段录音", "指令")
                    self._cancel_recording("放弃本条", "（在网页上点了按钮）")
                    back_to_idle()

                try:
                    _ts, captured_at, chunk = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                event = iterator(torch.from_numpy(chunk)) or {}

                # ---- 待机：滚动留声；听到人声才开始找唤醒词 ----
                if started_at is None:
                    rolling.append(chunk)
                    if "start" in event:
                        speech_active = True
                        pending_speech_start = captured_at   # 开口时刻，唤醒后拿它当 record_time
                        if not wake_mode:
                            # 免唤醒模式：听到人声直接开录（rolling 就是回溯缓冲）
                            begin_recording(pending_speech_start)
                    if "end" in event:
                        speech_active = False
                    if started_at is None and wake_mode:
                        # 说话途中反复扫；刚说完再补扫一次（唤醒词的最后一班车）
                        if (speech_active and self._scan_due(last_scan_at, cfg)) or "end" in event:
                            if self._start_scan(rolling, ("exit", "wake")):
                                last_scan_at = time.monotonic()
                    continue

                # ---- 录音中 ----
                buffer.append(chunk)
                duration = _chunks_duration(buffer, cfg.sample_rate)

                if "end" in event:
                    if duration * 1000 < cfg.min_segment_ms:
                        # 刚唤醒就安静下来（"小医小医"之后停了半天）—— 还没开始说事。
                        # 不当作一段内容，继续等，免得存下一堆只有唤醒词的空记录。
                        log(f"唤醒后静音 {duration:.2f}s，还没听到内容，继续等", "VAD")
                        continue
                    self._emit(buffer, started_at, captured_at, on_utterance, "静音断句")
                    back_to_idle()
                    continue

                if duration >= cfg.max_record_s:
                    self._emit(buffer, started_at, captured_at, on_utterance, "超过最长录音时长")
                    back_to_idle()
                    continue

                # ---- 录音途中扫描结束词/退出词（后台线程，不阻塞这个循环）----
                if self._scan_due(last_scan_at, cfg):
                    if self._start_scan(buffer[-tail_chunks:], self._recording_scan_groups(wake_mode)):
                        last_scan_at = time.monotonic()
        finally:
            self._stop.set()
            try:
                stream_ctx.__exit__(None, None, None)
            except Exception:
                pass
            if self._dropped:
                log(f"本次运行共丢弃 {self._dropped} 个音频块（消费端确实跟不上，受影响的片段已作废重来）", "采集")
            if self._skipped:
                skipped_s = self._skipped * self.cfg.chunk_size / float(self.cfg.sample_rate)
                log(f"处理期间暂停采集，跳过 {self._skipped} 个音频块（{skipped_s:.1f}s）"
                    f"—— 属正常现象，没有积压也没有丢音", "采集")

    # -- 后台指令扫描 ------------------------------------------------------- #
    def _recording_scan_groups(self, wake_mode: bool) -> tuple:
        """录音途中要盯着哪几类指令。

        分段结束词、作废词和退出词是唤醒词模式的核心交互，必须常开；代价只是每隔
        ``wake_scan_interval_s`` 秒对**尾部**几秒音频跑一次 tiny 转写，很轻。
        免唤醒模式没有唤醒词，但老式的"结束/完毕"提前截断按 ``early_stop_keyword``
        开关决定要不要一起扫。
        """
        if wake_mode:
            return ("exit", "cancel", "segment_end")
        if self.cfg.early_stop_keyword:
            return ("exit", "cancel", "segment_end", "stop")
        return ("exit", "cancel")

    def _scan_busy(self) -> bool:
        return self._scan is not None and self._scan.is_alive()

    def _scan_due(self, last_scan_at: float, cfg: "Config") -> bool:
        """到点且上一轮已经跑完 —— 扫描期间不排队，永远只有一次在飞。"""
        if self._scan_busy():
            return False
        return (time.monotonic() - last_scan_at) >= cfg.wake_scan_interval_s

    def _start_scan(self, chunks: Any, groups: tuple) -> bool:
        """后台线程里跑一次快速转写，命中控制词就放进 ``_scan_result``；起了线程返回 True。

        必须离开主循环：Whisper 再快也要几百毫秒，在主循环里跑会卡住 VAD 状态机
        → 队列积压 → 丢块 → 静音断句和指令判定一起失效（这正是老版本的病根）。
        这里只做一次拷贝 + 起线程，主循环立刻回去喂 VAD。
        """
        chunks = [c for c in chunks if c is not None]
        if not chunks or not groups or self._scan_busy():
            return False
        import numpy as np

        audio = np.concatenate(chunks).astype("float32", copy=False)
        epoch = self._scan_epoch

        def worker() -> None:
            try:
                text = self.transcriber.scan(audio)
            except Exception as exc:
                log(f"指令扫描跳过：{type(exc).__name__}: {exc}", "ASR")
                return
            # 这段录音可能已经结束、用户都开始说下一句了 —— 迟到的结果必须丢掉
            if epoch != self._scan_epoch:
                return
            hit = match_command_detail(text, self.cfg, groups)
            if hit is not None:
                self._scan_result = (hit[0], hit[1], text)

        self._scan = threading.Thread(target=worker, name="cmd-scan", daemon=True)
        self._scan.start()
        return True

    def _take_scan_result(self) -> Optional[tuple]:
        """取走后台扫描的结果（取完即清，同一句不会触发两次）。"""
        hit, self._scan_result = self._scan_result, None
        return hit

    # -- 网页按钮的手动控制（"说错了"时的双保险）---------------------------- #
    def request_finish(self) -> bool:
        """请求立刻截断当前录音并送去整理。返回 False 表示当下没在录音。

        只置一个标志位，真正干活的是主循环 —— 这样音频缓冲、VAD 状态、
        扫描 epoch 全都在同一个线程里收尾，不会出现半截状态。
        """
        if self.state != STATE_RECORDING:
            return False
        self._finish_request.set()
        return True

    def request_cancel(self) -> bool:
        """请求丢弃当前这段录音（不转写、不整理、不入库）。没在录音时返回 False。"""
        if self.state != STATE_RECORDING:
            return False
        self._cancel_request.set()
        return True

    def _take_manual(self) -> Optional[str]:
        """取走网页上按下的一次手动操作（取完即清，两次点击不会都生效）。"""
        if self._finish_request.is_set():
            self._finish_request.clear()
            self._cancel_request.clear()      # 同时按下时以"结束"为准
            return "finish"
        if self._cancel_request.is_set():
            self._cancel_request.clear()
            return "cancel"
        return None

    def _cancel_recording(self, keyword: str, text: str) -> None:
        """作废本条：音频到这里为止，往后一步都不走。

        与"保存"的区别就在**不调用 ``_emit``** —— 于是不转写、不调 DeepSeek、
        不写 SQLite、不追加 ``my_memos.md``，那几秒音频随缓冲区一起被丢掉。
        """
        self.cancelled += 1
        log(f"听到作废指令「{keyword}」→ {text!r}：本条已丢弃，"
            f"未调用 DeepSeek、未写入数据库", "指令")
        print(f"\n {ICON_CANCELLED} [本条已作废，未保存] 这段录音没有离开过本机内存",
              flush=True)
        if self.on_cancel is not None:
            try:
                self.on_cancel(keyword, text)
            except Exception:
                pass          # Web 界面出问题绝不能拖垮录音主循环

    def _emit(
        self,
        buffer: list,
        started_at: datetime,
        ended_at: datetime,
        on_utterance: Callable[[Utterance], None],
        reason: str,
    ) -> None:
        import numpy as np

        if not buffer:
            return
        audio = np.concatenate(buffer).astype("float32", copy=False)
        duration = len(audio) / float(self.cfg.sample_rate)
        if duration * 1000 < self.cfg.min_speech_ms:
            log(f"片段过短（{duration:.2f}s），按噪声丢弃", "VAD")
            return
        utterance = Utterance(audio=audio, started_at=started_at, ended_at=ended_at,
                              duration=duration, reason=reason)
        log(f"断句完成 · {duration:.2f}s · {reason}", "VAD")
        announce_thinking(self.llm_label)
        self._set_state(STATE_PROCESSING, f"录音结束（{reason}），正在整理")

        # 转写 + 整理要好几秒，这期间**暂停入队**：采集线程仍然把设备缓冲读走
        # （防止 PortAudio 溢出报警），但不再往队列里塞。队列因此不会积压，
        # 也就不会为了腾地方丢块、把 VAD 的静音计时打乱。
        self._processing.set()
        try:
            on_utterance(utterance)
        except Exception as exc:
            log(f"处理录音时出错：{type(exc).__name__}: {exc}", "错误")
        finally:
            self._discard_pending_queue()
            self._processing.clear()
        # 连续记录：状态机回到待机/倾听，下一句无需重启程序

    def _discard_pending_queue(self) -> None:
        """清掉队列里的残留块，让下一段录音从干净的状态开始。"""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def stop(self) -> None:
        self._stop.set()


# --------------------------------------------------------------------------- #
# 主流程编排
# --------------------------------------------------------------------------- #


class Broadcaster:
    """极简发布/订阅：状态变化和新卡片推给所有 Web 连接。

    只用 threading + queue 实现，**不 import Flask** —— 这样自检脚本不用装 Flask
    就能验证整条推送链路，Web 层（webui.py）只是它的一个消费者。
    """

    def __init__(self, maxsize: int = 64):
        self._lock = threading.Lock()
        self._subscribers: list = []
        self._maxsize = maxsize

    def subscribe(self) -> "queue.Queue[dict]":
        box: "queue.Queue[dict]" = queue.Queue(maxsize=self._maxsize)
        with self._lock:
            self._subscribers.append(box)
        return box

    def unsubscribe(self, box: "queue.Queue[dict]") -> None:
        with self._lock:
            if box in self._subscribers:
                self._subscribers.remove(box)

    def publish(self, event: dict) -> None:
        with self._lock:
            boxes = list(self._subscribers)
        for box in boxes:
            try:
                box.put_nowait(event)
            except queue.Full:
                # 某个页面卡住了就丢它的事件，绝不让 Web 拖累录音主循环
                pass

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


class AppState:
    """跨线程共享的运行状态（当前处于哪个状态 + 本次运行的统计）。

    ``VoiceListener`` 的 ``on_status`` 和 ``MemoPipeline`` 的 ``on_memo`` 都会写到这里，
    Web 界面通过 ``broadcaster`` 实时收到，不需要轮询。
    """

    def __init__(self, wake_mode: bool = True, wake_word: str = DEFAULT_WAKE_WORD):
        self._lock = threading.Lock()
        self.broadcaster = Broadcaster()
        self.wake_mode = bool(wake_mode)
        self.wake_word = wake_word or DEFAULT_WAKE_WORD
        self.state = STATE_STANDBY
        self.detail = ""
        self.updated_at = _timestamp()
        self.started_at = _timestamp()
        self.saved_count = 0
        self.cancelled_count = 0
        self.last_memo_id: Optional[int] = None

    # -- 读 ---------------------------------------------------------------- #
    def label(self) -> str:
        if self.state == STATE_STANDBY and not self.wake_mode:
            return STATE_STANDBY_NO_WAKE
        return STATE_LABELS.get(self.state, STATE_LABELS[STATE_STANDBY])

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "label": self.label(),
                "detail": self.detail,
                "updated_at": self.updated_at,
                "started_at": self.started_at,
                "saved_count": self.saved_count,
                "cancelled_count": self.cancelled_count,
                "last_memo_id": self.last_memo_id,
                "wake_mode": self.wake_mode,
                "wake_word": self.wake_word,
            }

    # -- 写 ---------------------------------------------------------------- #
    def set_state(self, state: str, detail: str = "") -> None:
        with self._lock:
            self.state = state
            self.detail = detail
            self.updated_at = _timestamp()
        event = {"type": "state"}
        event.update(self.snapshot())
        self.broadcaster.publish(event)

    def published_state(self) -> dict:
        """当前状态事件的副本（给刚连上来的页面补一份，免得空等）。"""
        event = {"type": "state"}
        event.update(self.snapshot())
        return event

    def publish_memo(self, memo: dict) -> None:
        with self._lock:
            self.saved_count += 1
            self.last_memo_id = memo.get("id")
        self.broadcaster.publish({"type": "memo", "memo": memo_to_dict(memo)})
        self.set_state(self.state, self.detail)     # 让页面上的统计数字立刻刷新

    def publish_cancel(self, keyword: str = "", text: str = "") -> None:
        """语音作废了一条录音 —— 页面弹个提示，并更新"作废"计数。"""
        with self._lock:
            self.cancelled_count += 1
        self.broadcaster.publish({
            "type": "cancel", "keyword": keyword, "text": text,
            "cancelled_count": self.cancelled_count,
        })
        self.set_state(self.state, self.detail)

    def publish_update(self, memo: dict) -> None:
        """某张卡片被改过了（网页上手工修改 / 重新润色）—— 广播新内容。

        刻意**不动** ``saved_count``：改一条不是存一条，计数一涨页面上那句
        "已保存 N 条"就骗人了。新卡片才走 ``publish_memo``。
        """
        self.broadcaster.publish({"type": "memo", "memo": memo_to_dict(memo)})

    def publish_delete(self, memo_id: Optional[int]) -> None:
        """删掉了一张卡片 —— 广播出去让所有打开的页面同步，并把统计减一。"""
        with self._lock:
            self.saved_count = max(0, self.saved_count - 1)
            if self.last_memo_id == memo_id:
                self.last_memo_id = None
        self.broadcaster.publish({"type": "delete", "id": memo_id})
        self.set_state(self.state, self.detail)


def memo_to_dict(memo: dict) -> dict:
    """把一条备忘录规整成可 JSON 序列化的纯字典（顺带补上 todos 的解析）。"""
    todos = memo.get("todos")
    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except (TypeError, ValueError):
            todos = []
    if not isinstance(todos, list):
        todos = []
    payload = {
        "id": memo.get("id"),
        "record_time": memo.get("record_time") or "",
        "raw_text": memo.get("raw_text") or "",
        "title": memo.get("title") or "",
        "category": memo.get("category") or "",
        "summary": memo.get("summary") or "",
        "todos": todos,
    }
    for key in SOAP_COLUMNS:
        payload[key] = memo.get(key) or ""
    payload["has_soap"] = has_soap(memo)
    # 前端按 SOAP_SECTIONS 的顺序渲染病历卡片，所以顺序也一起下发 ——
    # 免得"添加一段 SOAP"要在 JS 里再抄一份字段表，两边迟早对不上。
    payload["soap_sections"] = [{"key": key, "label": label} for key, label in SOAP_SECTIONS]
    return payload


class MemoPipeline:
    """录音 → 转写 → 整理 → 落库。"""

    def __init__(self, cfg: Config, conn: sqlite3.Connection,
                 transcriber: Transcriber, client: Optional[Any] = None,
                 on_memo: Optional[Callable[[dict], None]] = None):
        self.cfg = cfg
        self.conn = conn
        self.transcriber = transcriber
        self.client = client
        self.on_memo = on_memo          # 新卡片 → 推给 Web 界面

    def handle_audio(self, utterance: Utterance) -> Optional[int]:
        log("本地识别中…", "ASR")
        started = time.monotonic()
        raw_text = self.transcriber.transcribe(utterance.audio)
        log(f"识别完成（{time.monotonic() - started:.1f}s）：{raw_text or '（空）'}", "ASR")

        raw_text = clean_transcript(raw_text, self.cfg)
        if not raw_text:
            log("识别结果为空（或只剩指令词），跳过", "ASR")
            return None
        return self.handle_text(raw_text, utterance.started_at, cleaned=True)

    def handle_text(self, raw_text: str, recorded_at: Optional[datetime] = None,
                    cleaned: bool = False) -> Optional[int]:
        # 这里是"文字"和"录音"两条路径唯一的汇合点：控制词在这里兜底剥离一次，
        # 保证送进 DeepSeek、写进 raw_text 的都不带"结束/完毕/下一条/退出录制"。
        # （handle_audio 里已经清过一次，那次是为了判断"清完是不是空的"；幂等。）
        if not cleaned:
            raw_text = clean_transcript(raw_text, self.cfg)
        if not raw_text:
            log("内容只剩控制词，跳过", "整理")
            return None
        record_time = (recorded_at or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
        memo, source = organize(raw_text, record_time, self.client, self.cfg.llm_model)
        memo_id = insert_memo(self.conn, memo)
        memo["id"] = memo_id          # 供 Markdown 条目标号与 Web 卡片复用
        print_memo(memo, memo_id=memo_id, source=source)

        if self.cfg.markdown_path:
            if append_memo_markdown(memo, self.cfg.markdown_path):
                log(f"已追加到 {self.cfg.markdown_path}", "导出")
        beep(self.cfg.beep)
        print(f" {ICON_SAVED} [已保存 #{memo_id}]", flush=True)
        if self.on_memo is not None:
            try:
                self.on_memo(memo)
            except Exception as exc:      # Web 界面出问题绝不能影响落库
                log(f"推送新卡片失败：{type(exc).__name__}: {exc}", "Web")
        return memo_id


def print_memo(memo: dict, memo_id: Optional[int] = None, source: str = "") -> None:
    """把一条备忘录格式化打印到控制台。"""
    header = "备忘录" + (f" #{memo_id}" if memo_id else "")
    if source:
        header += f"（整理方式：{'DeepSeek' if source == 'deepseek' else '本地规则'}）"
    lines = [
        "",
        "┌" + "─" * 58,
        f"│ {header}",
        "├" + "─" * 58,
        f"│ 时间：{memo.get('record_time', '')}",
        f"│ 标题：{memo.get('title', '')}",
        f"│ 分类：{memo.get('category', '')}",
    ]
    if has_soap(memo):
        # 病历卡片：SOAP 五段按固定顺序输出，空段直接跳过（不打印"无"占位，
        # 五个"无"排下来反而看不出哪些是真的没问）
        for key, label in SOAP_SECTIONS:
            body = (memo.get(key) or "").strip()
            if body:
                lines.extend(_console_section(label, body))
    else:
        lines.append(f"│ 摘要：{memo.get('summary', '')}")
    todos = memo.get("todos") or []
    if todos:
        lines.append("│ 待办：")
        for index, todo in enumerate(todos, 1):
            due = f"  （截止 {todo['due']}）" if todo.get("due") else ""
            lines.append(f"│   {index}. {todo['task']}{due}")
    elif not has_soap(memo):
        lines.append("│ 待办：无")
    lines.append("└" + "─" * 58)
    print("\n".join(lines), flush=True)


def _console_section(label: str, body: str) -> list:
    """把一段多行正文渲染成控制台宽度内的竖线块（现病史常常是好几条）。"""
    output = [f"│ {label}：{body.splitlines()[0]}"]
    for extra in body.splitlines()[1:]:
        output.append(f"│   {extra}")
    return output


def make_pipeline(cfg: Config, conn: sqlite3.Connection) -> MemoPipeline:
    transcriber = Transcriber(cfg)
    client = build_llm_client(cfg)
    return MemoPipeline(cfg, conn, transcriber, client)


# --------------------------------------------------------------------------- #
# 命令行入口
# --------------------------------------------------------------------------- #


def load_wav_16k_mono(path: os.PathLike | str) -> Any:
    """读取 WAV 并转成 16k 单声道 float32。

    只用标准库 ``wave`` + numpy，不额外引入依赖。支持 8/16/32 位整数 PCM；
    WAV 里的浮点格式（``wave`` 模块不支持）请先用 ffmpeg 转成 16 位。
    """
    import wave

    import numpy as np

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
    elif width == 1:
        data = (np.frombuffer(frames, dtype="uint8").astype("float32") - 128.0) / 128.0
    elif width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype("float32") / 2147483648.0
    else:
        raise ValueError(f"不支持的 WAV 位深：{width * 8} 位（支持 8/16/32 位整数 PCM）")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if rate != 16000 and len(data):
        target = int(len(data) * 16000 / rate)
        data = np.interp(
            np.linspace(0, len(data) - 1, target), np.arange(len(data)), data
        ).astype("float32")
    return np.ascontiguousarray(data, dtype="float32")


class ReplayStream:
    """把一段离线音频伪装成 ``sounddevice.InputStream``，代替麦克风做回放测试。

    每遍之间插入一段静音，让 VAD 能自然触发"静音断句"，从而验证**连续多条**记录。
    按真实时间节奏出块（``time.sleep``），否则会瞬间灌满队列、丢掉音频。
    """

    def __init__(self, audio: Any, cfg: Config, loops: int = 1, gap_seconds: float = 0.6,
                 realtime: bool = True):
        import numpy as np

        rate = cfg.sample_rate
        lead = np.zeros(int(max(cfg.pre_speech_pad_ms / 1000.0, 0.4) * rate), dtype="float32")
        # 尾部静音要长于断句阈值，否则最后一遍不会触发 'end'
        tail = np.zeros(int((cfg.silence_ms / 1000.0 + gap_seconds) * rate), dtype="float32")
        self._data = np.concatenate([np.concatenate([lead, audio, tail]) for _ in range(max(1, loops))])
        self._pos = 0
        self._cfg = cfg
        self._closed = False
        self._realtime = realtime   # 关掉后不按真实时间节流，仅供离线自检快速跑完

    def __enter__(self) -> "ReplayStream":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        self._closed = True
        return False

    def read(self, frames: int) -> tuple:
        import numpy as np

        if self._closed:
            raise RuntimeError("stream closed")
        if self._realtime:
            time.sleep(frames / float(self._cfg.sample_rate))   # 模拟实时采集节奏
        block = self._data[self._pos : self._pos + frames]
        self._pos += len(block)
        if len(block) < frames:                             # 播完后返回无限静音
            block = np.concatenate([block, np.zeros(frames - len(block), dtype="float32")])
        return block.reshape(-1, 1), False


def run_replay_mode(cfg: Config, pipeline: "MemoPipeline", listener: "VoiceListener",
                    wav_path: str, loops: int) -> int:
    """用 WAV 文件代替麦克风跑一遍监听流程，验证断句与连续记录。"""
    try:
        audio = load_wav_16k_mono(wav_path)
    except Exception as exc:
        log(f"读取 WAV 失败：{type(exc).__name__}: {exc}", "回放")
        return 1
    if not len(audio):
        log(f"WAV 里没有音频数据：{wav_path}", "回放")
        return 1

    duration = len(audio) / cfg.sample_rate
    log(f"回放模式：{wav_path}（{duration:.1f}s），重复 {loops} 遍代替麦克风", "回放")

    received: list = []
    done = threading.Event()

    def on_utterance(utterance: Utterance) -> None:
        pipeline.handle_audio(utterance)
        received.append(utterance)
        if len(received) >= loops:
            done.set()

    factory = lambda: ReplayStream(audio, cfg, loops=loops)  # noqa: E731
    worker = threading.Thread(
        target=listener.run, args=(on_utterance, factory), name="replay-listener", daemon=True
    )
    worker.start()

    per_loop = duration + max(cfg.silence_ms / 1000.0 + 0.6, 1.0) + 1.0
    finished = done.wait(timeout=loops * per_loop + 30.0)
    listener.stop()
    worker.join(timeout=10)

    log(f"回放结束：共得到 {len(received)}/{loops} 条备忘录", "回放")
    if not finished:
        log("有片段未能断句（可调低 VAD_SILENCE_MS，或确认音频里确实有人声）", "回放")
        return 1
    return 0


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="免唤醒语音备忘录：silero-vad + faster-whisper + DeepSeek + SQLite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python main.py\n"
            "  python main.py --text \"明天下午三点提醒我跟客户开会\"\n"
            "  python main.py --list 20\n"
            "  python main.py --doctor\n"
        ),
    )
    parser.add_argument("--text", metavar="TEXT", help="跳过麦克风，直接把这段文字走完 LLM + 落库链路")
    parser.add_argument("--list", nargs="?", type=int, const=10, default=None, metavar="N",
                        help="列出最近 N 条备忘录（默认 10）后退出")
    parser.add_argument("--init-db", action="store_true", help="仅初始化数据库后退出")
    parser.add_argument("--doctor", action="store_true", help="检查依赖、配置与麦克风设备")
    parser.add_argument("--model", help="覆盖 ASR 模型，如 tiny/base/small/medium")
    parser.add_argument("--device", help="覆盖推理设备：auto/cpu/cuda")
    parser.add_argument("--language", help="覆盖识别语言，如 zh/en；留空用 .env 里的 ASR_LANGUAGE")
    parser.add_argument("--db", help="覆盖数据库路径")
    parser.add_argument("--no-llm", action="store_true", help="强制使用本地规则整理，不调用 DeepSeek")
    parser.add_argument("--early-stop", action="store_true",
                        help="开启\"结束/完毕\"提前截断（默认关：纯 CPU 环境建议只靠静音断句）")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="关闭\"结束/完毕\"提前截断（默认即关闭，保留此参数以兼容旧脚本）")
    parser.add_argument("--env", default=".env", help=".env 文件路径（默认 ./.env）")

    wake = parser.add_argument_group("唤醒词")
    wake.add_argument("--wake-word", metavar="WORD",
                      help=f"唤醒词（默认 {DEFAULT_WAKE_WORD}）。只有听到它才开始记录")
    wake.add_argument("--no-wake", action="store_true",
                      help="关闭唤醒词，回到\"一直听\"的免唤醒模式（安静环境更省事）")

    web = parser.add_argument_group("Web 界面")
    web.add_argument("--web", action="store_true",
                     help=f"同时启动网页看板 http://127.0.0.1:{DEFAULT_WEB_PORT}（需 pip install flask）")
    web.add_argument("--port", type=int, metavar="N", help=f"网页端口（默认 {DEFAULT_WEB_PORT}）")
    web.add_argument("--no-mic", action="store_true",
                     help="只跑 Web 界面，不开麦克风（用 --text 或手机端接口往里塞内容）")

    export = parser.add_argument_group("文档导出")
    export.add_argument("--export-md", nargs="?", const=DEFAULT_EXPORT_MD, default=None, metavar="PATH",
                        help=f"把数据库中的全部历史记录导出为 Markdown（默认 {DEFAULT_EXPORT_MD}）")
    export.add_argument("--export-docx", nargs="?", const=DEFAULT_EXPORT_DOCX, default=None, metavar="PATH",
                        help=f"导出为 Word 文档（默认 {DEFAULT_EXPORT_DOCX}，需 pip install python-docx）")
    export.add_argument("--md-path", metavar="PATH",
                        help=f"每条备忘录自动追加到的文件（默认 {DEFAULT_MARKDOWN_FILENAME}）")
    export.add_argument("--no-md", action="store_true", help="关闭\"每条自动追加 Markdown\"，只写数据库")

    live = parser.add_argument_group("监听调优与测试")
    live.add_argument("--no-beep", action="store_true", help="关闭录音开始/保存完成时的蜂鸣")
    live.add_argument("--replay", metavar="WAV",
                      help="用 WAV 文件代替麦克风回放，用于测试连续录制（无需出声）")
    live.add_argument("--replay-loops", type=int, default=2, metavar="N",
                      help="回放模式下重复多少遍（默认 2，用来验证连续多条）")

    parser.add_argument("--version", action="version", version=f"voice-memo {__version__}")
    return parser.parse_args(argv)


def run_doctor(cfg: Config) -> int:
    """环境自检：依赖、配置、麦克风。"""
    log(f"voice-memo {__version__} 环境自检", "自检")
    problems = 0

    print("\n[1/3] 第三方依赖")
    optional = {
        "numpy": "音频数组运算",
        "sounddevice": "麦克风采集",
        "torch": "silero-vad 运行时",
        "silero_vad": "VAD 免唤醒断句",
        "faster_whisper": "本地语音识别",
        "openai": "DeepSeek API 客户端",
    }
    import importlib.util

    for module, purpose in optional.items():
        spec = importlib.util.find_spec(module)
        if spec is None:
            problems += 1
            print(f"  [缺失] {module:<16} {purpose}    → pip install -r requirements.txt")
        else:
            print(f"  [ OK ] {module:<16} {purpose}")

    print("\n[2/3] 配置")
    if cfg.api_key:
        print(f"  [ OK ] DEEPSEEK_API_KEY 已配置（{cfg.api_key[:6]}…{cfg.api_key[-4:]}）")
    else:
        problems += 1
        print("  [缺失] DEEPSEEK_API_KEY 未配置 → 复制 .env.example 为 .env 后填写")
    print(f"  [ OK ] DEEPSEEK_BASE_URL = {cfg.base_url}")
    transcriber = Transcriber(cfg)
    backend = transcriber.resolve_backend()
    if backend == "funasr":
        print(f"  [ OK ] 语音识别 = {transcriber.describe()}"
              f"（ASR_BACKEND={cfg.asr_backend}）")
    else:
        print(f"  [ OK ] 语音识别 = {transcriber.describe()}"
              f"（ASR_BACKEND={cfg.asr_backend}，未检测到 funasr）")
        if not Transcriber.funasr_available():
            print("         → 想要中文识别率更高：pip install funasr，"
                  "装上后自动切到 SenseVoiceSmall（中文专用、CPU 上更快）")
    print(f"  [ OK ] {transcriber.describe_hotwords()}")
    print(f"         束搜索 beam_size = {cfg.asr_beam_size}（ASR_BEAM_SIZE：越大越准、越慢）")
    if cfg.audio_preprocess:
        print(f"  [ OK ] 音频预处理 = 开（去直流"
              f"{' + 降噪' if cfg.audio_denoise else ''}"
              f" + 响度归一（只补不压，上限 {cfg.audio_max_gain_db:.0f} dB）"
              f" + 削峰保护）")
    else:
        print("  [ -- ] 音频预处理 = 关（AUDIO_PREPROCESS=false）")
    if importlib.util.find_spec("pypinyin") is not None:
        print("  [ OK ] pypinyin 已安装：控制词会做拼音匹配"
              "（「錄音完畢」「坐着飞」这类同音错字也能认出来）")
    else:
        print("  [注意] pypinyin 未安装：控制词只做逐字模糊匹配，"
              "同音错字容易漏判 → pip install pypinyin")
    print(f"  [ OK ] 静音断句 = {cfg.silence_ms} ms · 提前截断 = {'开' if cfg.early_stop_keyword else '关'}")
    if cfg.early_stop_keyword:
        print("  [注意] 结束词提前截断会在录音途中反复调用 Whisper，纯 CPU 机器容易卡顿；"
              "只靠静音断句的话建议关掉它（EARLY_STOP_KEYWORD=false）")
    if cfg.wake_word_enabled and cfg.wake_word:
        print(f"  [ OK ] 唤醒词 = 「{cfg.wake_word}」（扫描模型 {cfg.wake_model}，"
              f"间隔 {cfg.wake_scan_interval_s}s）")
        print(f"         分段结束词 = {' / '.join(cfg.segment_end_keywords)}")
        print(f"         短结束词   = {' / '.join(cfg.short_end_keywords) or '（已关闭）'}"
              f"（只在句尾生效）")
        print(f"         退出词     = {' / '.join(cfg.exit_keywords)}")
        print(f"         作废词     = {' / '.join(cfg.cancel_keywords)}"
              f"（只在句尾生效，避免误伤正文）")
    else:
        print("  [ -- ] 唤醒词 = 已关闭（免唤醒模式：听到人声就直接记录）")
    if importlib.util.find_spec("flask") is not None:
        print(f"  [ OK ] flask 已安装，可用 --web → http://{cfg.web_host}:{cfg.web_port}")
    else:
        print("  [ -- ] flask 未安装，--web 不可用（pip install flask）")
    print(f"  [ OK ] 蜂鸣提示 = {'开' if cfg.beep else '关'}")
    if cfg.markdown_path:
        print(f"  [ OK ] 自动追加 Markdown = {Path(cfg.markdown_path).resolve()}")
    else:
        print("  [ -- ] 自动追加 Markdown = 已关闭（只写数据库）")
    if importlib.util.find_spec("docx") is not None:
        print("  [ OK ] python-docx 已安装，可用 --export-docx")
    else:
        print("  [ -- ] python-docx 未安装，--export-docx 不可用（pip install python-docx）")

    print("\n[3/3] 数据库与麦克风")
    try:
        conn = init_db(cfg.db_path)
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(memos)")]
        missing = [c for c in REQUIRED_COLUMNS if c not in columns]
        if missing:
            problems += 1
            print(f"  [异常] memos 表缺少字段：{missing}")
        else:
            print(f"  [ OK ] 数据库 {cfg.db_path} · memos 表字段齐全（{len(columns)} 列）")
            print(f"         {' / '.join(columns)}")
        conn.close()
    except Exception as exc:
        problems += 1
        print(f"  [异常] 数据库不可用：{exc}")

    try:
        import sounddevice as sd

        devices = [d for d in sd.query_devices() if d.get("max_input_channels", 0) > 0]
        if devices:
            default_in = sd.default.device[0]
            name = sd.query_devices(default_in)["name"] if isinstance(default_in, int) and default_in >= 0 else "未知"
            print(f"  [ OK ] 检测到 {len(devices)} 个输入设备，默认：{name}")
        else:
            problems += 1
            print("  [异常] 未检测到任何音频输入设备")
    except Exception as exc:
        print(f"  [跳过] 无法枚举麦克风（{exc}）")

    print()
    if problems:
        log(f"自检完成：发现 {problems} 个待处理项", "自检")
        return 1
    log("自检完成：全部通过", "自检")
    return 0


def print_memo_list(conn: sqlite3.Connection, limit: int) -> None:
    memos = fetch_memos(conn, limit)
    if not memos:
        log("暂无备忘录", "查询")
        return
    log(f"最近 {len(memos)} 条备忘录", "查询")
    for memo in memos:
        print_memo(memo, memo_id=memo.get("id"))


def start_web_server(cfg: Config, conn: sqlite3.Connection, state: "AppState",
                     pipeline: Optional["MemoPipeline"] = None,
                     listener: Optional["VoiceListener"] = None) -> Any:
    """在后台线程里起 Flask 看板，返回该线程（失败时返回 None）。

    依赖是懒加载的：没装 Flask 只影响 ``--web``，其余功能照常用。
    ``listener`` 允许先传 None——Web 先起来、监听器随后创建时，
    用 ``webui.set_listener(app, listener)`` 补上即可。
    """
    try:
        import webui
    except ImportError as exc:
        log(f"Web 界面需要 Flask：{exc}", "Web")
        log("请执行：pip install flask", "Web")
        return None

    app = webui.create_app(cfg, conn, state, pipeline=pipeline, listener=listener)
    thread = threading.Thread(
        target=lambda: app.run(host=cfg.web_host, port=cfg.web_port,
                               threaded=True, use_reloader=False),
        name="web-server", daemon=True,
    )
    thread.start()
    shown = "127.0.0.1" if cfg.web_host in ("0.0.0.0", "::") else cfg.web_host
    log(f"网页看板已启动：http://{shown}:{cfg.web_port}", "Web")
    log("实时状态与卡片会通过 SSE 自动刷新，无需手动刷新页面", "Web")
    return app


def main(argv: Optional[list] = None) -> int:
    _setup_console()
    args = parse_args(argv)
    load_dotenv(args.env)

    cfg = Config.from_env()
    if args.model:
        cfg.asr_model = args.model
    if args.device:
        cfg.asr_device = args.device
    if args.language:
        cfg.asr_language = args.language
    if args.db:
        cfg.db_path = args.db
    if args.early_stop:
        cfg.early_stop_keyword = True
    if args.no_early_stop:
        cfg.early_stop_keyword = False
    if args.wake_word:
        cfg.wake_word = args.wake_word
        cfg.wake_word_enabled = True
    if args.no_wake:
        cfg.wake_word_enabled = False
    if not cfg.wake_word.strip():
        cfg.wake_word_enabled = False      # WAKE_WORD 留空 = 免唤醒
    if args.replay and not args.doctor and not args.wake_word and not args.no_wake:
        # 回放用的 WAV 里一般没有唤醒词，开着唤醒词就永远待机、一条都出不来。
        # 除非用户显式指定了唤醒词（说明那段音频里确实录了），否则临时免唤醒。
        cfg.wake_word_enabled = False
        log("回放模式：WAV 里通常没有唤醒词，已临时切到免唤醒模式"
            "（想连唤醒词一起测就加 --wake-word \"小医小医\"）", "回放")
    if args.port:
        cfg.web_port = args.port
    if args.no_md:
        cfg.markdown_path = ""
    elif args.md_path:
        cfg.markdown_path = args.md_path
    if args.no_beep:
        cfg.beep = False

    if args.doctor:
        return run_doctor(cfg)

    conn = init_db(cfg.db_path)
    log(f"数据库就绪：{Path(cfg.db_path).resolve()}")

    if args.init_db:
        log("仅初始化数据库，退出")
        conn.close()
        return 0

    if args.export_md is not None:
        count = export_markdown(conn, args.export_md)
        log(f"已导出 {count} 条记录到 {Path(args.export_md).resolve()}", "导出")
        conn.close()
        return 0

    if args.export_docx is not None:
        try:
            count = export_docx(conn, args.export_docx)
        except ExportDependencyError as exc:
            log(str(exc), "导出")
            conn.close()
            return 1
        log(f"已导出 {count} 条记录到 {Path(args.export_docx).resolve()}", "导出")
        conn.close()
        return 0

    if args.list is not None:
        print_memo_list(conn, args.list)
        conn.close()
        return 0

    client = None if args.no_llm else build_llm_client(cfg)
    transcriber = Transcriber(cfg)
    log(f"语音识别后端：{transcriber.describe()}", "ASR")
    if args.model and transcriber.resolve_backend() == "funasr":
        log(f"当前用的是 FunASR/SenseVoice，--model {args.model} 不生效；"
            f"想用 faster-whisper 请加 ASR_BACKEND=whisper", "ASR")

    # 状态与推送：Web 界面靠它实时刷新，命令行模式下它只是个安静的中转站
    wake_mode = bool(cfg.wake_word_enabled and cfg.wake_word)
    state = AppState(wake_mode=wake_mode, wake_word=cfg.wake_word)
    pipeline = MemoPipeline(cfg, conn, transcriber, client, on_memo=state.publish_memo)
    listener: Optional[VoiceListener] = None
    web_app: Optional[Any] = None

    if args.web:
        web_app = start_web_server(cfg, conn, state, pipeline=pipeline)

    if args.text:
        # 走完整的 pipeline，这样 --text 也会同步追加到 Markdown
        pipeline.handle_text(clean_transcript(args.text, cfg))
        if web_app is None:
            conn.close()
            return 0
        log("这条已生成卡片，网页上能看到了；按 Ctrl+C 结束", "Web")

    if args.no_mic or args.text:
        if web_app is None:
            log("--no-mic / --text 需要配合 --web 使用（否则进程会立刻空转退出）", "Web")
            conn.close()
            return 1
        if not args.text:
            log("仅 Web 模式：没有开麦克风，可用 --text 或网页里的输入框往里塞内容", "Web")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            log("收到中断信号，正在退出…", "Web")
        conn.close()
        return 0

    # ---- 常驻监听模式 ----
    try:
        vad = build_vad()
    except Exception as exc:
        log(f"VAD 模型加载失败：{exc}", "VAD")
        log("请确认已安装 torch 与 silero-vad（pip install -r requirements.txt）", "VAD")
        conn.close()
        return 1

    if not transcriber.preload() and not Transcriber.available():
        log("未安装 faster-whisper，无法进行语音识别", "ASR")
        log("请执行：pip install -r requirements.txt", "ASR")
        conn.close()
        return 1

    llm_label = "DeepSeek " if client is not None else "本地规则"
    listener = VoiceListener(cfg, vad, transcriber=transcriber, llm_label=llm_label,
                             on_status=state.set_state, on_cancel=state.publish_cancel)
    if web_app is not None and not args.no_mic:
        # 页面上的"退出录制"按钮要能停掉监听器 —— 把 listener 补给已经起好的 Web
        import webui

        webui.set_listener(web_app, listener)

    if cfg.markdown_path:
        log(f"每条备忘录会同步追加到 {Path(cfg.markdown_path).resolve()}", "导出")

    if args.replay:
        try:
            return run_replay_mode(cfg, pipeline, listener, args.replay, max(1, args.replay_loops))
        finally:
            conn.close()

    try:
        listener.run(pipeline.handle_audio)
    except KeyboardInterrupt:
        log("收到中断信号，正在退出…", "监听")
        listener.stop()
    finally:
        conn.close()
        if listener.exit_requested.is_set():
            log("已按语音指令退出，数据已保存到 " + cfg.db_path, "系统")
        else:
            log("已退出，数据已保存到 " + cfg.db_path, "系统")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
