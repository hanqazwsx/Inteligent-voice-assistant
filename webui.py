"""网页看板：实时状态 + 备忘录卡片流 + 一键导出。

设计要点
--------
* **Flask 是懒加载的**：所有 ``from flask import ...`` 都写在 ``create_app()`` 内部。
  没装 Flask 时 ``import webui`` 依然成功，只是 ``create_app`` 抛 ImportError ——
  ``main.py`` 捕获后给一句安装提示，其余功能照常。自检脚本也靠这一点优雅跳过。
* **页面不轮询**：``/api/stream`` 是一条 SSE（``text/event-stream``）长连接，
  状态变化和"新卡片落库"由 ``main.Broadcaster`` 主动推过来，页面收到即渲染。
* **写入只走 main 里的函数**：转写、整理、落库全在 ``main.MemoPipeline`` 里；删除走
  ``main.remove_memo``（顺带把 ``my_memos.md`` 重刷对齐）。Web 层自己不写 SQL，
  只是那套状态机的一面显示器 + 一个删除按钮。

路由
----
``GET  /``                 卡片看板页面
``GET  /api/state``        当前状态快照（页面首次加载时用）
``GET  /api/memos``        最近的卡片（按时间倒序）
``GET  /api/stream``       SSE 实时推送（state / memo）
``POST /api/text``         手动塞一段文字进整理链路（--no-mic 时很好用）
``POST /api/stop``         停止监听（等同于语音说"退出录制"）
``POST /api/control/finish``  手动结束并整理当前录音（说不出指令词时的双保险）
``POST /api/control/cancel``  手动放弃当前录音（不转写、不入库）
``POST /api/memo/<id>/update``      保存网页上手工修改的卡片字段（contenteditable 回写）
``POST /api/memo/<id>/regenerate``  重新润色：把手改过的整张卡片再交给 DeepSeek 生成一遍
``POST|DELETE /api/delete/<id>``  删除一张卡片（数据库 + my_memos.md 同步）
``GET  /api/export``       导出全部记录：``?fmt=md`` 或 ``?fmt=docx``，直接下载
"""

from __future__ import annotations

import io
import sqlite3
from datetime import datetime
from typing import Any, Optional

DEFAULT_EXPORT_LIMIT = 500


def create_app(cfg: Any, conn: sqlite3.Connection, state: Any,
               pipeline: Any = None, listener: Any = None) -> Any:
    """构建并返回 Flask app。

    ``cfg``      : ``main.Config``
    ``conn``     : 已打开的 sqlite3 连接（``check_same_thread=False``）
    ``state``    : ``main.AppState``，提供 ``snapshot()`` 与 ``broadcaster``
    ``pipeline`` : ``main.MemoPipeline``，供 ``/api/text`` 手动写入
    ``listener`` : ``main.VoiceListener``，供 ``/api/stop`` 停录音
    """
    try:
        from flask import Flask, Response, jsonify, render_template, request, send_file
    except ImportError as exc:  # pragma: no cover - 取决于是否装了 Flask
        raise ImportError(
            "网页看板需要 Flask，请先安装：pip install flask"
        ) from exc

    import main as core          # 复用 main.py 里的导出/查询函数，避免逻辑重复

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["JSON_AS_ASCII"] = False
    app.config["STATE"] = state
    app.config["LISTENER"] = listener
    # 重新润色要用它身上的 DeepSeek 客户端（可能为 None —— 没 Key 或 --no-llm）
    app.config["PIPELINE"] = pipeline

    # ------------------------------------------------------------------ #
    # 页面
    # ------------------------------------------------------------------ #
    @app.get("/")
    def index() -> Any:
        snapshot = state.snapshot()
        return render_template(
            "index.html",
            snapshot=snapshot,
            memos=[core.memo_to_dict(m) for m in core.fetch_memos(conn, DEFAULT_EXPORT_LIMIT)],
            segment_end=" / ".join(cfg.segment_end_keywords[:3]),
            exit_words=" / ".join(cfg.exit_keywords[:2]),
            cancel_words=" / ".join(cfg.cancel_keywords[:2]),
            silence_s=f"{cfg.silence_ms / 1000:.1f}",
        )

    # ------------------------------------------------------------------ #
    # JSON 接口
    # ------------------------------------------------------------------ #
    @app.get("/api/state")
    def api_state() -> Any:
        return jsonify(state.snapshot())

    @app.get("/api/memos")
    def api_memos() -> Any:
        try:
            limit = int(request.args.get("limit", DEFAULT_EXPORT_LIMIT))
        except (TypeError, ValueError):
            limit = DEFAULT_EXPORT_LIMIT
        limit = max(1, min(limit, 5000))
        memos = [core.memo_to_dict(m) for m in core.fetch_memos(conn, limit)]
        return jsonify({"count": len(memos), "memos": memos})

    @app.get("/api/stream")
    def api_stream() -> Any:
        """SSE：状态变化与新卡片实时推给页面。"""

        def events() -> Any:
            box = state.broadcaster.subscribe()
            try:
                # 刚连上先补一份当前状态，免得页面在第一次变化前空着
                yield _sse(state.published_state())
                while True:
                    try:
                        event = box.get(timeout=15)
                    except Exception:            # queue.Empty
                        yield ": keep-alive\n\n"  # 心跳，防止代理掐断长连接
                        continue
                    yield _sse(event)
            finally:
                state.broadcaster.unsubscribe(box)

        return Response(
            events(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",       # 关掉 nginx 缓冲（若前面挂了代理）
                "Connection": "keep-alive",
            },
        )

    @app.post("/api/text")
    def api_text() -> Any:
        """手动喂一段文字，走完整的整理 + 落库 + 推送链路。"""
        payload = request.get_json(silent=True) or {}
        text = (payload.get("text") or request.form.get("text") or "").strip()
        if not text:
            return jsonify({"ok": False, "error": "text 不能为空"}), 400
        if pipeline is None:
            return jsonify({"ok": False, "error": "当前进程没有可用的整理管线"}), 503
        try:
            memo_id = pipeline.handle_text(core.clean_transcript(text, cfg))
        except Exception as exc:                                  # 不让界面 500
            return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500
        if memo_id is None:
            return jsonify({"ok": False, "error": "内容只剩指令词，已忽略"}), 200
        return jsonify({"ok": True, "id": memo_id})

    @app.route("/api/delete/<int:memo_id>", methods=["POST", "DELETE"])
    def api_delete(memo_id: int) -> Any:
        """删除一张卡片：先删数据库，再按数据库现状重刷 my_memos.md，最后广播。

        删完必须广播 —— 否则同一个后端的其它标签页/手机页面还挂着那张已经不存在
        的卡片，点进去导出的文档却已经没有它了。
        """
        try:
            removed = core.remove_memo(conn, memo_id, cfg.markdown_path)
        except Exception as exc:                                  # 不让界面 500
            return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500
        if removed is None:
            return jsonify({"ok": False, "error": f"没有 id={memo_id} 的卡片"}), 404
        state.publish_delete(memo_id)
        remaining = conn.execute("SELECT COUNT(*) FROM memos").fetchone()[0]
        return jsonify({
            "ok": True,
            "id": memo_id,
            "title": removed.get("title") or "",
            "remaining": int(remaining),
            "markdown_path": cfg.markdown_path or "",
            "markdown_synced": bool(cfg.markdown_path),
        })

    @app.post("/api/memo/<int:memo_id>/update")
    def api_memo_update(memo_id: int) -> Any:
        """保存网页上手工修改的字段（``contenteditable`` 回写）。

        字段名走 ``core.EDITABLE_FIELDS`` 白名单：它既是"哪些字段允许改"的声明，
        也是拼 UPDATE 语句时唯一的列名来源 —— 白名单之外的一律丢弃，绝不拼进 SQL。
        """
        payload = request.get_json(silent=True) or {}
        fields = payload.get("fields") if isinstance(payload.get("fields"), dict) else payload
        if not isinstance(fields, dict):
            return jsonify({"ok": False, "error": "请求体得是字段字典"}), 400
        unknown = [name for name in fields if name not in core.EDITABLE_FIELDS]
        if unknown and not (set(fields) & set(core.EDITABLE_FIELDS)):
            return jsonify({
                "ok": False,
                "error": f"没有可改的字段；允许：{', '.join(core.EDITABLE_FIELDS)}",
            }), 400
        try:
            memo = core.update_memo(conn, memo_id, fields)
        except Exception as exc:                                  # 不让界面 500
            return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500
        if memo is None:
            return jsonify({"ok": False, "error": f"没有 id={memo_id} 的卡片"}), 404
        _sync_markdown_and_broadcast(memo)
        return jsonify({"ok": True, "memo": core.memo_to_dict(memo), "ignored": unknown})

    @app.post("/api/memo/<int:memo_id>/regenerate")
    def api_memo_regenerate(memo_id: int) -> Any:
        """重新润色：把（手改过的）整张卡片再交给 DeepSeek 生成一遍。

        和"修改"是两回事 —— 修改是学生自己动手，这里是让模型把改动**贯穿**到
        整份记录（比如修好了现病史里的"反跳痛"，诊断思考该跟着重写）。
        """
        memo = core.fetch_memo(conn, memo_id)
        if memo is None:
            return jsonify({"ok": False, "error": f"没有 id={memo_id} 的卡片"}), 404
        # 页面上的未保存改动一起带过来：润色要基于"学生改过的那一版"而不是库里的旧版。
        # 这里只覆盖内存里的副本，真写库等模型返回之后 —— 模型调用失败时，
        # 页面上的改动仍然是"未保存"状态，学生不会以为已经存进去了。
        payload = request.get_json(silent=True) or {}
        edited = payload.get("fields") if isinstance(payload.get("fields"), dict) else None
        if edited:
            memo = dict(memo)
            memo.update({name: str(value or "") for name, value in edited.items()
                         if name in core.EDITABLE_FIELDS})
        target = app.config.get("PIPELINE")
        client = getattr(target, "client", None)
        if client is None:
            return jsonify({
                "ok": False,
                "error": "当前没有可用的 DeepSeek 客户端（没配 API Key，或用了 --no-llm），"
                         "重新润色需要它",
            }), 503
        try:
            fresh = core.regenerate_memo(client, cfg.llm_model, memo,
                                         memo.get("record_time") or "")
            updated = core.update_memo(conn, memo_id, fresh,
                                       allowed=core.REGENERATE_FIELDS)
        except Exception as exc:                                  # 网络/解析失败都走这里
            return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 502
        if updated is None:
            return jsonify({"ok": False, "error": f"没有 id={memo_id} 的卡片"}), 404
        _sync_markdown_and_broadcast(updated)
        return jsonify({"ok": True, "memo": core.memo_to_dict(updated)})

    def _sync_markdown_and_broadcast(memo: dict) -> None:
        """改完一条记录后的统一收尾：重刷 Markdown + 推给所有打开的页面。"""
        if cfg.markdown_path:
            try:
                core.export_markdown(conn, cfg.markdown_path)
            except Exception as exc:                    # 文档写失败不该让接口失败
                core.log(f"重刷 Markdown 失败：{type(exc).__name__}: {exc}", "导出")
        state.publish_update(memo)

    @app.post("/api/control/<action>")
    def api_control(action: str) -> Any:
        """网页上的「结束并整理 / 放弃本条」两个按钮。

        只置一个标志位就返回，真正的收尾（截断音频、转写、落库或丢弃）由录音主循环
        下一轮完成 —— Web 线程绝不直接碰音频缓冲，否则会跟 VAD 状态机打架。
        """
        if action not in ("finish", "cancel"):
            return jsonify({"ok": False, "error": "action 只能是 finish 或 cancel"}), 400
        target = app.config.get("LISTENER")
        if target is None:
            return jsonify({"ok": False,
                            "error": "当前进程没有开麦克风（--no-mic），没有可控制的录音"}), 503
        accepted = (target.request_finish() if action == "finish"
                    else target.request_cancel())
        if not accepted:
            return jsonify({
                "ok": False,
                "state": target.state,
                "error": "现在没有在录音（可能已经整理完，或还在待机）",
            }), 409
        return jsonify({"ok": True, "action": action, "state": target.state})

    @app.post("/api/stop")
    def api_stop() -> Any:
        """停止监听（等同语音说"退出录制"）。"""
        target = app.config.get("LISTENER")
        if target is None:
            return jsonify({"ok": False, "error": "当前没有在监听"}), 503
        target.exit_requested.set()
        target.stop()
        return jsonify({"ok": True})

    # ------------------------------------------------------------------ #
    # 导出下载
    # ------------------------------------------------------------------ #
    @app.get("/api/export")
    def api_export() -> Any:
        fmt = (request.args.get("fmt") or "md").lower()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if fmt in ("md", "markdown"):
            memos = core.fetch_all_memos(conn)
            subtitle = (f"共 {len(memos)} 条记录 · 导出时间 "
                        f"{datetime.now():%Y-%m-%d %H:%M:%S}")
            body = "".join(
                core.format_markdown_entry(m, index=i) for i, m in enumerate(memos, 1)
            )
            data = (core._markdown_header(core.MARKDOWN_DOC_TITLE, subtitle) + body)
            return send_file(
                io.BytesIO(data.encode("utf-8")),
                mimetype="text/markdown",       # Flask 会自动补 charset=utf-8
                as_attachment=True,
                download_name=f"memos-{stamp}.md",
            )
        if fmt in ("docx", "word"):
            try:
                from docx import Document
            except ImportError:
                return jsonify({
                    "ok": False,
                    "error": "导出 Word 需要 python-docx，请先安装：pip install python-docx",
                }), 501
            memos = core.fetch_all_memos(conn)
            document = Document()
            document.add_heading(core.MARKDOWN_DOC_TITLE, level=0)
            document.add_paragraph(
                f"由 voice-memo 自动生成 · 共 {len(memos)} 条记录 · "
                f"导出时间 {datetime.now():%Y-%m-%d %H:%M:%S}"
            )
            for index, memo in enumerate(memos, 1):
                document.add_heading(
                    f"{index}. {memo.get('title') or '未命名备忘录'}", level=2
                )
                document.add_paragraph(f"时间：{memo.get('record_time', '')}")
                document.add_paragraph(f"分类：{memo.get('category') or '其他'}")
                document.add_paragraph(f"摘要：{memo.get('summary') or '（无）'}")
                for todo in memo.get("todos") or []:
                    due = f"（截止 {todo['due']}）" if todo.get("due") else ""
                    document.add_paragraph(f"☐ {todo['task']}{due}", style="List Bullet")
                raw = " ".join((memo.get("raw_text") or "").split())
                if raw:
                    document.add_paragraph(f"原始识别原文：{raw}")
            buffer = io.BytesIO()
            document.save(buffer)
            buffer.seek(0)
            return send_file(
                buffer,
                mimetype="application/vnd.openxmlformats-officedocument"
                         ".wordprocessingml.document",
                as_attachment=True,
                download_name=f"memos-{stamp}.docx",
            )
        return jsonify({"ok": False, "error": "fmt 只能是 md 或 docx"}), 400

    return app


def set_listener(app: Any, listener: Any) -> None:
    """监听器晚于 Flask 创建时，用它补上（``/api/stop`` 需要）。"""
    app.config["LISTENER"] = listener


def _sse(event: dict) -> str:
    """把一条事件序列化成 SSE 帧。"""
    import json

    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def build_export_markdown(conn: sqlite3.Connection, core: Any) -> str:
    """整份 Markdown 文本（自检脚本直接断言内容，不用起 HTTP 服务）。"""
    memos = core.fetch_all_memos(conn)
    stamp = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    subtitle = f"共 {len(memos)} 条记录 · 导出时间 {stamp}"
    body = "".join(core.format_markdown_entry(m, index=i) for i, m in enumerate(memos, 1))
    return core._markdown_header(core.MARKDOWN_DOC_TITLE, subtitle) + body
