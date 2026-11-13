# -*- coding: utf-8 -*-
"""AI 選股小幫手｜給一般人用的網頁 App（需與 ai_stock_analyzer.py 放同一資料夾）"""
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from ai_stock_analyzer import (resolve, technical, chips, fundamentals, news,
                               verdict, WEIGHTS, AI_WATCHLIST, analyze)

st.set_page_config(page_title="AI 選股小幫手", page_icon="📈", layout="centered")
st.title("📈 AI 選股小幫手")
st.caption("輸入股票代碼或公司名稱，一鍵看懂 技術面・籌碼面・基本面・消息面")

tab_one, tab_rank = st.tabs(["🔍 個股分析", "🏆 AI 類股排行"])


@st.cache_data(ttl=900, show_spinner=False)
def load(tk):
    t = yf.Ticker(tk)
    df = t.history(period="2y")
    try:
        info = t.info or {}
    except Exception:
        info = {}
    return df, info


@st.cache_data(ttl=3600, show_spinner=False)
def run_scan():
    rows = []
    for name, (tk, cat) in AI_WATCHLIST.items():
        try:
            r = analyze(tk, quiet=True)
            if r:
                rows.append({"公司": name, "代碼": tk, "領域": cat,
                             "分數": round(r["score"]), "結論": r["verdict"]})
        except Exception:
            pass
    return sorted(rows, key=lambda x: -x["分數"])


def candle(df):
    d = df.tail(180)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
                                 increasing_line_color="#e53935", decreasing_line_color="#1e9e55",
                                 name="K線"))
    for n, col in ((20, "#f9a825"), (60, "#1e88e5")):
        fig.add_trace(go.Scatter(x=df.index[-180:], y=df["Close"].rolling(n).mean().tail(180),
                                 line=dict(color=col, width=1.5), name=f"{n}日均線"))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.08))
    return fig


with tab_one:
    q = st.text_input("股票代碼或公司名稱", placeholder="例如：2330、台積電、NVDA")
    pick = st.selectbox("或從 AI 熱門股挑選", ["（不選）"] + list(AI_WATCHLIST.keys()))
    if st.button("開始分析", type="primary", use_container_width=True):
        query = q.strip() or (pick if pick != "（不選）" else "")
        if not query:
            st.warning("請先輸入股票代碼或名稱")
        else:
            with st.spinner("分析中，約需 10~20 秒…"):
                tk = resolve(query)
                if not tk:
                    st.error(f"找不到「{query}」，請確認代碼或名稱")
                    st.stop()
                df, info = load(tk)
                if df.empty or len(df) < 130:
                    st.error("歷史資料不足，無法分析")
                    st.stop()
                t_s, t_n, t_f, px = technical(df)
                c_s, c_n, c_f, _ = chips(tk, info)
                f_s, f_n, f_f = fundamentals(info)
                n_s, n_n, n_f = news(yf.Ticker(tk))

            parts = {"技術面": t_s, "籌碼面": c_s, "基本面": f_s, "消息面": n_s}
            avail = {k: v for k, v in parts.items() if v is not None}
            total = sum(WEIGHTS[k] * v for k, v in avail.items()) / sum(WEIGHTS[k] for k in avail)
            flags = t_f + (c_f or []) + f_f + n_f
            if len(avail) < 3:
                flags.append("可用資料面向不足 3 個，結論可信度低")
            v, action = verdict(total, flags)

            st.subheader(f"{info.get('shortName') or tk}（{tk}）")
            c1, c2 = st.columns(2)
            c1.metric("現價", f"{px['price']:.2f}")
            c2.metric("綜合評分", f"{total:.0f} / 100")
            if total >= 70:
                st.success(f"結論：{v}")
            elif total >= 55:
                st.info(f"結論：{v}")
            elif total >= 40:
                st.warning(f"結論：{v}")
            else:
                st.error(f"結論：{v}")
            st.write(f"**操作建議：** {action}")

            for label, s in parts.items():
                st.write(f"{label}　{'資料不足' if s is None else f'{s:.0f} 分'}")
                st.progress(0 if s is None else int(s))

            if flags:
                st.markdown("#### ⚠️ 風險警示")
                for f in flags:
                    st.warning(f)

            st.markdown("#### K 線圖（紅漲綠跌）")
            st.plotly_chart(candle(df), use_container_width=True)

            for label, notes in (("技術面", t_n), ("籌碼面", c_n), ("基本面", f_n), ("消息面", n_n)):
                with st.expander(f"{label}細節"):
                    for n in notes:
                        st.write("• " + n)

            st.caption("【誠實聲明】本工具以公開數據做量化評分，不是預測，也無法保證獲利。"
                       "AI 類股齊漲齊跌，買多檔不等於分散風險。請只用賠得起的錢，並自行負責決策。")

with tab_rank:
    st.write("掃描內建的 AI 觀察清單並依分數排序（第一次約 1~2 分鐘，之後 1 小時內用快取）。")
    if st.button("開始掃描", use_container_width=True):
        with st.spinner("掃描中…"):
            rows = run_scan()
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption("分數高 ≠ 保證上漲，只代表目前條件相對較好，仍須自行研究。")
