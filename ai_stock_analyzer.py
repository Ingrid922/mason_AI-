#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 股票全方位分析器（技術面 / 籌碼面 / 基本面 / 消息面）

安裝：  pip install yfinance pandas numpy requests
用法：
  python ai_stock_analyzer.py 2330          # 輸入代碼
  python ai_stock_analyzer.py 台積電         # 輸入公司名稱
  python ai_stock_analyzer.py NVDA          # 美股
  python ai_stock_analyzer.py --scan        # 掃描 AI 觀察清單並排名
  python ai_stock_analyzer.py               # 互動模式（客戶直接輸入）

資料來源：Yahoo Finance（K線、財報、新聞）、FinMind（台股三大法人、融資）
※ 本程式為輔助工具，不是保證獲利的神器，詳見報告最後的風險聲明。
"""
import sys
import time
import datetime as dt

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------- AI 觀察清單 ----------------
AI_WATCHLIST = {
    "台積電": ("2330.TW", "晶圓代工"),
    "鴻海": ("2317.TW", "AI伺服器代工"),
    "廣達": ("2382.TW", "AI伺服器代工"),
    "緯創": ("3231.TW", "AI伺服器代工"),
    "緯穎": ("6669.TW", "雲端AI伺服器"),
    "技嘉": ("2376.TW", "AI伺服器/顯卡"),
    "華碩": ("2357.TW", "AI PC/伺服器"),
    "聯發科": ("2454.TW", "邊緣AI晶片"),
    "世芯": ("3661.TW", "ASIC設計服務"),
    "創意": ("3443.TW", "ASIC設計服務"),
    "智邦": ("2345.TW", "AI網通交換器"),
    "台達電": ("2308.TW", "電源/散熱"),
    "奇鋐": ("3017.TW", "AI散熱"),
    "雙鴻": ("3324.TWO", "AI散熱"),
    "輝達": ("NVDA", "GPU龍頭"),
    "微軟": ("MSFT", "雲端/AI平台"),
    "谷歌": ("GOOGL", "雲端/AI模型"),
    "超微": ("AMD", "GPU/CPU"),
    "博通": ("AVGO", "客製AI晶片/網通"),
    "帕蘭泰爾": ("PLTR", "AI軟體"),
    "美光": ("MU", "HBM記憶體"),
    "安謀": ("ARM", "晶片IP"),
}

POS_WORDS = ["surge", "beat", "record", "growth", "upgrade", "partnership", "win", "soar", "strong",
             "上修", "創高", "大單", "成長", "利多", "獲利", "擴產", "調升", "看好", "突破"]
NEG_WORDS = ["miss", "downgrade", "lawsuit", "probe", "ban", "cut", "plunge", "fall", "weak", "delay",
             "下修", "衰退", "利空", "調降", "虧損", "禁令", "調查", "延遲", "砍單", "看淡"]


# ---------------- 代碼解析 ----------------
def is_tw(tk):
    return tk.endswith(".TW") or tk.endswith(".TWO")


def resolve(q):
    q = q.strip()
    for name, (tk, _) in AI_WATCHLIST.items():
        if q == name or (len(q) >= 2 and q in name) or q.upper() in (tk.upper(), tk.split(".")[0].upper()):
            return tk
    if q.isdigit():
        for suf in (".TW", ".TWO"):
            try:
                if not yf.Ticker(q + suf).history(period="5d").empty:
                    return q + suf
            except Exception:
                pass
        return None
    if q.isascii() and q.replace("-", "").replace(".", "").isalpha() and len(q) <= 6:
        return q.upper()
    try:
        res = yf.Search(q, max_results=5).quotes
        if res:
            return res[0]["symbol"]
    except Exception:
        pass
    return None


# ---------------- 技術面 ----------------
def technical(df):
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    ma = {n: c.rolling(n).mean() for n in (5, 20, 60, 120)}
    d = c.diff()
    rs = d.clip(lower=0).ewm(alpha=1 / 14).mean() / (-d.clip(upper=0)).ewm(alpha=1 / 14).mean()
    rsi = 100 - 100 / (1 + rs)
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    hist = macd - macd.ewm(span=9).mean()
    rsv = (c - l.rolling(9).min()) / (h.rolling(9).max() - l.rolling(9).min()) * 100
    k = rsv.ewm(alpha=1 / 3).mean()
    dd = k.ewm(alpha=1 / 3).mean()

    last = c.iloc[-1]
    notes, flags, sig = [], [], []

    if last > ma[20].iloc[-1] > ma[60].iloc[-1]:
        sig.append(1); notes.append("均線多頭排列（股價>月線>季線）")
    elif last < ma[20].iloc[-1] < ma[60].iloc[-1]:
        sig.append(-1); notes.append("均線空頭排列（股價<月線<季線）")
    else:
        sig.append(0); notes.append("均線糾結，方向未明")

    if not np.isnan(ma[120].iloc[-1]):
        if last > ma[120].iloc[-1]:
            sig.append(1); notes.append("站上半年線，中期趨勢向上")
        else:
            sig.append(-1); notes.append("跌破半年線，中期趨勢偏弱")

    if hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2]:
        sig.append(1); notes.append("MACD 柱狀體為正且擴大")
    elif hist.iloc[-1] < 0 and hist.iloc[-1] < hist.iloc[-2]:
        sig.append(-1); notes.append("MACD 柱狀體為負且擴大")
    else:
        sig.append(0); notes.append("MACD 動能轉換中")

    r = rsi.iloc[-1]
    if 50 <= r <= 70:
        sig.append(1); notes.append(f"RSI {r:.0f}，強勢但未過熱")
    elif r > 80:
        sig.append(-1); flags.append(f"RSI {r:.0f} 嚴重過熱，追高風險大"); notes.append(f"RSI {r:.0f}，過熱")
    elif r > 70:
        sig.append(0); notes.append(f"RSI {r:.0f}，偏熱")
    elif r >= 30:
        sig.append(-1); notes.append(f"RSI {r:.0f}，動能偏弱")
    else:
        sig.append(0); notes.append(f"RSI {r:.0f}，超賣（可能反彈，但趨勢仍弱）")

    if k.iloc[-1] > dd.iloc[-1] and k.iloc[-1] < 80:
        sig.append(1); notes.append(f"KD 黃金交叉向上（K={k.iloc[-1]:.0f}）")
    elif k.iloc[-1] < dd.iloc[-1] and k.iloc[-1] > 20:
        sig.append(-1); notes.append(f"KD 死亡交叉向下（K={k.iloc[-1]:.0f}）")
    else:
        sig.append(0); notes.append(f"KD 在極端區（K={k.iloc[-1]:.0f}），訊號鈍化")

    vr = v.tail(5).mean() / v.tail(20).mean() if v.tail(20).mean() else 1
    chg5 = last / c.iloc[-6] - 1
    if vr > 1.2 and chg5 > 0:
        sig.append(1); notes.append(f"價漲量增（量比 {vr:.1f}）")
    elif vr > 1.2 and chg5 < 0:
        sig.append(-1); notes.append(f"價跌量增（量比 {vr:.1f}），賣壓沉重")
    else:
        sig.append(0); notes.append("量能無明顯變化")

    bias = last / ma[60].iloc[-1] - 1
    if bias > 0.25:
        sig.append(-1); flags.append(f"股價高於季線 {bias:.0%}，乖離過大，回檔風險高")
    pos = (last - c.tail(250).min()) / (c.tail(250).max() - c.tail(250).min())
    notes.append(f"位於近一年區間的 {pos:.0%} 位置")

    score = (sum(sig) + len(sig)) / (2 * len(sig)) * 100
    return score, notes, flags, {"price": last, "ma20": ma[20].iloc[-1], "ma60": ma[60].iloc[-1]}


# ---------------- 籌碼面 ----------------
def finmind(dataset, stock_id, days=60):
    start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    r = requests.get("https://api.finmindtrade.com/api/v4/data",
                     params={"dataset": dataset, "data_id": stock_id, "start_date": start}, timeout=15)
    return pd.DataFrame(r.json().get("data", []))


def chips(tk, info):
    notes, flags, sig = [], [], []
    if is_tw(tk):
        sid = tk.split(".")[0]
        try:
            df = finmind("TaiwanStockInstitutionalInvestorsBuySell", sid)
            if df.empty:
                return None, ["查無三大法人資料（可能為 API 限流）"], [], None
            df["net"] = (df["buy"] - df["sell"]) / 1000  # 張
            dates = sorted(df["date"].unique())
            d5, d20 = dates[-5:], dates[-20:]

            def net(name, ds):
                return df[(df["name"].str.contains(name)) & (df["date"].isin(ds))]["net"].sum()

            for label, key in (("外資", "Foreign"), ("投信", "Investment_Trust")):
                n5, n20 = net(key, d5), net(key, d20)
                notes.append(f"{label}：近5日 {n5:+,.0f} 張，近20日 {n20:+,.0f} 張")
                sig.append(1 if n20 > 0 else -1)
                sig.append(1 if n5 > 0 else -1)
        except Exception as e:
            notes.append(f"三大法人資料取得失敗：{e}")
        try:
            m = finmind("TaiwanStockMarginPurchaseShortSale", sid)
            if len(m) > 20:
                b = m["MarginPurchaseTodayBalance"]
                chg = b.iloc[-1] / b.iloc[-21] - 1 if b.iloc[-21] else 0
                notes.append(f"融資餘額近20日變化 {chg:+.0%}")
                if chg > 0.15:
                    sig.append(-1); flags.append("融資暴增，散戶追高，籌碼偏亂")
                elif chg < -0.10:
                    sig.append(1)
                else:
                    sig.append(0)
        except Exception:
            pass
    else:
        inst = info.get("heldPercentInstitutions")
        short = info.get("shortPercentOfFloat")
        if inst is not None:
            notes.append(f"機構持股比例 {inst:.0%}")
            sig.append(1 if inst > 0.6 else (-1 if inst < 0.3 else 0))
        if short is not None:
            notes.append(f"空單佔流通股 {short:.1%}")
            if short > 0.10:
                sig.append(-1); flags.append("空單比例偏高，市場分歧大")
            else:
                sig.append(1 if short < 0.03 else 0)
    if not sig:
        return None, notes or ["籌碼資料不足"], flags, None
    return (sum(sig) + len(sig)) / (2 * len(sig)) * 100, notes, flags, None


# ---------------- 基本面 ----------------
def fundamentals(info):
    notes, flags, sig = [], [], []

    def add(val, label, fmt, good, bad, pct=False):
        if val is None:
            return
        txt = f"{val:.1%}" if pct else f"{val:.1f}"
        notes.append(f"{label}：{txt}")
        sig.append(1 if good(val) else (-1 if bad(val) else 0))

    add(info.get("revenueGrowth"), "營收年增率", "", lambda x: x > .20, lambda x: x < 0, True)
    add(info.get("earningsGrowth"), "獲利年增率", "", lambda x: x > .15, lambda x: x < 0, True)
    add(info.get("grossMargins"), "毛利率", "", lambda x: x >= .40, lambda x: x < .15, True)
    add(info.get("returnOnEquity"), "ROE", "", lambda x: x > .15, lambda x: x < .08, True)
    add(info.get("debtToEquity"), "負債權益比(%)", "", lambda x: x < 60, lambda x: x > 150)
    fcf = info.get("freeCashflow")
    if fcf is not None:
        notes.append(f"自由現金流：{'為正' if fcf > 0 else '為負'}")
        sig.append(1 if fcf > 0 else -1)
        if fcf < 0:
            flags.append("自由現金流為負，燒錢型公司需留意資金風險")

    pe = info.get("forwardPE") or info.get("trailingPE")
    if pe is None or pe <= 0:
        notes.append("本益比：無（可能虧損）")
        sig.append(-1); flags.append("目前無正獲利，估值難以衡量")
    else:
        notes.append(f"預估本益比：{pe:.1f} 倍")
        sig.append(1 if pe <= 25 else (-1 if pe > 50 else 0))
        if pe > 50:
            flags.append(f"本益比 {pe:.0f} 倍偏高，成長必須完全兌現才撐得住")
    pb = info.get("priceToBook")
    if pb:
        notes.append(f"股價淨值比：{pb:.1f}")
    if not sig:
        return None, ["基本面資料不足"], flags
    notes.append("（註：AI代工廠毛利率本來就低，請與同業比較，勿只看單一門檻）")
    return (sum(sig) + len(sig)) / (2 * len(sig)) * 100, notes, flags


# ---------------- 消息面 ----------------
def news(tk_obj):
    try:
        items = tk_obj.news or []
    except Exception:
        items = []
    titles = []
    for it in items[:10]:
        c = it.get("content", it)
        t = c.get("title")
        if t:
            titles.append(t)
    if not titles:
        return None, ["近期無可取得的新聞（台股新聞源較少，請自行補看公開資訊觀測站）"], []
    pos = sum(any(w in t.lower() for w in POS_WORDS) for t in titles)
    neg = sum(any(w in t.lower() for w in NEG_WORDS) for t in titles)
    score = max(0, min(100, 50 + 12 * (pos - neg)))
    notes = [f"近 {len(titles)} 則新聞：偏多 {pos} 則 / 偏空 {neg} 則（關鍵字粗估）"]
    notes += [f"  • {t[:60]}" for t in titles[:5]]
    return score, notes, []


# ---------------- 綜合 ----------------
WEIGHTS = {"技術面": 0.30, "籌碼面": 0.20, "基本面": 0.35, "消息面": 0.15}


def verdict(score, flags):
    if score >= 70 and len(flags) <= 1:
        return "偏多｜可分批布局", "分 3 批進場，單一檔不超過總資金 10~15%；跌破季線或虧損約 8~10% 就檢討。"
    if score >= 55:
        return "中性偏多｜小量試單或等回檔", "最多 5% 資金試單，等回測月線/季線不破再加碼，不追急漲。"
    if score >= 40:
        return "中性觀望｜不建議現在進場", "加入觀察名單，等基本面或趨勢轉強再說。"
    return "偏空｜建議避開", "目前風險大於報酬；已持有者應檢視停損，不要攤平。"


def analyze(query, quiet=False):
    tk = resolve(query)
    if not tk:
        print(f"找不到「{query}」，請確認代碼或名稱。")
        return None
    t = yf.Ticker(tk)
    df = t.history(period="2y")
    if df.empty or len(df) < 130:
        print(f"{tk} 的歷史資料不足，無法分析。")
        return None
    try:
        info = t.info or {}
    except Exception:
        info = {}

    name = info.get("shortName") or info.get("longName") or tk
    sector = next((v[1] for v in AI_WATCHLIST.values() if v[0] == tk), None)

    t_s, t_n, t_f, px = technical(df)
    c_s, c_n, c_f, _ = chips(tk, info)
    f_s, f_n, f_f = fundamentals(info)
    n_s, n_n, n_f = news(t)

    parts = {"技術面": t_s, "籌碼面": c_s, "基本面": f_s, "消息面": n_s}
    avail = {k: v for k, v in parts.items() if v is not None}
    total = sum(WEIGHTS[k] * v for k, v in avail.items()) / sum(WEIGHTS[k] for k in avail)
    flags = t_f + (c_f or []) + f_f + n_f
    if len(avail) < 3:
        flags.append("可用資料面向不足 3 個，結論可信度低")
    v, action = verdict(total, flags)

    if quiet:
        return {"name": name, "ticker": tk, "score": total, "verdict": v.split("｜")[0], "parts": parts}

    line = "=" * 52
    print(f"\n{line}\n{name}（{tk}）" + (f"｜AI定位：{sector}" if sector else ""))
    print(f"現價 {px['price']:.2f}｜月線 {px['ma20']:.2f}｜季線 {px['ma60']:.2f}\n{line}")
    for label, s, notes in (("技術面", t_s, t_n), ("籌碼面", c_s, c_n), ("基本面", f_s, f_n), ("消息面", n_s, n_n)):
        print(f"\n【{label}】" + (f" {s:.0f}/100" if s is not None else " 資料不足"))
        for n in notes:
            print(f"  - {n}")
    print(f"\n{line}\n綜合評分：{total:.0f}/100  →  {v}")
    print(f"操作建議：{action}")
    if flags:
        print("\n⚠ 風險警示：")
        for f in flags:
            print(f"  ! {f}")
    print("""
【誠實聲明】
 1. 這是用公開數據做的量化評分，不是預測；任何人（包含我）都無法保證漲跌。
 2. AI 類股高度同質、齊漲齊跌，買多檔 AI 股 ≠ 分散風險。
 3. 消息面為關鍵字粗估，重大事件請閱讀原文；籌碼資料有時間落差。
 4. 只用你賠得起的錢，並自行決定，投資盈虧由自己負責。""")
    return total


def scan():
    rows = []
    for name, (tk, cat) in AI_WATCHLIST.items():
        try:
            r = analyze(tk, quiet=True)
            if r:
                rows.append((r["score"], name, tk, cat, r["verdict"]))
        except Exception as e:
            print(f"{name} 失敗：{e}")
        time.sleep(0.5)
    print("\nAI 觀察清單排名（分數越高越值得研究，不代表保證上漲）")
    print("-" * 60)
    for s, n, tk, cat, v in sorted(rows, reverse=True):
        print(f"{s:5.0f}  {n}（{tk}）{cat}  →  {v}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--scan":
        scan()
    elif args:
        analyze(" ".join(args))
    else:
        print("AI 股票分析器｜輸入股票代碼或公司名稱（q 離開，scan 掃描清單）")
        while True:
            q = input("\n> ").strip()
            if q.lower() in ("q", "quit", "exit"):
                break
            if q.lower() == "scan":
                scan()
            elif q:
                analyze(q)
