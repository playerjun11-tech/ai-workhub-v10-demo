
import html
import io
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_VERSION = "V10 DEMO"
TEST_BRANCH = "서울중앙지점(TEST)"
BRANCH_TOTAL = 78
DATA_DATE = "2025-06-25"
ACCESS_URL = "Streamlit Cloud 배포 후 생성 주소 사용"

SAMPLE_DATA = [
    {"제목":"고객 부재 시 반송 처리 기준","내용":"고객 부재 시에는 고객 연락 여부, 재방문 약속 여부, 방문 횟수를 확인한 뒤 재방문 또는 반송 처리합니다.","작성일":"2025-05-10","작성자":"AI WorkHub","분류":"고객응대"},
    {"제목":"MQT 측정 기준 및 절차","내용":"MQT 측정 시 장비 MAC 주소, 측정 위치, 측정 일시를 확인하고 기준값 미달 시 재측정 또는 조치 요청합니다.","작성일":"2025-05-08","작성자":"AI WorkHub","분류":"네트워크"},
    {"제목":"재장애 발생 시 처리 절차","내용":"동일 고객에게 반복 장애가 발생하면 이전 조치 이력, 장비 교체 여부, 선로 상태를 확인하고 재장애로 등록합니다.","작성일":"2025-05-05","작성자":"AI WorkHub","분류":"장애처리"},
    {"제목":"승주 설비 점검 기준","내용":"승주 작업 전 안전장비 착용 여부, 설비 상태, 작업 가능 여부를 확인하고 이상 시 작업을 보류합니다.","작성일":"2025-05-01","작성자":"AI WorkHub","분류":"설비관리"},
    {"제목":"개통 지연 시 고객 안내 기준","내용":"개통 지연 발생 시 지연 사유, 예상 완료 시간, 후속 조치 계획을 고객에게 안내합니다.","작성일":"2025-04-28","작성자":"AI WorkHub","분류":"개통업무"},
    {"제목":"장비 미보유 시 조치 방법","내용":"작업에 필요한 장비가 부족한 경우 지점 보유 재고 확인 후 대체 장비 또는 추가 불출을 요청합니다.","작성일":"2025-04-25","작성자":"AI WorkHub","분류":"자재"},
    {"제목":"요금 감면 요청 처리 기준","내용":"고객 요금 감면 문의는 장애 기간, 처리 결과, 보상 기준을 확인한 뒤 담당 부서 기준에 따라 안내합니다.","작성일":"2025-04-22","작성자":"AI WorkHub","분류":"FAQ"},
    {"제목":"A/S 처리 완료 후 확인사항","내용":"A/S 완료 후 고객 확인, 장비 정상 동작, 시스템 처리 상태, 재방문 필요 여부를 확인합니다.","작성일":"2025-04-20","작성자":"AI WorkHub","분류":"AS"},
    {"제목":"작업관리 상태 변경 기준","내용":"작업관리 상태는 대기, 처리중, 완료, 반송 기준에 따라 변경하며 동일 건 중복 처리 여부를 확인합니다.","작성일":"2025-04-18","작성자":"AI WorkHub","분류":"작업관리"},
    {"제목":"품질 개선 대상 선정 기준","내용":"재장애, 고객 불만, 반복 문의, 장비 교체 이력 등을 기준으로 품질 개선 대상을 선정합니다.","작성일":"2025-04-15","작성자":"AI WorkHub","분류":"품질"},
]

RECOMMENDED = ["고객 부재 반송", "MQT 기준", "재장애 처리", "승주 설비", "개통 지연", "장비 미보유", "요금 감면", "A/S 처리"]

st.set_page_config(page_title="Home&Service AI WorkHub V10 Demo", page_icon="🏠", layout="wide")

def safe(v):
    return html.escape(str(v))

def load_data():
    df = pd.DataFrame(SAMPLE_DATA)
    df["링크"] = ""
    df["검색내용"] = df["제목"] + " " + df["내용"] + " " + df["분류"]
    return df

def run_search(question, df, top_n=5):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(df["검색내용"].tolist() + [question])
    sim = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    result = df.copy()
    result["유사도"] = sim
    return result.sort_values("유사도", ascending=False).head(top_n)

def make_answer(question, result):
    main = result.iloc[0]
    return f"""문의하신 내용은 「{main['제목']}」 기준으로 우선 확인하면 됩니다.

• 관련 업무 기준을 확인합니다.
• 고객 또는 작업 상태를 확인합니다.
• 동일 문의가 반복될 경우 업무지식 링크를 공유합니다.
• 필요한 경우 FAQ로 등록하여 지점 내 응대 기준을 표준화합니다.

관련 근거: {main['제목']}
분류: {main['분류']}
※ 본 화면은 외부 접속용 TEST 데이터 기반 데모입니다."""

def render_html(df, query="", result=None):
    if result is None or result.empty:
        result = df.head(5).copy()
        answer = """고객이 부재중일 때의 반송 처리 기준은 다음과 같습니다.

• 고객 연락 후 재방문 약속이 있는 경우 → 재방문 처리
• 고객 연락 불가 또는 거부 의사 표시 → 반송 처리
• 2회 이상 방문 후 부재 시 → 반송 처리 가능

※ 본 화면은 외부 접속용 TEST 데이터 기반 데모입니다."""
    else:
        answer = make_answer(query, result)

    article_rows = ""
    for i, (_, row) in enumerate(result.iterrows(), start=1):
        article_rows += f"""
        <div class="article-row">
          <div class="rank">{i}</div>
          <div class="article-title">{safe(row['제목'])}</div>
          <div class="article-cat">{safe(row['분류'])}</div>
          <div class="article-date">{safe(row['작성일'])}</div>
        </div>
        """

    top_bars = [
        ("고객 부재 반송", 248),
        ("MQT 측정 기준", 196),
        ("재장애 처리", 178),
        ("요금 감면", 165),
        ("개통 지연", 132),
    ]
    maxv = max(v for _, v in top_bars)
    bars = ""
    for label, val in top_bars:
        bars += f"""
        <div class="bar-row">
          <div class="bar-label">{safe(label)}</div>
          <div class="bar-track"><div class="bar-fill" style="width:{int(val/maxv*100)}%"></div></div>
          <div class="bar-num">{val}</div>
        </div>
        """

    html_doc = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8"/>
<style>
*{{box-sizing:border-box}}
body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans KR",Arial,sans-serif;background:#f6f7fb;color:#111827}}
.app{{display:grid;grid-template-columns:250px 1fr;min-height:100vh}}
.sidebar{{background:#fff;border-right:1px solid #e5e7eb;padding:22px 18px}}
.brand{{display:flex;align-items:center;gap:10px;margin-bottom:10px}}
.logo{{width:42px;height:42px;border-radius:14px;background:#fff1f3;color:#d6001c;display:flex;align-items:center;justify-content:center;font-size:25px;font-weight:900}}
.brand-title{{font-size:20px;font-weight:950;line-height:1.05}}.brand-title span{{color:#d6001c}}
.caption{{color:#6b7280;font-size:12px;font-weight:700;margin-bottom:24px}}
.menu{{border-top:1px solid #edf0f3;border-bottom:1px solid #edf0f3;padding:10px 0;margin-bottom:20px}}
.menu-item{{padding:12px 14px;border-radius:13px;margin:4px 0;font-size:14px;font-weight:850;color:#374151}}
.menu-item.active{{background:#fff1f3;color:#991b1b}}
.side-section{{margin-top:18px;font-size:14px;font-weight:950}}
.selectbox{{margin-top:8px;border:1px solid #e5e7eb;border-radius:12px;padding:11px 12px;font-size:13px;font-weight:800;color:#374151}}
.system{{margin-top:12px;background:#fff;border:1px solid #edf0f3;border-radius:16px;padding:13px 15px;box-shadow:0 6px 16px rgba(15,23,42,.035)}}
.sys-row{{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #f1f5f9;color:#6b7280;font-size:13px}}
.sys-row:last-child{{border-bottom:0}}.sys-row b{{color:#111827}}.green{{color:#16a34a!important}}
.qr{{margin-top:18px;background:linear-gradient(180deg,#fff1f3,#fff);border:1px solid #fecaca;border-radius:18px;padding:18px;text-align:center}}
.fake-qr{{width:132px;height:132px;background:#fff;margin:8px auto;background-image:linear-gradient(90deg,#000 10px,transparent 10px),linear-gradient(#000 10px,transparent 10px);background-size:22px 22px;border:12px solid #fff;box-shadow:inset 0 0 0 8px #000}}
.main{{padding:28px 36px 26px}}
.hero{{background:linear-gradient(135deg,#fff 0%,#fff7f8 55%,#fff 100%);border:1px solid #fee2e2;border-radius:28px;padding:26px 28px;box-shadow:0 12px 28px rgba(15,23,42,.06);margin-bottom:20px;position:relative;overflow:hidden}}
.hero:after{{content:"";position:absolute;width:260px;height:260px;border-radius:50%;right:-110px;top:-120px;background:rgba(214,0,28,.06)}}
.header{{display:flex;justify-content:space-between;align-items:flex-start;gap:24px;position:relative;z-index:1}}
.title{{margin:0;font-size:38px;font-weight:950;letter-spacing:-.05em}}.title span{{color:#d6001c}}
.subtitle{{margin:8px 0 0;color:#4b5563;font-size:16px;font-weight:800}}
.desc{{margin:18px 0 0;color:#6b7280;font-size:13px;font-weight:650}}
.hero-chips{{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px}}
.chip{{border:1px solid #fecaca;background:#fff;color:#991b1b;border-radius:999px;padding:7px 12px;font-size:12px;font-weight:950}}.chip.green{{background:#dcfce7;border-color:#bbf7d0;color:#15803d}}
.branch-card{{position:relative;min-width:285px;background:#fff;border:1px solid #fee2e2;border-radius:22px;padding:20px 22px;box-shadow:0 10px 25px rgba(15,23,42,.07)}}
.pill{{position:absolute;right:16px;top:-11px;background:#dcfce7;color:#15803d;padding:6px 13px;border-radius:999px;font-size:12px;font-weight:950}}
.branch-flex{{display:flex;align-items:center;gap:14px}}.branch-icon{{width:54px;height:54px;border-radius:16px;background:#f3f4f6;display:flex;align-items:center;justify-content:center;font-size:31px}}
.branch-name{{font-size:17px;font-weight:950;margin:0 0 4px}}.branch-info{{font-size:13px;color:#4b5563;line-height:1.45;font-weight:750;margin:0}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin-bottom:20px}}
.kpi{{background:#fff;border:1px solid #e5e7eb;border-radius:22px;padding:22px;min-height:132px;box-shadow:0 10px 25px rgba(15,23,42,.07);display:flex;align-items:center;gap:18px}}
.kpi-icon{{width:62px;height:62px;border-radius:18px;display:flex;align-items:center;justify-content:center;font-size:31px}}.redicon{{background:#ffe4e8;color:#d6001c}}.purpleicon{{background:#ede9fe;color:#4f46e5}}.greenicon{{background:#dcfce7;color:#16a34a}}
.kpi-label{{font-size:14px;font-weight:950;margin:0 0 8px}}.kpi-value{{font-size:31px;font-weight:950;line-height:1;margin:0}}.kpi-value span{{font-size:15px;font-weight:850;margin-left:3px}}.kpi-change{{color:#16a34a;font-size:12px;font-weight:950;margin-top:8px}}
.search-card{{background:linear-gradient(180deg,#fff,#fff7f8);border:1px solid #fecaca;border-radius:24px;padding:24px 28px;box-shadow:0 10px 25px rgba(15,23,42,.07);margin-bottom:18px}}
.search-title{{text-align:center;font-size:26px;font-weight:950;letter-spacing:-.03em;margin:0 0 16px}}
.searchbox{{display:flex;gap:10px;justify-content:center}}.searchbox input{{width:62%;height:52px;border:1px solid #fecaca;border-radius:15px;padding:0 17px;font-size:15px;outline:none;box-shadow:0 8px 18px rgba(15,23,42,.045)}}.searchbox button{{min-width:74px;height:52px;border:0;border-radius:15px;background:#d6001c;color:#fff;font-size:16px;font-weight:950;box-shadow:0 10px 20px rgba(214,0,28,.24);cursor:pointer}}
.reco-title{{font-size:14px;font-weight:950;margin:18px 0 10px}}.reco-list{{display:flex;flex-wrap:wrap;gap:10px}}.reco{{border:1px solid #e5e7eb;background:#fff;border-radius:999px;padding:8px 14px;font-size:12px;font-weight:850;color:#374151;cursor:pointer}}
.content-grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px}}
.panel{{background:#fff;border:1px solid #e5e7eb;border-radius:22px;padding:18px 20px;box-shadow:0 10px 25px rgba(15,23,42,.07)}}
.panel-title{{display:flex;justify-content:space-between;align-items:center;font-size:18px;font-weight:950;margin:0 0 13px}}.panel-title.red{{color:#d6001c}}.copy{{border:1px solid #e5e7eb;background:#fff;border-radius:10px;padding:6px 10px;font-size:12px;font-weight:900}}
.notice{{background:#fff1f3;border:1px solid #fee2e2;border-radius:12px;padding:10px 12px;color:#7f1d1d;font-size:13px;font-weight:750;margin-bottom:12px}}
.answer-text{{color:#374151;font-size:14px;line-height:1.7;white-space:pre-line}}
.article-list{{border:1px solid #edf0f3;border-radius:14px;overflow:hidden}}.article-row{{display:grid;grid-template-columns:34px 1fr 100px 96px;gap:8px;align-items:center;padding:12px 14px;border-bottom:1px solid #edf0f3;font-size:13px}}.article-row:last-child{{border-bottom:0}}.rank{{width:23px;height:23px;border-radius:999px;background:#f9fafb;border:1px solid #e5e7eb;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:950}}.article-title{{font-weight:900;color:#111827;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.article-cat,.article-date{{color:#6b7280;font-weight:750}}.article-date{{text-align:right}}
.dashboard{{background:#fff;border:1px solid #e5e7eb;border-radius:22px;padding:18px 20px;box-shadow:0 10px 25px rgba(15,23,42,.07)}}.dash-title{{font-size:19px;font-weight:950;color:#d6001c;margin:0 0 14px}}.dash-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}}.chart-card{{background:#fff;border:1px solid #e5e7eb;border-radius:18px;padding:15px;min-height:178px}}.chart-title{{color:#111827;font-size:14px;font-weight:950;margin:0 0 12px}}.line-svg{{width:100%;height:118px}}.donut-wrap{{height:118px;display:flex;align-items:center;justify-content:center}}.donut{{width:106px;height:106px;border-radius:50%;background:conic-gradient(#ef4444 0 38%,#22c55e 38% 66%,#a855f7 66% 84%,#f59e0b 84% 94%,#cbd5e1 94% 100%);position:relative}}.donut:after{{content:"";position:absolute;inset:28px;background:#fff;border-radius:50%}}
.bars{{display:flex;flex-direction:column;gap:11px}}.bar-row{{display:grid;grid-template-columns:85px 1fr 42px;gap:8px;align-items:center;font-size:12px}}.bar-label{{font-weight:850;color:#374151;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.bar-track{{height:9px;background:#f1f5f9;border-radius:999px;overflow:hidden}}.bar-fill{{height:100%;background:linear-gradient(90deg,#fb7185,#d6001c);border-radius:999px}}.bar-num{{font-weight:900;color:#4b5563;text-align:right}}.bottom-note{{margin-top:12px;background:#fff;border:1px solid #e5e7eb;border-radius:18px;padding:14px 16px;color:#374151;font-weight:800;font-size:13px}}
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="brand"><div class="logo">⌂</div><div class="brand-title">Home&Service<br><span>AI WorkHub</span></div></div>
    <div class="caption">AI 기반 지점별 업무지원 플랫폼</div>
    <div class="menu">
      <div class="menu-item active">🏠 홈</div><div class="menu-item">🔍 업무지식 검색</div><div class="menu-item">🤖 AI 업무지원</div><div class="menu-item">📘 업무가이드</div><div class="menu-item">📊 통계/리포트</div><div class="menu-item">🏢 지점별 현황</div><div class="menu-item">📣 공지사항</div><div class="menu-item">⚙️ 설정</div>
    </div>
    <div class="side-section">● 운영 지점</div><div class="selectbox">서울중앙지점(TEST)⌄</div>
    <div class="side-section">● 시스템 정보</div>
    <div class="system">
      <div class="sys-row"><span>버전</span><b>V10 DEMO</b></div><div class="sys-row"><span>TEST 여부</span><b class="green">TEST 운영</b></div><div class="sys-row"><span>접속 지점</span><b>78개 지점</b></div><div class="sys-row"><span>데이터 기준일</span><b>2025-06-25</b></div>
    </div>
    <div class="qr"><b style="color:#d6001c;">모바일 접속 QR</b><div class="fake-qr"></div><div style="font-size:12px;color:#6b7280;font-weight:800;">배포 URL QR로 교체 가능</div></div>
  </aside>
  <main class="main">
    <section class="hero">
      <div class="header">
        <div><h1 class="title">홈&서비스 AI WorkHub <span>V10 DEMO</span></h1><p class="subtitle">AI 기반 지점별 업무지원 플랫폼</p><p class="desc">반복 문의를 AI가 분석·답변하고, 업무지식을 축적하여 지점별 업무 표준화를 지원합니다.</p><div class="hero-chips"><div class="chip green">● TEST 운영 완료</div><div class="chip">서울중앙지점</div><div class="chip">78개 지점 확장 가능</div><div class="chip">외부 접속용 데모</div></div></div>
        <div class="branch-card"><div class="pill">● DEMO</div><div class="branch-flex"><div class="branch-icon">🏢</div><div><p class="branch-name">서울중앙지점(TEST)</p><p class="branch-info">78개 지점 연결 예정<br>TEST 데이터 기준</p></div></div></div>
      </div>
    </section>
    <section class="kpis">
      <div class="kpi"><div class="kpi-icon redicon">📕</div><div><p class="kpi-label">업무지식</p><p class="kpi-value">{len(df)}<span>건</span></p><div class="kpi-change">▲ 샘플 데이터</div></div></div>
      <div class="kpi"><div class="kpi-icon redicon">💬</div><div><p class="kpi-label">질문 수</p><p class="kpi-value">1,248<span>건</span></p><div class="kpi-change">▲ DEMO 지표</div></div></div>
      <div class="kpi"><div class="kpi-icon purpleicon">🤖</div><div><p class="kpi-label">AI 답변 성공률</p><p class="kpi-value">96%</p><div class="kpi-change">▲ DEMO 지표</div></div></div>
      <div class="kpi"><div class="kpi-icon greenicon">📈</div><div><p class="kpi-label">답변 만족도</p><p class="kpi-value">98%</p><div class="kpi-change">▲ DEMO 지표</div></div></div>
    </section>
    <section class="search-card">
      <div class="search-title">✨ 무엇을 도와드릴까요? ✨</div>
      <div class="searchbox"><input id="qInput" value="{safe(query)}" placeholder="궁금한 업무 내용을 검색해보세요. (예: 고객 부재 반송 처리 기준)" /><button onclick="goSearch()">검색</button></div>
      <div class="reco-title">자주 찾는 질문</div>
      <div class="reco-list">{"".join([f'<div class="reco" onclick="chipSearch(\\'{safe(r)}\\')">{safe(r)}</div>' for r in RECOMMENDED])}</div>
    </section>
    <section class="content-grid">
      <div class="panel"><div class="panel-title red">🤖 AI 답변 <button class="copy">복사</button></div><div class="notice">AI가 업무지식을 분석하여 답변한 내용입니다.</div><div class="answer-text">{safe(answer)}</div><div style="margin-top:12px;color:#6b7280;font-size:13px;font-weight:750;">도움이 되셨나요? 👍 👎</div></div>
      <div class="panel"><div class="panel-title">📚 관련 업무지식 TOP 5 <span style="font-size:12px;color:#6b7280;">더보기 〉</span></div><div class="article-list">{article_rows}</div></div>
    </section>
    <section class="dashboard">
      <div class="dash-title">📊 AI 분석 대시보드</div>
      <div class="dash-grid">
        <div class="chart-card"><div class="chart-title">일별 질문 건수 추이</div><svg class="line-svg" viewBox="0 0 300 120"><polyline points="10,82 50,56 90,70 130,88 170,58 210,74 250,48 290,36" fill="none" stroke="#ef4444" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/><circle cx="10" cy="82" r="4" fill="#ef4444"/><circle cx="50" cy="56" r="4" fill="#ef4444"/><circle cx="90" cy="70" r="4" fill="#ef4444"/><circle cx="130" cy="88" r="4" fill="#ef4444"/><circle cx="170" cy="58" r="4" fill="#ef4444"/><circle cx="210" cy="74" r="4" fill="#ef4444"/><circle cx="250" cy="48" r="4" fill="#ef4444"/><circle cx="290" cy="36" r="4" fill="#ef4444"/></svg></div>
        <div class="chart-card"><div class="chart-title">문의 유형 분포</div><div class="donut-wrap"><div class="donut"></div></div></div>
        <div class="chart-card"><div class="chart-title">답변 성공률 추이</div><svg class="line-svg" viewBox="0 0 300 120"><polyline points="10,88 55,78 100,64 145,70 190,38 235,30 290,34" fill="none" stroke="#22c55e" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/><circle cx="10" cy="88" r="4" fill="#22c55e"/><circle cx="55" cy="78" r="4" fill="#22c55e"/><circle cx="100" cy="64" r="4" fill="#22c55e"/><circle cx="145" cy="70" r="4" fill="#22c55e"/><circle cx="190" cy="38" r="4" fill="#22c55e"/><circle cx="235" cy="30" r="4" fill="#22c55e"/><circle cx="290" cy="34" r="4" fill="#22c55e"/></svg></div>
        <div class="chart-card"><div class="chart-title">반복 문의 TOP 5</div><div class="bars">{bars}</div></div>
      </div>
    </section>
    <div class="bottom-note">✅ 본 데모는 외부 접속용 TEST 데이터만 사용합니다. 실제 운영 시 지점별 데이터로 확장 가능합니다.</div>
  </main>
</div>
<script>
function goSearch() {{
  const q = document.getElementById('qInput').value || '';
  const base = window.parent.location.pathname;
  window.parent.location.href = base + '?q=' + encodeURIComponent(q);
}}
function chipSearch(q) {{
  const base = window.parent.location.pathname;
  window.parent.location.href = base + '?q=' + encodeURIComponent(q);
}}
document.getElementById('qInput').addEventListener('keydown', function(e) {{
  if(e.key === 'Enter') goSearch();
}});
</script>
</body></html>
"""
    components.html(html_doc, height=1080, scrolling=True)

def main():
    st.markdown("""
    <style>
    .main .block-container {padding:0; max-width:100%;}
    [data-testid="stSidebar"]{display:none;}
    header, footer, #MainMenu {visibility:hidden;}
    </style>
    """, unsafe_allow_html=True)
    df = load_data()
    query = st.query_params.get("q", "")
    result = run_search(query, df) if query else None
    render(df, query, result)

if __name__ == "__main__":
    main()
