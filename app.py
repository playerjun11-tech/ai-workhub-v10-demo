
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_VERSION = "V10 Cloud Final"
TEST_BRANCH = "서울중앙지점(TEST)"
BRANCH_TOTAL = 78
DATA_DATE = "2025-06-25"

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

st.set_page_config(page_title="Home&Service AI WorkHub V10", page_icon="🏠", layout="wide")

CSS = """
<style>
:root{--red:#d6001c;--soft:#fff1f3;--line:#e5e7eb;--bg:#f6f7fb;--shadow:0 10px 25px rgba(15,23,42,.07);}
.stApp{background:var(--bg);}
.main .block-container{max-width:1320px;padding-top:1rem;padding-bottom:2rem;}
header,footer,#MainMenu{visibility:hidden;}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line);}
.side-brand{display:flex;gap:10px;align-items:center;margin-bottom:10px;}
.side-logo{width:42px;height:42px;border-radius:14px;background:var(--soft);color:var(--red);display:flex;align-items:center;justify-content:center;font-size:25px;font-weight:900;}
.side-title{font-size:20px;font-weight:950;line-height:1.05}.side-title span{color:var(--red);}
.side-caption{font-size:12px;color:#6b7280;font-weight:700;line-height:1.45;margin-bottom:18px;}
.side-item{padding:11px 12px;border-radius:13px;margin:4px 0;font-size:14px;font-weight:850;color:#374151;}
.side-item.active{background:var(--soft);color:#991b1b;}
.sys-box{margin-top:10px;background:#fff;border:1px solid #edf0f3;border-radius:16px;padding:13px 15px;box-shadow:0 6px 16px rgba(15,23,42,.035);}
.sys-row{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #f1f5f9;color:#6b7280;font-size:13px}.sys-row:last-child{border-bottom:0}.sys-row b{color:#111827}.green{color:#16a34a!important;}
.hero{background:linear-gradient(135deg,#fff 0%,#fff7f8 55%,#fff 100%);border:1px solid #fee2e2;border-radius:28px;padding:28px 30px;box-shadow:var(--shadow);margin-bottom:20px;}
.hero-grid{display:grid;grid-template-columns:1fr 290px;gap:24px;align-items:center;}
.hero h1{margin:0;font-size:38px;line-height:1.08;font-weight:950;letter-spacing:-.05em;color:#111827}.hero h1 span{color:var(--red);}
.subtitle{margin:8px 0 0;color:#4b5563;font-size:16px;font-weight:800}.desc{margin:18px 0 0;color:#6b7280;font-size:13px;font-weight:650}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px}.chip{border:1px solid #fecaca;background:#fff;color:#991b1b;border-radius:999px;padding:7px 12px;font-size:12px;font-weight:950}.chip.green{background:#dcfce7;border-color:#bbf7d0;color:#15803d}
.branch-card{position:relative;background:#fff;border:1px solid #fee2e2;border-radius:22px;padding:20px 22px;box-shadow:var(--shadow)}.branch-pill{position:absolute;right:16px;top:-11px;background:#dcfce7;color:#15803d;padding:6px 13px;border-radius:999px;font-size:12px;font-weight:950}
.branch-inner{display:flex;gap:14px;align-items:center}.branch-icon{width:54px;height:54px;border-radius:16px;background:#f3f4f6;display:flex;align-items:center;justify-content:center;font-size:31px}.branch-name{font-size:17px;font-weight:950;margin:0 0 4px}.branch-info{font-size:13px;color:#4b5563;line-height:1.45;font-weight:750;margin:0}
.kpi-card{background:#fff;border:1px solid var(--line);border-radius:22px;padding:22px;box-shadow:var(--shadow);min-height:130px;display:flex;align-items:center;gap:18px}.kpi-icon{width:62px;height:62px;border-radius:18px;display:flex;align-items:center;justify-content:center;font-size:31px}.i-red{background:#ffe4e8;color:var(--red)}.i-purple{background:#ede9fe;color:#4f46e5}.i-green{background:#dcfce7;color:#16a34a}.kpi-label{font-size:14px;font-weight:950;margin:0 0 8px}.kpi-value{font-size:31px;font-weight:950;line-height:1;margin:0}.kpi-value span{font-size:15px;font-weight:850;margin-left:3px}.kpi-change{color:#16a34a;font-size:12px;font-weight:950;margin-top:8px}
.search-card{background:linear-gradient(180deg,#fff,#fff7f8);border:1px solid #fecaca;border-radius:24px;padding:24px 28px;box-shadow:var(--shadow);margin:20px 0 18px}.search-title{text-align:center;font-size:26px;font-weight:950;letter-spacing:-.03em;margin:0 0 16px}
div[data-testid="stTextInput"] input{height:52px!important;border-radius:15px!important;border:1px solid #fecaca!important;background:#fff!important;font-size:15px!important;padding:0 17px!important;box-shadow:0 8px 18px rgba(15,23,42,.045)!important}
div.stButton>button[kind="primary"]{height:52px!important;border-radius:15px!important;background:var(--red)!important;border:0!important;color:#fff!important;font-weight:950!important;font-size:16px!important;box-shadow:0 10px 20px rgba(214,0,28,.24)!important}
div.stButton>button[kind="secondary"]{height:36px!important;border-radius:999px!important;background:#fff!important;border:1px solid #e5e7eb!important;color:#374151!important;font-size:12px!important;font-weight:850!important}
.panel{background:#fff;border:1px solid var(--line);border-radius:22px;padding:18px 20px;box-shadow:var(--shadow);margin-bottom:18px}.panel-title{display:flex;align-items:center;justify-content:space-between;font-size:18px;font-weight:950;margin:0 0 13px}.panel-title.red{color:var(--red)}.notice{background:#fff1f3;border:1px solid #fee2e2;border-radius:12px;padding:10px 12px;color:#7f1d1d;font-size:13px;font-weight:750;margin-bottom:12px}.answer-text{color:#374151;font-size:14px;line-height:1.7;white-space:pre-line}
.article-row{display:grid;grid-template-columns:36px 1fr 90px 96px;gap:8px;align-items:center;padding:12px 14px;border-bottom:1px solid #edf0f3;font-size:13px}.article-row:last-child{border-bottom:0}.rank{width:23px;height:23px;border-radius:999px;background:#f9fafb;border:1px solid #e5e7eb;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:950}.article-title{font-weight:900;color:#111827}.article-cat,.article-date{color:#6b7280;font-weight:750}.article-date{text-align:right}
.bar-row{display:grid;grid-template-columns:105px 1fr 42px;gap:8px;align-items:center;font-size:12px;margin-bottom:11px}.bar-label{font-weight:850;color:#374151;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.bar-track{height:9px;background:#f1f5f9;border-radius:999px;overflow:hidden}.bar-fill{height:100%;background:linear-gradient(90deg,#fb7185,#d6001c);border-radius:999px}.bar-num{font-weight:900;color:#4b5563;text-align:right}
@media(max-width:900px){.hero-grid{grid-template-columns:1fr}.hero h1{font-size:30px}}
</style>
"""

@st.cache_data
def load_data():
    df = pd.DataFrame(SAMPLE_DATA)
    df["검색내용"] = df["제목"] + " " + df["내용"] + " " + df["분류"]
    return df

def search_data(query, df, top_n=5):
    if not query.strip():
        return df.head(top_n).copy()
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(df["검색내용"].tolist() + [query])
    sim = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    out = df.copy()
    out["유사도"] = sim
    return out.sort_values("유사도", ascending=False).head(top_n)

def make_answer(query, results):
    main = results.iloc[0]
    if not query:
        query = "고객 부재 반송 처리 기준"
    return f"""문의하신 내용은 「{main['제목']}」 기준으로 우선 확인하면 됩니다.

• 관련 업무 기준을 확인합니다.
• 고객 또는 작업 상태를 확인합니다.
• 동일 문의가 반복될 경우 업무지식 링크를 공유합니다.
• 필요한 경우 FAQ로 등록하여 지점 내 응대 기준을 표준화합니다.

관련 근거: {main['제목']}
분류: {main['분류']}

※ 본 화면은 외부 접속용 TEST 데이터 기반 데모입니다."""

def render_sidebar():
    st.sidebar.markdown("""
    <div class="side-brand"><div class="side-logo">⌂</div><div class="side-title">Home&Service<br><span>AI WorkHub</span></div></div>
    <div class="side-caption">AI 기반 지점별 업무지원 플랫폼</div>
    <div class="side-item active">🏠 홈</div><div class="side-item">🔍 업무지식 검색</div><div class="side-item">🤖 AI 업무지원</div><div class="side-item">📘 업무가이드</div><div class="side-item">📊 통계/리포트</div><div class="side-item">🏢 지점별 현황</div>
    """, unsafe_allow_html=True)
    st.sidebar.selectbox("운영 지점", [TEST_BRANCH, "전체 지점(확장 예정)"], label_visibility="collapsed")
    st.sidebar.markdown(f"""
    <div class="sys-box">
      <div class="sys-row"><span>버전</span><b>{APP_VERSION}</b></div>
      <div class="sys-row"><span>TEST 여부</span><b class="green">TEST 운영</b></div>
      <div class="sys-row"><span>접속 지점</span><b>{BRANCH_TOTAL}개 지점</b></div>
      <div class="sys-row"><span>데이터 기준일</span><b>{DATA_DATE}</b></div>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.caption("배포 후 생성된 URL을 휴대폰으로 접속하세요.")

def hero():
    st.markdown(f"""
    <div class="hero"><div class="hero-grid"><div>
    <h1>홈&서비스 AI WorkHub <span>V10 DEMO</span></h1>
    <div class="subtitle">AI 기반 지점별 업무지원 플랫폼</div>
    <div class="desc">반복 문의를 AI가 분석·답변하고, 업무지식을 축적하여 지점별 업무 표준화를 지원합니다.</div>
    <div class="chips"><div class="chip green">● TEST 운영 완료</div><div class="chip">서울중앙지점</div><div class="chip">78개 지점 확장 가능</div><div class="chip">외부 접속용 데모</div></div>
    </div><div class="branch-card"><div class="branch-pill">● DEMO</div><div class="branch-inner"><div class="branch-icon">🏢</div><div><p class="branch-name">서울중앙지점(TEST)</p><p class="branch-info">78개 지점 연결 예정<br>TEST 데이터 기준</p></div></div></div></div></div>
    """, unsafe_allow_html=True)

def kpi_cards(df):
    cols = st.columns(4)
    data = [("📕","업무지식",str(len(df)),"건","▲ 샘플 데이터","i-red"),("💬","질문 수","1,248","건","▲ DEMO 지표","i-red"),("🤖","AI 답변 성공률","96%","","▲ DEMO 지표","i-purple"),("📈","답변 만족도","98%","","▲ DEMO 지표","i-green")]
    for col, (icon,label,value,unit,change,klass) in zip(cols, data):
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-icon {klass}">{icon}</div><div><p class="kpi-label">{label}</p><p class="kpi-value">{value}<span>{unit}</span></p><div class="kpi-change">{change}</div></div></div>', unsafe_allow_html=True)

def search_area():
    st.markdown('<div class="search-card"><div class="search-title">✨ 무엇을 도와드릴까요? ✨</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,5,1])
    with c2:
        qcol,bcol = st.columns([5,1])
        with qcol:
            query = st.text_input("질문 검색", placeholder="궁금한 업무 내용을 검색해보세요. 예: 고객 부재 반송", label_visibility="collapsed", key="query")
        with bcol:
            clicked = st.button("검색", type="primary", use_container_width=True)
    st.markdown("**자주 찾는 질문**")
    cols = st.columns(4)
    for i, text in enumerate(RECOMMENDED):
        with cols[i % 4]:
            if st.button(text, key=f"rec_{i}", type="secondary", use_container_width=True):
                st.session_state.query = text
                clicked = True
                query = text
    st.markdown("</div>", unsafe_allow_html=True)
    return query if clicked or query else ""

def render_answer_and_results(query, results):
    answer = make_answer(query, results)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="panel"><div class="panel-title red">🤖 AI 답변 <span style="font-size:12px;color:#6b7280;">복사</span></div><div class="notice">AI가 업무지식을 분석하여 답변한 내용입니다.</div><div class="answer-text">{answer}</div><div style="margin-top:12px;color:#6b7280;font-size:13px;font-weight:750;">도움이 되셨나요? 👍 👎</div></div>', unsafe_allow_html=True)
    with c2:
        rows = ""
        for idx, (_, row) in enumerate(results.iterrows(), start=1):
            rows += f'<div class="article-row"><div class="rank">{idx}</div><div class="article-title">{row["제목"]}</div><div class="article-cat">{row["분류"]}</div><div class="article-date">{row["작성일"]}</div></div>'
        st.markdown(f'<div class="panel"><div class="panel-title">📚 관련 업무지식 TOP 5 <span style="font-size:12px;color:#6b7280;">더보기 〉</span></div>{rows}</div>', unsafe_allow_html=True)

def dashboard():
    st.markdown('<div class="panel"><div class="panel-title red">📊 AI 분석 대시보드</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown("**일별 질문 건수 추이**")
        st.line_chart(pd.DataFrame({"질문":[95,140,118,88,146,121,172]}))
    with c2:
        st.markdown("**문의 유형 분포**")
        st.bar_chart(pd.DataFrame({"비율":[38,28,18,10,6]}, index=["고객응대","네트워크","장애처리","개통업무","기타"]))
    with c3:
        st.markdown("**답변 성공률 추이**")
        st.line_chart(pd.DataFrame({"성공률":[85,89,87,95,97,96,98]}))
    with c4:
        st.markdown("**반복 문의 TOP 5**")
        data=[("고객 부재 반송",248),("MQT 측정",196),("재장애",178),("요금 감면",165),("개통 지연",132)]
        max_v=max(v for _,v in data)
        bars=""
        for label,val in data:
            bars += f'<div class="bar-row"><div class="bar-label">{label}</div><div class="bar-track"><div class="bar-fill" style="width:{int(val/max_v*100)}%"></div></div><div class="bar-num">{val}</div></div>'
        st.markdown(bars, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def main():
    st.markdown(CSS, unsafe_allow_html=True)
    render_sidebar()
    df=load_data()
    hero()
    kpi_cards(df)
    query=search_area()
    results=search_data(query, df)
    render_answer_and_results(query, results)
    dashboard()
    st.success("본 데모는 외부 접속용 TEST 데이터만 사용합니다. 실제 운영 시 지점별 데이터로 확장 가능합니다.")

if __name__ == "__main__":
    main()
