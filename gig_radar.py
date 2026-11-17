"""
GIG RADAR — 내 취향에 맞는 공연을 찾아주고, 티켓팅 날짜까지 챙겨주는 AI

기능
  1) 공연 검색      : 가수/키워드로 다가오는 공연 일정 검색
  2) 취향으로 찾기  : 좋아하는 아티스트를 바탕으로 비슷한 아티스트 공연까지 추천
  3) 티켓팅 D-day   : 선예매/일반예매 오픈일을 D-day 배지로 표시
  4) 캘린더 추가    : 카드별 구글 캘린더 버튼 + 전체 일정 .ics 파일 다운로드
"""
import os, re, json, html, tempfile
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlencode

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
client = OpenAI()
MODEL = "gpt-4o-mini"
KST = timezone(timedelta(hours=9))   # 서버가 해외(UTC)에 있어도 한국 시간 기준으로 계산


def today_kst() -> date:
    return datetime.now(KST).date()


# ═════════════════════════════════════════════
# 1. LLM 프롬프트
# ═════════════════════════════════════════════
EVENT_SCHEMA = """{"title":"공연명","artist":"아티스트","type":"콘서트|페스티벌|공연",
"date_start":"YYYY-MM-DD","date_end":"YYYY-MM-DD (하루 공연이면 빈 문자열)",
"venue":"장소","city":"도시",
"presale_open":"선예매(팬클럽) 오픈 'YYYY-MM-DD HH:MM' (모르면 빈 문자열)",
"ticket_open":"일반예매 오픈 'YYYY-MM-DD HH:MM' (모르면 빈 문자열)",
"ticket_site":"예매처 (예: 인터파크, 멜론티켓, YES24, 티켓링크)",
"ticket_status":"오픈 예정|예매 중|매진|미정",
"url":"출처 링크"}"""

COMMON_RULES = """규칙:
- 웹 검색으로 확인한 '실제' 정보만. 확인 안 된 값은 지어내지 말고 빈 문자열("")
- 오늘 이후에 열리는 공연만
- 티켓 오픈 일시는 공식 공지/예매처 기준으로 최대한 정확히 (시간까지)
- 최대 8개
- 반드시 JSON만 출력하고 다른 말은 쓰지 마"""

SEARCH_PROMPT = f"""너는 공연 정보 큐레이터야. 사용자가 찾는 공연의 일정과 티켓팅 정보를 찾아줘.
출력 형식:
{{"summary":"검색 결과 한두 문장 요약","events":[{EVENT_SCHEMA}]}}
{COMMON_RULES}"""

TASTE_PROMPT = f"""너는 음악 취향 분석가이자 공연 큐레이터야.
1) 사용자가 좋아하는 아티스트와 취향 설명을 보고, 음악 스타일이 비슷한 아티스트 5~8명을 골라.
   (사용자가 이미 말한 아티스트는 제외, 국내 활동이 활발한 아티스트 우선)
2) 웹 검색으로 '좋아하는 아티스트'와 '비슷한 아티스트'의 다가오는 공연, 그리고 이들이 출연하는 페스티벌을 찾아.
3) 각 공연마다 이 사용자에게 왜 맞는지 reason을 한 문장으로 써.
   예) "실리카겔의 몽환적인 사운드를 좋아한다면 이 밴드의 라이브도 취향일 거예요"
출력 형식:
{{"taste":"사용자 취향 한 줄 분석","similar":["비슷한 아티스트", ...],
"summary":"추천 결과 한두 문장 요약",
"events":[{EVENT_SCHEMA[:-1]},"match":"좋아하는 아티스트|비슷한 아티스트|페스티벌","reason":"추천 이유 한 문장"}}]}}
{COMMON_RULES}"""


def call_llm(instructions: str, user_msg: str) -> dict:
    resp = client.responses.create(
        model=MODEL,
        tools=[{"type": "web_search_preview",
                "user_location": {"type": "approximate", "country": "KR"}}],
        instructions=instructions,
        input=user_msg,
    )
    text = resp.output_text
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group()) if m else {"summary": text}
    except json.JSONDecodeError:
        return {"summary": text}


def search_events(query, region="국내", category="전체") -> dict:
    msg = (f"오늘 날짜: {today_kst().isoformat()}\n"
           f"검색어: {query}\n지역: {region}\n종류: {category}")
    return call_llm(SEARCH_PROMPT, msg)


def recommend_by_taste(artists, mood="", region="국내") -> dict:
    msg = (f"오늘 날짜: {today_kst().isoformat()}\n"
           f"좋아하는 아티스트: {artists}\n"
           f"취향 설명: {mood or '(없음)'}\n지역: {region}")
    return call_llm(TASTE_PROMPT, msg)


# ═════════════════════════════════════════════
# 2. 날짜 처리 · D-day · 캘린더
# ═════════════════════════════════════════════
DATE_RE = re.compile(r"(\d{4})\s*[-./년]\s*(\d{1,2})\s*[-./월]\s*(\d{1,2})")
TIME_RE = re.compile(r"(\d{1,2})\s*[:시]\s*(\d{2})?")
WEEK = "월화수목금토일"


def parse_date(s):
    m = DATE_RE.search(s or "")
    if not m:
        return None
    try:
        return date(*map(int, m.groups()))
    except ValueError:
        return None


def parse_dt(s):
    """'2026-11-20 20:00' → datetime(KST). 시간이 없으면 (date, None)"""
    d = parse_date(s)
    if not d:
        return None, None
    rest = (s or "")[DATE_RE.search(s).end():]
    t = TIME_RE.search(rest)
    if t:
        h, mnt = int(t.group(1)), int(t.group(2) or 0)
        if h < 12 and re.search(r"오후|pm|PM|저녁|밤", rest):
            h += 12
        if 0 <= h < 24 and 0 <= mnt < 60:
            return d, datetime(d.year, d.month, d.day, h, mnt, tzinfo=KST)
    return d, None


def fmt_date(d):
    return f"{d:%Y.%m.%d} ({WEEK[d.weekday()]})"


def fmt_range(start, end):
    if not start:
        return "날짜 미정"
    if end and end != start:
        same_year = end.year == start.year
        return f"{fmt_date(start)} – {end:%m.%d} ({WEEK[end.weekday()]})" if same_year \
            else f"{fmt_date(start)} – {fmt_date(end)}"
    return fmt_date(start)


def fmt_dt(d, dt):
    if dt:
        return f"{d:%m.%d}({WEEK[d.weekday()]}) {dt:%H:%M}"
    return f"{d:%m.%d}({WEEK[d.weekday()]})" if d else ""


def next_ticket(e):
    """아직 안 열린 가장 가까운 티켓 오픈 (라벨, date, datetime)"""
    now = datetime.now(KST)
    cands = []
    for label, key in (("선예매", "presale_open"), ("일반예매", "ticket_open")):
        d, dt = parse_dt(e.get(key, ""))
        if not d:
            continue
        if (dt and dt > now) or (not dt and d >= now.date()):
            cands.append((d, dt, label))
    if not cands:
        return None
    d, dt, label = min(cands, key=lambda x: (x[0], x[1] or datetime.max.replace(tzinfo=KST)))
    return label, d, dt


def ticket_badge(e):
    nt = next_ticket(e)
    status = (e.get("ticket_status") or "").strip()
    if nt:
        label, d, _ = nt
        diff = (d - today_kst()).days
        if diff == 0:
            return f"<span class='tbadge hot'>🔥 {label} D-DAY</span>"
        cls = "tbadge hot" if diff <= 3 else "tbadge soon"
        return f"<span class='{cls}'>⏰ {label} D-{diff}</span>"
    if "매진" in status:
        return "<span class='tbadge sold'>매진</span>"
    if "중" in status:
        return "<span class='tbadge open'>예매 중</span>"
    return ""


def gcal_url(title, start, end=None, start_dt=None, details="", location=""):
    """구글 캘린더 '일정 추가' 링크"""
    if start_dt:  # 시간 있는 일정 (티켓팅 30분)
        utc = start_dt.astimezone(timezone.utc)
        dates = f"{utc:%Y%m%dT%H%M%SZ}/{utc + timedelta(minutes=30):%Y%m%dT%H%M%SZ}"
    else:         # 종일 일정 (공연)
        last = (end or start) + timedelta(days=1)
        dates = f"{start:%Y%m%d}/{last:%Y%m%d}"
    q = {"action": "TEMPLATE", "text": title, "dates": dates,
         "details": details, "location": location}
    return "https://calendar.google.com/calendar/render?" + urlencode(q)


def _ics_escape(s):
    return (s or "").replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def make_ics(events):
    """전체 공연 + 티켓 오픈 일정을 .ics 파일로 (티켓팅은 30분 전 알림)"""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//GIG RADAR//KO", "CALSCALE:GREGORIAN"]
    n = 0
    for i, e in enumerate(events):
        start, end = parse_date(e.get("date_start")), parse_date(e.get("date_end"))
        title, loc = e.get("title", "공연"), f"{e.get('venue','')} {e.get('city','')}".strip()
        if start:
            n += 1
            lines += ["BEGIN:VEVENT", f"UID:gig-{i}-show-{stamp}@gigradar", f"DTSTAMP:{stamp}",
                      f"DTSTART;VALUE=DATE:{start:%Y%m%d}",
                      f"DTEND;VALUE=DATE:{(end or start) + timedelta(days=1):%Y%m%d}",
                      f"SUMMARY:{_ics_escape('🎤 ' + title)}", f"LOCATION:{_ics_escape(loc)}",
                      f"DESCRIPTION:{_ics_escape(e.get('url',''))}", "END:VEVENT"]
        for label, key in (("선예매", "presale_open"), ("일반예매", "ticket_open")):
            d, dt = parse_dt(e.get(key, ""))
            if not dt:
                continue
            n += 1
            utc = dt.astimezone(timezone.utc)
            lines += ["BEGIN:VEVENT", f"UID:gig-{i}-{key}-{stamp}@gigradar", f"DTSTAMP:{stamp}",
                      f"DTSTART:{utc:%Y%m%dT%H%M%SZ}",
                      f"DTEND:{utc + timedelta(minutes=30):%Y%m%dT%H%M%SZ}",
                      f"SUMMARY:{_ics_escape(f'🎟 {label} 오픈 · {title}')}",
                      f"DESCRIPTION:{_ics_escape(e.get('ticket_site',''))}",
                      "BEGIN:VALARM", "ACTION:DISPLAY", "TRIGGER:-PT30M",
                      f"DESCRIPTION:{_ics_escape(f'{label} 30분 전!')}", "END:VALARM",
                      "END:VEVENT"]
    lines.append("END:VCALENDAR")
    if n == 0:
        return None
    path = os.path.join(tempfile.gettempdir(), f"gigradar_{datetime.now():%Y%m%d_%H%M%S}.ics")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")
    return path


def clean_events(events):
    """지난 공연 제거 + 날짜순 정렬"""
    t = today_kst()
    out = []
    for e in events or []:
        if not isinstance(e, dict):
            continue
        s, en = parse_date(e.get("date_start")), parse_date(e.get("date_end"))
        if (en or s) and (en or s) < t:
            continue
        out.append(e)
    return sorted(out, key=lambda e: parse_date(e.get("date_start")) or date.max)


# ═════════════════════════════════════════════
# 3. 화면(HTML) 렌더링
# ═════════════════════════════════════════════
TYPE_COLOR = {"콘서트": "#ff3df5", "페스티벌": "#c6ff00", "공연": "#00e5ff"}
MATCH_ICON = {"좋아하는": "💖", "비슷한": "🎯", "페스티벌": "🎪"}
esc = lambda s: html.escape(str(s or ""))


def render_card(e):
    t = e.get("type") or "공연"
    color = next((c for k, c in TYPE_COLOR.items() if k in t), "#00e5ff")
    start, end = parse_date(e.get("date_start")), parse_date(e.get("date_end"))
    loc = f"{e.get('venue','')} {e.get('city','')}".strip()

    # 티켓 정보 줄
    tlines = []
    for label, key in (("선예매", "presale_open"), ("일반예매", "ticket_open")):
        d, dt = parse_dt(e.get(key, ""))
        if d:
            tlines.append(f"<p class='tline'><b>{label}</b> {esc(fmt_dt(d, dt))}</p>")
    site = e.get("ticket_site") or ""
    if site:
        tlines.append(f"<p class='tline'><b>예매처</b> {esc(site)}</p>")
    if not tlines:
        tlines.append(f"<p class='tline muted'>🎟 {esc(e.get('ticket_status') or '티켓 정보 미정')}</p>")

    # 추천 이유 (취향 탭)
    reason = ""
    if e.get("reason"):
        icon = next((v for k, v in MATCH_ICON.items() if k in (e.get("match") or "")), "💡")
        reason = f"<div class='reason'>{icon} {esc(e['reason'])}</div>"

    # 버튼: 공연 캘린더 / 티켓팅 알림 / 출처
    btns = []
    if start:
        url = gcal_url(f"🎤 {e.get('title','공연')}", start, end,
                       details=e.get("url", ""), location=loc)
        btns.append(f"<a class='btn' href='{esc(url)}' target='_blank'>📅 캘린더</a>")
    nt = next_ticket(e)
    if nt and nt[2]:
        label, d, dt = nt
        url = gcal_url(f"🎟 {label} 오픈 · {e.get('title','')}", d, start_dt=dt,
                       details=f"{site}\n{e.get('url','')}")
        btns.append(f"<a class='btn hotbtn' href='{esc(url)}' target='_blank'>⏰ 티켓팅 알림</a>")
    src = e.get("url", "")
    if src.startswith("http"):
        btns.append(f"<a class='btn ghost' href='{esc(src)}' target='_blank'>출처 →</a>")

    return f"""
    <div class='card'>
      <div class='badges'>
        <span class='badge' style='background:{color}'>{esc(t)}</span>{ticket_badge(e)}
      </div>
      <div class='date'>{esc(fmt_range(start, end))}</div>
      <h3>{esc(e.get('title'))}</h3>
      <p class='artist'>{esc(e.get('artist'))}</p>
      <p class='venue'>📍 {esc(loc or '장소 미정')}</p>
      <div class='tickets'>{''.join(tlines)}</div>
      {reason}
      <div class='btns'>{''.join(btns)}</div>
    </div>"""


def render_result(data, events, header_extra=""):
    if not events:
        msg = data.get("summary") or "다가오는 일정을 찾지 못했어요"
        return f"<div class='empty'>🔍 {esc(msg)}</div>"
    hot = sum(1 for e in events if (nt := next_ticket(e)) and (nt[1] - today_kst()).days <= 7)
    hot_html = f"<span class='hotcount'>🔥 일주일 안에 티켓팅 {hot}건</span>" if hot else ""
    return (f"{header_extra}"
            f"<div class='summary'><span>{esc(data.get('summary'))}</span>{hot_html}</div>"
            f"<div class='grid'>{''.join(render_card(e) for e in events)}</div>")


def taste_header(data):
    chips = "".join(f"<span class='chip'>{esc(a)}</span>" for a in (data.get("similar") or []))
    taste = esc(data.get("taste"))
    if not (taste or chips):
        return ""
    return (f"<div class='taste'><div class='taste-title'>🎧 당신의 취향</div>"
            f"<p>{taste}</p>"
            f"{'<div class=taste-sub>이런 아티스트도 좋아할 거예요</div><div class=chips>' + chips + '</div>' if chips else ''}"
            f"</div>")


# ═════════════════════════════════════════════
# 4. 이벤트 핸들러
# ═════════════════════════════════════════════
ERR = "<div class='empty'>⚠️ 잠시 문제가 생겼어요. 조금 뒤에 다시 시도해 주세요.</div>"


def on_search(q, region, category):
    if not q or not q.strip():
        return "<div class='empty'>검색어를 입력해 주세요 🙂</div>", gr.File(visible=False)
    if len(q) > 100:
        return "<div class='empty'>검색어는 100자 이내로 입력해 주세요</div>", gr.File(visible=False)
    try:
        data = search_events(q.strip(), region, category)
    except Exception as ex:
        print("search error:", ex)
        return ERR, gr.File(visible=False)
    events = clean_events(data.get("events"))
    ics = make_ics(events)
    return render_result(data, events), gr.File(value=ics, visible=bool(ics))


def on_taste(artists, mood, region):
    if not artists or not artists.strip():
        return "<div class='empty'>좋아하는 아티스트를 1명 이상 적어 주세요 🎶</div>", gr.File(visible=False)
    if len(artists) + len(mood or "") > 200:
        return "<div class='empty'>입력은 200자 이내로 적어 주세요</div>", gr.File(visible=False)
    try:
        data = recommend_by_taste(artists.strip(), (mood or "").strip(), region)
    except Exception as ex:
        print("taste error:", ex)
        return ERR, gr.File(visible=False)
    events = clean_events(data.get("events"))
    ics = make_ics(events)
    return render_result(data, events, taste_header(data)), gr.File(value=ics, visible=bool(ics))


# ═════════════════════════════════════════════
# 5. 사용 방법
# ═════════════════════════════════════════════
GUIDE_MD = """
### 🎧 GIG RADAR는 이런 서비스예요
**내 취향에 맞는 공연을 찾아주고, 티켓팅 날짜까지 챙겨주는 AI**예요. 가수 이름을 몰라도 취향만 알려주면 몰랐던 공연까지 찾아줘요.

### 두 가지 찾기 방법
- **🔍 공연 검색**: 보고 싶은 가수나 공연을 알고 있을 때 써요. 예) `아이유 콘서트`, `여름 락 페스티벌`
- **💖 취향으로 찾기**: 좋아하는 아티스트를 적으면 비슷한 아티스트 공연까지 추천해 줘요. 예) `잔나비, 실리카겔, 혁오`

### 결과 카드 보는 법
- **⏰ 선예매 / 일반예매 D-n**: 티켓 오픈까지 남은 날이에요. 3일 이하로 남으면 🔥 표시로 바뀌어요.
- **📅 캘린더**: 공연 날짜를 구글 캘린더에 추가해요.
- **⏰ 티켓팅 알림**: 티켓 오픈 시간을 구글 캘린더에 추가해요.
- **📥 .ics 파일**: 결과 맨 아래에서 전체 일정을 한 번에 받을 수 있어요. 아이폰, 갤럭시, 아웃룩 캘린더에서 열면 공연과 티켓 오픈 일정이 추가되고, 티켓팅 30분 전에 알림이 와요.
- **출처 →**: AI가 정보를 가져온 원래 페이지로 이동해요.

### 꼭 알아두세요
- AI가 웹에서 모은 정보라서 **날짜, 시간, 장소가 틀리거나 바뀌었을 수 있어요.** 예매하기 전에 공식 예매처 공지를 꼭 확인하세요.
- 결과가 나오기까지 10~30초 정도 걸려요.
- 입력한 내용은 결과를 찾는 데만 쓰이고, 따로 저장하지 않아요.
"""

# ═════════════════════════════════════════════
# 6. 디자인
# ═════════════════════════════════════════════
CSS = """
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&display=swap');

.gradio-container{max-width:1200px!important; width:92%!important; margin:0 auto!important;
  padding-top:6vh!important; font-family:'Pretendard',sans-serif!important; font-size:17px}
.hero{text-align:center; padding:8px 0 24px}
.hero h1{font-family:'Space Grotesk'; font-size:clamp(56px,9vw,112px); line-height:1; margin:0; letter-spacing:-3px;
  background:linear-gradient(90deg,#ff3df5 0%,#b46cff 50%,#c6ff00 100%); -webkit-background-clip:text; color:transparent}
.hero p{color:#b4b4c2; font-size:20px; margin-top:16px}
.hero .tag{display:inline-block; margin-top:4px; color:#c6ff00; font-size:15px; font-weight:600}

/* 탭 */
.gradio-container button[role=tab]{color:#9a9aa8!important; font-size:17px!important; font-weight:600!important; padding:12px 20px!important}
.gradio-container button[role=tab][aria-selected=true]{color:#fff!important; border-bottom:3px solid #ff3df5!important}

/* 입력 */
.gradio-container textarea, .gradio-container input[type=text]{font-size:18px!important; padding:16px 18px!important;
  border-radius:14px!important; color:#fff!important}
.gradio-container span[data-testid="block-info"]{color:#c9c9d6!important; font-size:15px!important; font-weight:600}
.go-btn{background:linear-gradient(90deg,#ff3df5,#7c4dff)!important; color:#fff!important; border:none!important;
  font-weight:800!important; font-size:19px!important; min-height:60px!important; border-radius:14px!important}
.go-btn:hover{filter:brightness(1.12)}

/* 라디오 */
.gradio-container label:has(input[type=radio]){background:#1f1f2b!important; border:1px solid #3a3a48!important;
  border-radius:999px!important; padding:10px 20px!important; transition:.15s}
.gradio-container label:has(input[type=radio]) span{color:#e8e8f0!important; font-size:16px!important; font-weight:500}
.gradio-container label:has(input[type=radio]):hover{border-color:#ff3df5!important}
.gradio-container label:has(input[type=radio]:checked){background:linear-gradient(90deg,#ff3df5,#7c4dff)!important; border-color:transparent!important}
.gradio-container label:has(input[type=radio]:checked) span{color:#fff!important; font-weight:700!important}

/* 사용 방법 */
#guide{border:1px solid #2e2e3a!important; border-radius:16px!important}
#guide .prose, #guide .prose *{color:#d6d6e0!important; line-height:1.7}
#guide .prose h3{color:#ff7af8!important; margin-top:18px}
#guide .prose code{background:#26263a!important; color:#c6ff00!important; padding:2px 6px; border-radius:6px}

/* 취향 분석 */
.taste{background:linear-gradient(135deg,#24112b,#14142a); border:1px solid #4a2a5a; border-radius:20px; padding:22px 24px; margin-bottom:18px}
.taste-title{color:#ff7af8; font-weight:800; font-size:15px; letter-spacing:.5px}
.taste p{color:#fff; font-size:19px; margin:8px 0 14px; line-height:1.5}
.taste-sub{color:#a8a8b5; font-size:14px; margin-bottom:8px}
.chips{display:flex; flex-wrap:wrap; gap:8px}
.chip{background:#2c1d3a; color:#f0d6ff; border:1px solid #5a3a72; padding:6px 14px; border-radius:999px; font-size:14px; font-weight:600}

/* 결과 */
.summary{display:flex; flex-wrap:wrap; align-items:center; gap:12px; color:#d0d0da; margin:8px 0 18px; font-size:16px}
.hotcount{background:#ff3df5; color:#0b0b10; font-weight:800; font-size:13px; padding:5px 12px; border-radius:999px}
.grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:18px}
.card{display:flex; flex-direction:column; background:#15151d; border:1px solid #26262f; border-radius:20px; padding:22px; transition:.2s}
.card:hover{transform:translateY(-4px); border-color:#ff3df5; box-shadow:0 10px 30px rgba(255,61,245,.18)}
.badges{display:flex; flex-wrap:wrap; gap:6px}
.badge, .tbadge{font-weight:800; font-size:12px; padding:4px 10px; border-radius:999px}
.badge{color:#0b0b10}
.tbadge.soon{background:#2a2a10; color:#c6ff00; border:1px solid #c6ff00}
.tbadge.hot{background:#ff3df5; color:#0b0b10; animation:pulse 1.4s infinite}
.tbadge.open{background:#0f2a2e; color:#00e5ff; border:1px solid #00e5ff}
.tbadge.sold{background:#2a2a33; color:#8a8a96}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(255,61,245,.6)}50%{box-shadow:0 0 0 6px rgba(255,61,245,0)}}
.date{font-family:'Space Grotesk'; color:#c6ff00; font-size:20px; margin-top:14px}
.card h3{color:#fff; font-size:19px; margin:6px 0 4px; line-height:1.35}
.card .artist{color:#ff3df5; font-weight:600; margin:2px 0}
.card .venue{color:#a8a8b5; font-size:15px; margin:4px 0 10px}
.tickets{background:#1b1b25; border-radius:12px; padding:10px 12px}
.tline{color:#d0d0da; font-size:14px; margin:3px 0}
.tline b{color:#8a8a96; font-weight:600; display:inline-block; width:62px}
.tline.muted{color:#8a8a96}
.reason{margin-top:10px; background:#221630; border-left:3px solid #ff3df5; color:#f0d6ff; font-size:14px; padding:10px 12px; border-radius:8px; line-height:1.5}
.btns{display:flex; flex-wrap:wrap; gap:8px; margin-top:auto; padding-top:14px}
.btn{font-size:13px; font-weight:700; padding:7px 12px; border-radius:10px; text-decoration:none!important;
  background:#22222e; color:#e8e8f0!important; border:1px solid #3a3a48; transition:.15s}
.btn:hover{border-color:#00e5ff; color:#00e5ff!important}
.btn.hotbtn{background:#3a1440; border-color:#ff3df5; color:#ffb3fb!important}
.btn.ghost{background:transparent; color:#00e5ff!important; border-color:transparent}
.empty{color:#b4b4c2; padding:56px; text-align:center; border:1px dashed #3a3a48; border-radius:20px; font-size:17px}
.footer{text-align:center; color:#6f6f7d; font-size:13px; margin-top:40px}
"""

colors = dict(
    body_background_fill="#0b0b10", block_background_fill="#15151d",
    block_border_color="#26262f", input_background_fill="#1c1c26",
    body_text_color="#e8e8f0", block_label_text_color="#c9c9d6",
    block_title_text_color="#c9c9d6", border_color_primary="#26262f",
    checkbox_label_background_fill="#1f1f2b",
    checkbox_label_background_fill_hover="#26263a",
    checkbox_label_background_fill_selected="#3a1d4a",
    checkbox_label_text_color="#e8e8f0",
    checkbox_label_text_color_selected="#ffffff",
    checkbox_label_border_color="#3a3a48",
)
theme = gr.themes.Base(primary_hue="pink", neutral_hue="zinc").set(
    **colors, **{k + "_dark": v for k, v in colors.items()}
)

GR6 = int(gr.__version__.split(".")[0]) >= 6
blocks_kwargs = {} if GR6 else {"css": CSS, "theme": theme}
launch_kwargs = {"css": CSS, "theme": theme} if GR6 else {}

# ═════════════════════════════════════════════
# 7. 화면 구성
# ═════════════════════════════════════════════
EMPTY = "<div class='empty'>{}</div>"

with gr.Blocks(title="GIG RADAR", **blocks_kwargs) as demo:
    gr.HTML("""<div class='hero'><h1>GIG RADAR</h1>
               <p>내 취향에 맞는 공연을 찾아주고, 티켓팅 날짜까지 챙겨주는 AI 🎧</p>
               <span class='tag'>⏰ 티켓팅 D-day · 📅 캘린더 추가 · 💖 취향 추천</span></div>""")

    with gr.Tabs():
        # ── 탭 1: 공연 검색 ──
        with gr.Tab("🔍 공연 검색"):
            with gr.Row():
                query = gr.Textbox(placeholder="예) 데이식스 콘서트, 여름 락 페스티벌, 내한 공연",
                                   show_label=False, scale=5, max_lines=1)
                s_btn = gr.Button("검색", elem_classes="go-btn", scale=1)
            with gr.Row():
                s_region = gr.Radio(["국내", "해외", "전체"], value="국내", label="지역")
                s_cat = gr.Radio(["전체", "콘서트", "페스티벌", "뮤지컬/공연"], value="전체", label="종류")
            gr.Examples(["아이유 콘서트", "락 페스티벌", "내한 공연", "이번 달 힙합 공연"], inputs=query)
            s_out = gr.HTML(EMPTY.format("아티스트나 공연 이름을 검색해 보세요 ✨"))
            s_file = gr.File(label="📥 전체 일정 캘린더 파일 (.ics)", visible=False)

        # ── 탭 2: 취향으로 찾기 ──
        with gr.Tab("💖 취향으로 찾기"):
            artists = gr.Textbox(label="좋아하는 아티스트 (쉼표로 구분)",
                                 placeholder="예) 잔나비, 실리카겔, 혁오", max_lines=1)
            mood = gr.Textbox(label="요즘 듣고 싶은 느낌 (선택)",
                              placeholder="예) 몽환적인 밴드 사운드, 신나는 페스티벌 분위기", max_lines=1)
            with gr.Row():
                t_region = gr.Radio(["국내", "해외", "전체"], value="국내", label="지역", scale=3)
                t_btn = gr.Button("내 취향 공연 찾기", elem_classes="go-btn", scale=2)
            gr.Examples([["잔나비, 실리카겔, 혁오", "몽환적인 밴드 사운드"],
                         ["뉴진스, 르세라핌", "트렌디한 K-POP"],
                         ["빈지노, 이센스", "힙합 공연"],
                         ["Oasis, Coldplay", "떼창하는 대형 공연"]],
                        inputs=[artists, mood])
            t_out = gr.HTML(EMPTY.format("좋아하는 아티스트를 알려주면 비슷한 공연까지 찾아드려요 💖"))
            t_file = gr.File(label="📥 전체 일정 캘린더 파일 (.ics)", visible=False)

    with gr.Accordion("📖 사용 방법", open=False, elem_id="guide"):
        gr.Markdown(GUIDE_MD)

    gr.HTML("<div class='footer'>AI가 웹에서 모은 정보예요 · 예매 전에 공식 공지를 꼭 확인하세요</div>")

    # concurrency_limit: 동시에 처리하는 요청 수 제한 (API 비용 보호)
    s_btn.click(on_search, [query, s_region, s_cat], [s_out, s_file], concurrency_limit=3)
    query.submit(on_search, [query, s_region, s_cat], [s_out, s_file], concurrency_limit=3)
    t_btn.click(on_taste, [artists, mood, t_region], [t_out, t_file], concurrency_limit=3)
    artists.submit(on_taste, [artists, mood, t_region], [t_out, t_file], concurrency_limit=3)

demo.queue(max_size=20)

if __name__ == "__main__":
    demo.launch(**launch_kwargs)
