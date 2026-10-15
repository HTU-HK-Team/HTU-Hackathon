"""Government Services Chatbot - 42 Irbid AI Hackathon.

    python main.py questions.json            -> writes answers.json in the same folder
    python main.py questions.json --debug    -> also prints routing + contexts per question

Pure Python standard library (no pip packages). Needs only Ollama running with the model pulled.

How it works
  1. Data: each of the 12 files is parsed into its 9 fields (menus removed, whitespace cleaned,
     duplicate lines removed; facts untouched).
  2. Routing (rules, not embeddings): the question is mapped to (document, situation) pairs,
     e.g. "الباسبور ضاع" -> passport + lost -> passport_lost. Dialect words are covered.
     Each situation is paired with the nearest document mentioned, so multi-part questions work.
     Follow-up messages inherit the document / situation of the previous message.
  3. Generation: the model gets ONLY the routed service(s) text and must answer from it.
     Unclear (no document named) -> the model asks which document.
     Out of scope -> the model says it doesn't have that information.
  4. contexts: the lines of the service text that the answer actually used (word/number overlap).
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ============================== SETTINGS ==============================
MODEL = os.environ.get("MODEL", "command-r7b-arabic:7b")     # <- the organisers' model
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
NUM_CTX = 8192            # Ollama's default context is small and truncates silently (largest prompt here ~4K tokens)
SEED = 42
CALL_TIMEOUT = 180        # seconds per model call (2 retries)
NUM_PREDICT = 450         # max answer tokens
TEMPERATURE = 0.0
HISTORY_TURNS = 2         # previous Q/A pairs of the same chat sent to the model
# Tunable knobs (grid.py searches them and writes best_config.json, which is loaded here automatically)
CONFIG = {
    "context": "focused",     # focused: only the fields the question asks about (+conditions, notes) | full: whole service
    "style": "detailed",      # concise | detailed
    "facts": True,            # extract age / validity / abroad / nth time from the question and give them to the model
    "temperature": 0.0,
    "decompose": True,        # one model call per service for multi-document questions
}
MAX_CONTEXT_CHARS = 9000
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
MOCK = os.environ.get("MOCK_LLM") == "1"           # offline test of everything except the model
DEBUG = "--debug" in sys.argv
STATS = {"messages": 0, "rules": 0, "llm_fallback": 0, "clarify": 0, "refuse": 0, "fee_rule_in_code": 0,
         "number_check_regenerated": 0, "number_check_still_bad": 0, "errors": 0, "max_prompt_tokens": 0,
         "near_ctx_limit": 0}
try:
    CONFIG.update(json.loads((HERE / "best_config.json").read_text(encoding="utf-8")))
except Exception:
    pass

# ============================== TEXT NORMALISATION ==============================
TASHKEEL = re.compile(r"[ً-ٰٟـ]")
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def norm(s):
    s = TASHKEEL.sub("", s).translate(AR_DIGITS).lower()
    s = re.sub("[أإآٱ]", "ا", s)
    return s.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و")


def is_arabic(s):
    return bool(re.search(r"[؀-ۿ]", s))


# ============================== DATA ==============================
DOCS = {"passport": "جواز السفر", "family_book": "دفتر العائلة", "id_card": "البطاقة الذكية (الهوية)"}
SITS = {"first_issue": "لأول مرة", "renewal": "تجديد", "lost": "بدل فاقد", "damaged": "بدل تالف"}
TITLES = {
    "passport_first_issue": "جواز سفر لأول مرة", "passport_renewal": "تجديد جواز السفر",
    "passport_lost": "جواز سفر بدل فاقد", "passport_damaged": "جواز سفر بدل تالف",
    "family_book_first_issue": "دفتر عائلة لأول مرة", "family_book_renewal": "تجديد دفتر العائلة",
    "family_book_lost": "دفتر عائلة بدل فاقد", "family_book_damaged": "دفتر عائلة بدل تالف",
    "id_card_first_issue": "بطاقة ذكية لأول مرة", "id_card_renewal": "تجديد البطاقة الذكية",
    "id_card_lost": "بطاقة ذكية بدل فاقد", "id_card_damaged": "بطاقة ذكية بدل تالف",
}
# (label prefix in the file, field name we use)
FIELDS = [("الفئة المستفيدة", "الفئة المستفيدة"), ("أماكن تقديم الخدمة", "أماكن تقديم الخدمة"),
          ("شروط تقديم الخدمة", "الشروط"), ("الوثائق المطلوبة", "الوثائق المطلوبة"),
          ("إجراءات تقديم الخدمة", "الإجراءات"), ("الزمن المستغرق", "الزمن المستغرق"),
          ("شركاء الخدمة", "شركاء الخدمة"), ("ملاحظات", "ملاحظات"), ("الرسوم", "الرسوم")]
MENU = {"الرئيسية", "الخدمات", "دليل الخدمات", "شارك"}


def clean_line(s):
    s = s.replace("\t", " ").replace("]", "").replace("[", "")
    return re.sub(r"\s+", " ", s).strip(" .")


def parse_file(path):
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if "شارك" in [l.strip() for l in lines]:                    # website menu ends with "شارك" + title
        i = [l.strip() for l in lines].index("شارك")
        lines = lines[i + 2:]
    else:
        lines = [l for l in lines if l.strip() not in MENU]
    fields, cur, seen = {}, None, set()
    for raw in lines:
        st = raw.strip()
        hit = None
        for label, name in FIELDS:
            if name not in seen and norm(st).startswith(norm(label)):
                hit = (label, name)
                break
        if hit:
            cur = hit[1]
            seen.add(cur)
            fields[cur] = []
            rest = raw.split("\t", 1)[1] if "\t" in raw else ""
            if not rest and len(st) > 45:                       # label and value on one line, no tab
                rest = st[len(hit[0]):]
            raw = rest
        if cur is None:
            continue
        line = clean_line(raw)
        if line and norm(line) not in {norm(x) for x in fields[cur]}:
            fields[cur].append(line)
    return {k: v for k, v in fields.items() if v}


def load_services():
    services = {}
    for path in sorted(DATA_DIR.glob("*.txt")):
        sid = path.stem
        doc = next((d for d in DOCS if sid.startswith(d)), None)
        sit = next((s for s in SITS if sid.endswith(s)), None)
        services[sid] = {"id": sid, "doc": doc, "sit": sit, "title": TITLES.get(sid, sid),
                         "fields": parse_file(path)}
    return services


def service_text(svc, only_fields=None):
    parts = [f"=== الخدمة: {svc['title']} ==="]
    for name, lines in svc["fields"].items():
        if only_fields and name not in only_fields:
            continue
        parts.append(f"[{name}]\n" + "\n".join(lines))
    return "\n".join(parts)


# ============================== ROUTING ==============================
def rx(p):
    return re.compile(norm(p))


DOC_RX = {
    "passport": rx(r"جواز|باسبور|passport"),
    "family_book": rx(r"دفتر|family book|family-book"),
    "id_card": rx(r"هوي[هت]|الشخصيه|الذكيه|id card|identity card|national id|smart card|\bid\b"),
}
DOC_WEAK_RX = {"family_book": rx(r"عائل|عيل|اسره|اسرتي|رب الاسره|family")}
CARD_RX = rx(r"بطاق\w*|كرت")
CARD_NOT = rx(r"^\s*(ال)?(جسور|احصاءات|اصفر|صفراء|فلسطيني|اسرائيلي|بنك|ائتمان|صراف)")
SIT_RX = {
    "lost": rx(r"فاقد|فقد|مفقود|ضاع|ضايع|ضياع|ضيع|سرق|مسروق|انسرق|lost|stolen|missing"),
    "damaged": rx(r"تالف|تلف(?!ون|زيون)|اهتر|اهرت|مهتري|ممزق|تمزق|مشقوق|انشق|مقطوع|انقطع|تقطع|مخزوق|انخزق|مكسور|انكسر|"
                  r"خربان|خرب|محروق|انحرق|مبلول|انبل|تبلل|غسال|damaged|torn|broken|ruined"),
    "renewal": rx(r"تجديد|جدد|renew"),
    "first_issue": rx(r"اول مر[هت]|لاول مر[هت]|first time|first issue|for the first"),
}
SIT_WEAK_RX = {   # only used when no strong situation word is attached to that document
    "renewal": rx(r"منتهي|انتهي|انتهت|خلص|صلاحي|ممتلئ|امتلا|تعبت|expired|expire"),
    "first_issue": rx(r"اصدار|استخراج|اطلع|بطلع|نطلع|يطلع|طلع|جديد|فتح|افتح|تزوج|زواج|new|issue|get a|apply for a"),
}
OOS_RX = rx(r"رخص|سواق|قياد|شهاده (جامع|ميلاد|خبر|عدم)|ميلاد|تصديق|فيزا|تاشير|اقام[هت]|طيران|تذكر|سيار|ضريب|"
            r"جامع|محكومي|تجنيس|بنك|كهرباء|مياه|فاتور|مخالف|سجل تجاري|license|visa|flight|ticket|tax|degree")

FIELD_RX = {
    "الرسوم": rx(r"رسوم|رسم|بدفع|ادفع|يدفع|ندفع|تكلف|بكلف|كلف|سعر|دينار|دنانير|مصاري|فلوس|غرام|كفال|اغلي|ارخص|"
                 r"cost|fee|price|pay|how much"),
    "الوثائق المطلوبة": rx(r"وثائق|اوراق|ورق|مستند|مطلوب|بلزم|يلزم|لازم|احضر|اجيب|بجيب|نجيب|بحتاج|احتاج|يحتاج|صور|"
                           r"document|paper|bring|need|photo"),
    "الشروط": rx(r"شروط|شرط|يحق|بقدر|يقدر|نقدر|بيقدر|مسموح|ممكن|يمكن|موافق|مين|من يحق|عمر|سن |condition|eligible|who|allowed|can i"),
    "أماكن تقديم الخدمة": rx(r"وين|اين|مكان|اماكن|سفار|قنصلي|مكتب|برا|خارج|where|embassy|abroad"),
    "الزمن المستغرق": rx(r"يستغرق|بستغرق|وقت|مد[هت]|بطول|بتاخذ|بتاخد|بياخذ|بياخد|دقيق|ساع|how long|time"),
    "الإجراءات": rx(r"اجراء|خطوات|كيف|شو بعمل|شو اعمل|شو لازم اعمل|ماذا افعل|steps|how do|procedure|process"),
    "شركاء الخدمة": rx(r"شركاء|جهات|partner"),
}


def _matches(text, rxs):
    out = []
    for key, r in rxs.items():
        for m in r.finditer(text):
            out.append((key, m.start(), m.end()))
    return out


def find_docs(text):
    docs = _matches(text, DOC_RX)
    for m in CARD_RX.finditer(text):
        if not CARD_NOT.search(text[m.end():m.end() + 20]):
            docs.append(("id_card", m.start(), m.end()))
    if not docs:
        docs = _matches(text, DOC_WEAK_RX)
    return sorted(docs, key=lambda x: x[1])


def gap(a, b):
    return max(0, max(a[1], b[1]) - min(a[2], b[2]))


def find_sits(text):
    return sorted(_matches(text, SIT_RX), key=lambda x: x[1]), sorted(_matches(text, SIT_WEAK_RX), key=lambda x: x[1])


def route(question, state):
    """Returns (services, mode, new_state). mode: answer | clarify | refuse | classify."""
    t = norm(question)
    docs = find_docs(t)
    strong, weak = find_sits(t)
    pairs = set()
    if docs:
        for s in strong:                                            # each situation -> nearest document
            d = min(docs, key=lambda d: gap(d, s))
            pairs.add((d[0], s[0]))
        paired_docs = {d for d, _ in pairs}
        for d in docs:                                              # documents without a strong situation
            if d[0] in paired_docs:
                near = min(strong, key=lambda s: gap(d, s)) if strong else None
                if near is None or gap(d, near) > 40:
                    continue
                pairs.add((d[0], near[0]))
                continue
            near_s = min(strong, key=lambda s: gap(d, s)) if strong else None
            w = min(weak, key=lambda s: gap(d, s)) if weak else None
            if near_s is not None and gap(d, near_s) <= 40:
                pairs.add((d[0], near_s[0]))
            elif w is not None and gap(d, w) <= 40 and (
                    min(docs, key=lambda x: gap(x, w))[0] == d[0] or gap(d, min(docs, key=lambda x: gap(x, w))) <= 6):
                pairs.add((d[0], w[0]))                             # "هوية وجواز" share the verb
            elif strong or weak:
                continue                                            # mentioned as a requirement, not asked about
            else:
                pairs.add((d[0], None))
        # "ضاع دفتر العائلة أول مرة": "first time" is a count when lost/damaged is there
        pairs = {(d, s) for d, s in pairs if not (s == "first_issue" and ((d, "lost") in pairs or (d, "damaged") in pairs))}
        if not pairs:
            pairs = {(docs[0][0], None)}
        # same document both with and without a situation -> keep the specific one
        pairs = {(d, s) for d, s in pairs if s is not None or not any(dd == d and ss for dd, ss in pairs)}
        no_sit = [p for p in pairs if p[1] is None]
        if no_sit and state.get("pairs"):                           # follow-up: "وكم رسوم الجواز؟"
            prev_sit = {d: s for d, s in state["pairs"] if s}
            prev_sits = {s for _, s in state["pairs"] if s}
            for d, _ in no_sit:
                pairs.discard((d, None))
                if d in prev_sit:
                    pairs.add((d, prev_sit[d]))
                elif len(prev_sits) == 1:
                    pairs.add((d, next(iter(prev_sits))))
                else:
                    pairs.add((d, None))
        if len({d for d, _ in pairs}) > 1 and all(s is None for _, s in pairs):
            first = docs[0][0]                                      # "documents for X? do I need my ID?"
            close = [d for d in docs if gap(d, docs[0]) <= 15]
            pairs = {(d[0], None) for d in close} or {(first, None)}
    else:
        sits = {s for s, _, _ in strong} or {s for s, _, _ in weak}
        if OOS_RX.search(t):                                        # out of scope wins over follow-up memory
            return [], "refuse", {}
        if state.get("pairs"):                                      # follow-up without a document
            prev_docs = sorted({d for d, _ in state["pairs"]})
            if sits:
                pairs = {(d, s) for d in prev_docs for s in sits}
            else:
                pairs = set(state["pairs"])
        if not pairs:
            if OOS_RX.search(t):
                return [], "refuse", {}
            if sits or {s for s, _, _ in weak}:
                return [], "clarify", {"pending_sits": sorted(sits), "pending_q": question}
            if wanted_fields(question) and len(t.split()) <= 6:   # "وكم بتكلف؟" with nothing before it
                return [], "clarify", {"pending_sits": [], "pending_q": question}
            return [], "classify", {}
    # a clarification answer ("الجواز") completes the pending situation
    if state.get("pending_sits") and all(s is None for _, s in pairs):
        pairs = {(d, s) for d, _ in pairs for s in state["pending_sits"]}
    services = []
    for d, s in sorted(pairs, key=lambda p: (p[0], p[1] or "")):
        for sit in ([s] if s else list(SITS)):
            sid = f"{d}_{sit}"
            if sid not in services:
                services.append(sid)
    return services, "answer", {"pairs": sorted(pairs, key=lambda p: (p[0], p[1] or ""))}


def wanted_fields(question):
    t = norm(question)
    return [f for f, r in FIELD_RX.items() if r.search(t)]


def build_context(question, sids, services):
    """Returns (text, fields shown). A focused question (fees / documents / place / time ...) gets only
    those fields + conditions + notes, so the 7B model can't drift into unrelated lists."""
    fields = wanted_fields(question)
    if CONFIG["context"] == "full" and len(sids) <= 2:
        fields = []
    if fields and "الإجراءات" not in fields:
        keep = set(fields) | {"ملاحظات", "الشروط"}
    else:
        keep = None                                   # broad question ("what do I do?") -> whole service
    if keep is None and len(sids) > 2:
        keep = {"الرسوم", "الوثائق المطلوبة", "الشروط", "ملاحظات"}
    text = "\n\n".join(service_text(services[s], keep) for s in sids)
    return text[:MAX_CONTEXT_CHARS], keep


# ============================== LLM ==============================
GLOSSARY = ("مفردات عامية: الباسبور/الباسبورت = جواز السفر، الهوية/البطاقة = البطاقة الذكية (البطاقة الشخصية)، "
            "ضاع/ضايع/انسرق = بدل فاقد، اهترى/تقطع/تلف/انبل = بدل تالف، خلص/منتهي = انتهت صلاحيته (تجديد)، "
            "قديش/كم = كم، شو = ماذا، بلزمني/بدي = أحتاج/أريد، برا البلد = خارج الأردن، أطلع = أستخرج.")

SYSTEM_ANSWER = """أنت مساعد دائرة الأحوال المدنية والجوازات في الأردن. تجيب المواطنين اعتماداً فقط على نص الخدمات المرفق.
القواعد:
1. حدّد الخدمة التي يسأل عنها المستخدم واستخدم معلومات تلك الخدمة فقط. لا تخلط رسوم أو وثائق أو شروط خدمة بخدمة أخرى.
2. الرسوم قد تختلف حسب العمر أو صلاحية الوثيقة (منتهية أو متبقٍ عليها أقل/أكثر من مدة معينة) أو مكان التقديم (داخل الأردن أو خارجه) أو عدد مرات الفقدان. طبّق القاعدة المطابقة لحالة المستخدم واذكر المبلغ الصحيح له. إذا لم يذكر حالته والمبلغ يختلف، اذكر الحالات باختصار.
3. اقرأ [ملاحظات] دائماً: فيها رسوم إضافية ومهل وغرامات واستثناءات. اذكر ما يتعلق بالسؤال منها.
4. إذا سأل عن الوثائق أو الشروط أو الإجراءات فاذكر القائمة كاملة لتلك الخدمة.
5. إذا كان في السؤال افتراض خاطئ فصحّحه من النص.
6. إذا سأل عن أكثر من خدمة أو أكثر من شخص أو أكثر من جزء، أجب عن كل جزء على حدة.
6ب. في المقارنة (أغلى/أرخص/أسرع/الفرق): اكتب أولاً رقم كل خدمة، ثم قارن الرقمين بدقة (الرقم الأكبر هو الأغلى)، ثم اكتب النتيجة.
6ج. إذا جمعت مبالغ فاذكر كل مبلغ ومصدره ثم المجموع. لا تكتب أي رقم غير موجود في النص إلا ناتج جمع واضح.
6د. إذا وردت "الرسوم المنطبقة على حالة المستخدم" فهي السطر الصحيح من النص لحالته: اعتمد عليها.
7. إذا كانت المعلومة غير موجودة في النص فقل إنها غير متوفرة لديك، ولا تخترع شيئاً ولا تستخدم معلوماتك العامة. لا تضف أي وثيقة أو رسم أو شرط غير مكتوب حرفياً في النص.
8. أجب عن السؤال المطروح فقط، ولا تكرر معلومات سبق ذكرها في المحادثة، ولا تضف معلومات لم يُسأل عنها إلا ملاحظة مهمة تخص السؤال مباشرة.
9. فقط إذا كان السؤال يُجاب بنعم أو لا (مثل "صح؟" أو "بقدر؟") فابدأ بـ "نعم" أو "لا" ثم وضّح بجملة.
10. لا تفترض عمر المستخدم أو حالته. إذا قال إن وثيقته "خلصت" أو "منتهية" فطبّق رسوم الوثيقة المنتهية فقط (ولا تذكر رسوم الوثيقة غير المنتهية)، وإذا لم يذكر عمره فاذكر رسوم كل فئة عمرية باختصار. إذا لم يذكر هل وثيقته منتهية أم لا والرسوم تختلف، فاذكر الحالتين.
11. {style_rule} لا تكرر المعلومة نفسها ولا تعِد صياغة السؤال. لا تستخدم عناوين أو رموز تنسيق مثل # أو **.
12. إذا وردت "معلومات مستخرجة من سؤال المستخدم" فاعتمد عليها في اختيار الرسوم والقاعدة المطابقة.
13. أجب مباشرة وباختصار بالعربية الفصحى البسيطة، دون مقدمات ودون ذكر "النص" أو "المقاطع" أو أرقام مصادر.
""" + GLOSSARY

STYLE_RULES = {
    "concise": 'لا تزد عن 100 كلمة إلا إذا طُلبت قائمة كاملة. إذا سأل "شو بعمل؟" فاذكر الخدمة المناسبة والوثائق الأساسية والرسوم باختصار.',
    "detailed": ('أعطِ إجابة كاملة عن المطلوب فقط: المعلومة المطلوبة (المبلغ/المدة/القائمة) مع القاعدة أو الحالة التي تنطبق، '
                 'ثم أضف فقط الغرامة أو المهلة أو الشرط أو الاستثناء من [الشروط] و[ملاحظات] الذي يغيّر الإجابة عن هذا السؤال تحديداً. '
                 'لا تذكر الرسوم إلا إذا سأل عن التكلفة، ولا تذكر الوثائق إلا إذا سأل عنها، ولا تضف معلومات من الخدمة لم يُسأل عنها. '
                 'لا تزد عن 150 كلمة إلا إذا طُلبت قائمة كاملة. إذا سأل "شو بعمل؟" فاذكر الخدمة المناسبة والوثائق الأساسية والرسوم.'),
}


def system_answer(en):
    s = SYSTEM_ANSWER.replace("{style_rule}", STYLE_RULES[CONFIG["style"]])
    return s + ("\n14. The user wrote in English: answer in English (translate the Arabic facts)." if en else "")

SYSTEM_REFUSE = """أنت مساعد دائرة الأحوال المدنية والجوازات في الأردن. لديك معلومات فقط عن خدمات: جواز السفر، دفتر العائلة، البطاقة الذكية (إصدار لأول مرة، تجديد، بدل فاقد، بدل تالف).
سؤال المستخدم خارج هذه المعلومات. أجب بجملتين قصيرتين فقط:
الأولى: اعتذار بأنه لا تتوفر لديك معلومات عن هذا الموضوع.
الثانية: أنك تستطيع المساعدة فقط في خدمات جواز السفر ودفتر العائلة والبطاقة الذكية.
لا تجب عن السؤال، لا تقدم نصائح أو مصادر أو مواقع أخرى، ولا تخمّن أي رقم أو معلومة.
مثال: «عذراً، لا تتوفر لدي معلومات عن هذا الموضوع. أستطيع مساعدتك فقط في خدمات جواز السفر ودفتر العائلة والبطاقة الذكية.»"""

SYSTEM_CLARIFY = """أنت مساعد دائرة الأحوال المدنية والجوازات في الأردن. لديك معلومات عن خدمات: جواز السفر، دفتر العائلة، البطاقة الذكية (إصدار لأول مرة، تجديد، بدل فاقد، بدل تالف).
سؤال المستخدم غير واضح لأنه لم يحدد الوثيقة المقصودة. اكتب سؤالاً واحداً قصيراً فقط تسأله فيه أي وثيقة يقصد، ولا شيء غيره.
لا تجب عن سؤاله ولا تذكر أي وثائق مطلوبة أو رسوم أو شروط.
مثال: «بدي أجدد، شو المطلوب؟» -> «هل تقصد تجديد جواز السفر، أم دفتر العائلة، أم البطاقة الذكية؟»"""

CLASSIFY = """صنّف سؤال المستخدم إلى واحدة من التسميات التالية فقط:
""" + "\n".join(f"{sid} = {t}" for sid, t in TITLES.items()) + """
passport / family_book / id_card = عن هذه الوثيقة لكن الحالة غير محددة
unclear = عن هذه الخدمات لكن لا يمكن معرفة أي وثيقة
none = أي موضوع آخر (خدمة أخرى، سفر، أسعار، أي شيء غير هذه الوثائق الثلاث)
اكتب التسمية فقط.

السؤال: {q}
التسمية:"""


def _post(path, payload, timeout=600):
    req = urllib.request.Request(OLLAMA_URL + path, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _get(path, timeout=10):
    with urllib.request.urlopen(OLLAMA_URL + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def ensure_ollama():
    global MODEL
    if MOCK:
        return
    try:
        tags = _get("/api/tags")
    except Exception:
        print("Ollama not reachable - trying to start `ollama serve` ...")
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"  could not start ollama: {e}")
        for _ in range(30):
            time.sleep(1)
            try:
                tags = _get("/api/tags")
                break
            except Exception:
                pass
        else:
            print("  ! Ollama is still not reachable. Open the Ollama app and run again.")
            return
    names = [m.get("name") or m.get("model") for m in tags.get("models", [])]
    if MODEL not in names and MODEL + ":latest" not in names:
        base = MODEL.split(":")[0]
        same = [n for n in names if n.split(":")[0] == base]          # e.g. :7b requested, :latest installed
        chat = [n for n in names if not re.search(r"embed|bge|nomic|e5|minilm|rerank", n, re.I)]
        pick = (same or chat or [MODEL])[0]
        print(f"  ! model {MODEL} not installed; using {pick}   -> run: ollama pull {MODEL}")
        MODEL = pick
    print(f"Model: {MODEL}   (main.py: {Path(__file__).resolve()})")


FOREIGN = re.compile(r"[ᄀ-ᇿ　-〿぀-ヿ㄰-㆏㐀-䶿一-鿿가-힯]+")


def llm(messages, num_predict=NUM_PREDICT):
    if MOCK:
        return "MOCK: " + messages[-1]["content"][-300:]
    payload = {"model": MODEL, "messages": messages, "stream": False, "keep_alive": "30m",
               "options": {"temperature": CONFIG["temperature"], "num_ctx": NUM_CTX, "num_predict": num_predict,
                           "seed": SEED, "repeat_penalty": 1.05}}
    out, err = None, None
    for attempt in range(3):                      # timeout + 2 retries per call
        try:
            out = _post("/api/chat", payload, timeout=CALL_TIMEOUT)
            break
        except Exception as e:
            err = e
            time.sleep(3)
    if out is None:
        raise RuntimeError(f"model call failed 3 times: {err}")
    n = out.get("prompt_eval_count") or 0          # tokens Ollama actually read: detects silent truncation
    STATS["max_prompt_tokens"] = max(STATS["max_prompt_tokens"], n)
    if n >= NUM_CTX * 0.95:
        STATS["near_ctx_limit"] += 1
        print(f"  ! prompt used {n}/{NUM_CTX} tokens - it may have been truncated")
    text = out["message"]["content"]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)     # reasoning models
    text = FOREIGN.sub("", text)
    text = re.sub(r"[\U0001F000-\U0001FFFF]", "", text)
    text = re.sub(r"\s*\[\d+(,\s*\d+)*\]", "", text)                  # stray citation marks
    return text.strip()


def classify(question):
    """Fallback when no routing rule fires: the model picks a label from a fixed list (classification, not search)."""
    if MOCK:
        return "none"
    out = llm([{"role": "user", "content": CLASSIFY.format(q=question)}], num_predict=12).lower()
    for sid in TITLES:
        if sid in out:
            return sid
    for k in ("family_book", "passport", "id_card", "unclear", "none"):
        if k in out:
            return k
    return "none"


# ============================== CONTEXTS (what the answer used) ==============================
STOP = set(norm(w) for w in """في من على الى إلى عن او أو و ان أن إن ما لا هل كم هذا هذه ذلك التي الذي الذين مع بعد قبل كل اي أي
يتم يجب حال حاله حالة لمن له لها هو هي هم انت أنت انا أنا كان يكون تكون ذات غير بين عند حتى قد لم لن ثم او اذا إذا
الخدمه الخدمة الطلب جواز السفر سفر دفتر العائله العائلة البطاقه البطاقة الذكيه الذكية الشخصيه الشخصية تجديد بدل فاقد تالف
لاول لأول مره مرة اول أول يمكنك تحتاج تحتاجين المطلوب المطلوبة الوثائق ايضا أيضاً فقط يمكن عليك لك""".split())


def stem(w):
    for p in ("وال", "بال", "فال", "كال", "لل", "ال"):
        if w.startswith(p) and len(w) - len(p) >= 2:
            w = w[len(p):]
            break
    if w[:1] in ("و", "ل", "ب") and len(w) > 4:
        w = w[1:]
    return w.rstrip("ه") if len(w) > 3 else w


STOP_STEMS = {stem(w) for w in STOP}


def toks(s):
    out = []
    for w in re.findall(r"[\w]+", norm(s)):
        w = stem(w)
        if w and w not in STOP_STEMS and len(w) > 2 and not w.isdigit():
            out.append(w)
    return out


def fact_numbers(text):
    """Numbers that are facts (fees, days, counts) - not list numbering like '1.' or '2-'."""
    t = re.sub(r"(^|\n)\s*\d+\s*[.\-)]", " ", norm(text))
    return set(re.findall(r"\d+", t))


def pick_contexts(answer, question, sids, services, shown=None):
    a_toks = set(toks(answer))
    q_toks = set(toks(question))
    nums = fact_numbers(answer)
    cands = []                                   # (score, sid, field, line)
    for sid in sids:
        for field, lines in services[sid]["fields"].items():
            if shown and field not in shown:
                continue
            for line in lines:
                lt = toks(line)
                if not lt:
                    continue
                overlap = len(set(lt) & a_toks)
                num_hit = len(fact_numbers(line) & nums)
                score = overlap + 2 * min(num_hit, 2)
                need = len(set(lt)) if len(set(lt)) <= 2 else max(2, 0.34 * len(set(lt)))
                if overlap >= 1 and (score >= need or num_hit):
                    cands.append((score, sid, field, line, num_hit))
    if not cands:
        return []
    by_sid = {}
    num_sids = set()
    for sc, sid, _, _, nh in cands:
        by_sid[sid] = by_sid.get(sid, 0) + sc
        if nh:
            num_sids.add(sid)
    best = max(by_sid.values())
    keep_sids = [s for s in sids if by_sid.get(s, 0) >= 0.45 * best or s in num_sids]
    out, seen_lines = {}, set()
    for sid in sorted(keep_sids, key=lambda s: -by_sid[s]):
        for sc, s, field, line, _ in cands:
            if s != sid or norm(line) in seen_lines:
                continue
            seen_lines.add(norm(line))
            out.setdefault((sid, field), []).append(line)
    for (sid, field), lines in out.items():          # most of a list used -> give the whole list
        full = services[sid]["fields"][field]
        if len(full) > 1 and len(lines) >= 0.5 * len(full):
            out[(sid, field)] = list(full)
    order = {f: i for i, (_, f) in enumerate(FIELDS)}
    res = []
    for (sid, field), lines in sorted(out.items(), key=lambda x: (keep_sids.index(x[0][0]), order.get(x[0][1], 99))):
        sep = " " if all(re.match(r"\s*\d", l) for l in lines) else "، "
        res.append(f"{services[sid]['title']} - {field}: " + sep.join(lines))
    return res


# ============================== ONE MESSAGE ==============================
ADVICE = re.compile(norm(r"يمكنك|ننصح|انصح|اوصي|مواقع|موقع|استشار|راجع|تواصل|الاتصال|you can|website|contact|check"))
EMOJI = re.compile(r"[\U0001F000-\U0001FFFF\u2600-\u27BF\uFE0F]")


def sentences(text):
    return [x.strip() for x in re.split(r"(?<=[.!؟?])\s+|\n+", text) if x.strip()]


def clean_refusal(ans):
    keep = [x for x in sentences(EMOJI.sub("", ans)) if not ADVICE.search(norm(x))]
    return " ".join(keep[:2]) or ans


DOC_WORDS = re.compile(norm(r"جواز|دفتر|بطاق|هوي|passport|family|id card"))


def clean_clarify(ans):
    """Keep only the question that asks which document; None if the model didn't write one."""
    qs = re.findall(r"[^.!؟?\n]*[؟?]", EMOJI.sub("", ans))
    good = [x.strip() for x in qs if len(DOC_WORDS.findall(norm(x))) >= 2]
    return good[0] if good else None


ORD = {"اول": 1, "الاول": 1, "اولي": 1, "الاولي": 1, "ثاني": 2, "تاني": 2, "الثانيه": 2, "الثاني": 2, "التانيه": 2,
       "ثالث": 3, "تالت": 3, "الثالثه": 3, "الثالث": 3, "التالته": 3, "رابع": 4, "الرابعه": 4}
UNIT_DAYS = {"يوم": 1, "ايام": 1, "اسبوع": 7, "اسابيع": 7, "شهر": 30, "اشهر": 30, "شهور": 30, "سنه": 365, "سنين": 365,
             "سنوات": 365, "day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30, "year": 365, "years": 365}


HOME_RX = re.compile(norm(r"الاردن|عمان|اربد|الزرقاء|العقبه|السلط|الكرك|مادبا|جرش|عجلون|المفرق|معان|الطفيله|jordan|amman"))


def _days(n, unit):
    if unit in ("شهرين",):
        return 60
    if unit in ("سنتين",):
        return 730
    return (n or 1) * UNIT_DAYS.get(unit, 30)


def situation(question):
    """Structured facts about the user's case, parsed in code (the 7B model is bad at this)."""
    t = norm(question)
    f = {}
    m = re.search(r"(?:عمر\w*|سن\w*|age[d]?|i am|i'm)\D{0,12}?(\d{1,3})", t) or \
        re.search(r"(\d{1,3})\s*(?:سنه|سنين|سنوات|عام|years? old)", t)
    if m and int(m.group(1)) < 120:
        f["age"] = int(m.group(1))
    m = re.search(r"(?:باقي|بايق|متبقي|ضايل|ظايل|بقي|يخلص بعد|بخلص بعد|بينتهي بعد|ينتهي بعد|left|remaining|expires in)\D{0,15}?"
                  r"(\d+)?\s*(يوم|ايام|اسبوع|اسابيع|شهرين|شهر|اشهر|شهور|سنتين|سنه|سنين|سنوات|days?|weeks?|months?|years?)", t)
    if m:
        f["days_left"] = _days(int(m.group(1)) if m.group(1) else None, m.group(2))
    elif re.search(norm(r"منتهي|انتهت|انتهى|خلص|خلصت|expired"), t) and not re.search(norm(r"غير منتهي|مش منتهي|ما خلص|مو منتهي"), t):
        f["expired"] = True
    elif re.search(norm(r"ساري|غير منتهي|مش منتهي|ما خلص|valid"), t):
        f["expired"] = False
    res = re.search(r"(?:مقيم|ساكن|عايش|اعيش|اسكن|بشتغل|اشتغل|living|live|reside)\s*(?:في|ب|in)?\s*(\S+)", t)
    if (res and not HOME_RX.search(res.group(1))) or re.search(norm(
            r"برا البلد|برا الاردن|بره|خارج|بالغربه|مغترب|بالسعوديه|بالامارات|بدبي|بقطر|بالكويت|بامريكا|بالمانيا|بتركيا|abroad|outside"), t):
        f["abroad"] = True
    elif re.search(norm(r"بالاردن|داخل الاردن|في الاردن|بعمان|in jordan"), t):
        f["abroad"] = False
    m = re.search(r"(?:المره|مره|للمره)\s*([ء-ي]+)|([ء-ي]+)\s*مره", t)
    if m:
        w = m.group(1) or m.group(2) or ""
        n = ORD.get(w) or ORD.get(w[1:]) or ORD.get(w[2:])
        if n:
            f["nth"] = n
    if re.search(norm(r"عسكري|الجيش|القوات المسلحه|الامن العام|الدفاع المدني|الدرك|military|army"), t):
        f["military"] = True
    if re.search(norm(r"ممتلئ|امتلا|تعبت الصفحات|خلصت الصفحات|خلصوا الصفحات|ما في صفحات|full"), t):
        f["full_pages"] = True
    m = re.search(r"(?:تزوج|زواج|عقد)\D{0,15}?(\d+)?\s*(يوم|ايام|شهرين|شهر|اشهر|شهور|سنتين|سنه|سنين|سنوات)", t)
    if m:
        f["married_days"] = _days(int(m.group(1)) if m.group(1) else None, m.group(2))
    return f


def user_facts(question):
    f = situation(question)
    facts = []
    if "age" in f:
        a = f["age"]
        rel = lambda x: "أكبر من" if a > x else "أصغر من" if a < x else "يساوي"
        facts.append(f"العمر {a} سنة: {rel(16)} 16 سنة، و{rel(18)} 18 سنة")
    if "days_left" in f:
        d = f["days_left"]
        facts.append(f"المتبقي على صلاحية الوثيقة حوالي {d} يوماً، أي {'أقل' if d < 180 else 'أكثر'} من 180 يوماً، والوثيقة غير منتهية")
    elif f.get("expired") is True:
        facts.append("الوثيقة منتهية الصلاحية")
    elif f.get("expired") is False:
        facts.append("الوثيقة غير منتهية (سارية)")
    if f.get("abroad") is True:
        facts.append("المستخدم خارج الأردن")
    elif f.get("abroad") is False:
        facts.append("المستخدم داخل الأردن")
    if f.get("nth", 1) >= 2:
        facts.append(f"هذه المرة رقم {f['nth']} (الفقدان/الطلب رقم {f['nth']})")
    if f.get("military"):
        facts.append("المستخدم عسكري")
    if f.get("full_pages"):
        facts.append("صفحات الجواز ممتلئة")
    if "married_days" in f:
        d = f["married_days"]
        facts.append(f"مضى على عقد الزواج حوالي {d} يوماً، أي {'أكثر' if d > 90 else 'أقل'} من 90 يوماً")
    return facts


def _has(line, *words):
    n = norm(line)
    return any(norm(w) in n for w in words)


def fee_lines(sid, services, question):
    """The fee line(s) of the data that apply to the user's situation, chosen in code. [] = let the model read all."""
    f = situation(question)
    fees = services[sid]["fields"].get("الرسوم", [])
    notes = services[sid]["fields"].get("ملاحظات", [])
    pick = lambda *w: [l for l in fees if _has(l, *w)]
    out = []
    age, abroad = f.get("age"), f.get("abroad")
    if sid in ("passport_renewal", "passport_first_issue"):
        if f.get("military"):
            out = pick("القوات المسلحه")
        elif abroad:
            out = pick("خارج البلاد")
        elif age is not None and age < 16:
            out = pick("دون")
        elif sid == "passport_renewal":
            if f.get("full_pages") and not f.get("expired"):
                out = pick("الممتلئ")
            elif f.get("expired") or f.get("days_left", 999) < 180:
                out = pick("منتهي او متبقي به اقل") + ([] if age is not None else pick("دون"))
            elif f.get("days_left", 0) >= 180 or f.get("expired") is False:
                out = pick("غير المنتهي")
        elif sid == "passport_first_issue" and (age is not None or abroad is False):
            out = pick("داخل البلاد")
    elif sid == "passport_lost" and "nth" in f:
        out = [l for l in fees if (_has(l, "تلي المره") if f["nth"] >= 2 else not _has(l, "تلي المره"))]
    elif sid == "family_book_renewal":
        if "expired" in f or "days_left" in f:
            exp = f.get("expired", False)
            out = [l for l in fees if _has(l, "غير منتهي") != exp]
        else:
            out = list(fees)                          # unknown: both cases (expired 2 JD / not expired 3 JD)
    elif sid == "id_card_first_issue" and age is not None:
        out = fees if age >= 18 else [l for l in fees if not _has(l, "غرامه")]
    elif sid == "family_book_first_issue" and "married_days" in f:
        out = fees if f["married_days"] > 90 else [l for l in fees if not _has(l, "غرامه")]
    if f.get("nth", 1) >= 2 and sid in ("id_card_lost", "family_book_lost"):
        word = {2: "الثانيه", 3: "الثالثه"}.get(f["nth"], "الثالثه")
        out = fees + [l for l in notes if _has(l, word)]
    return out


SPLIT_RX = re.compile(r"[،,.؛;!؟?\n]+|\s+(?=و(?:كمان|ابن|بنت|مرت|زوج|اخ|اب|ام|انا|احنا|هوي|جواز|دفتر|بطاق|الهوي|الجواز|الدفتر|البطاق))")


def sub_question(question, doc):
    """The clauses of a multi-part question that belong to one document (+ clauses that name no document)."""
    q = re.sub("[إأآ]", "ا", question)
    parts = [p.strip() for p in SPLIT_RX.split(q) if p and p.strip()]
    mine = [p for p in parts if any(d == doc for d, _, _ in find_docs(norm(p)))]
    shared = [p for p in parts if not find_docs(norm(p))]
    return "، ".join(mine + shared) if mine else question


YESNO = re.compile(norm(r"صح|صحيح|هل|بقدر|يقدر|ممكن|يمكن|بيصير|مسموح|بنفع|لازم|ضروري|يحق|can |is |do |does |should |must "))
WH = re.compile(norm(r"شو|كم|قديش|وين|ايش|مين|كيف|متي|اين|ماذا|ما هي|ما هو|what|how|where|who|when"))


def tidy(ans, question):
    ans = re.sub(r"^\s*#+\s*", "", ans, flags=re.M)              # markdown headers
    ans = ans.replace("**", "").replace("__", "")
    ans = "\n".join(l for l in ans.split("\n") if not re.search(            # internal hint copied by the model
        r"الرسوم المنطبقة على حالة|معلومات مستخرجة من سؤال", l))
    ans = re.sub(r"^\s*-\s*[^:\n]{2,40}:\s*«[^»]*»(\s*/\s*«[^»]*»)*\s*$", "", ans, flags=re.M)
    ans = re.sub(r"\[(ملاحظات|الرسوم|الشروط|الوثائق المطلوبة|الإجراءات|الزمن المستغرق|أماكن تقديم الخدمة|شركاء الخدمة|الفئة المستفيدة)\]\s*:?\s*",
                 lambda m: "ملاحظة: " if m.group(1) == "ملاحظات" else "", ans)
    ans = re.sub(r"\(\s*انظر\s*\)", "", ans)
    ans = re.sub(r"ملاحظة:\s*-\s*", "ملاحظة: ", ans)
    t = norm(question)
    if WH.search(t) and not re.search(norm(r"صح|صحيح|\bهل\b"), t):  # not a yes/no question -> drop a leading "نعم،"
        ans = re.sub(r"^\s*(نعم|أجل|yes)[،,.!]?\s*", "", ans, flags=re.I)
    return re.sub(r"\n{3,}", "\n\n", ans).strip()


SYSTEM_PART = ("\nالسؤال قد يتضمن عدة مواضيع أو أشخاص. لديك هنا معلومات خدمة واحدة فقط: «{title}». "
               "أجب فقط عن الجزء من السؤال الذي يخص هذه الخدمة، في جملة أو جملتين، وابدأ باسم الخدمة. "
               "لا تجمع مبالغ خدمات أخرى.")


def _fee_hint(question, sids, services):
    t = norm(question)
    if not (FIELD_RX["الرسوم"].search(t) or re.search(norm(r"شو بعمل|شو اعمل|ماذا افعل|شو بصير|ماذا يحدث"), t)):
        return [], []
    lines, used = [], []
    for sid in sids:
        fl = fee_lines(sid, services, question)
        if fl:
            lines.append(f"- {services[sid]['title']}: " + " / ".join(f"«{l}»" for l in fl))
            used.append((sid, fl))
    if used:
        STATS["fee_rule_in_code"] += 1
    return lines, used


AMOUNT_RX = re.compile(r"(\d+)\s*\)?\s*(?:دينار|دنانير|دنانبر|دينارا)|(?:دينار|دنانير)\s*\(?(\d+)|بقيمه\s*\(?(\d+)")


def allowed_numbers(context, question, extra=()):
    """Numbers an answer may contain: any number in the service text / question, or a total of 2-3 money amounts."""
    nc = norm(context)
    nums = {int(n) for n in re.findall(r"\d+", nc + " " + norm(question) + " " + norm(" ".join(extra))) if len(n) < 7}
    amounts = {int(next(g for g in m.groups() if g)) for m in AMOUNT_RX.finditer(nc)}
    words = {"دينارين": 2, "ديناران": 2, "ثلاثه دنانير": 3, "خمسه دنانير": 5, "عشره دنانير": 10, "عشرين دينار": 20}
    amounts |= {v for k, v in words.items() if k in nc}
    nums |= amounts
    amounts = sorted(amounts)
    sums = {amounts[i] + amounts[j] for i in range(len(amounts)) for j in range(i + 1, len(amounts))}
    sums |= {amounts[i] + amounts[j] + amounts[k] for i in range(len(amounts))
             for j in range(i + 1, len(amounts)) for k in range(j + 1, len(amounts))}
    return nums | sums


def bad_numbers(answer, context, question, extra=()):
    ok = allowed_numbers(context, question, extra)
    return sorted({int(n) for n in fact_numbers(answer) if len(n) < 7 and int(n) not in ok and int(n) > 1})


NUMBER_RETRY = {"ar": "\n\nتنبيه: الأرقام التالية في إجابتك غير موجودة في نص الخدمة: {nums}. أعد كتابة الإجابة باستخدام أرقام النص فقط.",
                "en": "\n\nWarning: these numbers in your answer are not in the service text: {nums}. Rewrite the answer using only numbers from the text."}


def generate(question, sids, services, hist_msgs, en, focus=None):
    context, shown = build_context(question, sids, services)
    facts = user_facts(focus["sub"] if focus else question) if CONFIG["facts"] else []
    hint_q = focus["sub"] if focus else question
    hints, _ = _fee_hint(hint_q if wanted_fields(hint_q) else question, sids, services) if CONFIG["facts"] else ([], [])
    ask = question
    if focus:
        ask = focus["sub"] if wanted_fields(focus["sub"]) else question
    user = f"معلومات الخدمات:\n{context}\n\nسؤال المستخدم: {ask}"
    if facts:
        user += "\n\nمعلومات مستخرجة من سؤال المستخدم: " + "؛ ".join(facts) + "."
    if hints:
        user += "\n\nالرسوم المنطبقة على حالة المستخدم (من النص):\n" + "\n".join(hints)
    num_predict = NUM_PREDICT
    if focus:
        if wanted_fields(focus["sub"]):
            part = f"(الجزء: «{focus['sub']}»)"
        else:                                         # "وقديش لتجديد جواز السفر" -> same thing asked, other service
            part = "(نفس الشيء الذي يسأل عنه المستخدم في سؤاله، لكن لهذه الخدمة)"
        user += (f"\n\nأجب الآن فقط عن «{focus['title']}» {part}. "
                 f"أجب بجملة أو جملتين فقط، وابدأ بعبارة «{focus['title']}:». لا تذكر أي خدمة أو شخص آخر.")
        num_predict = 160
    msgs = [{"role": "system", "content": system_answer(en)}, *hist_msgs, {"role": "user", "content": user}]
    ans = llm(msgs, num_predict)
    # number check: every number must come from the service text, the question, or a sum of text amounts
    hist_text = " ".join(m["content"] for m in hist_msgs)
    bad = bad_numbers(ans, context, ask + " " + hist_text)
    if bad and not MOCK:
        STATS["number_check_regenerated"] += 1
        if DEBUG:
            print(f"   ! number check: {bad} not in the data -> regenerating")
        retry = [dict(m) for m in msgs]
        retry[-1]["content"] += NUMBER_RETRY["en" if en else "ar"].format(nums=", ".join(map(str, bad)))
        ans2 = llm(retry, num_predict)
        if len(bad_numbers(ans2, context, ask + " " + hist_text)) < len(bad):
            ans = ans2
        if bad_numbers(ans, context, ask + " " + hist_text):
            STATS["number_check_still_bad"] += 1
    return ans, shown


def answer_message(question, history, state, services):
    sids, mode, new_state = route(question, state)
    if mode == "classify":
        c = classify(question)
        STATS["llm_fallback"] += 1
        if c in TITLES:
            sids, mode, new_state = [c], "answer", {"pairs": [(services[c]["doc"], services[c]["sit"])]}
        elif c in DOCS:
            sids, mode, new_state = route(question + " " + DOCS[c].split(" (")[0], state)
            if mode != "answer":
                sids, mode = [f"{c}_{s}" for s in SITS], "answer"
                new_state = {"pairs": [(c, None)]}
        else:
            mode = "clarify" if c == "unclear" else "refuse"
    hist_msgs = []
    for q, a in history[-HISTORY_TURNS:]:
        hist_msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    en = not is_arabic(question)
    STATS["messages"] += 1
    STATS[{"refuse": "refuse", "clarify": "clarify"}.get(mode, "rules")] += 1 if mode in ("refuse", "clarify", "answer") else 0
    if mode == "refuse":
        sys_p = SYSTEM_REFUSE + ("\nThe user wrote in English: reply in English." if en else "")
        ans = llm([{"role": "system", "content": sys_p}, {"role": "user", "content": question}], num_predict=90)
        return clean_refusal(ans), [], state, mode, []      # keep the earlier topic for later follow-ups
    if mode == "clarify":
        sys_p = SYSTEM_CLARIFY + ("\nThe user wrote in English: reply in English." if en else "")
        ans = llm([{"role": "system", "content": sys_p}, {"role": "user", "content": question}], num_predict=60)
        out = clean_clarify(ans)
        if not out:                               # retry with a stricter, direct instruction
            ask = ("Write only one short question asking the user which document they mean: passport, family book, or smart ID card."
                   if en else "اكتب فقط سؤالاً واحداً قصيراً تسأل فيه المستخدم: أي وثيقة يقصد، جواز السفر أم دفتر العائلة أم البطاقة الذكية؟")
            ans = llm([{"role": "user", "content": ask}], num_predict=40)
            out = clean_clarify(ans) or ans.strip()
        return out, [], new_state, mode, []
    q = question
    if state.get("pending_q"):                      # answer to our clarification: "الهوية"
        q = f"{state['pending_q']} ({question})"
        hist_msgs = []
    docs = {services[s]["doc"] for s in sids}
    pairs = new_state.get("pairs", [])
    if CONFIG["decompose"] and len(docs) > 1 and len(pairs) > 1 and all(sit for _, sit in pairs):
        # several different documents (long story / several people): one focused call per service
        parts, shown_all = [], set()
        for sid in sids:
            title = services[sid]["title"]
            part, shown = generate(q, [sid], services, hist_msgs, en,
                                   {"title": title, "sub": sub_question(q, services[sid]["doc"])})
            part = tidy(part, q)
            if not norm(part).startswith(norm(title)[:6]):
                part = f"{title}: {part}"
            parts.append(part)
            shown_all |= set(shown or [f for f, _ in FIELDS])
        ans, shown = "\n".join(parts), shown_all
    else:
        ans, shown = generate(q, sids, services, hist_msgs, en)
    ans = tidy(ans, question)
    ctxs = pick_contexts(ans, q, sids, services, shown)
    if not ctxs:                                  # nothing matched: give the most relevant fields
        fields = wanted_fields(q)[:2] or ["الرسوم"]
        for sid in sids[:2]:
            for f in fields:
                if f in services[sid]["fields"]:
                    ctxs.append(f"{services[sid]['title']} - {f}: " + " ".join(services[sid]["fields"][f]))
    return ans, ctxs, new_state, mode, sids


# ============================== MAIN ==============================
def one_line(text):
    """answers.json gets single-line answers like answers_example.json (no \\n in the JSON)."""
    lines = [l.strip(" -•\t") for l in text.splitlines() if l.strip(" -•\t")]
    out = ""
    for l in lines:
        if out and not re.search(r"[.:،؛!؟?]$", out):
            out += "،" if re.match(r"^\d+[.\-)]", l) is None else ""
        out += (" " if out else "") + l
    return re.sub(r"\s{2,}", " ", out).strip()


def run_chats(data, services, log=print):
    chats = data.get("chats", [])
    total = sum(len(c.get("messages", [])) for c in chats)
    done = 0
    for chat in chats:
        history, state = [], {}                   # every chat is a new user
        for msg in chat.get("messages", []):
            q = msg.get("question", "")
            t1 = time.time()
            try:
                ans, ctxs, state, mode, sids = answer_message(q, history, state, services)
            except Exception as e:                # never lose the whole file because of one message
                log(f"  ! {msg.get('message_id')}: {type(e).__name__}: {e}")
                STATS["errors"] += 1
                try:
                    ans, ctxs, mode, sids = llm([{"role": "system", "content": SYSTEM_REFUSE},
                                                 {"role": "user", "content": q}], 120), [], "error", []
                except Exception:
                    ans, ctxs, mode, sids = "error", [], "error", []
            msg["answer"] = one_line(ans) if ans else "error"
            msg["contexts"] = ctxs
            history.append((q, msg["answer"]))
            done += 1
            log(f"[{done}/{total}] {msg.get('message_id')} {mode:<8} {','.join(sids)[:70]}  ({time.time() - t1:.1f}s)")
            if DEBUG:
                log(f"   Q: {q}\n   A: {msg['answer']}\n   C: {json.dumps(ctxs, ensure_ascii=False)[:600]}\n")
        yield chat


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    q_path = Path(args[0] if args else "questions.json").resolve()
    out_path = Path(args[1]).resolve() if len(args) > 1 else q_path.parent / "answers.json"
    data = json.loads(q_path.read_text(encoding="utf-8-sig"))
    services = load_services()
    print(f"Loaded {len(services)} services from {DATA_DIR}")
    print(f"Config: {CONFIG}")
    ensure_ollama()
    t0 = time.time()
    for _ in run_chats(data, services):          # save after every chat: a crash still leaves a valid file
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    n = sum(len(c.get("messages", [])) for c in data.get("chats", []))
    print(f"\nWrote {out_path}  ({n} answers, {time.time() - t0:.0f}s total)")
    print("Stats:", json.dumps(STATS))


if __name__ == "__main__":
    main()
