import streamlit as st
import pandas as pd
import io
import math
import re
from rapidfuzz import fuzz
from collections import Counter, defaultdict
import time

st.set_page_config(page_title="⚡ Ultra-Fast Product Matcher", page_icon="🎯", layout="wide")

# ══════════════════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════════════════

DEFAULT_EXCLUSIONS = ['프로미라이프', '통합', '보장', '건강', '보험', '가입', '종합']

# ══════════════════════════════════════════════════════════
# 보종(상품 성격) 정의
# 매칭 파이프라인: 보험사 → 코드 → 보종 → 고유명
# ══════════════════════════════════════════════════════════

# HARD_STRICT: 양방향 절대 규칙 — 완화 없음
# 상품 끝에 붙는 보종 키워드 중심 (종신보험, 암보험, 실손보험 등)
HARD_STRICT = frozenset([
    '실손', '태아', '암', '연금', '저축', '재물',
    '경영', '변액', '치아', '여성',
])

# SOFT_STRICT: 한 방향 절대 + 반대 방향 완화
# 가입 방식/형태 수식어 — 해당 버전이 없는 보험사 존재 가능
SOFT_STRICT = frozenset([
    '갱신', '간편', '정기',
])

STRICT_KEYWORDS = HARD_STRICT | SOFT_STRICT

# ── 동의어 그룹 ──────────────────────────────────────────
# 같은 그룹 = 같은 보종으로 간주 → 서로 매칭 허용
# 그룹 전체가 HARD 규칙 적용 (그룹 있음↔있음, 없음↔없음)
SYNONYM_GROUPS = [
    # 운전/교통
    frozenset(['운전자', '교통']),
    # 해지환급금 계열 — prefix 매칭 사용
    # '해약환급금미지급', '해약환급금일부지급', '해지환급금미지급' 등 모든 변형 포함
    frozenset(['무해지', '저해지', '해약환급금', '해지환급금']),
    # 상조/장례
    frozenset(['상조', '장례']),
    # 어린이/아이
    frozenset(['어린이', '아이']),
    # 간병/치매
    frozenset(['간병', '치매']),
    # 통합/종합
    frozenset(['통합', '종합']),
    # 종신/상속
    frozenset(['종신', '상속']),
]

# prefix 매칭: 이 단어로 시작하는 텍스트가 있으면 그룹 소속으로 간주
# 예) '해약환급금미지급', '해약환급금일부지급' → '해약환급금' 그룹
_SYNONYM_PREFIX = frozenset(['해약환급금', '해지환급금'])

_SYNONYM_MEMBERS = frozenset(w for grp in SYNONYM_GROUPS for w in grp)

# COMMON_MODIFIERS: 수식어 제거용 — STRICT/동의어 멤버 절대 포함 금지
COMMON_MODIFIERS = [
    '심사', '가입',
    '갱신형', '비갱신형',
    '연만기', '연납', '월납', '일시납',
    '만기', '특약', '기본형', '표준형',
    '실속형', '플러스형', '플러스',
    '온라인', '다이렉트', '무진단', '무심사',
    '슬림', '베이직', '스마트', '프리미엄',
    '골드', '실버', '블루', '그린',
]
_MODIFIERS_SORTED = sorted(COMMON_MODIFIERS, key=len, reverse=True)


# ══════════════════════════════════════════════════════════
# 텍스트 정제
# ══════════════════════════════════════════════════════════

def clean_text(text):
    """공백 및 보이지 않는 문자 제거"""
    if pd.isna(text) or text in ('', 'nan', 'None'):
        return ''
    s = str(text)
    return ''.join(
        c for c in s
        if not (c.isspace() or c in '\xa0\u3000\u200b\u200c\u200d\ufeff')
    )

def normalize_insurer(text):
    """보험사명 정규화 — 한글·영문·숫자만, 소문자"""
    if not text:
        return ''
    return ''.join(
        c for c in text
        if ('\uAC00' <= c <= '\uD7A3') or c.isdigit() or ('a' <= c.lower() <= 'z')
    ).lower()

def remove_insurer_name(text, insurer):
    if not text or not insurer:
        return text
    result = text.replace(insurer, '')
    for i in range(2, min(5, len(insurer))):
        partial = insurer[:i]
        if partial in result:
            result = result.replace(partial, '')
    return result

# 정규식 사전 컴파일 (매 호출마다 컴파일 방지)
_RE_MU_PREFIX    = re.compile(r'^[\(\[\（\(][무無][\)\]\）\)]\s*')
_RE_PAREN_INNER  = re.compile(r'\([^\)]*\)')
_RE_BRACKET_OPEN = re.compile(r'[\(\（]')

def remove_brackets_prefix(text):
    """
    괄호 기호 및 내부 구분자 제거 — 내용은 보존.

    - 소괄호 () : 내용 통째로 제거 (부가설명)
    - 대괄호 [] : 기호만 제거, 내용 보존 ([약:속] → 약속)
    - 콜론·쉼표 등 구분자 → 공백
    """
    if not text:
        return ''
    text = _RE_PAREN_INNER.sub('', text)
    text = text.replace('[', ' ').replace(']', ' ')
    for ch in (':', ',', '.', '/'):
        text = text.replace(ch, ' ')
    for term in ('무배당', '(무)', '(無)', '배당'):
        text = text.replace(term, '')
    for ch in ('(', ')', '（', '）'):
        text = text.replace(ch, '')
    return text

def remove_modifiers(text):
    """
    COMMON_MODIFIERS 제거.
    STRICT_KEYWORDS는 절대 건드리지 않음 — 상수 분리로 충돌 원천 차단.
    """
    if not text:
        return text
    result = text
    for mod in _MODIFIERS_SORTED:
        result = result.replace(mod, '')
    return result

def remove_special_chars(text):
    """특수문자 제거 — 한글·영문·숫자만"""
    if not text:
        return ''
    return ''.join(
        c for c in text
        if ('\uAC00' <= c <= '\uD7A3') or ('a' <= c <= 'z') or ('A' <= c <= 'Z') or c.isdigit()
    )

def keep_text_only(text):
    """한글·영문만 (숫자 제거)"""
    if not text:
        return ''
    return ''.join(
        c for c in text
        if ('\uAC00' <= c <= '\uD7A3') or ('a' <= c <= 'z') or ('A' <= c <= 'Z')
    )

def remove_exclusion_terms(text, exclusion_set):
    if not text:
        return ''
    for term in exclusion_set:
        if term:
            text = text.replace(term, '')
    return text

def preprocess_text(text, insurer, exclusion_set, keep_exclusions=True):
    """
    정제 파이프라인:
    1. clean_text (공백·보이지않는문자)
    2. 보험사명 제거
    3. 괄호·무배당 접두어 제거
    4. COMMON_MODIFIERS 제거 (STRICT_KEYWORDS 보존)
    5. 특수문자 제거
    6. 제외문구 제거 (선택)
    """
    if not text:
        return ''
    r = clean_text(text)
    r = remove_insurer_name(r, insurer)
    r = remove_brackets_prefix(r)
    r = remove_modifiers(r)
    r = remove_special_chars(r)
    if not keep_exclusions:
        r = remove_exclusion_terms(r, exclusion_set)
    return r


def extract_unique_name(clean_text_str, exclusion_set):
    """
    2단계: 정제된 텍스트에서 고유명만 추출.

    STRICT_KEYWORDS + 동의어 멤버 + 제외문구를 모두 제거하면
    남는 텍스트가 진짜 고유 식별자.

    예) "더경증간편건강"  → "더경증"
        "마음든든암갱신"  → "마음든든"
        "약속종신"        → "약속"
        "참편한"          → "참편한"  (제외문구 건강·보험 제거 후)
    """
    if not clean_text_str:
        return ''
    r = clean_text_str
    # STRICT_KEYWORDS 제거
    for kw in STRICT_KEYWORDS:
        r = r.replace(kw, '')
    # 동의어 멤버 제거 (STRICT에 없는 것만)
    for w in _SYNONYM_MEMBERS:
        if w not in STRICT_KEYWORDS:
            r = r.replace(w, '')
    # 제외문구 제거
    r = remove_exclusion_terms(r, exclusion_set)
    return r.strip()

def extract_main_name(original_text):
    """STRICT 키워드 체크용 메인 상품명 추출."""
    if not original_text:
        return ''
    s = _RE_MU_PREFIX.sub('', original_text.strip())
    m = _RE_BRACKET_OPEN.search(s)
    if m:
        s = s[:m.start()]
    s = s.replace('[', ' ').replace(']', ' ')
    for ch in (':', ',', '.', '/'):
        s = s.replace(ch, ' ')
    return remove_modifiers(clean_text(s))


def _has_group(text, grp):
    """
    그룹 소속 여부 판단.
    - 일반 멤버: 텍스트에 포함되면 True
    - prefix 멤버: 텍스트에 해당 prefix로 시작하는 단어가 있으면 True
      예) '해약환급금' prefix → '해약환급금미지급', '해약환급금일부지급' 모두 해당
    """
    for w in grp:
        if w in _SYNONYM_PREFIX:
            # prefix 매칭: 해당 prefix를 포함하는지 체크
            if w in text:
                return True
        else:
            if w in text:
                return True
    return False


# ══════════════════════════════════════════════════════════
# STRICT 키워드 필터
# ══════════════════════════════════════════════════════════

def filter_by_strict_keywords(target_original, ref_products):
    """
    HARD_STRICT : 양방향 절대 규칙
    SOFT_STRICT : 단방향 절대 + 역방향 완화
    SYNONYM_GROUPS : 그룹 단위 매칭
      예) 타겟에 '운전자' → ref에 '교통'도 허용 (같은 그룹)
          타겟에 '무해지' → ref에 '저해지', '해약환급금미지급' 허용
          타겟에 그룹 없음 → ref에 그룹 단어 있으면 제외
    """
    target_main = extract_main_name(target_original)

    target_synonym_groups = [
        grp for grp in SYNONYM_GROUPS
        if _has_group(target_main, grp)
    ]

    def passes_hard_strict(ref_raw):
        for kw in HARD_STRICT:
            if kw in _SYNONYM_MEMBERS: continue
            t_has = kw in target_main
            r_has = kw in ref_raw
            if t_has != r_has:
                return False
        for grp in SYNONYM_GROUPS:
            t_has_grp = _has_group(target_main, grp)
            r_has_grp = _has_group(ref_raw, grp)
            if t_has_grp != r_has_grp:
                return False
        return True

    hard = [p for p in ref_products if passes_hard_strict(p['clean_no_excl'])]

    if not hard:
        def passes_hard_absent_only(ref_raw):
            for kw in HARD_STRICT:
                if kw in _SYNONYM_MEMBERS: continue
                if kw not in target_main and kw in ref_raw:
                    return False
            for grp in SYNONYM_GROUPS:
                t_has_grp = _has_group(target_main, grp)
                r_has_grp = _has_group(ref_raw, grp)
                if not t_has_grp and r_has_grp:
                    return False
            return True
        hard = [p for p in ref_products if passes_hard_absent_only(p['clean_no_excl'])]
        if not hard:
            hard = ref_products

    # SOFT_STRICT absent (타겟에 없는데 ref에 있으면 제외)
    def passes_soft_absent(ref_raw):
        for kw in SOFT_STRICT:
            if kw not in target_main and kw in ref_raw:
                return False
        return True

    candidates = [p for p in hard if passes_soft_absent(p['clean_no_excl'])]
    if not candidates:
        candidates = hard

    # SOFT_STRICT present (타겟에 있는데 ref에 없으면 제외, 완화 가능)
    def passes_soft_present(ref_raw):
        for kw in SOFT_STRICT:
            if kw in target_main and kw not in ref_raw:
                return False
        return True

    final = [p for p in candidates if passes_soft_present(p['clean_no_excl'])]
    return final if final else candidates


# ══════════════════════════════════════════════════════════
# 점수 계산
# ══════════════════════════════════════════════════════════

def tokenize(text):
    """2~4글자 n-gram"""
    tokens = set()
    for n in (2, 3, 4):
        for i in range(len(text) - n + 1):
            tokens.add(text[i:i+n])
    return tokens

def build_tfidf_weights(all_texts):
    N = len(all_texts)
    df_counter = Counter()
    for text in all_texts:
        for tok in tokenize(text):
            df_counter[tok] += 1
    return {tok: math.log((N + 1) / (df + 1)) + 1 for tok, df in df_counter.items()}

def tfidf_similarity(t_tokens, r_tokens, idf):
    if not t_tokens or not r_tokens:
        return 0.0
    common = t_tokens & r_tokens
    if not common:
        return 0.0
    num   = sum(idf.get(t, 1.0) for t in common)
    t_n   = math.sqrt(sum(idf.get(t, 1.0) ** 2 for t in t_tokens))
    r_n   = math.sqrt(sum(idf.get(t, 1.0) ** 2 for t in r_tokens))
    denom = t_n * r_n
    return num / denom if denom > 0 else 0.0

def find_longest_substring(s1, s2):
    if not s1 or not s2:
        return '', 0
    m, n = len(s1), len(s2)
    if m > n:
        s1, s2 = s2, s1
        m, n = n, m
    max_len = end_pos = 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1] + 1
                if curr[j] > max_len:
                    max_len = curr[j]
                    end_pos = i
        prev = curr
    return (s1[end_pos - max_len:end_pos], max_len) if max_len > 0 else ('', 0)

def count_prefix_match(s1, s2):
    count = 0
    for a, b in zip(s1, s2):
        if a == b:
            count += 1
        else:
            break
    return count

def calc_match_score(t_unique, r_unique, t_unique_tok, r_unique_tok,
                     t_full, r_full, idf, is_active=False):
    """
    점수 계산 — LCS 제외 (승자에게만 1회 계산).

    고유명 기준:
      A. TF-IDF × 40   — 희귀 고유명 n-gram 일치
      B. prefix × 2    — 고유명 앞부분 일치

    전체 텍스트 기준:
      C. token_set_ratio × 0.20
      D. partial_ratio   × 0.15

    판매중 보너스: +5.0
    """
    if not t_unique and not t_full:
        return 0.0, 0

    tfidf  = tfidf_similarity(t_unique_tok, r_unique_tok, idf)
    prefix = count_prefix_match(t_unique, r_unique)
    t_set  = fuzz.token_set_ratio(t_full, r_full)
    part   = fuzz.partial_ratio(t_full, r_full)
    active_bonus = 5.0 if is_active else 0.0

    score = (tfidf  * 40
             + prefix * 2
             + t_set  * 0.20
             + part   * 0.15
             + active_bonus)

    return score, prefix


# ══════════════════════════════════════════════════════════
# 후보 탐색
# ══════════════════════════════════════════════════════════

def find_best_in_products(t_unique, t_unique_tok, t_full, ref_products, idf):
    """
    점수로 승자 결정 → 승자에게만 LCS 1회 계산.
    """
    best_match, best_score = None, -1
    best_prefix = 0
    matched_text = ''

    for ref in ref_products:
        is_active = ref['sales_status'] == '판매중'
        score, prefix = calc_match_score(
            t_unique, ref['unique_name'],
            t_unique_tok, ref['tokens'],
            t_full, ref['text_only'],
            idf, is_active
        )
        if score > best_score:
            best_score   = score
            best_prefix  = prefix
            best_match   = ref
            matched_text = ref['clean_no_excl']

    # LCS는 승자에게만 1회
    best_lcs = 0
    if best_match:
        _, best_lcs = find_longest_substring(t_unique, best_match['unique_name'])

    return best_match, best_score, best_prefix, best_lcs, matched_text


# ══════════════════════════════════════════════════════════
# 메인 매칭 로직
# ══════════════════════════════════════════════════════════

def match_product(target_original, target_clean, ref_products,
                  target_is_real, exclusion_set, idf):
    if not ref_products:
        return None, None, 0, 0, '', '', 'No Products'

    # 1단계: 실손 필터
    if target_is_real:
        filtered = [p for p in ref_products if p['is_real_expense']]
        if filtered:
            ref_products = filtered

    # 1단계: 상품군 분류 (STRICT + 동의어 필터)
    candidates = filter_by_strict_keywords(target_original, ref_products)

    # 2단계: 고유명 추출 (STRICT+동의어+제외문구 제거)
    t_unique     = extract_unique_name(target_clean, exclusion_set)
    t_unique_tok = tokenize(t_unique)
    t_full       = keep_text_only(target_clean)  # 전체 텍스트 (fuzzy용)

    # 3단계: 고유명 기반 최종 매칭
    best, best_score, best_prefix, best_lcs, matched_text = \
        find_best_in_products(t_unique, t_unique_tok, t_full, candidates, idf)

    # 약한 매칭(TF-IDF < 0.1) → 제외문구 제거 버전으로 재시도
    tfidf_best = tfidf_similarity(t_unique_tok, best['tokens'] if best else set(), idf)

    if tfidf_best < 0.1:
        t_ne         = remove_exclusion_terms(target_clean, exclusion_set)
        t_ne_unique  = extract_unique_name(t_ne, set())
        t_ne_tok     = tokenize(t_ne_unique)
        t_ne_full    = keep_text_only(t_ne)

        b2_match, b2_score = None, -1
        b2_prefix = b2_lcs = 0
        b2_text = ''

        for ref in candidates:
            is_active = ref['sales_status'] == '판매중'
            score2, p2 = calc_match_score(
                t_ne_unique, ref['unique_name'],
                t_ne_tok, ref['tokens'],
                t_ne_full, ref['text_only_excl'],
                idf, is_active
            )
            if score2 > b2_score:
                b2_score  = score2
                b2_prefix = p2
                b2_match  = ref
                b2_text   = ref['clean_with_excl']

        # 2패스 승자 LCS 1회
        if b2_match:
            _, b2_lcs = find_longest_substring(t_ne_unique, b2_match['unique_name'])
            if b2_score > best_score:
                best, best_score = b2_match, b2_score
                best_prefix, best_lcs = b2_prefix, b2_lcs
                matched_text = b2_text

    if not best:
        return None, None, 0, 0, '', '', 'No Match'

    # LCS = 0 → 제외문구 제거 버전으로 재계산
    if best_lcs == 0:
        t_ne_u = extract_unique_name(
            remove_exclusion_terms(target_clean, exclusion_set), set())
        if t_ne_u:
            _, lcs = find_longest_substring(t_ne_u, best['unique_name'])
            if lcs > best_lcs:
                best_lcs     = lcs
                matched_text = best['clean_with_excl']

    t_tok2  = set(target_clean)
    r_tok2  = set(best['clean_no_excl'])
    overlap = len(t_tok2 & r_tok2)
    clean_used = (f'score={best_score:.2f} unique="{t_unique}" '
                  f'lcs={best_lcs} prefix={best_prefix}')

    return (best['prod_code'], best['prod_name'],
            overlap, best_lcs, matched_text, clean_used, best['sales_status'])


# ══════════════════════════════════════════════════════════
# 인덱스 빌드
# ══════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def build_reference_index(ref_df, exclusion_tuple):
    exclusion_set = set(exclusion_tuple)
    ref_dict = defaultdict(list)

    for _, row in ref_df.iterrows():
        insurer    = clean_text(str(row.get('보험사', '')))
        prod_name  = clean_text(str(row.get('상품명', '')))
        prod_code  = str(row.get('상품코드', '')).strip()
        ins_code   = str(row.get('보험사상품코드', '')).strip()
        sales_stat = str(row.get('판매상태', '')).strip()
        is_real    = ('실손' in prod_name or '실손' in str(row.get('IIMS 상품명', '')))

        if not insurer or not prod_name:
            continue

        no_excl   = preprocess_text(prod_name, insurer, exclusion_set, keep_exclusions=True)
        with_excl = preprocess_text(prod_name, insurer, exclusion_set, keep_exclusions=False)

        # 고유명 추출 (STRICT+동의어+제외문구 제거 후 남은 것)
        unique_name      = extract_unique_name(no_excl, exclusion_set)
        unique_name_text = keep_text_only(unique_name)

        ref_dict[insurer].append({
            'prod_name':        prod_name,
            'prod_code':        prod_code,
            'insurer_code':     ins_code,
            'clean_no_excl':    no_excl,
            'clean_with_excl':  with_excl,
            'text_only':        keep_text_only(no_excl),
            'text_only_excl':   keep_text_only(with_excl),
            'unique_name':      unique_name_text,           # 고유명만
            'tokens':           tokenize(unique_name_text), # TF-IDF는 고유명 기준
            'tokens_full':      tokenize(keep_text_only(no_excl)),  # 전체 텍스트 토큰
            'tokens_no_excl':   tokenize(keep_text_only(with_excl)),
            'sales_status':     sales_stat,
            'is_real_expense':  is_real,
        })

    for insurer in ref_dict:
        ref_dict[insurer].sort(key=lambda x: (
            0 if x['sales_status'] == '판매중' else 1,
            -int(''.join(filter(str.isdigit, x['prod_code'])) or 0)
        ))

    # TF-IDF는 고유명 기준으로 빌드 — 희귀한 고유명일수록 높은 가중치
    idf_dict = {ins: build_tfidf_weights([p['unique_name'] for p in prods])
                for ins, prods in ref_dict.items()}

    normalized_map = {normalize_insurer(k): k for k in ref_dict}

    return dict(ref_dict), idf_dict, normalized_map

def match_insurer(target_insurer, ref_dict, normalized_map):
    if not target_insurer:
        return None, 0
    if target_insurer in ref_dict:
        return target_insurer, 100
    norm = normalize_insurer(target_insurer)
    if norm in normalized_map:
        return normalized_map[norm], 100
    return None, 0


# ══════════════════════════════════════════════════════════
# 행 단위 처리
# ══════════════════════════════════════════════════════════

def process_target_row(row, ref_dict, idf_dict, normalized_map, exclusion_set, target_df):
    idx            = row.name
    target_insurer = clean_text(str(row.iloc[0]))
    target_product = clean_text(str(row.iloc[1]))

    if '보험사상품코드' in target_df.columns:
        raw = row.get('보험사상품코드', '')
        target_ins_code = '' if pd.isna(raw) else str(raw).strip()
        if target_ins_code in ('nan', 'None', '-', 'N/A'):
            target_ins_code = ''
    else:
        target_ins_code = ''

    target_is_real  = '실손' in target_product
    matched_ins, ins_score = match_insurer(target_insurer, ref_dict, normalized_map)

    if not matched_ins:
        return {
            'idx': idx, '상품코드': '', 'Matched_상품명': '', 'Matched_보험사': '',
            'Insurer_Score': 0, 'Overlap_Count': 0, 'Substring_Length': 0,
            'Matched_Text': '', 'Clean_Version': '', 'Match_Method': 'No Insurer Match',
            'Confidence': 'No Data', '판매상태': 'N/A',
            'Target_Clean_Text': '', 'Target_Original': target_product
        }

    ref_products = ref_dict.get(matched_ins, [])
    idf          = idf_dict.get(matched_ins, {})

    # 보험사 상품코드 직접 매칭
    if target_ins_code:
        for ref in ref_products:
            if ref['insurer_code'].strip() and ref['insurer_code'].strip() == target_ins_code:
                return {
                    'idx': idx, '상품코드': ref['prod_code'], 'Matched_상품명': ref['prod_name'],
                    'Matched_보험사': matched_ins, 'Insurer_Score': ins_score,
                    'Overlap_Count': 999, 'Substring_Length': 999,
                    'Matched_Text': ref['prod_name'], 'Clean_Version': 'Insurer Code',
                    'Match_Method': 'Insurer Code Match', 'Confidence': 'High',
                    '판매상태': ref['sales_status'],
                    'Target_Clean_Text': preprocess_text(
                        target_product, target_insurer, exclusion_set, keep_exclusions=True),
                    'Target_Original': target_product
                }

    target_clean = preprocess_text(target_product, target_insurer, exclusion_set, keep_exclusions=True)

    prod_code, prod_name, overlap, sublen, matched_text, clean_ver, sales_stat = match_product(
        target_product, target_clean, ref_products, target_is_real, exclusion_set, idf)

    if prod_code:
        confidence = 'High' if sublen >= 5 else 'Medium' if sublen >= 3 else 'Low'
        return {
            'idx': idx, '상품코드': prod_code, 'Matched_상품명': prod_name,
            'Matched_보험사': matched_ins, 'Insurer_Score': ins_score,
            'Overlap_Count': overlap, 'Substring_Length': sublen,
            'Matched_Text': matched_text, 'Clean_Version': clean_ver,
            'Match_Method': 'Text Match', 'Confidence': confidence,
            '판매상태': sales_stat, 'Target_Clean_Text': target_clean,
            'Target_Original': target_product
        }

    return {
        'idx': idx, '상품코드': '', 'Matched_상품명': '', 'Matched_보험사': matched_ins,
        'Insurer_Score': ins_score, 'Overlap_Count': 0, 'Substring_Length': 0,
        'Matched_Text': '', 'Clean_Version': '', 'Match_Method': 'No Match',
        'Confidence': 'No Data', '판매상태': 'N/A',
        'Target_Clean_Text': target_clean, 'Target_Original': target_product
    }


def cross_match_low_results(results, exclusion_set):
    """
    2차 비교: Low/NoMatch → 같은 보험사 High 결과와 텍스트 유사도 비교.
    보종 일치 확인 후 비교 (변액연금↔종신 같은 오매칭 방지).
    """
    HIGH_MIN_LCS    = 5
    CROSS_MIN_RATIO = 75

    # 보험사별 High 풀 구성 — 보종 판단용 원본 상품명도 저장
    high_pool = defaultdict(list)
    for r in results:
        if (r.get('Confidence') == 'High'
                and r.get('Substring_Length', 0) >= HIGH_MIN_LCS
                and r.get('상품코드', '')
                and r.get('Target_Clean_Text', '')
                and r.get('Matched_보험사', '')):
            t_orig = r.get('Target_Original', '')
            high_pool[r['Matched_보험사']].append({
                'clean':          keep_text_only(r['Target_Clean_Text']),
                'target_original': t_orig,
                'target_main':    extract_main_name(t_orig) if t_orig else '',  # 미리 계산
                'prod_code':      r['상품코드'],
                'prod_name':      r.get('Matched_상품명', ''),
                'sales_status':   r.get('판매상태', ''),
            })

    if not high_pool:
        return results, 0

    updated = 0
    for r in results:
        if r.get('Confidence') not in ('Low', 'No Data'):
            continue
        insurer = r.get('Matched_보험사', '')
        if not insurer or insurer not in high_pool:
            continue
        t_text     = keep_text_only(r.get('Target_Clean_Text', ''))
        t_original = r.get('Target_Original', '')
        t_main     = extract_main_name(t_original) if t_original else ''  # Low 타겟 1회만 계산
        if not t_text:
            continue

        best_ratio, best_ref = 0, None
        for h in high_pool[insurer]:
            if not h['clean']:
                continue
            # 보종 필터: 미리 계산된 main_name 사용
            if not _same_product_group_mains(t_main, h['target_main']):
                continue
            ratio = fuzz.partial_ratio(t_text, h['clean'])
            if ratio > best_ratio:
                best_ratio = ratio
                best_ref   = h
                if ratio == 100:
                    break

        if best_ratio >= CROSS_MIN_RATIO and best_ref:
            r['상품코드']         = best_ref['prod_code']
            r['Matched_상품명']   = best_ref['prod_name']
            r['판매상태']         = best_ref['sales_status']
            r['Match_Method']     = 'Cross Match (High→Low)'
            r['Confidence']       = 'Medium'
            r['Substring_Length'] = best_ratio
            r['Clean_Version']    = f'cross_ratio={best_ratio}'
            r['Matched_Text']     = best_ref['prod_name']
            updated += 1

    return results, updated


def _same_product_group_mains(main_a, main_b):
    """
    미리 계산된 main_name 두 개로 보종 일치 여부 판단.
    extract_main_name 반복 호출 없음.
    """
    if not main_a or not main_b:
        return True
    for kw in HARD_STRICT:
        if kw in _SYNONYM_MEMBERS:
            continue
        if (kw in main_a) != (kw in main_b):
            return False
    for grp in SYNONYM_GROUPS:
        if _has_group(main_a, grp) != _has_group(main_b, grp):
            return False
    for kw in SOFT_STRICT:
        if (kw in main_a) != (kw in main_b):
            return False
    return True

def create_target_template():
    return pd.DataFrame({
        '보험사': ['DB손보','KB손보','한화생명','삼성화재','메리츠화재',
                   '흥국화재','현대해상','농협생명','신한라이프','교보생명'],
        '상품명': [
            '무배당 DB 실손의료비보험(갱신형)',
            'KB 무배당 실손의료비보험',
            '무배당 한화 더경증 간편건강보험Ⅱ(세만기형) 2601',
            '삼성화재 무배당 참편한건강보험',
            '메리츠 무배당 마음든든암보험(연만기 갱신형)',
            '흥국 무배당 실손의료보험[태아]',
            '현대해상 무배당 굿앤굿어린이CI보험',
            '농협생명 무배당 NH저축보험',
            '신한라이프 무배당 연금보험',
            '교보생명 무배당 교보종신보험',
        ],
        '보험사상품코드': ['3059002','24482','','L1716','','','HH2023','','SL001','KB001'],
    })

def create_reference_template():
    return pd.DataFrame({
        '보험사': ['DB손보','DB손보','KB손보','한화생명','한화생명',
                   '삼성화재','메리츠화재','흥국화재','현대해상','농협생명','신한라이프','교보생명'],
        '상품명': [
            '(무)DB실손의료비보험(갱신형)',
            '(무)DB간편실손의료비보험(갱신형)',
            '(무)KB실손의료비보험',
            '무배당 한화 더경증 간편건강보험Ⅱ(세만기형)',
            '무배당 한화 참편한건강보험',
            '무배당 삼성 참편한건강보험',
            '무배당 메리츠 마음든든암보험(연만기갱신형)',
            '무배당 흥국 실손의료보험[태아]',
            '무배당 굿앤굿어린이CI보험',
            '무배당 NH저축보험',
            '무배당 신한 연금보험',
            '무배당 교보종신보험',
        ],
        '상품코드': [f'1100{i}' for i in range(1, 13)],
        '보험사상품코드': ['3059002','','24482','','','L1716','','','HH2023','','SL001','KB001'],
        '판매상태': ['판매중','판매중지','판매중','판매중','판매중지',
                     '판매중','판매중','판매중','판매중','판매중','판매중','판매중'],
        'IIMS 상품명': [''] * 12,
    })

def to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as w:
        df.to_excel(w, index=False)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════

def main():
    st.title("🎯 Ultra-Fast Product Code Matcher")
    st.success("✅ TF-IDF + rapidfuzz 하이브리드 | ⚡ 고유명 우선 매칭")

    with st.expander("📋 파일 양식 다운로드 (처음 사용 시 확인)", expanded=False):
        st.markdown("""
**파일 업로드 전 아래 양식을 먼저 확인하세요.**

| 파일 | 필수 컬럼 |
|------|-----------|
| **Target** | `보험사`, `상품명`, `보험사상품코드`(선택) |
| **Reference** | `보험사`, `상품명`, `상품코드`, `보험사상품코드`(선택), `판매상태`, `IIMS 상품명`(선택) |

- `판매상태` : `판매중` / `판매중지` 중 하나로 입력
- `보험사상품코드` 가 있으면 텍스트 매칭보다 우선 적용
- `IIMS 상품명` 에 `실손` 포함 시 실손 상품으로 인식
        """)
        tc1, tc2 = st.columns(2)
        with tc1:
            st.download_button("⬇️ Target 양식 다운로드",
                data=to_excel_bytes(create_target_template()),
                file_name="target_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.dataframe(create_target_template(), use_container_width=True)
        with tc2:
            st.download_button("⬇️ Reference 양식 다운로드",
                data=to_excel_bytes(create_reference_template()),
                file_name="reference_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.dataframe(create_reference_template(), use_container_width=True)
    st.divider()

    with st.sidebar:
        st.header("🚫 제외문구")
        exclusion_input = st.text_area("쉼표로 구분", value=','.join(DEFAULT_EXCLUSIONS), height=150)
        exclusion_terms = [t.strip() for t in exclusion_input.split(',') if t.strip()]
        st.write(f"✅ {len(exclusion_terms)}개: {exclusion_terms}")
        st.divider()
        st.warning(
            "⚠️ **보험사명 완전 일치 필수**\n\n"
            "타겟과 레퍼런스의 보험사명이 **정확히 동일**해야 매칭됩니다.\n\n"
            "예) `DB손보` ↔ `DB손보` ✅\n\n"
            "예) `DB손해보험` ↔ `DB손보` ❌\n\n"
            "업로드 전 두 파일의 보험사명을 통일해주세요."
        )
        st.divider()
        st.info(
            "💡 매칭 로직\n"
            "1. 보험사명 완전 일치 범위 내에서만 검색\n"
            "2. 보험사상품코드 직접 매칭\n"
            "3. 보종 필터 (상품 성격 매칭)\n"
            "   [절대] 실손·태아·암·연금·저축·재물·경영·변액·치아·여성\n"
            "   [완화] 갱신·간편·정기\n"
            "   [동의어] 운전자↔교통 / 무해지↔저해지↔해약환급금미지급\n"
            "   　　　　상조↔장례 / 어린이↔아이 / 간병↔치매\n"
            "   　　　　통합↔종합 / 종신↔상속\n"
            "4. [1단계] 상품군 분류\n"
            "   STRICT+동의어로 후보 풀 확정\n"
            "   [2단계] 고유명 추출\n"
            "   STRICT·동의어·제외문구 제거 → 순수 고유명\n"
            "   [3단계] 고유명 매칭\n"
            "   TF-IDF×40 + LCS×3 + prefix×2\n"
            "   + token_set×0.2 + partial×0.15\n"
            "   + 판매중 보너스+5\n"
            "5. TF-IDF<0.1 → 제외문구 제거 재시도\n"
            "6. LCS=0 → 제외문구 제거 버전 재계산\n"
            "7. **2차 교차매칭**: Low/NoMatch →\n"
            "   같은 보험사 High 결과와 텍스트 비교\n"
            "   LCS≥4 이면 해당 상품코드 부여\n"
            "8. 판매중 우선 / 상품코드 높은 순"
        )

    st.error("⚠️ 타겟과 레퍼런스의 **보험사명이 정확히 일치**해야 합니다.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📁 Target 파일")
        file_target = st.file_uploader("Target (.xlsx)", type=['xlsx'], key='target')
        if file_target:
            df_target = pd.read_excel(file_target, engine='openpyxl', dtype=str)
            st.write(f"✅ {len(df_target)}행 로드")
            st.dataframe(df_target.head())
    with col2:
        st.subheader("📁 Reference 파일")
        file_ref = st.file_uploader("Reference (.xlsx)", type=['xlsx'], key='ref')
        if file_ref:
            df_ref = pd.read_excel(file_ref, engine='openpyxl', dtype=str)
            st.write(f"✅ {len(df_ref)}행 로드")
            st.dataframe(df_ref.head())

    if st.button("🚀 매칭 시작", type="primary", disabled=not (file_target and file_ref)):
        start_time = time.time()

        with st.spinner("TF-IDF 인덱스 빌드 중..."):
            ref_dict, idf_dict, normalized_map = build_reference_index(df_ref, tuple(exclusion_terms))
            total_ref = sum(len(v) for v in ref_dict.values())
            st.write(f"✅ {len(ref_dict)}개 보험사 / {total_ref}개 상품 인덱싱 완료")

        results      = []
        progress_bar = st.progress(0)
        total_rows   = len(df_target)

        for idx, row in df_target.iterrows():
            results.append(process_target_row(
                row, ref_dict, idf_dict, normalized_map, set(exclusion_terms), df_target))
            if (idx + 1) % 100 == 0:
                progress_bar.progress((idx + 1) / total_rows)
        progress_bar.progress(1.0)

        # 2차 비교: Low/NoMatch → High 결과와 텍스트 유사도 비교
        results, cross_updated = cross_match_low_results(results, set(exclusion_terms))

        df_result = pd.DataFrame(results)
        df_result = df_result.set_index('idx')

        df_output = df_target.copy()
        for col in ['상품코드','Matched_상품명','Matched_보험사','Match_Method',
                    'Overlap_Count','Substring_Length','Confidence','판매상태',
                    'Matched_Text','Clean_Version','Target_Clean_Text']:
            if col in df_result.columns:
                df_output[col] = df_result[col].values

        # 검토필요 컬럼 — 벡터 연산으로 빠르게 처리
        method  = df_output['Match_Method'].astype(str)
        conf    = df_output['Confidence'].astype(str)
        sublen  = pd.to_numeric(df_output['Substring_Length'], errors='coerce').fillna(0).astype(int)

        cond_unmatch = method.isin(['No Match', 'No Insurer Match']) | (conf == 'No Data')
        cond_review  = (conf == 'Low') | (method == 'Cross Match (High→Low)') | ((conf == 'Medium') & (sublen < 4))

        df_output['검토필요'] = ''
        df_output.loc[cond_review,  '검토필요'] = '🔍 수기검증필수'
        df_output.loc[cond_unmatch, '검토필요'] = '⚠️ 미매칭'

        review_cnt = (df_output['검토필요'] != '').sum()

        elapsed  = time.time() - start_time
        total    = len(df_output)
        matched  = df_output['상품코드'].astype(str).str.strip().ne('').sum()
        code_m   = (df_output['Match_Method'] == 'Insurer Code Match').sum()
        text_m   = (df_output['Match_Method'] == 'Text Match').sum()
        cross_m  = (df_output['Match_Method'] == 'Cross Match (High→Low)').sum()
        active   = (df_output['판매상태'] == '판매중').sum()
        disc     = (df_output['판매상태'] == '판매중지').sum()

        st.success(f"✅ {elapsed:.1f}s | {matched}/{total} 매칭 ({matched/total*100:.1f}%)")

        no_ins = df_output[df_output['Match_Method'] == 'No Insurer Match']
        if len(no_ins) > 0:
            unmatched = no_ins['보험사'].unique().tolist()
            st.error(f"❌ 보험사명 불일치: {len(no_ins)}건 — `{'`, `'.join(unmatched)}`")

        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("⏱️ 시간",       f"{elapsed:.1f}s")
        c2.metric("🔑 코드매칭",   f"{code_m}")
        c3.metric("📝 텍스트매칭", f"{text_m}")
        c4.metric("🔗 교차매칭",   f"{cross_m}")
        c5.metric("🟢 판매중",     f"{active}")
        c6.metric("🔴 판매중지",   f"{disc}")
        c7.metric("🔍 검토필요",   f"{review_cnt}")

        st.subheader("📋 결과")
        st.dataframe(df_output[['검토필요','보험사','상품명','Target_Clean_Text','Matched_상품명',
                                 '상품코드','판매상태','Match_Method',
                                 'Substring_Length','Confidence','Clean_Version']].head(50))

        st.subheader("🔍 상세 비교 (상위 20건)")
        try:
            for i in range(min(20, len(df_output))):
                row = df_output.iloc[i]
                with st.container():
                    st.divider()
                    ca, cb = st.columns(2)
                    with ca:
                        st.write("**📌 Target:**")
                        st.info(f"**보험사:** {row['보험사']}\n\n"
                                f"**상품명:** {row['상품명']}\n\n"
                                f"**정제:** `{row['Target_Clean_Text']}`")
                    with cb:
                        code_val = str(row['상품코드']).strip()
                        if code_val and code_val not in ('', 'nan', 'None'):
                            emoji = "🟢" if row['판매상태'] == '판매중' else "🔴"
                            st.write(f"**{emoji} 매칭 결과:**")
                            st.success(f"**코드:** `{code_val}`\n\n"
                                       f"**상품명:** {row['Matched_상품명']}\n\n"
                                       f"**매칭텍스트:** `{row['Matched_Text']}`\n\n"
                                       f"**상태:** {row['판매상태']}")
                            st.caption(str(row['Clean_Version']))
                        else:
                            st.error("**❌ 미매칭**")
        except Exception as e:
            st.warning(f"상세 비교 표시 중 오류: {e}")

        # 다운로드 — 항상 표시
        excel_data = io.BytesIO()
        with pd.ExcelWriter(excel_data, engine='xlsxwriter') as writer:
            df_output.to_excel(writer, index=False, sheet_name='Results')

        st.download_button(
            "📥 결과 다운로드", excel_data.getvalue(),
            f"matched_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )


if __name__ == "__main__":
    main()
