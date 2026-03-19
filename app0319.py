import streamlit as st
import pandas as pd
import io
import math
import re
from rapidfuzz import fuzz
from collections import Counter, defaultdict
import time

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ Ultra-Fast Product Matcher",
    page_icon="🎯",
    layout="wide"
)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
DEFAULT_EXCLUSIONS = ['프로미라이프', '통합', '보장', '건강', '보험']

COMMON_MODIFIERS = [
    '갱신형', '비갱신형', '연만기', '연납', '월납', '일시납',
    '만기', '종신', '정기', '특약', '기본형', '표준형',
    '간편심사', '간편가입', '간편', '실속형', '플러스형', '플러스',
    '온라인', '다이렉트', '무진단', '무심사', '슬림', '베이직',
    '스마트', '프리미엄', '골드', '실버', '블루', '그린',
]

# 타겟에 없으면 ref에도 없어야 / 타겟에 있으면 ref에도 있어야
STRICT_KEYWORDS = ['실손', '태아', '갱신', '간편', '암', '연금', '저축', '재물', '치매', '정기']


# ──────────────────────────────────────────────
# 텍스트 정제
# ──────────────────────────────────────────────
def clean_text(text):
    """
    공백 및 보이지 않는 문자 제거 + 정규화.
    엑셀에서 읽으면 NBSP(\\xa0), 전각공백(\\u3000), 제로폭공백 등이 섞일 수 있음.
    """
    if pd.isna(text) or text in ['', 'nan', 'None']:
        return ''
    s = str(text)
    # 모든 종류의 공백·제어문자 제거 (일반 공백, NBSP, 전각공백, 탭, 개행 등)
    s = ''.join(c for c in s if not (c.isspace() or c in '\xa0\u3000\u200b\u200c\u200d\ufeff'))
    return s

def normalize_insurer(text):
    """보험사명 비교용 정규화 — 특수기호·공백 완전 제거, 소문자 통일"""
    if not text:
        return ''
    # 한글·영문·숫자만 남김
    return ''.join(c for c in text if
                   ('\uAC00' <= c <= '\uD7A3') or
                   c.isdigit() or
                   ('a' <= c.lower() <= 'z')).lower()

def remove_insurer_name(text, insurer):
    if not text or not insurer:
        return text
    result = text.replace(insurer, '')
    if len(insurer) >= 2:
        for i in range(2, min(5, len(insurer))):
            partial = insurer[:i]
            if partial in result:
                result = result.replace(partial, '')
    return result

def remove_special_chars(text):
    if not text:
        return ''
    return ''.join(c for c in text if
                   ('\uAC00' <= c <= '\uD7A3') or
                   ('a' <= c <= 'z') or ('A' <= c <= 'Z') or
                   ('0' <= c <= '9'))

def keep_text_only(text):
    """한글·영문만 — 숫자 제거"""
    if not text:
        return ''
    return ''.join(c for c in text if
                   ('\uAC00' <= c <= '\uD7A3') or
                   ('a' <= c <= 'z') or ('A' <= c <= 'Z'))

def remove_exclusion_terms(text, exclusion_set):
    if not text:
        return ''
    for term in exclusion_set:
        if term:
            text = text.replace(term, '')
    return text

def remove_common_terms(text):
    if not text:
        return ''
    for term in ['무배당', '(무)', '배당', '()', '(', ')', '[', ']']:
        text = text.replace(term, '')
    return text

def remove_modifiers(text):
    """
    COMMON_MODIFIERS 제거 — 긴 것부터 제거해야 중복 방지.
    단, STRICT_KEYWORDS와 겹치는 항목은 제거하지 않음.
    예) "간편가입H3종신" → "H3종신"
        "간편H3종신"    → "H3종신"  (간편은 STRICT이지만 "간편가입"으로 묶이면 제거)
    """
    if not text:
        return text
    result = text
    for mod in sorted(COMMON_MODIFIERS, key=len, reverse=True):
        # STRICT_KEYWORDS 단독 항목은 건드리지 않음
        # 단, "간편가입"처럼 STRICT 키워드를 포함하는 더 긴 수식어는 제거
        is_exact_strict = mod in STRICT_KEYWORDS
        if is_exact_strict:
            continue
        result = result.replace(mod, '')
    return result

def extract_core_name(text):
    """수식어 제거 → 고유명만 추출 (점수 계산용)"""
    return remove_modifiers(text) if text else text

def extract_main_name(text):
    """
    괄호 앞 메인 상품명 추출.
    STRICT 키워드 체크용 — 수식어는 제거하되 STRICT_KEYWORDS 자체는 유지.

    예) "(무)슬기로운 건강생활보험(22.01)(1종경증간편고지Ⅰ형)"
     →  괄호 앞: "슬기로운건강생활보험"  ("간편" 없음 ✅)

    예) "간편 H3종신(무)"
     →  괄호 앞: "간편H3종신"  ("간편" 있음 → ref에도 간편 있어야 ✅)

    예) "플러스정기보험"
     →  "플러스" 제거(COMMON_MODIFIERS) → "정기보험"  ("정기" 있음 ✅)
    """
    if not text:
        return text
    s = re.sub(r'^[\(\[\(（][무無][\)\]\)）]\s*', '', text.strip())
    m = re.search(r'[\(\[（\[]', s)
    if m:
        s = s[:m.start()]
    # 수식어 제거 (STRICT_KEYWORDS는 remove_modifiers 내에서 보존됨)
    return remove_modifiers(clean_text(s))


def preprocess_text(text, insurer, exclusion_set, keep_exclusions=True):
    """
    공통 전처리.
    정제 순서:
    1. 공백/보이지않는문자 제거
    2. 보험사명 제거
    3. 무배당·괄호 등 공통 접두어 제거
    4. COMMON_MODIFIERS 제거 (STRICT_KEYWORDS 제외)
    5. 특수문자 제거
    6. 제외문구 제거 (선택)
    """
    if not text:
        return ''
    result = clean_text(text)
    result = remove_insurer_name(result, insurer)
    result = remove_common_terms(result)
    result = remove_modifiers(result)
    result = remove_special_chars(result)
    if not keep_exclusions:
        result = remove_exclusion_terms(result, exclusion_set)
    return result


def tokenize(text):
    """
    2~4글자 n-gram 토큰으로 분리.
    예) "더경증간편건강" → {"더경", "경증", "증간", "간편", "편건", "건강", ...}
    """
    tokens = set()
    for n in (2, 3, 4):
        for i in range(len(text) - n + 1):
            tokens.add(text[i:i+n])
    return tokens


# ──────────────────────────────────────────────
# TF-IDF 가중치 빌드
# ──────────────────────────────────────────────
def build_tfidf_weights(all_texts):
    """
    보험사 내 전체 상품명에서 n-gram IDF 계산.
    희귀한 토큰(="더경증", "마음든든" 등 고유명)일수록 높은 가중치.
    흔한 토큰(="간편", "건강", "보험" 등)은 낮은 가중치.

    IDF(t) = log((N + 1) / (df(t) + 1)) + 1
    """
    N = len(all_texts)
    df_counter = Counter()
    for text in all_texts:
        tokens = tokenize(text)
        for tok in tokens:
            df_counter[tok] += 1

    idf = {}
    for tok, df in df_counter.items():
        idf[tok] = math.log((N + 1) / (df + 1)) + 1

    return idf


def tfidf_similarity(target_tokens, ref_tokens, idf):
    """
    TF-IDF 코사인 유사도.
    두 토큰 집합의 교집합 토큰에 대해 IDF 가중치를 합산 후 정규화.
    """
    if not target_tokens or not ref_tokens:
        return 0.0

    common = target_tokens & ref_tokens
    if not common:
        return 0.0

    numerator   = sum(idf.get(t, 1.0) for t in common)
    t_norm      = math.sqrt(sum(idf.get(t, 1.0) ** 2 for t in target_tokens))
    r_norm      = math.sqrt(sum(idf.get(t, 1.0) ** 2 for t in ref_tokens))
    denominator = t_norm * r_norm

    return numerator / denominator if denominator > 0 else 0.0


# ──────────────────────────────────────────────
# 보조 점수 함수
# ──────────────────────────────────────────────
def find_longest_substring(s1, s2):
    if not s1 or not s2:
        return '', 0
    m, n = len(s1), len(s2)
    if m > n:
        s1, s2 = s2, s1
        m, n = n, m
    max_len, end_pos = 0, 0
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
    """앞에서부터 연속 일치 글자 수"""
    if not s1 or not s2:
        return 0
    count = 0
    for a, b in zip(s1, s2):
        if a == b:
            count += 1
        else:
            break
    return count


# ──────────────────────────────────────────────
# 종합 매칭 점수
# ──────────────────────────────────────────────
def calc_match_score(target_clean, ref_clean, target_tokens, ref_tokens, idf):
    """
    TF-IDF + rapidfuzz + 구조적 점수 조합.

    [점수 구성]
    A. TF-IDF 코사인 유사도 (0~1)
       → n-gram 기반. 희귀한 고유명 토큰("더경증", "마음든든") 매칭 시 고점수.
       → 흔한 수식어("간편", "건강") 는 낮은 가중치.

    B. rapidfuzz token_set_ratio (0~100)
       → 어순이 달라도 토큰 집합 기준 유사도. 뒤섞인 상품명에도 강함.

    C. rapidfuzz partial_ratio (0~100)
       → 짧은 쪽이 긴 쪽에 포함되는 정도. 축약형 상품명 처리.

    D. prefix_match — 앞부분 연속 일치
       → "더경증간편건강" 처럼 고유명이 앞에 오는 경우 강력히 구분.

    E. core_sublen — 수식어 제거 후 고유명 최장 연속 일치
       → "마음든든암" 같은 핵심 이름 일치.

    최종 score = A×40 + B×0.3 + C×0.2 + D×1.5 + E×2
    가중치 해석:
      TF-IDF가 주도 (0~40점) — 고유명 희귀 토큰이 많이 겹칠수록 압도적 우세
      rapidfuzz가 보조 (0~50점) — 어순 변형·축약 커버
      구조 점수가 동점 해소 (prefix, core)
    """
    if not target_clean or not ref_clean:
        return 0.0, 0, 0

    # A. TF-IDF
    tfidf_sim = tfidf_similarity(target_tokens, ref_tokens, idf)

    # B. token_set_ratio (어순 무관 유사도)
    t_text = keep_text_only(target_clean)
    r_text = keep_text_only(ref_clean)
    token_ratio = fuzz.token_set_ratio(t_text, r_text)

    # C. partial_ratio (부분 포함 유사도)
    partial = fuzz.partial_ratio(t_text, r_text)

    # D. prefix (한글·영문 앞부분 일치)
    prefix = count_prefix_match(t_text, r_text)

    # E. core substring (수식어 제거 후 고유명)
    _, core_sub = find_longest_substring(
        keep_text_only(extract_core_name(target_clean)),
        keep_text_only(extract_core_name(ref_clean))
    )

    score = (tfidf_sim * 40
             + token_ratio * 0.3
             + partial * 0.2
             + prefix * 1.5
             + core_sub * 2)

    return score, prefix, core_sub


def extract_main_name(text):
    """
    괄호 앞 메인 상품명 추출 + 수식어 제거.
    STRICT 키워드 체크용 — ref의 clean_no_excl(수식어 제거됨)과 공평하게 비교.

    예) "(무)슬기로운 건강생활보험(22.01)(1종경증간편고지Ⅰ형)"
     →  괄호 앞: "슬기로운 건강생활보험"
     →  수식어 제거 후: "슬기로운건강생활보험"  ("간편" 없음 ✅)

    예) "간편 H3종신(무)"
     →  괄호 앞: "간편 H3종신"
     →  수식어 제거 후: "H3종신"  ("간편" 없음 → ref에 간편 없어도 매칭 허용 ✅)
    """
    if not text:
        return text
    s = re.sub(r'^[\(\[\(（][무無][\)\]\)）]\s*', '', text.strip())
    m = re.search(r'[\(\[（\[]', s)
    if m:
        s = s[:m.start()]
    # 수식어 제거 후 반환 (ref clean_no_excl과 동일 기준)
    return remove_modifiers(clean_text(s))

# ──────────────────────────────────────────────
# STRICT 키워드 필터
# ──────────────────────────────────────────────
def filter_by_strict_keywords(target_original, ref_products):
    """
    STRICT_KEYWORDS 필터.

    핵심 원칙:
      키워드 체크는 **괄호 앞 메인 상품명** 기준으로만 수행.
      "(무)슬기로운 건강생활보험(22.01)(1종경증간편고지Ⅰ형)" 에서
      → 메인명 = "슬기로운 건강생활보험"
      → "간편"은 괄호 안에만 있으므로 키워드로 인식하지 않음 ✅

    규칙:
      - 메인명에 없는 키워드가 ref에 있으면 절대 제외 (hard)
      - 메인명에 있는 키워드가 ref에 없으면 제외 (soft — 후보 없으면 완화)
    """
    target_main = extract_main_name(target_original)

    def not_has_absent_kw(ref_raw):
        for kw in STRICT_KEYWORDS:
            if kw not in target_main and kw in ref_raw:
                return False
        return True

    hard_filtered = [p for p in ref_products if not_has_absent_kw(p['clean_no_excl'])]
    if not hard_filtered:
        return ref_products  # 엣지케이스

    def has_present_kw(ref_raw):
        for kw in STRICT_KEYWORDS:
            if kw in target_main and kw not in ref_raw:
                return False
        return True

    soft_filtered = [p for p in hard_filtered if has_present_kw(p['clean_no_excl'])]
    return soft_filtered if soft_filtered else hard_filtered


# ──────────────────────────────────────────────
# 후보 탐색
# ──────────────────────────────────────────────
def find_best_in_products(target_clean, ref_products, idf):
    """TF-IDF+rapidfuzz 점수 기준 최고점 상품 반환. 실제 longest substring도 함께 계산."""
    target_tokens = tokenize(keep_text_only(target_clean))
    t_text_only   = keep_text_only(target_clean)

    best_match, best_score = None, -1
    best_prefix, best_core = 0, 0
    best_lcs = 0          # 실제 최장 연속일치 길이
    matched_text = ''

    for ref in ref_products:
        score, prefix, core_sub = calc_match_score(
            target_clean, ref['clean_no_excl'],
            target_tokens, ref['tokens'], idf
        )
        if score > best_score:
            best_score   = score
            best_prefix  = prefix
            best_core    = core_sub
            best_match   = ref
            matched_text = ref['clean_no_excl']
            _, best_lcs  = find_longest_substring(t_text_only,
                                                   keep_text_only(ref['clean_no_excl']))

    return best_match, best_score, best_prefix, best_core, best_lcs, matched_text


def best_lcs_across_versions(target_clean, ref, exclusion_set):
    """
    한 ref 상품에 대해 가능한 모든 텍스트 버전 조합으로 longest substring 계산.
    타겟: 원본 / 제외문구 제거
    ref : 원본(clean_no_excl) / 제외문구 제거(clean_with_excl)
    → 4가지 조합 중 가장 긴 연속일치 반환
    """
    target_no_excl = remove_exclusion_terms(target_clean, exclusion_set)
    versions = [
        (keep_text_only(target_clean),    keep_text_only(ref['clean_no_excl'])),
        (keep_text_only(target_clean),    keep_text_only(ref['clean_with_excl'])),
        (keep_text_only(target_no_excl),  keep_text_only(ref['clean_no_excl'])),
        (keep_text_only(target_no_excl),  keep_text_only(ref['clean_with_excl'])),
    ]
    best = 0
    for t, r in versions:
        _, lcs = find_longest_substring(t, r)
        if lcs > best:
            best = lcs
    return best


# ──────────────────────────────────────────────
# 메인 매칭 로직
# ──────────────────────────────────────────────
def match_product(target_original, target_clean, ref_products,
                  target_is_real_expense, exclusion_set, idf):
    if not ref_products:
        return None, None, 0, 0, '', '', 'No Products'

    # 실손 필터
    if target_is_real_expense:
        filtered = [p for p in ref_products if p['is_real_expense']]
        if filtered:
            ref_products = filtered

    # STRICT 키워드 필터
    candidates = filter_by_strict_keywords(target_original, ref_products)

    # 1차: 원본(exclusion 포함) 텍스트로 매칭
    best_match, best_score, best_prefix, best_core, best_lcs, matched_text = \
        find_best_in_products(target_clean, candidates, idf)

    # 2차: 약한 매칭(TF-IDF < 0.15) → 제외문구 제거 버전으로 재시도
    tfidf_sim_best = tfidf_similarity(
        tokenize(keep_text_only(target_clean)),
        best_match['tokens'] if best_match else set(),
        idf
    )

    if tfidf_sim_best < 0.15:
        target_no_excl   = remove_exclusion_terms(target_clean, exclusion_set)
        target_tokens_ne = tokenize(keep_text_only(target_no_excl))
        t_text_ne        = keep_text_only(target_no_excl)

        best2_match, best2_score = None, -1
        best2_prefix, best2_core, best2_lcs, best2_text = 0, 0, 0, ''

        for ref in candidates:
            score2, prefix2, core2 = calc_match_score(
                target_no_excl, ref['clean_with_excl'],
                target_tokens_ne, ref['tokens_no_excl'], idf
            )
            if score2 > best2_score:
                best2_score  = score2
                best2_prefix = prefix2
                best2_core   = core2
                best2_match  = ref
                best2_text   = ref['clean_with_excl']
                _, best2_lcs = find_longest_substring(
                    t_text_ne, keep_text_only(ref['clean_with_excl']))

        if best2_match and best2_score > best_score:
            best_match   = best2_match
            best_score   = best2_score
            best_prefix  = best2_prefix
            best_core    = best2_core
            best_lcs     = best2_lcs
            matched_text = best2_text

    if not best_match:
        return None, None, 0, 0, '', '', 'No Match'

    # 3차: 여전히 연속일치 0글자면 모든 버전 조합으로 최선 탐색
    if best_lcs == 0:
        # 후보 전체를 대상으로 모든 텍스트 버전 조합에서 lcs 가장 긴 상품 선택
        fallback_best_match = best_match
        fallback_best_lcs   = 0
        fallback_text       = matched_text

        for ref in candidates:
            lcs = best_lcs_across_versions(target_clean, ref, exclusion_set)
            if lcs > fallback_best_lcs:
                fallback_best_lcs   = lcs
                fallback_best_match = ref
                # matched_text는 가장 잘 맞는 버전으로
                t_ne = keep_text_only(remove_exclusion_terms(target_clean, exclusion_set))
                versions_text = [
                    (keep_text_only(target_clean),  ref['clean_no_excl']),
                    (keep_text_only(target_clean),  ref['clean_with_excl']),
                    (t_ne,                          ref['clean_no_excl']),
                    (t_ne,                          ref['clean_with_excl']),
                ]
                best_v_lcs, best_v_text = 0, ref['clean_no_excl']
                for t, r in versions_text:
                    _, lcs_v = find_longest_substring(t, keep_text_only(r))
                    if lcs_v > best_v_lcs:
                        best_v_lcs  = lcs_v
                        best_v_text = r
                fallback_text = best_v_text

        best_match   = fallback_best_match
        best_lcs     = fallback_best_lcs
        matched_text = fallback_text

    clean_used = (f'score={best_score:.2f} lcs={best_lcs} '
                  f'prefix={best_prefix} core={best_core}')
    t_tok  = set(target_clean)
    r_tok  = set(best_match['clean_no_excl'])
    overlap = len(t_tok & r_tok)

    return (best_match['prod_code'], best_match['prod_name'],
            overlap, best_lcs,
            matched_text, clean_used, best_match['sales_status'])


# ──────────────────────────────────────────────
# 인덱스 빌드
# ──────────────────────────────────────────────
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
        is_real    = ('실손' in prod_name or
                      '실손' in str(row.get('IIMS 상품명', '')))

        if not insurer or not prod_name:
            continue

        no_excl   = preprocess_text(prod_name, insurer, exclusion_set, keep_exclusions=True)
        with_excl = preprocess_text(prod_name, insurer, exclusion_set, keep_exclusions=False)

        ref_dict[insurer].append({
            'prod_name':       prod_name,
            'prod_code':       prod_code,
            'insurer_code':    ins_code,
            'clean_no_excl':   no_excl,
            'clean_with_excl': with_excl,
            # 토큰은 인덱스 빌드 시 미리 계산 (속도 최적화)
            'tokens':          tokenize(keep_text_only(no_excl)),
            'tokens_no_excl':  tokenize(keep_text_only(with_excl)),
            'sales_status':    sales_stat,
            'is_real_expense': is_real,
        })

    # 판매중 우선 → 상품코드 숫자 높은 순
    for insurer in ref_dict:
        ref_dict[insurer].sort(key=lambda x: (
            0 if x['sales_status'] == '판매중' else 1,
            -int(''.join(filter(str.isdigit, x['prod_code'])) or 0)
        ))

    # 보험사별 TF-IDF 가중치 빌드
    idf_dict = {}
    for insurer, products in ref_dict.items():
        all_texts = [p['clean_no_excl'] for p in products]
        idf_dict[insurer] = build_tfidf_weights(all_texts)

    # 정규화 키 → 원본 보험사명 매핑 (보이지않는문자·대소문자 차이 허용)
    normalized_map = {normalize_insurer(k): k for k in ref_dict.keys()}

    return dict(ref_dict), idf_dict, normalized_map


def match_insurer(target_insurer, ref_dict, normalized_map):
    """
    보험사 완전 일치 매칭.
    1. 원본 그대로 dict 키 직접 조회 (가장 빠름)
    2. 안되면 정규화(한글·영문·숫자만, 소문자) 기준으로 재조회
       → 보이지 않는 공백·특수문자·대소문자 차이 허용
    """
    if not target_insurer:
        return None, 0
    # 1차: 완전 일치
    if target_insurer in ref_dict:
        return target_insurer, 100
    # 2차: 정규화 일치
    norm = normalize_insurer(target_insurer)
    if norm in normalized_map:
        return normalized_map[norm], 100
    return None, 0


# ──────────────────────────────────────────────
# 행 단위 처리
# ──────────────────────────────────────────────
def process_target_row(row, ref_dict, idf_dict, normalized_map, exclusion_set, target_df):
    idx            = row.name
    target_insurer = clean_text(str(row.iloc[0]))
    target_product = clean_text(str(row.iloc[1]))

    # 보험사상품코드 컬럼이 실제로 존재할 때만 읽음
    # nan / 'nan' / '' / 'None' 은 모두 코드 없음으로 처리
    if '보험사상품코드' in target_df.columns:
        raw_code = row.get('보험사상품코드', '')
        target_insurer_code = '' if pd.isna(raw_code) else str(raw_code).strip()
        if target_insurer_code in ('nan', 'None', '-', 'N/A'):
            target_insurer_code = ''
    else:
        target_insurer_code = ''

    target_is_real = '실손' in target_product
    matched_insurer, ins_score = match_insurer(target_insurer, ref_dict, normalized_map)

    if not matched_insurer:
        return {
            'idx': idx, '상품코드': '', 'Matched_상품명': '', 'Matched_보험사': '',
            'Insurer_Score': 0, 'Overlap_Count': 0, 'Substring_Length': 0,
            'Matched_Text': '', 'Clean_Version': '', 'Match_Method': 'No Insurer Match',
            'Confidence': 'No Data', '판매상태': 'N/A', 'Target_Clean_Text': ''
        }

    ref_products = ref_dict.get(matched_insurer, [])
    idf          = idf_dict.get(matched_insurer, {})

    # 보험사 상품코드 직접 매칭
    # 타겟 코드가 있고(비어있지 않고), ref 코드도 있고, 서로 일치할 때만 적용
    if target_insurer_code:
        for ref in ref_products:
            ref_code = ref['insurer_code'].strip()
            if ref_code and ref_code == target_insurer_code:
                return {
                    'idx': idx,
                    '상품코드': ref['prod_code'],
                    'Matched_상품명': ref['prod_name'],
                    'Matched_보험사': matched_insurer,
                    'Insurer_Score': ins_score,
                    'Overlap_Count': 999,
                    'Substring_Length': 999,
                    'Matched_Text': ref['prod_name'],
                    'Clean_Version': 'Insurer Code',
                    'Match_Method': 'Insurer Code Match',
                    'Confidence': 'High',
                    '판매상태': ref['sales_status'],
                    'Target_Clean_Text': preprocess_text(
                        target_product, target_insurer, exclusion_set, keep_exclusions=True)
                }

    target_clean = preprocess_text(
        target_product, target_insurer, exclusion_set, keep_exclusions=True)

    prod_code, prod_name, overlap, sublen, matched_text, clean_ver, sales_stat = match_product(
        target_product, target_clean, ref_products, target_is_real, exclusion_set, idf
    )

    if prod_code:
        confidence = 'High' if sublen >= 5 else 'Medium' if sublen >= 3 else 'Low'
        return {
            'idx': idx, '상품코드': prod_code, 'Matched_상품명': prod_name,
            'Matched_보험사': matched_insurer, 'Insurer_Score': ins_score,
            'Overlap_Count': overlap, 'Substring_Length': sublen,
            'Matched_Text': matched_text, 'Clean_Version': clean_ver,
            'Match_Method': 'Text Match', 'Confidence': confidence,
            '판매상태': sales_stat, 'Target_Clean_Text': target_clean
        }

    return {
        'idx': idx, '상품코드': '', 'Matched_상품명': '', 'Matched_보험사': matched_insurer,
        'Insurer_Score': ins_score, 'Overlap_Count': 0, 'Substring_Length': 0,
        'Matched_Text': '', 'Clean_Version': '', 'Match_Method': 'No Match',
        'Confidence': 'No Data', '판매상태': 'N/A', 'Target_Clean_Text': target_clean
    }


# ──────────────────────────────────────────────
# 템플릿 생성
# ──────────────────────────────────────────────
def create_target_template():
    """타겟 파일 양식 — 필수 컬럼 3개"""
    df = pd.DataFrame({
        '보험사': [
            'DB손보', 'KB손보', '한화생명', '삼성화재', '메리츠화재',
            '흥국화재', '현대해상', '농협생명', '신한라이프', '교보생명',
        ],
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
        '보험사상품코드': [
            '3059002', '24482', '', 'L1716', '',
            '', 'HH2023', '', 'SL001', 'KB001',
        ],
    })
    return df

def create_reference_template():
    """레퍼런스 파일 양식 — 필수 컬럼 구조"""
    df = pd.DataFrame({
        '보험사': [
            'DB손보', 'DB손보', 'KB손보', '한화생명', '한화생명',
            '삼성화재', '메리츠화재', '흥국화재', '현대해상', '농협생명',
            '신한라이프', '교보생명',
        ],
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
        '상품코드': [
            '11001', '11002', '11003', '11004', '11005',
            '11006', '11007', '11008', '11009', '11010',
            '11011', '11012',
        ],
        '보험사상품코드': [
            '3059002', '', '24482', '', '',
            'L1716', '', '', 'HH2023', '',
            'SL001', 'KB001',
        ],
        '판매상태': [
            '판매중', '판매중지', '판매중', '판매중', '판매중지',
            '판매중', '판매중', '판매중', '판매중', '판매중',
            '판매중', '판매중',
        ],
        'IIMS 상품명': [''] * 12,
    })
    return df

def to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()


# ──────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────
def main():
    st.title("🎯 Ultra-Fast Product Code Matcher")
    st.success("✅ TF-IDF + rapidfuzz 하이브리드 | ⚡ 고유명 우선 매칭")

    # ── 템플릿 다운로드 ──────────────────────────
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
            st.download_button(
                "⬇️ Target 양식 다운로드",
                data=to_excel_bytes(create_target_template()),
                file_name="target_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            st.dataframe(create_target_template(), use_container_width=True)
        with tc2:
            st.download_button(
                "⬇️ Reference 양식 다운로드",
                data=to_excel_bytes(create_reference_template()),
                file_name="reference_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            st.dataframe(create_reference_template(), use_container_width=True)
    st.divider()

    with st.sidebar:
        st.header("🚫 제외문구")
        exclusion_input = st.text_area(
            "쉼표로 구분",
            value=','.join(DEFAULT_EXCLUSIONS),
            height=150
        )
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
            "3. STRICT 키워드 필터\n"
            "   실손·태아·갱신·간편·암·연금·저축·재물·치매·정기\n"
            "   타겟 기준 엄격 적용 (한쪽에만 있으면 제외)\n"
            "4. **TF-IDF × 40** (희귀 고유명 토큰 우선)\n"
            "   + token_set_ratio × 0.3\n"
            "   + partial_ratio × 0.2\n"
            "   + prefix × 1.5\n"
            "   + core_sub × 2\n"
            "5. 약한 매칭(TF-IDF<0.15) → 제외문구 제거 재시도\n"
            "6. 판매중 우선 / 상품코드 높은 순"
        )

    st.error("⚠️ 타겟과 레퍼런스의 **보험사명이 정확히 일치**해야 합니다. 다르면 매칭되지 않습니다.")
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

        results = []
        progress_bar = st.progress(0)
        total_rows = len(df_target)

        for idx, row in df_target.iterrows():
            result = process_target_row(
                row, ref_dict, idf_dict, normalized_map, set(exclusion_terms), df_target)
            results.append(result)
            if (idx + 1) % 100 == 0:
                progress_bar.progress((idx + 1) / total_rows)
        progress_bar.progress(1.0)

        df_result = pd.DataFrame(results)
        df_output = df_target.copy()
        for col in ['상품코드', 'Matched_상품명', 'Matched_보험사', 'Match_Method',
                    'Overlap_Count', 'Substring_Length', 'Confidence', '판매상태',
                    'Matched_Text', 'Clean_Version', 'Target_Clean_Text']:
            df_output[col] = df_result[col]

        elapsed  = time.time() - start_time
        total    = len(df_output)
        matched  = df_output['상품코드'].astype(str).str.strip().ne('').sum()
        code_m   = (df_output['Match_Method'] == 'Insurer Code Match').sum()
        text_m   = (df_output['Match_Method'] == 'Text Match').sum()
        active   = (df_output['판매상태'] == '판매중').sum()
        disc     = (df_output['판매상태'] == '판매중지').sum()

        st.success(f"✅ {elapsed:.1f}s 완료 | {matched}/{total} 매칭 ({matched/total*100:.1f}%)")

        # 미매칭 보험사 목록 표시
        no_insurer = df_output[df_output['Match_Method'] == 'No Insurer Match']
        if len(no_insurer) > 0:
            unmatched_insurers = no_insurer['보험사'].unique().tolist()
            st.error(
                f"❌ 보험사명 불일치로 미매칭: {len(no_insurer)}건\n\n"
                f"해당 보험사: `{'`, `'.join(unmatched_insurers)}`\n\n"
                "레퍼런스 파일의 보험사명과 정확히 일치하는지 확인해주세요."
            )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("⏱️ 시간",       f"{elapsed:.1f}s")
        c2.metric("🔑 코드매칭",   f"{code_m}")
        c3.metric("📝 텍스트매칭", f"{text_m}")
        c4.metric("🟢 판매중",     f"{active}")
        c5.metric("🔴 판매중지",   f"{disc}")

        st.subheader("📋 결과")
        display_cols = ['보험사', '상품명', 'Target_Clean_Text', 'Matched_상품명',
                        '상품코드', '판매상태', 'Match_Method',
                        'Substring_Length', 'Confidence', 'Clean_Version']
        st.dataframe(df_output[display_cols].head(50))

        st.subheader("🔍 상세 비교 (상위 20건)")
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
                    if row['상품코드']:
                        emoji = "🟢" if row['판매상태'] == '판매중' else "🔴"
                        st.write(f"**{emoji} 매칭 결과:**")
                        st.success(f"**코드:** `{row['상품코드']}`\n\n"
                                   f"**상품명:** {row['Matched_상품명']}\n\n"
                                   f"**매칭텍스트:** `{row['Matched_Text']}`\n\n"
                                   f"**상태:** {row['판매상태']}")
                        st.caption(row['Clean_Version'])
                    else:
                        st.error("**❌ 미매칭**")

        excel_data = io.BytesIO()
        with pd.ExcelWriter(excel_data, engine='xlsxwriter') as writer:
            df_output.to_excel(writer, index=False, sheet_name='Results')

        st.download_button(
            "📥 결과 다운로드",
            excel_data.getvalue(),
            f"matched_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )


if __name__ == "__main__":
    main()
