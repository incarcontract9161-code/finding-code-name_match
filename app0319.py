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

DEFAULT_EXCLUSIONS = ['프로미라이프', '통합', '보장', '건강', '보험']

# STRICT_KEYWORDS : 타겟에 있으면 ref에도 있어야, 타겟에 없으면 ref에도 없어야
# ※ COMMON_MODIFIERS와 겹치지 않도록 완전히 분리
STRICT_KEYWORDS = frozenset([
    '실손', '태아', '갱신', '간편', '암', '연금', '저축', '재물',
    '치매', '정기', '간병', '경영','변액'
])

# COMMON_MODIFIERS : 수식어 제거용 — STRICT_KEYWORDS 항목은 절대 포함하지 않음
# 긴 것이 앞에 있어야 "간편가입" 제거 후 "간편"이 남는 문제를 방지
COMMON_MODIFIERS = [
    '심사', '가입',          # "간편" 포함 복합어 — 단독 "간편"보다 먼저 처리
    '갱신형', '비갱신형',
    '연만기', '연납', '월납', '일시납',
    '만기', '종신', '특약', '기본형', '표준형',
    '실속형', '플러스형', '플러스',
    '온라인', '다이렉트', '무진단', '무심사',
    '슬림', '베이직', '스마트', '프리미엄',
    '골드', '실버', '블루', '그린',
]
# 정렬은 build 시 한 번만 수행
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

def remove_brackets_prefix(text):
    """맨 앞 (무)/[무] 제거 + 전체 괄호 내용 정리"""
    if not text:
        return ''
    for term in ['무배당', '(무)', '[무]', '(無)', '배당', '()', '(', ')', '[', ']']:
        text = text.replace(term, '')
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

def extract_main_name(original_text):
    """
    STRICT 키워드 체크용 메인 상품명 추출.
    괄호 앞까지만 + COMMON_MODIFIERS 제거 + STRICT_KEYWORDS 보존.

    예) "(무)슬기로운건강생활보험(22.01)(1종경증간편고지Ⅰ형)" → "슬기로운건강생활보험"
    예) "간편H3종신(무)"   → "간편H3종신"  ("간편" 보존 ✅)
    예) "플러스정기보험"   → "정기보험"    ("정기" 보존 ✅)
    예) "간편가입H3종신"   → "H3종신"      ("간편가입" 제거, "간편" 단독 없음 ✅)
    """
    if not original_text:
        return ''
    s = re.sub(r'^[\(\[\(（][무無][\)\]\)）]\s*', '', original_text.strip())
    m = re.search(r'[\(\[（\[]', s)
    if m:
        s = s[:m.start()]
    return remove_modifiers(clean_text(s))


# ══════════════════════════════════════════════════════════
# STRICT 키워드 필터
# ══════════════════════════════════════════════════════════

def filter_by_strict_keywords(target_original, ref_products):
    """
    Hard rule  : 타겟 메인명에 없는 키워드가 ref에 있으면 무조건 제외 (완화 없음)
    Soft rule  : 타겟 메인명에 있는 키워드가 ref에 없으면 제외 (후보 없을 때만 완화)
    """
    target_main = extract_main_name(target_original)

    def passes_hard(ref_raw):
        for kw in STRICT_KEYWORDS:
            if kw not in target_main and kw in ref_raw:
                return False
        return True

    hard = [p for p in ref_products if passes_hard(p['clean_no_excl'])]
    if not hard:
        return ref_products  # 보험사 전체가 특정 상품군뿐인 엣지케이스

    def passes_soft(ref_raw):
        for kw in STRICT_KEYWORDS:
            if kw in target_main and kw not in ref_raw:
                return False
        return True

    soft = [p for p in hard if passes_soft(p['clean_no_excl'])]
    return soft if soft else hard


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

def calc_match_score(target_clean, ref_clean, t_tokens, r_tokens, idf):
    """
    종합 매칭 점수.

    A. TF-IDF 코사인 유사도 × 40  — 희귀 고유명 토큰 우선
    B. token_set_ratio      × 0.30 — 어순 무관 유사도
    C. partial_ratio        × 0.20 — 부분 포함(축약형)
    D. prefix_match         × 1.50 — 앞부분 일치
    E. core_sub             × 2.00 — 수식어 제거 후 고유명 연속일치
    """
    if not target_clean or not ref_clean:
        return 0.0, 0, 0

    tfidf  = tfidf_similarity(t_tokens, r_tokens, idf)
    t_text = keep_text_only(target_clean)
    r_text = keep_text_only(ref_clean)
    t_set  = fuzz.token_set_ratio(t_text, r_text)
    part   = fuzz.partial_ratio(t_text, r_text)
    prefix = count_prefix_match(t_text, r_text)
    _, core_sub = find_longest_substring(t_text, r_text)

    score = tfidf * 40 + t_set * 0.30 + part * 0.20 + prefix * 1.5 + core_sub * 2.0
    return score, prefix, core_sub


# ══════════════════════════════════════════════════════════
# 후보 탐색
# ══════════════════════════════════════════════════════════

def find_best_in_products(target_clean, ref_products, idf):
    t_tokens = tokenize(keep_text_only(target_clean))
    t_text   = keep_text_only(target_clean)

    best_match, best_score = None, -1
    best_prefix = best_core = best_lcs = 0
    matched_text = ''

    for ref in ref_products:
        score, prefix, core = calc_match_score(
            target_clean, ref['clean_no_excl'], t_tokens, ref['tokens'], idf)
        if score > best_score:
            best_score   = score
            best_prefix  = prefix
            best_core    = core
            best_match   = ref
            matched_text = ref['clean_no_excl']
            _, best_lcs  = find_longest_substring(t_text, keep_text_only(ref['clean_no_excl']))

    return best_match, best_score, best_prefix, best_core, best_lcs, matched_text

def best_lcs_all_versions(target_clean, ref, exclusion_set):
    """4가지 버전 조합 중 가장 긴 LCS 반환"""
    t_ne = remove_exclusion_terms(target_clean, exclusion_set)
    versions = [
        (keep_text_only(target_clean), keep_text_only(ref['clean_no_excl'])),
        (keep_text_only(target_clean), keep_text_only(ref['clean_with_excl'])),
        (keep_text_only(t_ne),         keep_text_only(ref['clean_no_excl'])),
        (keep_text_only(t_ne),         keep_text_only(ref['clean_with_excl'])),
    ]
    best = 0
    for t, r in versions:
        _, lcs = find_longest_substring(t, r)
        best = max(best, lcs)
    return best


# ══════════════════════════════════════════════════════════
# 메인 매칭 로직
# ══════════════════════════════════════════════════════════

def match_product(target_original, target_clean, ref_products,
                  target_is_real, exclusion_set, idf):
    if not ref_products:
        return None, None, 0, 0, '', '', 'No Products'

    # 1. 실손 필터
    if target_is_real:
        filtered = [p for p in ref_products if p['is_real_expense']]
        if filtered:
            ref_products = filtered

    # 2. STRICT 키워드 필터
    candidates = filter_by_strict_keywords(target_original, ref_products)

    # 3. 1차 매칭 (원본 기준)
    best, best_score, best_prefix, best_core, best_lcs, matched_text = \
        find_best_in_products(target_clean, candidates, idf)

    # 4. TF-IDF 약한 매칭 → 제외문구 제거 버전으로 재시도
    t_tok = tokenize(keep_text_only(target_clean))
    tfidf_best = tfidf_similarity(t_tok, best['tokens'] if best else set(), idf)

    if tfidf_best < 0.15:
        t_ne    = remove_exclusion_terms(target_clean, exclusion_set)
        t_ne_tok = tokenize(keep_text_only(t_ne))
        t_ne_txt = keep_text_only(t_ne)

        b2_match, b2_score = None, -1
        b2_prefix = b2_core = b2_lcs = 0
        b2_text = ''

        for ref in candidates:
            score2, p2, c2 = calc_match_score(
                t_ne, ref['clean_with_excl'], t_ne_tok, ref['tokens_no_excl'], idf)
            if score2 > b2_score:
                b2_score  = score2
                b2_prefix = p2
                b2_core   = c2
                b2_match  = ref
                b2_text   = ref['clean_with_excl']
                _, b2_lcs = find_longest_substring(t_ne_txt, keep_text_only(ref['clean_with_excl']))

        if b2_match and b2_score > best_score:
            best, best_score = b2_match, b2_score
            best_prefix, best_core, best_lcs = b2_prefix, b2_core, b2_lcs
            matched_text = b2_text

    if not best:
        return None, None, 0, 0, '', '', 'No Match'

    # 5. LCS = 0 → 모든 버전 조합으로 최선 탐색
    if best_lcs == 0:
        fb_match, fb_lcs, fb_text = best, 0, matched_text
        for ref in candidates:
            lcs = best_lcs_all_versions(target_clean, ref, exclusion_set)
            if lcs > fb_lcs:
                fb_lcs   = lcs
                fb_match = ref
                # 가장 잘 맞는 버전 텍스트 선택
                t_ne = keep_text_only(remove_exclusion_terms(target_clean, exclusion_set))
                best_v, best_v_lcs = ref['clean_no_excl'], 0
                for tv, rv in [
                    (keep_text_only(target_clean), ref['clean_no_excl']),
                    (keep_text_only(target_clean), ref['clean_with_excl']),
                    (t_ne, ref['clean_no_excl']),
                    (t_ne, ref['clean_with_excl']),
                ]:
                    _, v_lcs = find_longest_substring(tv, keep_text_only(rv))
                    if v_lcs > best_v_lcs:
                        best_v_lcs = v_lcs
                        best_v     = rv
                fb_text = best_v
        best, best_lcs, matched_text = fb_match, fb_lcs, fb_text

    t_tok2 = set(target_clean)
    r_tok2 = set(best['clean_no_excl'])
    overlap = len(t_tok2 & r_tok2)
    clean_used = f'score={best_score:.2f} lcs={best_lcs} prefix={best_prefix} core={best_core}'

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

        ref_dict[insurer].append({
            'prod_name':        prod_name,
            'prod_code':        prod_code,
            'insurer_code':     ins_code,
            'clean_no_excl':    no_excl,
            'clean_with_excl':  with_excl,
            'tokens':           tokenize(keep_text_only(no_excl)),
            'tokens_no_excl':   tokenize(keep_text_only(with_excl)),
            'sales_status':     sales_stat,
            'is_real_expense':  is_real,
        })

    for insurer in ref_dict:
        ref_dict[insurer].sort(key=lambda x: (
            0 if x['sales_status'] == '판매중' else 1,
            -int(''.join(filter(str.isdigit, x['prod_code'])) or 0)
        ))

    idf_dict = {ins: build_tfidf_weights([p['clean_no_excl'] for p in prods])
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
            'Confidence': 'No Data', '판매상태': 'N/A', 'Target_Clean_Text': ''
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
                        target_product, target_insurer, exclusion_set, keep_exclusions=True)
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
            '판매상태': sales_stat, 'Target_Clean_Text': target_clean
        }

    return {
        'idx': idx, '상품코드': '', 'Matched_상품명': '', 'Matched_보험사': matched_ins,
        'Insurer_Score': ins_score, 'Overlap_Count': 0, 'Substring_Length': 0,
        'Matched_Text': '', 'Clean_Version': '', 'Match_Method': 'No Match',
        'Confidence': 'No Data', '판매상태': 'N/A', 'Target_Clean_Text': target_clean
    }


# ══════════════════════════════════════════════════════════
# 템플릿
# ══════════════════════════════════════════════════════════

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
            "3. STRICT 키워드 필터\n"
            "   실손·태아·갱신·간편·암·연금·저축·재물·치매·정기·간병\n"
            "   (타겟 메인명 기준 — 괄호 안 부가설명 제외)\n"
            "4. TF-IDF×40 + token_set_ratio×0.3\n"
            "   + partial_ratio×0.2 + prefix×1.5 + core_sub×2\n"
            "5. TF-IDF<0.15 → 제외문구 제거 재시도\n"
            "6. LCS=0 → 전체 버전 조합 재시도\n"
            "7. 판매중 우선 / 상품코드 높은 순"
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

        df_result = pd.DataFrame(results)
        df_output = df_target.copy()
        for col in ['상품코드','Matched_상품명','Matched_보험사','Match_Method',
                    'Overlap_Count','Substring_Length','Confidence','판매상태',
                    'Matched_Text','Clean_Version','Target_Clean_Text']:
            df_output[col] = df_result[col]

        elapsed  = time.time() - start_time
        total    = len(df_output)
        matched  = df_output['상품코드'].astype(str).str.strip().ne('').sum()
        code_m   = (df_output['Match_Method'] == 'Insurer Code Match').sum()
        text_m   = (df_output['Match_Method'] == 'Text Match').sum()
        active   = (df_output['판매상태'] == '판매중').sum()
        disc     = (df_output['판매상태'] == '판매중지').sum()

        st.success(f"✅ {elapsed:.1f}s 완료 | {matched}/{total} 매칭 ({matched/total*100:.1f}%)")

        no_ins = df_output[df_output['Match_Method'] == 'No Insurer Match']
        if len(no_ins) > 0:
            unmatched = no_ins['보험사'].unique().tolist()
            st.error(
                f"❌ 보험사명 불일치 미매칭: {len(no_ins)}건\n\n"
                f"해당 보험사: `{'`, `'.join(unmatched)}`\n\n"
                "레퍼런스 파일의 보험사명과 정확히 일치하는지 확인해주세요."
            )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("⏱️ 시간",       f"{elapsed:.1f}s")
        c2.metric("🔑 코드매칭",   f"{code_m}")
        c3.metric("📝 텍스트매칭", f"{text_m}")
        c4.metric("🟢 판매중",     f"{active}")
        c5.metric("🔴 판매중지",   f"{disc}")

        st.subheader("📋 결과")
        st.dataframe(df_output[['보험사','상품명','Target_Clean_Text','Matched_상품명',
                                 '상품코드','판매상태','Match_Method',
                                 'Substring_Length','Confidence','Clean_Version']].head(50))

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
            "📥 결과 다운로드", excel_data.getvalue(),
            f"matched_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )


if __name__ == "__main__":
    main()
