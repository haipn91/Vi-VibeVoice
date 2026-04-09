"""
Chuan hoa phan so / ngay thang noi bang chu (tieng Viet) — phuong an A: regex + tu dien.
Dung sau ASR: thang (mot … muoi hai) -> so; nam + bon tu so doc tung chu so -> YYYY;
nam + 'hai nghin hai muoi [moc…]' -> 2020–2029.
So dem + tu lien sau (vi du 'sau chuong trinh', 'ba phan mem') -> chu so; co denylist de tranh sai.
Cac cum doc liên tiep ≥2 tu (tu tap chu so/doc CCCD, dien thoai) khong tach thanh so dem.

Tat bang bien moi truong: GIPFORMER_SPOKEN_ITN=0
"""
from __future__ import annotations

import re
import unicodedata

# Thu tu dai truoc de tranh 'muoi' an 'muoi mot'
_MONTH_PAIRS: list[tuple[str, str]] = [
    ("mười hai", "12"),
    ("mười một", "11"),
    ("mười", "10"),
    ("chín", "9"),
    ("tám", "8"),
    ("bảy", "7"),
    ("bậy", "7"),
    ("sáu", "6"),
    ("năm", "5"),
    ("tư", "4"),
    ("bốn", "4"),
    ("ba", "3"),
    ("hai", "2"),
    ("một", "1"),
]

_DIGIT_WORDS: dict[str, str] = {
    "không": "0",
    "linh": "0",
    "o": "0",
    "một": "1",
    "hai": "2",
    "ba": "3",
    "bốn": "4",
    "tư": "4",
    "năm": "5",
    "sáu": "6",
    "bảy": "7",
    "bậy": "7",
    "tám": "8",
    "chín": "9",
}

# Chu so trong regex: thu tu dai truoc (khong, linh, mot, ...)
_digit_alt = "|".join(
    sorted(_DIGIT_WORDS.keys(), key=lambda s: (-len(s), s))
)

# nghìn / nghin (khong dau)
_NGHIN = r"ngh[ìi]n"

# Duoi 'hai muoi' cho 2021–2029
_202X_TAIL_MAP: dict[str, int] = {
    "mốt": 1,
    "hai": 2,
    "ba": 3,
    "bốn": 4,
    "lăm": 5,
    "sáu": 6,
    "bảy": 7,
    "tám": 8,
    "chín": 9,
}
_202X_TAIL_ALT = "|".join(sorted(_202X_TAIL_MAP.keys(), key=lambda s: (-len(s), s)))

_RE_COMPOUND_202X = re.compile(
    rf"(?iu)\b(năm)\s+hai\s+{_NGHIN}\s+hai\s+mươi(?:\s+({_202X_TAIL_ALT}))?\b",
    re.UNICODE,
)

_RE_YEAR_4DIGIT_SPOKEN = re.compile(
    rf"(?iu)\b(năm)\s+(({_digit_alt})(?:\s+({_digit_alt})){{3}})\b",
    re.UNICODE,
)

# So dem (1–19 + mười) + mot tu tiep theo: dai truoc de tranh 'muoi mot' / 'muoi hai'.
_CARDINAL_QUANTIFIERS: list[tuple[str, str]] = [
    ("mười chín", "19"),
    ("mười tám", "18"),
    ("mười bảy", "17"),
    ("mười bậy", "17"),
    ("mười sáu", "16"),
    ("mười lăm", "15"),
    ("mười năm", "15"),
    ("mười bốn", "14"),
    ("mười tư", "14"),
    ("mười ba", "13"),
    ("mười hai", "12"),
    ("mười một", "11"),
    ("mười", "10"),
    ("chín", "9"),
    ("tám", "8"),
    ("bảy", "7"),
    ("bậy", "7"),
    ("sáu", "6"),
    ("năm", "5"),
    ("tư", "4"),
    ("bốn", "4"),
    ("ba", "3"),
    ("hai", "2"),
    ("một", "1"),
]
_CARDINAL_ALT = "|".join(re.escape(w) for w, _ in _CARDINAL_QUANTIFIERS)
_CARDINAL_TO_NUM: dict[str, str] = {w: n for w, n in _CARDINAL_QUANTIFIERS}

# 'nam' = 5 nhung 'nam hoc', 'nam nay' la thoi gian.
_NAM_NOT_QUANTIFIER_FOLLOWING: frozenset[str] = frozenset(
    {
        "học",
        "nay",
        "ngoái",
        "sau",
        "trước",
        "nào",
        "ấy",
        "kia",
        "gì",
        "tháng",
        "quý",
        "vừa",
        "qua",
        "leo",
    }
)

# Giu 'hai tram', 'ba nghin' (doc so phuc tap).
_LARGE_NUMBER_CONTINUATION: frozenset[str] = frozenset(
    {"trăm", "nghìn", "ngàn", "triệu", "tỷ", "ty"}
)

_RE_CARDINAL_QUANTIFIER = re.compile(
    rf"(?iu)\b({_CARDINAL_ALT})\s+(\w+)",
    re.UNICODE,
)

# Tu chi co the la chu so doc roi (dung nhan dien cum CCCD/dien thoai/dia chi).
_SPOKEN_SINGLE_DIGIT_WORDS: frozenset[str] = frozenset(_DIGIT_WORDS.keys())

_RE_TOKEN_GAP_OK = re.compile(r"^[\s,;.]+$")


def _strip_trailing_punct(token: str) -> str:
    return re.sub(r"[.,;:!?]+$", "", token)


def _spoken_digit_spans(text: str, min_tokens: int = 2) -> list[tuple[int, int]]:
    """
    Tim cac doan [start, end) trong text ma gom >= min_tokens tu lien tiep,
    moi tu (sau khi cat dau cau) thuoc _SPOKEN_SINGLE_DIGIT_WORDS, chi ngan cach
    boi khoang trang hoac dau phay/cham ngan.
    """
    if min_tokens < 2:
        min_tokens = 2
    words: list[tuple[int, int, str]] = []
    for m in re.finditer(r"\S+", text):
        core = _strip_trailing_punct(m.group(0))
        mm = re.match(r"(?iu)([\wÀ-ỹ]+)", core)
        if not mm:
            continue
        w = _fold_vi_word(mm.group(1))
        if w in _SPOKEN_SINGLE_DIGIT_WORDS:
            words.append((m.start(), m.end(), w))

    if not words:
        return []

    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(words):
        j = i
        while j + 1 < len(words):
            gap = text[words[j][1] : words[j + 1][0]]
            if not _RE_TOKEN_GAP_OK.fullmatch(gap):
                break
            j += 1
        ntok = j - i + 1
        if ntok >= min_tokens:
            spans.append((words[i][0], words[j][1]))
        i = j + 1
    return spans


def _span_covers_any(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    for rs, rend in spans:
        if start < rend and end > rs:
            return True
    return False


def _last_word_before_pos(text: str, pos: int) -> str | None:
    before = text[:pos].rstrip()
    toks = list(re.finditer(r"\S+", before))
    if not toks:
        return None
    return _strip_trailing_punct(toks[-1].group(0))


def _is_spoken_digit_word_token(surface: str) -> bool:
    mm = re.match(r"(?iu)([\wÀ-ỹ]+)", surface.strip())
    if not mm:
        return False
    return _fold_vi_word(mm.group(1)) in _SPOKEN_SINGLE_DIGIT_WORDS


def _fold_vi_word(w: str) -> str:
    w = unicodedata.normalize("NFC", w.strip().lower())
    return w


def apply_vi_spoken_number_rules(text: str) -> str:
    if not text or not text.strip():
        return text
    # Thang truoc 'nam + 4 chu so' de tranh 'thang nam' (thang 5) bi tom nhu tu khoa nam.
    text = _replace_compound_years_202x(text)
    text = _replace_months_spoken(text)
    text = _replace_years_spoken_4digits(text)
    text = _replace_spoken_cardinal_quantifiers(text)
    return text


def _replace_compound_years_202x(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        prev = _last_word_before_pos(m.string, m.start())
        if prev is not None and _is_spoken_digit_word_token(prev):
            return m.group(0)
        prefix = m.group(1)
        tail = m.group(2)
        if tail is None:
            year = 2020
        else:
            key = _fold_vi_word(tail)
            off = _202X_TAIL_MAP.get(key)
            if off is None:
                return m.group(0)
            year = 2020 + off
        return f"{prefix} {year}"

    return _RE_COMPOUND_202X.sub(repl, text)


def _replace_years_spoken_4digits(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        prev = _last_word_before_pos(m.string, m.start())
        if prev is not None and _is_spoken_digit_word_token(prev):
            return m.group(0)
        prefix = m.group(1)
        body = m.group(2)
        parts = [_fold_vi_word(p) for p in body.split() if p.strip()]
        if len(parts) != 4:
            return m.group(0)
        digits: list[str] = []
        for p in parts:
            d = _DIGIT_WORDS.get(p)
            if d is None:
                return m.group(0)
            digits.append(d)
        return f"{prefix} {''.join(digits)}"

    return _RE_YEAR_4DIGIT_SPOKEN.sub(repl, text)


def _replace_months_spoken(text: str) -> str:
    out = text
    for name, num in _MONTH_PAIRS:
        pat = re.compile(
            rf"(?iu)\b(tháng)\s+{re.escape(name)}\b",
            re.UNICODE,
        )

        def repl(m: re.Match[str], n: str = num) -> str:
            return f"{m.group(1)} {n}"

        out = pat.sub(repl, out)
    return out


def _should_skip_cardinal_quantifier(cardinal: str, next_word: str, before_text: str = "") -> bool:
    c = _fold_vi_word(cardinal)
    w2 = _fold_vi_word(next_word)
    if not w2:
        return True
    if w2.isdigit():
        return True
    if c == "một" and w2 == "số":
        return True
    # Doan 'hai khong ...' thuong la doc nam (vi du sau 'nam hoc'); tranh -> '2 khong'.
    if c == "hai" and w2 == "không":
        return True
    # '... khong hai sau' (202x doc roi): tranh '2 sau'.
    if (
        c == "hai"
        and w2 == "sáu"
        and before_text
        and re.search(r"(?iu)\bkhông\s+$", before_text)
    ):
        return True
    if c == "năm" and w2 in _NAM_NOT_QUANTIFIER_FOLLOWING:
        return True
    if w2 in _LARGE_NUMBER_CONTINUATION:
        return True
    return False


def _replace_spoken_cardinal_quantifiers(text: str) -> str:
    """Chuyen 'sau chuong trinh', 'ba phan mem' -> '6 ...', '3 ...'."""

    digit_spans = _spoken_digit_spans(text, min_tokens=2)

    def repl(m: re.Match[str]) -> str:
        raw_c, raw_next = m.group(1), m.group(2)
        c0, c1 = m.start(1), m.end(1)
        if _span_covers_any(digit_spans, c0, c1):
            return m.group(0)
        before = m.string[: m.start()]
        if _should_skip_cardinal_quantifier(raw_c, raw_next, before):
            return m.group(0)
        key = _fold_vi_word(raw_c)
        num = _CARDINAL_TO_NUM.get(key)
        if num is None:
            return m.group(0)
        return f"{num} {raw_next}"

    return _RE_CARDINAL_QUANTIFIER.sub(repl, text)


def _run_vi_spoken_itn_tests() -> tuple[int, list[tuple[str, str, str, str]]]:
    """Tra ve (so test dung, danh sach loi (nhan, raw, expected, got))."""
    groups: list[tuple[str, list[tuple[str, str]]]] = [
        (
            "ngay_thang_nam",
            [
                ("Tháng ba năm hai không hai sáu", "Tháng 3 năm 2026"),
                ("tháng mười hai năm một chín chín chín", "tháng 12 năm 1999"),
                ("THÁNG NĂM NĂM HAI KHÔNG HAI SÁU", "THÁNG 5 NĂM 2026"),
                ("năm hai nghìn hai mươi sáu chi bộ đã", "năm 2026 chi bộ đã"),
                ("năm hai nghìn hai mươi lăm đã triển khai", "năm 2025 đã triển khai"),
                ("năm học hai không hai sáu", "năm học hai không hai sáu"),
            ],
        ),
        (
            "so_dem_van_ban",
            [
                ("vào sáu chương trình kế hoạch", "vào 6 chương trình kế hoạch"),
                ("xây dựng ba phần mềm tổng hợp", "xây dựng 3 phần mềm tổng hợp"),
                ("triển khai tám đợt tập huấn", "triển khai 8 đợt tập huấn"),
                ("mười hai đề tài cấp cơ sở", "12 đề tài cấp cơ sở"),
            ],
        ),
        (
            "dien_thoai_cccd_khong_tach",
            [
                (
                    "số điện thoại không chín một hai ba bốn năm sáu bảy tám",
                    "số điện thoại không chín một hai ba bốn năm sáu bảy tám",
                ),
                (
                    "hotline không chín một tám tám tám, không không không, hai một",
                    "hotline không chín một tám tám tám, không không không, hai một",
                ),
                (
                    "CCCD một hai ba bốn năm sáu bảy tám chín không một hai ba",
                    "CCCD một hai ba bốn năm sáu bảy tám chín không một hai ba",
                ),
                (
                    "căn cước "
                    "công dân là không chín một hai ba bốn năm sáu bảy tám chín, không một hai ba",
                    "căn cước "
                    "công dân là không chín một hai ba bốn năm sáu bảy tám chín, không một hai ba",
                ),
                (
                    "mã số không chín một, hai hai, ba ba bốn năm",
                    "mã số không chín một, hai hai, ba ba bốn năm",
                ),
                (
                    "OTP là một hai ba bốn năm sáu, nhập vào hệ thống",
                    "OTP là một hai ba bốn năm sáu, nhập vào hệ thống",
                ),
                (
                    "số nhà một hai ba, đường Nguyễn Trãi",
                    "số nhà một hai ba, đường Nguyễn Trãi",
                ),
                (
                    "địa chỉ tổ một hai, thôn ba, xã bốn năm",
                    "địa chỉ tổ một hai, thôn ba, xã bốn năm",
                ),
            ],
        ),
        (
            "tu_choi_sua_dinh_kem",
            [
                ("một số nhiệm vụ mới", "một số nhiệm vụ mới"),
                ("hai trăm đảng viên", "hai trăm đảng viên"),
                ("năm nay đạt hai điểm A", "năm nay đạt 2 điểm A"),
                ("đơn vị có hai lần kiểm tra", "đơn vị có 2 lần kiểm tra"),
            ],
        ),
        (
            "bien_gioi_hop_le",
            [
                (
                    "mã một hai ba bốn năm sáu và bảy đợt thanh tra",
                    "mã một hai ba bốn năm sáu và 7 đợt thanh tra",
                ),
                (
                    "điện thoại không chín một hai ba xong rồi ba người vào",
                    "điện thoại không chín một hai ba xong rồi 3 người vào",
                ),
            ],
        ),
        (
            "hoi_quy_sau_thay_doi",
            [
                # Nam + 4 chu so: 'nam' sau chu so noi tiep (ID) khong duoc hieu nhu 'nam' lich
                (
                    "dãy bốn năm sáu bảy tám chín không một",
                    "dãy bốn năm sáu bảy tám chín không một",
                ),
                # Van ban: 'nam' + nam doc roi, khong co chu so lien truoc 'nam'
                (
                    "báo cáo năm một chín chín chín",
                    "báo cáo năm 1999",
                ),
            ],
        ),
    ]
    failed: list[tuple[str, str, str, str]] = []
    ok = 0
    for label, cases in groups:
        for raw, expected in cases:
            got = apply_vi_spoken_number_rules(raw)
            if got == expected:
                ok += 1
            else:
                failed.append((label, raw, expected, got))
    return ok, failed


if __name__ == "__main__":
    _ok, _bad = _run_vi_spoken_itn_tests()
    if _bad:
        for lab, raw, exp, got in _bad:
            print("FAIL", lab)
            print("  in :", raw)
            print("  exp:", exp)
            print("  got:", got)
        raise SystemExit(1)
    print("vi_spoken_itn: ok", _ok, "tests")
