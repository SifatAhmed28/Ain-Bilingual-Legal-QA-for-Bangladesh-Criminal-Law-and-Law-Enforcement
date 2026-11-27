"""Stage 1 configuration: paths, act whitelist, and text helpers.

The whitelist is the single source of truth for which statutes enter the
corpus.  ``act_id`` is the bdlaws act id, which is also the id used by BLAD
and UKIL-DB-EN, so all three sources can be joined without name matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "clean"
DATA_LEXICON = ROOT / "data" / "lexicon"
DATA_BENCHMARK = ROOT / "data" / "benchmark"
RESULTS = ROOT / "results"
REPORT = ROOT / "report"

# Stage 2: retrieval units are sized for the embedding model that Stage 3 will
# use.  400 tokens leaves room for the query and special tokens under a 512
# token limit, and 50 tokens of overlap keeps a clause that straddles a chunk
# boundary retrievable from either side.
EMBED_MODEL = "BAAI/bge-m3"
CHUNK_MAX_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50

BLAD_DIR = ROOT / "BLAD"
BLAD_ACTS_DIR = BLAD_DIR / "acts"
BLAD_ACT_LIST = BLAD_DIR / "filtered_act_list.csv"
UKIL_FILE = ROOT / "UKIL-DB-EN" / "acts_en.json"

BDLAWS_BASE = "http://bdlaws.minlaw.gov.bd"
BDLAWS_PRINT = BDLAWS_BASE + "/act-print-{act_id}.html"
BDLAWS_CHRONO = BDLAWS_BASE + "/laws-of-bangladesh-chronological-index.html"

RAW_BDLAWS = DATA_RAW / "bdlaws"
RAW_BDLAWS_HTML = RAW_BDLAWS / "html"
RAW_ACTS = DATA_RAW / "acts"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
    "bilingual-legal-qa-stage1/1.0"
)
FETCH_DELAY_SECONDS = 0.6


@dataclass(frozen=True)
class ActSpec:
    act_id: int
    code: str
    name_en: str
    name_bn: str
    category: str
    # Acts that were repealed are still downloaded so the audit log can show
    # why they were dropped.  They never reach sections.jsonl.
    audit_only: bool = False


# Criminal law + criminal procedure + law enforcement, curated.  Order here is
# the order used in acts.jsonl.  Repealed acts are kept for the audit trail.
ACTS: tuple[ActSpec, ...] = (
    # --- core criminal justice codes -------------------------------------
    ActSpec(11, "pc1860", "Penal Code, 1860", "দণ্ডবিধি, ১৮৬০", "criminal_code"),
    ActSpec(75, "crpc1898", "Code of Criminal Procedure, 1898", "ফৌজদারি কার্যবিধি, ১৮৯৮", "procedure"),
    ActSpec(24, "ev1872", "Evidence Act, 1872", "সাক্ষ্য আইন, ১৮৭২", "evidence"),
    ActSpec(12, "police1861", "Police Act, 1861", "পুলিশ আইন, ১৮৬১", "law_enforcement"),
    ActSpec(38, "arms1878", "Arms Act, 1878", "অস্ত্র আইন, ১৮৭৮", "offences"),
    ActSpec(52, "explosives1884", "Explosives Act, 1884", "বিস্ফোরক আইন, ১৮৮৪", "offences"),
    ActSpec(87, "esa1908", "Explosive Substances Act, 1908", "বিস্ফোরক দ্রব্য আইন, ১৯০৮", "offences"),
    ActSpec(69, "prisons1894", "Prisons Act, 1894", "কারাগার আইন, ১৮৯৪", "law_enforcement"),
    ActSpec(80, "prisoners1900", "Prisoners Act, 1900", "বন্দী আইন, ১৯০০", "law_enforcement"),
    ActSpec(120, "idprisoners1920", "Identification of Prisoners Act, 1920", "বন্দীদের শনাক্তকরণ আইন, ১৯২০", "law_enforcement"),
    ActSpec(217, "poca1947", "Prevention of Corruption Act, 1947", "দুর্নীতি প্রতিরোধ আইন, ১৯৪৭", "offences"),
    ActSpec(462, "spa1974", "Special Powers Act, 1974", "বিশেষ ক্ষমতা আইন, ১৯৭৪", "law_enforcement"),
    ActSpec(479, "extradition1974", "Extradition Act, 1974", "প্রত্যর্পণ আইন, ১৯৭৪", "procedure"),
    # --- modern substantive criminal law --------------------------------
    ActSpec(1256, "dowry2018", "Dowry Prohibition Act, 2018", "যৌতুক নিরোধ আইন, ২০১৮", "offences"),
    ActSpec(835, "wc2000", "Women and Children Repression Prevention Act, 2000", "নারী ও শিশু নির্যাতন দমন আইন, ২০০০", "offences"),
    ActSpec(883, "acid2002", "Acid Crime Prevention Act, 2002", "এসিড অপরাধ দমন আইন, ২০০২", "offences"),
    ActSpec(888, "speedy2002", "Speedy Trial Act, 2002", "আইন-শৃঙ্খলা বিঘ্নকারী অপরাধ (দ্রুত বিচার) আইন, ২০০২", "procedure"),
    ActSpec(914, "acc2004", "Anti-Corruption Commission Act, 2004", "দুর্নীতি দমন কমিশন আইন, ২০০৪", "law_enforcement"),
    ActSpec(1009, "ata2009", "Anti-Terrorism Act, 2009", "সন্ত্রাস বিরোধী আইন, ২০০৯", "offences"),
    ActSpec(1063, "dv2010", "Domestic Violence (Prevention and Protection) Act, 2010", "পারিবারিক সহিংসতা (প্রতিরোধ ও সুরক্ষা) আইন, ২০১০", "offences"),
    ActSpec(1086, "trafficking2012", "Human Trafficking Prevention and Suppression Act, 2012", "মানব পাচার প্রতিরোধ ও দমন আইন, ২০১২", "offences"),
    ActSpec(1087, "mla2012", "Mutual Legal Assistance in Criminal Matters Act, 2012", "অপরাধ সম্পর্কিত বিষয়ে পারস্পরিক সহায়তা আইন, ২০১২", "procedure"),
    ActSpec(1088, "mlpa2012", "Money Laundering Prevention Act, 2012", "মানিলন্ডারিং প্রতিরোধ আইন, ২০১২", "offences"),
    ActSpec(1119, "children2013", "Children Act, 2013", "শিশু আইন, ২০১৩", "procedure"),
    ActSpec(1133, "torture2013", "Torture and Custodial Death (Prevention) Act, 2013", "নির্যাতন এবং হেফাজতে মৃত্যু (নিবারণ) আইন, ২০১৩", "law_enforcement"),
    ActSpec(1276, "narcotics2018", "Narcotics Substances Control Act, 2018", "মাদকদ্রব্য নিয়ন্ত্রণ আইন, ২০১৮", "offences"),
    # --- policing and security forces ------------------------------------
    ActSpec(511, "dmp1976", "Dhaka Metropolitan Police Ordinance, 1976", "ঢাকা মহানগরী পুলিশ অধ্যাদেশ, ১৯৭৬", "law_enforcement"),
    ActSpec(578, "cmp1978", "Chittagong Metropolitan Police Ordinance, 1978", "চট্টগ্রাম মহানগরী পুলিশ অধ্যাদেশ, ১৯৭৮", "law_enforcement"),
    ActSpec(692, "kmp1985", "Khulna Metropolitan Police Ordinance, 1985", "খুলনা মহানগরী পুলিশ অধ্যাদেশ, ১৯৮৫", "law_enforcement"),
    ActSpec(1067, "bgb2010", "Border Guard Bangladesh Act, 2010", "বর্ডার গার্ড বাংলাদেশ আইন, ২০১০", "law_enforcement"),
    ActSpec(1181, "coastguard2016", "Bangladesh Coast Guard Act, 2016", "বাংলাদেশ কোস্ট গার্ড আইন, ২০১৬", "law_enforcement"),
    ActSpec(1484, "ansarbn2023", "Ansar Battalion Act, 2023", "আনসার ব্যাটালিয়ন আইন, ২০২৩", "law_enforcement"),
    ActSpec(1485, "policewelfare2023", "Bangladesh Police Welfare Fund Act, 2023", "বাংলাদেশ পুলিশ কল্যাণ তহবিল আইন, ২০২৩", "law_enforcement"),
    # --- post-BLAD / current law (fetched from bdlaws) --------------------
    ActSpec(1589, "policecommission2025", "Police Commission Ordinance, 2025", "পুলিশ কমিশন অধ্যাদেশ, ২০২৫", "law_enforcement"),
    ActSpec(1684, "trafficking2026", "Human Trafficking and Migrant Smuggling (Prevention and Suppression) Act, 2026", "মানব পাচার ও অভিবাসী চোরাচালান প্রতিরোধ ও দমন আইন, ২০২৬", "offences"),
    ActSpec(1710, "cyber2026", "Cyber Protection Act, 2026", "সাইবার সুরক্ষা আইন, ২০২৬", "offences"),
    ActSpec(1737, "gum2026", "GUM Prevention and Redress Act, 2026", "গুম প্রতিরোধ ও প্রতিকার আইন, ২০২৬", "law_enforcement"),
    ActSpec(1739, "srb2026", "Special Response Battalion (SRB) Act, 2026", "স্পেশাল রেসপন্স ব্যাটালিয়ন (SRB) আইন, ২০২৬", "law_enforcement"),
    ActSpec(1740, "apbn2026", "Armed Police Battalion (APBn) Act, 2026", "আর্মড পুলিশ ব্যাটালিয়ন (APBn) আইন, ২০২৬", "law_enforcement"),
    # --- repealed: downloaded, audited, then excluded ---------------------
    ActSpec(607, "dowry1980", "Dowry Prohibition Act, 1980", "যৌতুক নিরোধ আইন, ১৯৮০", "offences", audit_only=True),
    ActSpec(593, "apbn1979", "Armed Police Battalions Ordinance, 1979", "আর্মড পুলিশ ব্যাটালিয়ন অধ্যাদেশ, ১৯৭৯", "law_enforcement", audit_only=True),
    ActSpec(793, "battalionansar1995", "Battalion Ansar Act, 1995", "ব্যাটালিয়ন আনসার আইন, ১৯৯৫", "law_enforcement", audit_only=True),
    ActSpec(1261, "dsa2018", "Digital Security Act, 2018", "ডিজিটাল নিরাপত্তা আইন, ২০১৮", "offences", audit_only=True),
    ActSpec(1457, "csa2023", "Cyber Security Act, 2023", "সাইবার নিরাপত্তা আইন, ২০২৩", "offences", audit_only=True),
)

ACT_BY_ID: dict[int, ActSpec] = {a.act_id: a for a in ACTS}

# Bengali digits -> ASCII digits
BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# Bengali letters used as section suffixes (ধারা ৯ক -> 9A)
BN_SECTION_SUFFIX = {
    "ক": "A",
    "খ": "B",
    "গ": "C",
    "ঘ": "D",
    "ঙ": "E",
    "চ": "F",
    "ছ": "G",
    "জ": "H",
}

# Bengali month names -> month number (for amendment dates)
BN_MONTHS = {
    "জানুয়ারি": 1,
    "জানুয়ারি": 1,
    "ফেব্রুয়ারি": 2,
    "ফেব্রুয়ারি": 2,
    "মার্চ": 3,
    "এপ্রিল": 4,
    "মে": 5,
    "জুন": 6,
    "জুলাই": 7,
    "আগস্ট": 8,
    "সেপ্টেম্বর": 9,
    "অক্টোবর": 10,
    "নভেম্বর": 11,
    "ডিসেম্বর": 12,
}

EN_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
