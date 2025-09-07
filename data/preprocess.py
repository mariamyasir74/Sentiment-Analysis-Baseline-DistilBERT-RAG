import re
# lowercase (optionally - for transformers you don't lowercase before tokenization if using cased models)
# remove HTML tags and URLS
# normalize whitespaces, optionally expand contractions (ex: aren't->are not)
# keep emoji
# keep basic punctuation + hashtags
def clean_text(s: str) -> str:
    s = s.replace("\n", " ")
    s = re.sub(r"<.?>", " ", s)
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^\w\s'.,!?:;#@-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
