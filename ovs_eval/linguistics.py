import requests
from nltk.corpus import wordnet
from functools import lru_cache
import nltk
import os

# Ensure wordnet is downloaded
try:
    wordnet.ensure_loaded()
except LookupError:
    nltk.download('wordnet', quiet=True)

BABELNET_URL = "https://babelnet.io/v9/"
BABELNET_API_KEY = os.getenv("BABELNET_API_KEY")

@lru_cache(maxsize=512)
def _babelnet_get_synset_id(word: str):
    """Get the best BabelNet synset ID for an English word."""
    try:
        r = requests.get(
            f"{BABELNET_URL}getSynsetIds",
            params={"lemma": word, "searchLang": "EN", "key": BABELNET_API_KEY},
            timeout=8
        )
        if r.status_code != 200:
            return None
        data = r.json()
        for item in data:
            if item.get("source") == "WN":
                return item["id"]
        return data[0]["id"] if data else None
    except Exception:
        return None

@lru_cache(maxsize=512)
def _babelnet_get_synset_data(synset_id: str) -> dict:
    """Get full synset data — lemmas, glosses, relations."""
    try:
        r = requests.get(
            f"{BABELNET_URL}getSynset",
            params={"id": synset_id, "targetLang": "EN", "key": BABELNET_API_KEY},
            timeout=8
        )
        if r.status_code != 200:
            return {}
        return r.json()
    except Exception:
        return {}

@lru_cache(maxsize=512)
def _babelnet_get_edges(synset_id: str) -> list:
    """Get semantic edges (hypernyms, hyponyms) for a synset."""
    try:
        r = requests.get(
            f"{BABELNET_URL}getOutgoingEdges",
            params={"id": synset_id, "key": BABELNET_API_KEY},
            timeout=8
        )
        if r.status_code != 200:
            return []
        return r.json()
    except Exception:
        return []

def _babelnet_synonym(word: str):
    """Get a synonym from BabelNet — other English lemmas in the same synset."""
    synset_id = _babelnet_get_synset_id(word)
    if not synset_id:
        return None
    data = _babelnet_get_synset_data(synset_id)
    for sense in data.get("senses", []):
        lemma = sense.get("properties", {}).get("simpleLemma", "")
        lang  = sense.get("properties", {}).get("language", "")
        if lang == "EN" and lemma.lower() != word.lower() and "_" not in lemma:
            return lemma.lower().replace(" ", "_")
    return None

def babelnet_synonym_list(word: str) -> list:
    """Get all synonyms from BabelNet for a word."""
    synset_id = _babelnet_get_synset_id(word)
    if not synset_id:
        return []
        
    synonyms = []
    data = _babelnet_get_synset_data(synset_id)
    for sense in data.get("senses", []):
        lemma = sense.get("properties", {}).get("simpleLemma", "")
        lang  = sense.get("properties", {}).get("language", "")
        if lang == "EN" and lemma.lower() != word.lower() and "_" not in lemma:
            syn_name = lemma.lower().replace(" ", "_")
            if syn_name not in synonyms:
                synonyms.append(syn_name)
                
    return synonyms

def babelnet_definition(word: str) -> str:
    """Get the definition/gloss from BabelNet for a word."""
    synset_id = _babelnet_get_synset_id(word)
    if not synset_id:
        return ""
    data = _babelnet_get_synset_data(synset_id)
    for gloss in data.get("glosses", []):
        if gloss.get("language") == "EN":
            return gloss.get("gloss", "")
    return ""


def _babelnet_hypernym(word: str):
    """Get hypernym via BabelNet edges."""
    synset_id = _babelnet_get_synset_id(word)
    if not synset_id:
        return None
    edges = _babelnet_get_edges(synset_id)
    for edge in edges:
        if edge.get("pointer", {}).get("shortName") in ("+@", "is-a"):
            target_id = edge.get("target")
            if target_id:
                target_data = _babelnet_get_synset_data(target_id)
                for sense in target_data.get("senses", []):
                    lemma = sense.get("properties", {}).get("simpleLemma", "")
                    lang  = sense.get("properties", {}).get("language", "")
                    if lang == "EN" and lemma:
                        return lemma.lower().replace(" ", "_")
    return None

def babelnet_hypernym_list(word: str) -> list:
    """Get all hypernyms via BabelNet edges."""
    synset_id = _babelnet_get_synset_id(word)
    if not synset_id:
        return []
    hypernyms = []
    edges = _babelnet_get_edges(synset_id)
    for edge in edges:
        if edge.get("pointer", {}).get("shortName") in ("+@", "is-a"):
            target_id = edge.get("target")
            if target_id:
                target_data = _babelnet_get_synset_data(target_id)
                for sense in target_data.get("senses", []):
                    lemma = sense.get("properties", {}).get("simpleLemma", "")
                    lang  = sense.get("properties", {}).get("language", "")
                    if lang == "EN" and lemma:
                        hyp_name = lemma.lower().replace(" ", "_")
                        if hyp_name not in hypernyms:
                            hypernyms.append(hyp_name)
    return hypernyms

def _babelnet_hyponym(word: str):
    """Get hyponym via BabelNet edges."""
    synset_id = _babelnet_get_synset_id(word)
    if not synset_id:
        return None
    edges = _babelnet_get_edges(synset_id)
    for edge in edges:
        if edge.get("pointer", {}).get("shortName") in ("+~", "has-kind"):
            target_id = edge.get("target")
            if target_id:
                target_data = _babelnet_get_synset_data(target_id)
                for sense in target_data.get("senses", []):
                    lemma = sense.get("properties", {}).get("simpleLemma", "")
                    lang  = sense.get("properties", {}).get("language", "")
                    if lang == "EN" and lemma:
                        return lemma.lower().replace(" ", "_")
    return None

def babelnet_hyponym_list(word: str) -> list:
    """Get all hyponyms via BabelNet edges."""
    synset_id = _babelnet_get_synset_id(word)
    if not synset_id:
        return []
    hyponyms = []
    edges = _babelnet_get_edges(synset_id)
    for edge in edges:
        if edge.get("pointer", {}).get("shortName") in ("+~", "has-kind"):
            target_id = edge.get("target")
            if target_id:
                target_data = _babelnet_get_synset_data(target_id)
                for sense in target_data.get("senses", []):
                    lemma = sense.get("properties", {}).get("simpleLemma", "")
                    lang  = sense.get("properties", {}).get("language", "")
                    if lang == "EN" and lemma:
                        hyp_name = lemma.lower().replace(" ", "_")
                        if hyp_name not in hyponyms:
                            hyponyms.append(hyp_name)
    return hyponyms


import threading

nltk_lock = threading.Lock()

def get_syn(word):
    with nltk_lock:
        synsets = wordnet.synsets(word)
        if synsets:
            return synsets[0].lemma_names()[0]
        return word

def get_linguistic_cats(cats):
    """Simple WordNet-only lookup for synonym, hypernym, and hyponym."""
    with nltk_lock:
        orig_cats, syn_cats, hyper_cats, hypo_cats = {}, {}, {}, {}
        for cat in cats:
            synset = wordnet.synsets(cat)[0] if wordnet.synsets(cat) else None
            if synset:
                synonym = next((lemma.name() for lemma in synset.lemmas() if lemma.name() != cat), cat)
                hypernyms = synset.hypernyms()
                hypernym = next((lemma.name() for lemma in hypernyms[0].lemmas()), cat) if hypernyms else cat
                hyponyms = synset.hyponyms()
                hyponym = next((lemma.name() for lemma in hyponyms[0].lemmas()), cat) if hyponyms else cat
            else:
                synonym = cat
                hypernym = cat
                hyponym = cat
            orig_cats[cat] = cat
            syn_cats[cat] = synonym
            hyper_cats[cat] = hypernym
            hypo_cats[cat] = hyponym
        return orig_cats, syn_cats, hyper_cats, hypo_cats

def get_linguistic_cats_v2(cats):
    """Linguistic lookup with BabelNet fallback."""
    with nltk_lock:
        orig_cats, syn_cats, hyper_cats, hypo_cats = {}, {}, {}, {}
        for cat in cats:
            orig_cats[cat] = cat
            synset = wordnet.synsets(cat)[0] if wordnet.synsets(cat) else None
            
            # Synonym lookup
            synonym = None
            if synset:
                synonym = next((l.name() for l in synset.lemmas() if l.name().lower() != cat.lower()), None)
            if synonym is None:
                synonym = _babelnet_synonym(cat)
            syn_cats[cat] = synonym if synonym is not None else cat

            # Hypernym lookup
            hypernym = None
            if synset:
                hypernyms = synset.hypernyms()
                if hypernyms:
                    hypernym = next((l.name() for l in hypernyms[0].lemmas()), None)
            if hypernym is None:
                hypernym = _babelnet_hypernym(cat)
            hyper_cats[cat] = hypernym if hypernym is not None else cat

            # Hyponym lookup
            hyponym = None
            if synset:
                hyponyms = synset.hyponyms()
                if hyponyms:
                    hyponym = next((l.name() for l in hyponyms[0].lemmas()), None)
            if hyponym is None:
                hyponym = _babelnet_hyponym(cat)
            hypo_cats[cat] = hyponym if hyponym is not None else cat

        return orig_cats, syn_cats, hyper_cats, hypo_cats

def build_valid_category_types(orig, syn, hyper, hypo) -> tuple:
    """
    Returns only variants where ALL classes have a valid label.
    """
    category_types = {"Original": orig}
    coverage = {}
    
    for variant_name, variant_dict in [("Synonyms", syn),
                                        ("Hypernyms", hyper),
                                        ("Hyponyms",  hypo)]:
        valid = {k: v for k, v in variant_dict.items() if v is not None}
        missing = [k for k, v in variant_dict.items() if v is None]
        coverage[variant_name] = {
            "valid": len(valid),
            "missing": missing,
        }
        if valid:
            category_types[variant_name] = valid

    return category_types, coverage
