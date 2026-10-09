#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scrape.py - Scraping du site de l'ANIL vers des fichiers Word (.docx).

Fonctionnalités :
- les 4 sections historiques du Centre de ressources, en reprenant la logique
  "legacy" du script initial : liens <h3><a> sur www.anil.org ;
- les 8 sections "Parole d'expert" ;
- programmation concurrente pour accélérer le téléchargement des articles ;
- nettoyage renforcé du texte : espaces insécables, retours ligne parasites,
  caractères invisibles, blocs de navigation/footer ANIL ;
- checkpoints JSON pour reprendre après interruption ;
- fusion finale en fichiers data/corpus/anil_complet*.docx (le corpus du RAG).

Exemples :
    python -m rag.scrape --crawl --workers 4 --delay 0.25

    python -m rag.scrape --list-urls-only

    python -m rag.scrape --crawl \
      --sections parole-location --workers 4 --delay 0.25

    python -m rag.scrape --merge-only --articles-per-file 800
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag
from docx import Document

from rag.config import CORPUS_DIR, DATA_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Configuration générale
# ─────────────────────────────────────────────────────────────────────────────

BASE_DOMAIN = "www.anil.org"
BASE_URL = f"https://{BASE_DOMAIN}"
DEFAULT_DELAY = 0.25
DEFAULT_WORKERS = 4
DEFAULT_TIMEOUT = 25

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) "
        "Gecko/20100101 Firefox/120.0"
    )
}

_NON_HTML_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".7z", ".jpg", ".jpeg", ".png", ".gif", ".webp",
    ".svg", ".mp4", ".mp3", ".avi", ".mov"
}


@dataclass(frozen=True)
class SectionConfig:
    url: str
    mode: str = "legacy"  # legacy | parole
    allowed_prefixes: Tuple[str, ...] = ()


# IMPORTANT :
# - Centre de ressources : on garde la découverte legacy du script initial.
#   Elle récupère les anciens formats d'URL ANIL : /aj-..., /jurisprudences-...,
#   /publications/..., /documentation-experte/...
# - Parole d'expert : on borne au préfixe /parole-expert-logement pour éviter
#   de partir vers les outils, ADIL locales, pages génériques, etc.
SECTIONS: dict[str, SectionConfig] = {
    # Centre de ressources
    "analyses-juridiques": SectionConfig(
        "/centre-de-ressources/votre-recherche/analyses-juridiques/",
        mode="legacy",
    ),
    "etudes-eclairages": SectionConfig(
        "/centre-de-ressources/votre-recherche/etudes-eclairages/",
        mode="legacy",
    ),
    "habitat-actualite": SectionConfig(
        "/centre-de-ressources/votre-recherche/habitat-actualite/",
        mode="legacy",
    ),
    "jurisprudence": SectionConfig(
        "/centre-de-ressources/votre-recherche/jurisprudence/",
        mode="legacy",
    ),

    # Parole d'expert
    "parole-achat-vente-immobiliere": SectionConfig(
        "/parole-expert-logement/achat-vente-immobiliere/",
        mode="parole",
        allowed_prefixes=("/parole-expert-logement",),
    ),
    "parole-assurance": SectionConfig(
        "/parole-expert-logement-assurance/",
        mode="parole",
        allowed_prefixes=("/parole-expert-logement",),
    ),
    "parole-copropriete": SectionConfig(
        "/parole-expert-logement-copropriete/",
        mode="parole",
        allowed_prefixes=("/parole-expert-logement",),
    ),
    "parole-habitat-indigne": SectionConfig(
        "/parole-expert-logement-habitat-indigne/",
        mode="parole",
        allowed_prefixes=("/parole-expert-logement",),
    ),
    "parole-habiter-vie-pratique": SectionConfig(
        "/parole-expert-logement-habiter-vie-pratique/",
        mode="parole",
        allowed_prefixes=("/parole-expert-logement",),
    ),
    "parole-location": SectionConfig(
        "/parole-expert-logement-location/",
        mode="parole",
        allowed_prefixes=("/parole-expert-logement",),
    ),
    "parole-renovation-energetique": SectionConfig(
        "/parole-expert-logement-renovation-energetique/",
        mode="parole",
        allowed_prefixes=("/parole-expert-logement",),
    ),
    "parole-urbanisme": SectionConfig(
        "/parole-expert-logement-urbanisme/",
        mode="parole",
        allowed_prefixes=("/parole-expert-logement",),
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Nettoyage du texte
# ─────────────────────────────────────────────────────────────────────────────

NOISE_EXACT_LINES = {
    "Accueil",
    "Outils pratiques",
    "Centre de ressources",
    "Questions fréquentes",
    "Votre ADIL",
    "Comment pouvons-nous vous aider ?",
    "Les contenus les plus populaires :",
    "Les contenus les plus populaires",
    "Choisir votre ADIL",
    "Partager cette page",
    "Imprimer cette page",
    "Retour en haut de page",
    "Menu",
    "Fermer",
}

NOISE_START_MARKERS = (
    "Comment pouvons-nous vous aider ?",
    "Les contenus les plus populaires",
    "Choisir votre ADIL",
    "Sélectionnez votre département pour accéder aux contenus",
    "Nos conseillers locaux vous apporteront",
)

# Certains textes apparaissent dans les pages sous forme de menu / fil d'Ariane.
BREADCRUMB_OR_MENU_LINES = {
    "Accueil",
    "Outils pratiques",
    "Centre de ressources",
    "Questions fréquentes",
    "Documentation experte",
    "Analyses juridiques",
    "Jurisprudence",
    "Études et Éclairages",
    "Habitat Actualité",
    "Votre besoin",
}


def clean_text(text: str) -> str:
    """Nettoie un texte extrait du HTML avant écriture dans le .docx.

    Objectif : produire des phrases continues et exploitables pour le RAG,
    sans casser les paragraphes logiques.
    """
    if not text:
        return ""

    # Espaces invisibles / insécables / BOM / soft hyphen
    replacements = {
        "\xa0": " ",
        "\u202f": " ",
        "\u2009": " ",
        "\u2007": " ",
        "\u200b": "",
        "\ufeff": "",
        "\xad": "",
        "￾": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Caractères de contrôle incompatibles XML / Word
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Répare les mots coupés par retour ligne ou tiret parasite.
    # Exemple : "ré-\nnovation" -> "rénovation"
    text = re.sub(
        r"([A-Za-zÀ-ÖØ-öø-ÿ])-+\s*\n\s*([A-Za-zÀ-ÖØ-öø-ÿ])",
        r"\1\2",
        text,
    )

    # Les retours ligne simples deviennent des espaces.
    text = re.sub(r"[ \t]*\r?\n[ \t]*", " ", text)

    # Certains scraps collent les sigles : "CCH: R.300-2" -> "CCH : R.300-2"
    text = re.sub(r"\b([A-Z]{2,6}):", r"\1 :", text)

    # Espaces multiples
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Ponctuation française
    text = re.sub(r"\s+([,.;])", r"\1", text)
    text = re.sub(r"\s+(:)", r" :", text)
    text = re.sub(r"\s+([!?])", r" \1", text)

    # Parenthèses / crochets
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\[\s+", "[", text)
    text = re.sub(r"\s+\]", "]", text)

    # Guillemets français : on normalise sans surcorriger
    text = re.sub(r"«\s*", "« ", text)
    text = re.sub(r"\s*»", " »", text)

    return text.strip()


def is_noise_text(text: str) -> bool:
    """Détecte les lignes/blocs de navigation ou footer ANIL."""
    text = clean_text(text)
    if not text:
        return True

    if text in NOISE_EXACT_LINES:
        return True

    if text in BREADCRUMB_OR_MENU_LINES:
        return True

    if any(text.startswith(marker) for marker in NOISE_START_MARKERS):
        return True

    # Liens / blocs de service hors article
    if text.startswith("Sélectionnez votre département"):
        return True
    if text.startswith("Nos conseillers locaux"):
        return True

    # Micro-blocs de navigation très courts
    if len(text) <= 2:
        return True

    return False


def should_stop_article_extraction(text: str) -> bool:
    """Retourne True quand on atteint le footer ANIL de fin d'article."""
    text = clean_text(text)
    return any(marker in text for marker in NOISE_START_MARKERS)


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# HTTP
# ─────────────────────────────────────────────────────────────────────────────

def normalize_url(url: str) -> str:
    """URL canonique simple : absolute + suppression fragment."""
    url, _frag = urldefrag(url)
    parsed = urlparse(url)
    if parsed.scheme == "http":
        parsed = parsed._replace(scheme="https")
    return parsed.geturl()


def is_html_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return not any(path.endswith(ext) for ext in _NON_HTML_EXTENSIONS)


def fetch_html(
    url: str,
    *,
    delay: float = 0.0,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = 3,
) -> str:
    """Télécharge une page HTML avec retry simple."""
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            if delay:
                time.sleep(delay)
            response = requests.get(url, headers=_HEADERS, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            last_error = e
            if attempt < max_retries:
                wait = min(2 ** attempt, 8)
                print(f"  Erreur réseau ({attempt}/{max_retries}) sur {url}: {e}")
                print(f"  Retry dans {wait}s...")
                time.sleep(wait)

    raise last_error  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# Découverte des URLs d'articles
# ─────────────────────────────────────────────────────────────────────────────

def get_max_page(html: str) -> int:
    max_page = 1
    for m in re.finditer(r"/page-(\d+)/", html):
        try:
            max_page = max(max_page, int(m.group(1)))
        except ValueError:
            pass
    return max_page


def _accept_link(
    full_url: str,
    *,
    mode: str,
    allowed_prefixes: Tuple[str, ...] = (),
) -> bool:
    parsed = urlparse(full_url)

    if parsed.netloc != BASE_DOMAIN:
        return False

    if not is_html_url(full_url):
        return False

    path = parsed.path

    # Centre de ressources : on reprend la logique legacy de ton script initial.
    # On garde tout lien HTML ANIL trouvé dans les <h3>.
    if mode == "legacy":
        return True

    # Parole d'expert : on borne aux pages de ce périmètre.
    if mode == "parole":
        if not allowed_prefixes:
            return "/parole-expert-logement" in path
        return any(path.startswith(prefix) for prefix in allowed_prefixes)

    return False


def get_article_links_from_listing(
    html: str,
    page_url: str,
    *,
    mode: str,
    allowed_prefixes: Tuple[str, ...] = (),
) -> List[str]:
    """Extrait les liens d'articles depuis une page listing.

    Centre de ressources :
      logique legacy = liens dans les <h3>, comme ton script initial.

    Parole d'expert :
      les pages ANIL listent les questions directement dans des <li><a>
      après le H1. On prend donc les liens du <main>, puis on filtre.
    """
    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []

    candidates: List[Tag] = []

    if mode == "legacy":
        # Même logique que ton ancien script.
        for h3 in soup.find_all("h3"):
            candidates.extend(h3.find_all("a", href=True))

    elif mode == "parole":
        root = soup.find("main") or soup.find("article") or soup

        # On prend tous les liens du contenu principal.
        candidates.extend(root.find_all("a", href=True))

    else:
        candidates.extend(soup.find_all("a", href=True))

    for a in candidates:
        href = a.get("href")
        if not href:
            continue

        full_url = normalize_url(urljoin(page_url, href))
        parsed = urlparse(full_url)
        path = parsed.path

        if parsed.netloc != BASE_DOMAIN:
            continue

        if not is_html_url(full_url):
            continue

        if mode == "legacy":
            links.append(full_url)
            continue

        if mode == "parole":
            # On garde uniquement les pages Parole d'expert.
            if "/parole-expert-logement" not in path:
                continue

            # On exclut les pages racines / thématiques.
            if path.rstrip("/") in {
                "/parole-expert-logement",
                "/parole-expert-logement/achat-vente-immobiliere",
                "/parole-expert-logement-assurance",
                "/parole-expert-logement-copropriete",
                "/parole-expert-logement-habitat-indigne",
                "/parole-expert-logement-habiter-vie-pratique",
                "/parole-expert-logement-location",
                "/parole-expert-logement-renovation-energetique",
                "/parole-expert-logement-urbanisme",
            }:
                continue

            # On exclut le retour à toutes les thématiques si présent.
            label = clean_text(a.get_text(" ", strip=True))
            if label.lower().startswith("retour à toutes les thématiques"):
                continue

            links.append(full_url)
            continue

    return dedupe_preserve_order(links)


def discover_article_urls(
    section_url: str,
    *,
    mode: str,
    allowed_prefixes: Tuple[str, ...] = (),
    delay: float = DEFAULT_DELAY,
    max_articles: Optional[int] = None,
) -> List[str]:
    print(f"  Analyse listing : {section_url}")
    html = fetch_html(section_url, delay=delay)
    max_page = get_max_page(html)
    print(f"  {max_page} page(s) de listing détectée(s)")

    all_links = get_article_links_from_listing(
        html,
        section_url,
        mode=mode,
        allowed_prefixes=allowed_prefixes,
    )

    for page_num in range(2, max_page + 1):
        if max_articles and len(all_links) >= max_articles:
            break

        page_url = section_url.rstrip("/") + f"/page-{page_num}/"
        print(f"  Listing {page_num}/{max_page}...", end="\r", flush=True)

        try:
            page_html = fetch_html(page_url, delay=delay)
        except requests.RequestException as e:
            print(f"\n  ERREUR listing {page_num}: {e}")
            continue

        all_links.extend(
            get_article_links_from_listing(
                page_html,
                page_url,
                mode=mode,
                allowed_prefixes=allowed_prefixes,
            )
        )

    deduped = dedupe_preserve_order(all_links)
    if max_articles:
        deduped = deduped[:max_articles]

    print(f"\n  {len(deduped)} article(s) découvert(s)")
    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# Extraction d'article
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractedSection:
    title: str
    paragraphs: List[str] = field(default_factory=list)


@dataclass
class ExtractedArticle:
    url: str
    title: str
    intro: List[str] = field(default_factory=list)
    sections: List[ExtractedSection] = field(default_factory=list)
    error: str = ""


BLOCK_TAGS = {
    "p", "li", "h2", "h3", "h4", "blockquote",
    "td", "th", "caption"
}


def remove_noise_tags(soup: BeautifulSoup) -> None:
    """Supprime des blocs HTML qui ne font pas partie de l'article."""
    for tag in list(soup(["script", "style", "noscript", "svg", "form", "button", "iframe"])):
        try:
            tag.decompose()
        except Exception:
            pass

    # Balises structurelles souvent hors contenu.
    for tag in list(soup.find_all(["header", "footer", "nav", "aside"])):
        try:
            tag.decompose()
        except Exception:
            pass

    # Classes / ids de navigation fréquents.
    patterns = re.compile(
        r"(breadcrumb|ariane|menu|nav|footer|header|search|social|share|"
        r"newsletter|pagination|cookie|modal|toolbar|print|adil|popular)",
        re.I,
    )

    for tag in list(soup.find_all(True)):
        try:
            # Après un decompose(), certains tags peuvent avoir attrs=None.
            if tag is None or getattr(tag, "attrs", None) is None:
                continue

            classes = " ".join(tag.get("class") or [])
            ident = tag.get("id") or ""
            role = tag.get("role") or ""

            if patterns.search(classes) or patterns.search(ident) or role in {"navigation", "search"}:
                tag.decompose()

        except Exception:
            continue

def _root_for_article(soup: BeautifulSoup) -> Tag:
    return soup.find("main") or soup.find("article") or soup.body or soup


def _iter_content_blocks_after_h1(root: Tag, h1: Optional[Tag]) -> Iterable[Tag]:
    """Itère les blocs de contenu dans l'ordre, en commençant après le H1."""
    started = h1 is None

    for tag in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote", "td", "th", "caption"]):
        if tag.name == "h1":
            if h1 is not None and tag == h1:
                started = True
            continue

        if not started:
            continue

        # Évite de lire un bloc dont un parent est déjà un bloc candidat,
        # ce qui limite les doublons p/li imbriqués dans des cartes.
        parent = tag.parent
        skip = False
        while isinstance(parent, Tag) and parent != root:
            if parent.name in {"nav", "aside", "footer", "header"}:
                skip = True
                break
            parent = parent.parent
        if skip:
            continue

        yield tag


def extract_article(html: str, url: str) -> ExtractedArticle:
    """Extrait un article ANIL nettoyé et structuré."""
    soup = BeautifulSoup(html, "html.parser")
    remove_noise_tags(soup)

    root = _root_for_article(soup)
    h1 = root.find("h1") or soup.find("h1")
    title = clean_text(h1.get_text(" ", strip=True) if h1 else url) or url

    article = ExtractedArticle(url=url, title=title)
    current_section: Optional[ExtractedSection] = None

    seen_paragraphs: Set[str] = set()

    for tag in _iter_content_blocks_after_h1(root, h1):
        raw_text = tag.get_text(" ", strip=True)
        text = clean_text(raw_text)

        if should_stop_article_extraction(text):
            break

        if is_noise_text(text):
            continue

        # Supprime les titres de breadcrumb/menu qui réapparaissent parfois.
        if text in BREADCRUMB_OR_MENU_LINES:
            continue

        if tag.name in {"h2", "h3"}:
            # Évite de recréer une section avec un titre identique au H1.
            if text and text != title:
                current_section = ExtractedSection(title=text)
                article.sections.append(current_section)
            continue

        # Les h4 sont plutôt des sous-titres internes : on les garde comme
        # paragraphes pour préserver le contexte.
        if tag.name == "h4":
            text = text

        # Déduplication locale pour éviter les répétitions de cartes/liens.
        dedupe_key = re.sub(r"\s+", " ", text).strip().lower()
        if dedupe_key in seen_paragraphs:
            continue
        seen_paragraphs.add(dedupe_key)

        if current_section is not None:
            current_section.paragraphs.append(text)
        else:
            article.intro.append(text)

    # Fallback : si l'extraction structurée donne trop peu, on reprend les p
    # de main/article, nettoyés, sans section.
    n_chars = sum(len(p) for p in article.intro) + sum(len(p) for s in article.sections for p in s.paragraphs)
    if n_chars < 120:
        article.intro.clear()
        article.sections.clear()
        for p in root.find_all("p"):
            text = clean_text(p.get_text(" ", strip=True))
            if should_stop_article_extraction(text):
                break
            if not is_noise_text(text):
                article.intro.append(text)

    return article


def fetch_and_extract(url: str, *, delay: float) -> ExtractedArticle:
    try:
        html = fetch_html(url, delay=delay)
        return extract_article(html, url)
    except Exception as e:
        return ExtractedArticle(url=url, title=url, error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Word / checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(path: Path) -> Set[str]:
    if path.exists():
        try:
            return set(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return set()
    return set()


def save_checkpoint(path: Path, visited: Set[str]) -> None:
    path.write_text(
        json.dumps(sorted(visited), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def add_article_to_doc(doc: Document, article: ExtractedArticle) -> bool:
    """Ajoute un article au document Word. Retourne True si contenu utile."""
    if article.error:
        return False

    content_count = len(article.intro) + sum(len(s.paragraphs) for s in article.sections)
    if content_count == 0:
        return False

    doc.add_heading(article.title, level=1)
    url_para = doc.add_paragraph(article.url)
    if url_para.runs:
        url_para.runs[0].italic = True

    for p in article.intro:
        p = clean_text(p)
        if p and not is_noise_text(p):
            doc.add_paragraph(p)

    for section in article.sections:
        section_title = clean_text(section.title)
        clean_paragraphs = [
            clean_text(p)
            for p in section.paragraphs
            if clean_text(p) and not is_noise_text(p)
        ]

        if not section_title and not clean_paragraphs:
            continue

        if section_title and section_title != article.title:
            doc.add_heading(section_title, level=2)

        for p in clean_paragraphs:
            doc.add_paragraph(p)

    return True


def scrape_section_to_docx(
    section_name: str,
    config: SectionConfig,
    output_dir: Path,
    *,
    max_articles: Optional[int],
    delay: float,
    workers: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    docx_path = output_dir / f"{section_name}.docx"
    checkpoint_path = output_dir / f"{section_name}_checkpoint.json"

    section_url = normalize_url(urljoin(BASE_URL, config.url))
    article_urls = discover_article_urls(
        section_url,
        mode=config.mode,
        allowed_prefixes=config.allowed_prefixes,
        delay=delay,
        max_articles=max_articles,
    )

    visited = load_checkpoint(checkpoint_path)
    if visited and docx_path.exists():
        print(f"  Reprise : {len(visited)} article(s) déjà traité(s)")
        doc = Document(str(docx_path))
    else:
        visited = set()
        doc = Document()
        doc.add_heading(f"ANIL - {section_name.replace('-', ' ').title()}", level=0)

    remaining = [u for u in article_urls if u not in visited]
    print(f"  {len(remaining)} article(s) à traiter")

    if not remaining:
        doc.save(str(docx_path))
        save_checkpoint(checkpoint_path, visited)
        return docx_path

    scraped = 0
    errors = 0

    # Important : on récupère en parallèle, mais on écrit dans Word dans l'ordre
    # de article_urls pour obtenir des fichiers reproductibles.
    results: dict[str, ExtractedArticle] = {}

    max_workers = max(1, workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(fetch_and_extract, url, delay=delay): url
            for url in remaining
        }

        done = 0
        total = len(remaining)

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            done += 1

            try:
                article = future.result()
            except Exception as e:
                article = ExtractedArticle(url=url, title=url, error=str(e))

            results[url] = article

            if article.error:
                errors += 1
                print(f"  [{done}/{total}] ERREUR {url}: {article.error}")
            else:
                print(f"  [{done}/{total}] OK {article.title[:80]}")

    for url in remaining:
        article = results.get(url)
        if not article:
            continue

        if add_article_to_doc(doc, article):
            visited.add(url)
            scraped += 1
        else:
            errors += 1
            print(f"  Article ignoré ou vide : {url}")

        if scraped and scraped % 50 == 0:
            doc.save(str(docx_path))
            save_checkpoint(checkpoint_path, visited)
            print(f"  Checkpoint : {scraped} nouveaux articles sauvegardés")

    doc.save(str(docx_path))
    save_checkpoint(checkpoint_path, visited)

    print(
        f"  Section '{section_name}' terminée : "
        f"{scraped} nouveau(x) article(s), {errors} erreur(s) -> {docx_path}"
    )
    return docx_path


# ─────────────────────────────────────────────────────────────────────────────
# Fusion Word
# ─────────────────────────────────────────────────────────────────────────────

def _new_empty_doc() -> Document:
    d = Document()
    for p in list(d.paragraphs):
        p._element.getparent().remove(p._element)
    return d


def merge_docx_files_split(
    source_paths: List[Path],
    output_stem: Path,
    *,
    articles_per_file: int = 800,
) -> List[Path]:
    """Fusionne les fichiers par section en plusieurs docx si nécessaire."""
    produced: List[Path] = []

    current = _new_empty_doc()
    file_idx = 1
    articles_in_file = 0
    total_articles = 0

    def flush() -> None:
        nonlocal current, file_idx, articles_in_file

        if articles_in_file <= 0:
            return

        suffix = "" if file_idx == 1 else f"_partie{file_idx}"
        out = Path(str(output_stem) + suffix + ".docx")
        current.save(str(out))
        produced.append(out)
        print(f"  Partie {file_idx} -> {out.name} ({articles_in_file} articles)")

        current = _new_empty_doc()
        file_idx += 1
        articles_in_file = 0

    for path in source_paths:
        print(f"  Fusion : {path.name}")
        sub = Document(str(path))
        style_map = {p._element: (p.style.name if p.style else "") for p in sub.paragraphs}

        for element in sub.element.body:
            is_h1 = style_map.get(element, "") == "Heading 1"

            if is_h1:
                if articles_in_file >= articles_per_file:
                    flush()
                articles_in_file += 1
                total_articles += 1

            current.element.body.append(deepcopy(element))

    flush()

    print(
        f"Fusion terminée : {total_articles} article(s) répartis en "
        f"{len(produced)} fichier(s)"
    )
    return produced


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires CLI
# ─────────────────────────────────────────────────────────────────────────────

def selected_section_names(names: Optional[List[str]]) -> List[str]:
    if not names:
        return list(SECTIONS.keys())
    return names


def list_urls_only(output_dir: Path, section_names: List[str], delay: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for name in section_names:
        cfg = SECTIONS[name]
        section_url = normalize_url(urljoin(BASE_URL, cfg.url))

        urls = discover_article_urls(
            section_url,
            mode=cfg.mode,
            allowed_prefixes=cfg.allowed_prefixes,
            delay=delay,
            max_articles=None,
        )

        out = output_dir / f"{name}_urls.txt"
        out.write_text("\n".join(urls) + ("\n" if urls else ""), encoding="utf-8")
        print(f"  URLs écrites -> {out} ({len(urls)})")
        total += len(urls)

    print(f"\nTotal URLs détectées : {total}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--crawl", action="store_true", help="Scrape les sections sélectionnées")
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=list(SECTIONS.keys()),
        default=None,
        help="Sections à scraper. Défaut : toutes les sections.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "anil_sections",
        help="Dossier de sortie des .docx par section (defaut : data/anil_sections)",
    )
    parser.add_argument(
        "--merged-output",
        type=Path,
        default=None,
        help="Nom de base du fichier fusionné, sans extension. Défaut : data/corpus/anil_complet",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Fusionne uniquement les .docx existants sans scraper",
    )
    parser.add_argument(
        "--list-urls-only",
        action="store_true",
        help="Liste les URLs détectées sans scraper les articles",
    )
    parser.add_argument(
        "--articles-per-file",
        type=int,
        default=800,
        help="Nombre max d'articles par fichier Word fusionné",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Limite d'articles par section, utile pour tester",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Délai par requête HTTP, en secondes. Défaut : {DEFAULT_DELAY}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Nombre de téléchargements concurrents. Défaut : {DEFAULT_WORKERS}",
    )

    # Mode page unique optionnel
    parser.add_argument("url", nargs="?", help="URL d'article unique à scraper")
    parser.add_argument("output", nargs="?", type=Path, help="Fichier .docx de sortie en mode page unique")

    args = parser.parse_args()

    section_names = selected_section_names(args.sections)

    # Par défaut, les fichiers fusionnés vont dans data/corpus/ et les fichiers
    # par section dans un dossier séparé, pour ne pas chunker deux fois les
    # mêmes articles.
    merged_stem = args.merged_output or (CORPUS_DIR / "anil_complet")
    if str(merged_stem).endswith(".docx"):
        merged_stem = Path(str(merged_stem)[:-5])
    merged_stem.parent.mkdir(parents=True, exist_ok=True)

    if args.list_urls_only:
        list_urls_only(args.output_dir, section_names, args.delay)
        return

    if args.merge_only:
        source_files = [
            args.output_dir / f"{name}.docx"
            for name in section_names
            if (args.output_dir / f"{name}.docx").exists()
        ]

        if not source_files:
            print(f"Aucun fichier .docx de section trouvé dans {args.output_dir}")
            raise SystemExit(1)

        merge_docx_files_split(
            source_files,
            merged_stem,
            articles_per_file=args.articles_per_file,
        )
        return

    if args.crawl:
        produced_files: List[Path] = []

        for name in section_names:
            print(f"\n{'=' * 80}")
            print(f"Section : {name}")
            print(f"{'=' * 80}")

            try:
                produced = scrape_section_to_docx(
                    name,
                    SECTIONS[name],
                    args.output_dir,
                    max_articles=args.max_articles,
                    delay=args.delay,
                    workers=args.workers,
                )
                produced_files.append(produced)
            except Exception as e:
                print(f"ERREUR FATALE section '{name}' : {e}")

        if produced_files:
            print(f"\nFusion de {len(produced_files)} fichier(s)...")
            merge_docx_files_split(
                produced_files,
                merged_stem,
                articles_per_file=args.articles_per_file,
            )

        print("\nScraping terminé.")
        return

    # Mode page unique
    if args.url and args.output:
        html = fetch_html(args.url, delay=args.delay)
        article = extract_article(html, normalize_url(args.url))

        doc = Document()
        ok = add_article_to_doc(doc, article)
        if not ok:
            print("Aucun contenu utile extrait.")
            raise SystemExit(1)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(args.output))
        print(f"Écrit : {args.output}")
        return

    parser.print_help()
    raise SystemExit(1)


if __name__ == "__main__":
    main()
