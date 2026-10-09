"""
chunk.py - Decoupage hierarchique des .docx en chunks pour RAG.

Strategie de chunking, du plus prioritaire au moins prioritaire :

  1. Si l'article entier (Titre + URL + corps) tient dans `--max-chars`,
     on le garde en un seul chunk : c'est le cas ideal, le contexte de
     l'article est preserve.

  2. Sinon, on coupe par sous-titres (Heading 2). Chaque section
     [intro avant le 1er H2] et chaque section [H2 + paragraphes] devient
     un chunk independant, prefixe par "Titre / URL / Section".

  3. Si une section depasse encore `--max-chars`, on la redecoupe en
     accumulant les paragraphes (separes par "\\n\\n") avec un overlap de
     `--overlap-chars` caracteres pour ne pas casser la continuite.

Chaque chunk est enrichi en tete avec :

    Titre : <titre de l'article>
    URL : <url source>
    Section : <H2 si dispo>

    <contenu du chunk>

Format de sortie (JSONL, 1 chunk par ligne) :

    {"id": "...", "text": "Titre : ...\\nURL : ...\\nSection : ...\\n\\n<corps>",
     "metadata": {"source_file", "article_title", "article_url",
                  "section_title", "chunk_index", "char_count",
                  "year", "archived"}}

Datation des articles (champ `year`, 0 si inconnue) - utile pour ecarter au
retrieval les analyses juridiques anterieures a une reforme (ex : loi ALUR
de 2014), voir `--min-year` dans rag.ask :

  1. mention "A jour au <jour> <mois> <annee>" en tete d'article ;
  2. sinon, annee presente dans l'URL (".../analyses-juridiques-2010/...") ;
  3. sinon, date de la decision citee (jurisprudence : "Cass. Civ III : 21.11.19").

Le champ `archived` signale les pages issues des archives de l'ANIL.

Utilisation :

    # Chunker le corpus data/corpus/ vers data/chunks.jsonl
    python -m rag.chunk

    # Ajuster les tailles
    python -m rag.chunk --max-chars 3000 --overlap-chars 500 --min-chars 300

    # Apercu des 3 premiers chunks sans ecrire le fichier
    python -m rag.chunk --preview
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, List, Optional

from rag.config import CHUNKS_FILE, CORPUS_DIR


# Valeurs par defaut exposees pour pouvoir les surcharger via l'interface CLI.
DEFAULT_MAX_CHARS = 3000
DEFAULT_OVERLAP_CHARS = 500
DEFAULT_MIN_CHARS = 300


@dataclass
class Section:
    """Une section H2 d'un article : titre + paragraphes."""
    title: str
    paragraphs: List[str] = field(default_factory=list)


@dataclass
class Article:
    """Un article : titre H1, URL source, paragraphes d'intro avant tout H2,
    et liste de sections H2."""
    title: str
    url: str = ""
    # Paragraphes situes avant le premier H2 (introduction de l'article).
    intro_paragraphs: List[str] = field(default_factory=list)
    sections: List[Section] = field(default_factory=list)


@dataclass
class Chunk:
    """Un chunk pret a etre embedde."""
    id: str
    text: str
    metadata: dict = field(default_factory=dict)

    def to_jsonl(self) -> str:
        # ensure_ascii=False preserve les accents dans le fichier de sortie.
        return json.dumps(asdict(self), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Lecture des .docx -> Article(s)
# ---------------------------------------------------------------------------

def _is_italic_paragraph(paragraph) -> bool:
    """Retourne True si tous les runs non vides du paragraphe sont en italique.

    On ne se base pas sur le style du paragraphe car Word peut appliquer
    l'italique au niveau du run plutot que du paragraphe entier. On filtre
    d'abord les runs vides pour ne pas fausser le test.
    """
    runs = [r for r in paragraph.runs if r.text.strip()]
    if not runs:
        return False
    return all(bool(r.italic) for r in runs)


def _style_name(paragraph) -> str:
    """Renvoie le nom de style d'un paragraphe, ou une chaine vide en cas d'erreur.

    Le try/except protege contre les documents corrompus ou les styles
    personnalises qui ne possedent pas d'attribut 'name'.
    """
    try:
        return paragraph.style.name or ""
    except Exception:
        return ""


def parse_articles_from_docx(docx_path: Path) -> List[Article]:
    """Extrait la structure hierarchique du .docx en une liste d'Article.

    En mode crawl, plusieurs articles (un par H1).
    En mode page unique (sans H2), un seul article : H1 = titre,
    paragraphes = intro_paragraphs.

    La fonction parcourt les paragraphes lineairement et maintient un
    "curseur" (current, current_section) qui avance au fil des titres
    rencontres. C'est une machine a etats implicite :
        - Heading 1 : demarre un nouvel article.
        - Heading 2 : demarre une nouvelle section dans l'article courant.
        - Paragraphe italique juste apres un H1 : identifie comme URL source.
        - Paragraphe normal : affecte a la section courante ou a l'intro.
    """
    from docx import Document

    doc = Document(str(docx_path))
    articles: List[Article] = []
    current: Optional[Article] = None
    current_section: Optional[Section] = None
    # Drapeau qui signale qu'on vient de lire un H1 et qu'on attend
    # eventuellement l'URL (paragraphe italique qui suit immediatement).
    expect_url_after_h1 = False

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        style = _style_name(paragraph)

        # "Title" est le style du titre global du document (ex: "ANIL - ..."),
        # "Heading" sans numero est un style generique a ignorer.
        if style in ("Title", "Heading"):
            continue

        if style == "Heading 1":
            # On sauvegarde l'article precedent avant d'en commencer un nouveau.
            if current is not None:
                articles.append(current)
            current = Article(title=text or docx_path.stem)
            current_section = None
            # Active l'attente de l'URL sur le prochain paragraphe.
            expect_url_after_h1 = True
            continue

        if style == "Heading 2":
            if current is None:
                # Cas degenere : H2 avant tout H1 (ne devrait pas arriver
                # avec les .docx produits par le scraper).
                current = Article(title=docx_path.stem)
            # Chaque H2 cree une nouvelle section et la rattache a l'article.
            current_section = Section(title=text)
            current.sections.append(current_section)
            # Un H2 apres un H1 sans URL intermediaire ferme la fenetre d'attente.
            expect_url_after_h1 = False
            continue

        # --- Paragraphe normal ---
        if not text:
            # On ignore les paragraphes vides pour ne pas polluer les chunks.
            continue

        # Detection de l'URL : le scraper ecrit l'URL en italique
        # immediatement apres le H1. On verifie aussi le prefixe http
        # pour ne pas confondre avec une note de bas de page en italique.
        if (
            expect_url_after_h1
            and current is not None
            and _is_italic_paragraph(paragraph)
            and (text.startswith("http://") or text.startswith("https://"))
        ):
            current.url = text
            expect_url_after_h1 = False
            continue

        if current is None:
            # Texte situe avant le premier H1 du document (en-tete global,
            # mentions legales, etc.). On cree un article-fantome pour ne
            # pas perdre ce contenu.
            current = Article(title=docx_path.stem)

        if current_section is not None:
            # On est a l'interieur d'une section H2 : le texte lui appartient.
            current_section.paragraphs.append(text)
        else:
            # On est apres le H1 mais avant tout H2 : c'est l'intro de l'article.
            current.intro_paragraphs.append(text)
        # Un paragraphe normal consomme la fenetre d'attente de l'URL.
        expect_url_after_h1 = False

    # Le dernier article en cours n'a pas encore ete sauvegarde (il n'y a
    # pas de H1 suivant pour declencher l'ajout).
    if current is not None:
        articles.append(current)

    return articles


# ---------------------------------------------------------------------------
# Datation d'un article
# ---------------------------------------------------------------------------

_UPDATED_RE = re.compile(r"jour au (?:\d{1,2})?\s*(?:er)?\s*[a-zéû]+\s+((?:19|20)\d{2})", re.I)
_URL_YEAR_RE = re.compile(r"(?:-|/)((?:19|20)\d{2})(?:/|$)")
_DECISION_DATE_RE = re.compile(r"\b\d{1,2}\.\d{1,2}\.(\d{2})\b")


def article_year(article: Article) -> int:
    """Estime l'annee de redaction ou de mise a jour d'un article (0 si inconnue)."""
    head = " ".join(article.intro_paragraphs[:3])
    if not head and article.sections:
        head = " ".join(article.sections[0].paragraphs[:3])
    head = head[:400]

    match = _UPDATED_RE.search(head)
    if match:
        return int(match.group(1))

    url_years = [int(y) for y in _URL_YEAR_RE.findall(article.url)]
    if url_years:
        return min(url_years)

    # Jurisprudence : la premiere date est celle de la decision. Ailleurs, elle
    # designe souvent le texte de loi cite, pas l'article : on ne l'utilise pas.
    match = _DECISION_DATE_RE.search(head)
    if match and "jurisprudence" in article.url:
        yy = int(match.group(1))
        return 2000 + yy if yy <= date.today().year % 100 else 1900 + yy

    return 0


def is_archived(article: Article) -> bool:
    return "archives" in article.url or "avant-" in article.url


# ---------------------------------------------------------------------------
# Decoupage hierarchique d'un article
# ---------------------------------------------------------------------------

def chunk_text_by_paragraphs(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> List[str]:
    """Decoupe un texte en chunks en accumulant les paragraphes.

    Reprise quasi-litterale de la fonction de reference fournie : on accumule
    paragraphe par paragraphe (separateur "\\n\\n") tant qu'on tient sous
    max_chars, puis on flush avec un overlap des `overlap_chars` derniers
    caracteres pour preserver la continuite entre chunks consecutifs.

    Note : si un paragraphe est lui-meme plus grand que max_chars, il sera
    place dans son propre chunk (qui depassera max_chars). C'est rare et
    acceptable, l'API OpenAI tolere jusqu'a ~30000 caracteres.
    """
    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # +2 pour le separateur "\n\n" qui sera insere entre les blocs.
        if len(current_chunk) + len(paragraph) + 2 <= max_chars:
            # Le paragraphe tient encore dans le chunk courant : on l'ajoute.
            current_chunk = (
                current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            )
        else:
            # Le chunk courant est plein : on le sauvegarde.
            if current_chunk:
                chunks.append(current_chunk)
            # On demarre un nouveau chunk avec un overlap sur la fin du
            # chunk precedent, pour que le modele de retrieval puisse
            # retrouver des informations a cheval entre deux chunks.
            overlap = current_chunk[-overlap_chars:] if current_chunk else ""
            current_chunk = (
                overlap + "\n\n" + paragraph if overlap else paragraph
            )

    # Ne pas oublier le dernier chunk accumule.
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _build_article_full_text(article: Article) -> str:
    """Assemble le texte plein d'un article : intro puis sections (H2 + corps).

    Les sections sans titre et sans corps sont ignorees pour eviter
    les chunks vides ou redondants.
    """
    parts: List[str] = []

    if article.intro_paragraphs:
        parts.append("\n\n".join(article.intro_paragraphs))

    for section in article.sections:
        section_body = "\n\n".join(section.paragraphs).strip()
        if section.title and section_body:
            # Titre de section suivi de son contenu.
            parts.append(section.title + "\n\n" + section_body)
        elif section.title:
            # Section sans corps (juste un titre H2).
            parts.append(section.title)
        elif section_body:
            # Corps sans titre (ne devrait pas arriver, mais on le garde).
            parts.append(section_body)

    return "\n\n".join(parts).strip()


def _make_prefix(article: Article, section_title: Optional[str]) -> str:
    """Construit l'entete 'Titre / URL / Section' qui precede chaque chunk.

    Cet entete est injecte dans chaque chunk pour que le modele de language
    dispose du contexte minimal (d'ou vient l'information) meme si le chunk
    est fourni isole, sans les chunks adjacents de l'article.
    """
    lines = [f"Titre : {article.title}"]
    if article.url:
        lines.append(f"URL : {article.url}")
    if section_title:
        # On n'affiche le champ Section que quand il y a un decoupage H2
        # effectif, pour ne pas ajouter de bruit dans les chunks complets.
        lines.append(f"Section : {section_title}")
    # Ligne vide finale pour separer visuellement l'entete du corps.
    return "\n".join(lines) + "\n\n"


def _build_chunk(
    *,
    source_file: str,
    article: Article,
    section_title: Optional[str],
    body: str,
    chunk_index: int,
) -> Chunk:
    """Construit un Chunk avec son texte enrichi et son ID deterministe.

    L'ID inclut l'URL et le contenu du chunk pour éviter les collisions.
    """
    text = _make_prefix(article, section_title) + body

    raw_id = "|".join([
        source_file,
        article.url or "",
        article.title,
        section_title or "",
        str(chunk_index),
        hashlib.sha1(text.encode("utf-8")).hexdigest(),
    ])

    chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:32]

    metadata = {
        "source_file": source_file,
        "article_title": article.title,
        "article_url": article.url or "",
        "section_title": section_title or "",
        "chunk_index": chunk_index,
        "char_count": len(text),
        "year": article_year(article),
        "archived": is_archived(article),
    }

    return Chunk(id=chunk_id, text=text, metadata=metadata)


def chunk_article(
    article: Article,
    source_file: str,
    max_chars: int,
    overlap_chars: int,
) -> List[Chunk]:
    """Applique la strategie hierarchique a un article.

    1. Article entier tient dans max_chars (en comptant le prefixe) : 1 chunk.
    2. Sinon : 1 chunk par section (intro + chaque H2). Si une section depasse,
       elle est elle-meme redecoupee par paragraphes avec overlap.
    """
    full_body = _build_article_full_text(article)
    if not full_body:
        # Article vide (ex: page de sommaire sans contenu) : on ne produit rien.
        return []

    # --- Niveau 1 : l'article entier tient en un seul chunk ---
    full_prefix = _make_prefix(article, section_title=None)
    if len(full_prefix) + len(full_body) <= max_chars:
        return [_build_chunk(
            source_file=source_file,
            article=article,
            section_title=None,
            body=full_body,
            chunk_index=0,
        )]

    # --- Niveau 2 : decoupage par section H2 ---
    chunks: List[Chunk] = []

    # On constitue la liste des sections a traiter : d'abord l'intro
    # (paragraphes avant le premier H2), puis chaque section H2.
    sections_to_process: List[tuple] = []  # (section_title, body_text)
    if article.intro_paragraphs:
        intro_body = "\n\n".join(article.intro_paragraphs).strip()
        if intro_body:
            # L'intro n'a pas de titre de section : on passe None.
            sections_to_process.append((None, intro_body))
    for section in article.sections:
        body = "\n\n".join(section.paragraphs).strip()
        if not body and not section.title:
            continue
        # Si la section n'a pas de corps, on utilise son titre comme contenu
        # pour ne pas produire un chunk vide.
        sections_to_process.append((section.title or None, body or section.title))

    # Index global incremente a travers toutes les sections de l'article,
    # ce qui garantit l'unicite de chunk_index meme apres redecoupage interne.
    global_idx = 0

    for section_title, body in sections_to_process:
        prefix = _make_prefix(article, section_title)
        # On deduit la longueur du prefixe du budget disponible pour le corps.
        # Le max(... , 200) evite un budget negatif si le prefixe est deja tres long.
        budget = max(max_chars - len(prefix), 200)

        if len(body) <= budget:
            # --- Niveau 2 : la section tient en un seul chunk ---
            chunks.append(_build_chunk(
                source_file=source_file,
                article=article,
                section_title=section_title,
                body=body,
                chunk_index=global_idx,
            ))
            global_idx += 1
        else:
            # --- Niveau 3 : redecoupage interne de la section par paragraphes ---
            # On utilise le budget calcule ci-dessus (pas max_chars) pour que
            # le chunk final (prefixe + corps) reste sous max_chars.
            sub_bodies = chunk_text_by_paragraphs(
                body, max_chars=budget, overlap_chars=overlap_chars
            )
            for sub_body in sub_bodies:
                chunks.append(_build_chunk(
                    source_file=source_file,
                    article=article,
                    section_title=section_title,
                    body=sub_body,
                    chunk_index=global_idx,
                ))
                global_idx += 1

    return chunks


# ---------------------------------------------------------------------------
# Iteration sur les .docx
# ---------------------------------------------------------------------------

def iter_docx_files(input_path: Path) -> Iterable[Path]:
    """Produit les chemins des fichiers .docx a traiter.

    Accepte un dossier (tous les .docx du dossier, tries par nom) ou
    un fichier .docx unique. Les fichiers temporaires Word (prefixe "~$")
    crees par Word lors de l'edition sont ignores.
    """
    if input_path.is_dir():
        for path in sorted(input_path.glob("*.docx")):
            # Word cree un fichier cache "~$nom.docx" pendant l'edition.
            # Ce fichier est vide ou corrompu et ferait planter python-docx.
            if path.name.startswith("~$"):
                continue
            yield path
    elif input_path.is_file() and input_path.suffix.lower() == ".docx":
        yield input_path
    else:
        raise FileNotFoundError(f"Aucun .docx trouve a partir de : {input_path}")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=CORPUS_DIR,
        help="Dossier contenant les .docx, ou un .docx unique (defaut : data/corpus)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=CHUNKS_FILE,
        help="Fichier .jsonl de sortie (defaut : data/chunks.jsonl)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=f"Taille max d'un chunk en caracteres (defaut : {DEFAULT_MAX_CHARS})",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=DEFAULT_OVERLAP_CHARS,
        help=(
            "Chevauchement en caracteres lors du redecoupage interne d'une section "
            f"(defaut : {DEFAULT_OVERLAP_CHARS})"
        ),
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help=(
            "Taille minimale d'un chunk pour qu'il soit conserve "
            f"(defaut : {DEFAULT_MIN_CHARS}). Mettre 0 pour desactiver."
        ),
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Affiche les 3 premiers chunks au lieu d'ecrire le fichier",
    )
    args = parser.parse_args()

    print(f"Lecture des .docx depuis : {args.input}")
    all_chunks: List[Chunk] = []
    n_articles = 0

    # On traite chaque .docx independamment pour pouvoir afficher
    # les statistiques par fichier et isoler les eventuelles erreurs.
    for docx_path in iter_docx_files(args.input):
        articles = parse_articles_from_docx(docx_path)
        n_articles += len(articles)
        file_chunks: List[Chunk] = []
        for article in articles:
            file_chunks.extend(chunk_article(
                article,
                source_file=docx_path.name,
                max_chars=args.max_chars,
                overlap_chars=args.overlap_chars,
            ))
        print(
            f"  {docx_path.name}: {len(articles)} article(s) -> "
            f"{len(file_chunks)} chunk(s)"
        )
        all_chunks.extend(file_chunks)

    if not all_chunks:
        print("Aucun chunk produit. Verifie la structure des .docx.")
        raise SystemExit(1)

    # Filtrage des chunks trop courts : ils apportent peu d'information au
    # modele et augmentent inutilement le nombre d'appels d'embedding.
    if args.min_chars > 0:
        before = len(all_chunks)
        all_chunks = [c for c in all_chunks if len(c.text) >= args.min_chars]
        dropped = before - len(all_chunks)
        if dropped:
            print(f"  {dropped} chunk(s) ecarte(s) car < {args.min_chars} caracteres")

    # Statistiques de distribution des tailles : utile pour ajuster
    # --max-chars et s'assurer que la plupart des chunks ne sont pas tronques.
    sizes = [len(c.text) for c in all_chunks]
    if sizes:
        avg = sum(sizes) / len(sizes)
        print(
            f"Total : {n_articles} article(s) -> {len(all_chunks)} chunk(s)  "
            f"(min={min(sizes)}, moy={avg:.0f}, max={max(sizes)} car.)"
        )

    if args.preview:
        # Mode apercu : on affiche les 3 premiers chunks sans ecrire de fichier.
        # Utile pour valider la structure avant de lancer un embedding couteux.
        print("\n--- Apercu des 3 premiers chunks ---")
        for ch in all_chunks[:3]:
            print(f"\nID : {ch.id}  ({ch.metadata['char_count']} car.)")
            print(f"  metadata : {ch.metadata}")
            print(f"  texte    :\n{ch.text[:500]}{'...' if len(ch.text) > 500 else ''}")
        return

    # Ecriture en JSONL : un chunk par ligne, encodage UTF-8 sans BOM
    # pour compatibilite maximale avec les pipelines de traitement.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(ch.to_jsonl() + "\n")
    print(f"Ecrit : {args.output} ({len(all_chunks)} chunk(s))")


if __name__ == "__main__":
    main()
