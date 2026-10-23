"""
check_tex.py — integridade do fonte LaTeX antes de compilar.

Varre o .tex por defeitos que o pdflatex NAO acusa: caracteres de controle
soltos, comandos truncados por escape comido (ver a memoria sobre \ref virando
carriage return), referencias sem label e citacoes sem bibitem.

Mora em src/ por ser generico: recebe o .tex como argumento e serve a qualquer
artigo. Ficou em papers/gtdla/scripts/ ate 19/09/2026 por arrasto da separacao
do commit b9da297, que moveu em bloco os sete scripts do artigo submetido; este
nao confere texto contra numeros como os outros, so valida sintaxe, entao
pertence a camada compartilhada. Corrigir aqui vale para os dois artigos.

Sai com codigo 1 quando encontra defeito que quebra o documento --- e para isso
que serve o alvo de make. Label sem referencia e bibitem sem citacao sao avisos
e nao derrubam o build.

Uso::

    python src/check_tex.py papers/journal/main.tex
"""
import pathlib
import re
import sys

falhas = []

p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "papers/gtdla/main.tex")
b = p.read_bytes()
s = b.decode("utf-8")

print(f"== {p} ==")
cr = b.count(b"\r") - b.count(b"\r\n")
print("CR solitario:", cr)
if cr:
    falhas.append(f"{cr} CR solitario")
for code, name in ((9, "TAB"), (8, "BS"), (12, "FF"), (11, "VT"), (7, "BEL")):
    n = b.count(bytes([code]))
    if n:
        print(f"  controle {name}: {n}")
        if name != "TAB":          # TAB e legitimo em alguns fontes; os outros nao
            falhas.append(f"{n} caractere(s) de controle {name}")

# comandos cujo primeiro caractere pode ter sido comido por um escape (\r, \t, \n, \b, \f, \v, \a)
FRAG = ["ef", "extbf", "extit", "esizebox", "oprule", "ottomrule", "idrule", "abular", "ext",
        "egin", "ibitem", "ootnote", "rac", "ightarrow", "ime", "iny", "op", "ow", "ule"]
bad = []
for f in FRAG:
    for m in re.finditer(r"(?<![A-Za-z\\])" + f + r"[{\[~]", s):
        i = m.start()
        if i == 0 or s[i - 1] != "\\":
            bad.append((i, s[max(0, i - 40):i + 25].replace("\n", "|")))
if bad:
    print("SUSPEITAS de comando truncado:")
    for i, ctx in bad:
        print(f"  pos {i}: ...{ctx}...")
    falhas.append(f"{len(bad)} comando(s) truncado(s)")
else:
    print("nenhum comando truncado")

# referencias declaradas x usadas
labels = set(re.findall(r"\\label\{([^}]+)\}", s))
refs = set(re.findall(r"\\ref\{([^}]+)\}", s)) | set(re.findall(r"\\eqref\{([^}]+)\}", s))
print("labels:", len(labels), "| refs:", len(refs))
if refs - labels:
    print("  REF SEM LABEL:", sorted(refs - labels))
    falhas.append(f"{len(refs - labels)} referencia(s) sem label")
if labels - refs:
    print("  label nao referenciado:", sorted(labels - refs))

cites = set(re.findall(r"\\cite\{([^}]+)\}", s))
cited = {c.strip() for grp in cites for c in grp.split(",")}
items = set(re.findall(r"\\bibitem\{([^}]+)\}", s))
if cited - items:
    print("  CITE SEM BIBITEM:", sorted(cited - items))
    falhas.append(f"{len(cited - items)} citacao(oes) sem bibitem")
if items - cited:
    print("  bibitem nao citado:", sorted(items - cited))
print("citacoes:", len(cited), "| bibitems:", len(items))

if falhas:
    print("\nFALHOU: " + "; ".join(falhas))
    sys.exit(1)
print("\nOK")
