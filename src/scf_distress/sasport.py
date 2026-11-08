"""Run the Fed's SCF summary-variable SAS macro (``bulletin.macro.txt``) in Python.

The macro is one record-by-record SAS DATA step, with macro-level branches on the
survey year. This module

1. resolves the macro layer for a given year with ``PUBLIC=YES`` (``%IF``/``%ELSE``,
   ``%LET``, ``%EVAL``, ``&var`` substitution, the ``%MCONV`` helper macro),
2. parses the resulting DATA-step statements (assignments, ``IF/THEN/ELSE``,
   ``DO`` blocks and ``DO I=1 TO ...`` loops, ``ARRAY``, subsetting ``IF``), and
3. executes them vectorized over all records with numpy, using masks for
   conditional branches.

SAS semantics that matter here and are reproduced:
* numeric missing is NaN; arithmetic propagates it; ``MAX``/``MIN``/``SUM``
  ignore missing arguments; division by zero gives missing;
* in comparisons missing sorts below every number (``. < 0`` is true, ``. = .``
  is true); special missings ``.B``/``.Z`` are treated as plain missing;
* an ``IF`` condition is true when non-missing and non-zero;
* chained comparisons (``10 <= X < 25``) mean the conjunction;
* variables created in the step start missing on every record;
* operator precedence follows SAS (``**`` and prefix operators bind tightest,
  then ``* /``, ``+ -``, comparisons and ``IN``, ``AND``, ``OR``).

``PUT``, ``CALL``, ``LENGTH``, ``DROP``, ``KEEP`` and ``SET`` statements are
ignored; they do not change values in the public-data path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

MACRO_PATH = Path(__file__).resolve().parents[2] / "docs" / "bulletin.macro.txt"

# --------------------------------------------------------------------------
# Macro layer
# --------------------------------------------------------------------------


def _split_statements(text: str) -> list[str]:
    """Split SAS source into statements; drop /* */, '* ...;' and '%* ...;' comments."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    out, i, n = [], 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        if text.startswith("*", i) or text.startswith("%*", i):  # comment statement
            j = text.find(";", i)
            i = n if j < 0 else j + 1
            continue
        buf, q = [], None
        while i < n:
            c = text[i]
            if q:
                if c == q:
                    q = None
            elif c in "'\"":
                q = c
            elif c == ";":
                i += 1
                break
            buf.append(c)
            i += 1
        stmt = " ".join("".join(buf).split())
        if stmt:
            out.append(stmt)
    return out


def _eval_macro_condition(cond: str, mvars: dict) -> bool:
    cond = _substitute(cond, mvars)
    toks = re.findall(r"\d+(?:\.\d+)?|[A-Za-z_]\w*|<=|>=|\^=|~=|=|<|>|&|\||\(|\)", cond)
    py = []
    ops = {"GE": ">=", "LE": "<=", "GT": ">", "LT": "<", "EQ": "==", "NE": "!=", "=": "==",
           "^=": "!=", "~=": "!=", "AND": "and", "&": "and", "OR": "or", "|": "or", "NOT": "not"}
    for t in toks:
        u = t.upper()
        if u in ops:
            py.append(ops[u])
        elif t in "()" or t in ("<=", ">=", "<", ">"):
            py.append(t)
        elif re.fullmatch(r"\d+(?:\.\d+)?", t):
            py.append(t)
        else:
            py.append(repr(u))
    return bool(eval(" ".join(py), {"__builtins__": {}}, {}))


def _substitute(s: str, mvars: dict) -> str:
    """Resolve &VAR references (longest name match), then %EVAL(...)."""
    for _ in range(10):
        new = re.sub(r"&([A-Za-z_]\w*)\.?",
                     lambda m: str(mvars[m.group(1).upper()]) if m.group(1).upper() in mvars else m.group(0), s)
        if new == s:
            break
        s = new
    if re.search(r"&[A-Za-z_]", s):
        raise KeyError(f"Unresolved macro variable in: {s[:120]}")

    def _ev(m):
        expr = m.group(1)
        if not re.fullmatch(r"[\d\s+\-*/()]+", expr):
            raise ValueError(f"Unsupported %EVAL: {expr}")
        return str(int(eval(expr, {"__builtins__": {}}, {})))

    return re.sub(r"%EVAL\(([^()]*)\)", _ev, s, flags=re.I)


def _expand_mconv(s: str, body: str) -> str:
    """Inline %MCONV(F) / %MCONV(F=expr)."""
    def rep(m):
        arg = m.group(1).strip()
        arg = re.sub(r"^F\s*=\s*", "", arg, flags=re.I)
        return body.replace("&F", f"({arg})")
    return re.sub(r"%MCONV\(([^()]*)\)", rep, s, flags=re.I)


def _macro_constants(src: str, year: int) -> dict:
    m = re.search(r"%IF\s*\(&YEAR=" + str(year) + r"\)\s*%THEN\s*%DO;\s*%LET CPIADJ=&CPIBASE/(\d+);"
                  r"\s*%LET CPILAG=(\d+)/(\d+);", src) or \
        re.search(r"%ELSE\s*%IF\s*\(&YEAR=" + str(year) + r"\)\s*%THEN\s*%DO;\s*%LET CPIADJ=&CPIBASE/(\d+);"
                  r"\s*%LET CPILAG=(\d+)/(\d+);", src)
    if not m:
        raise ValueError(f"No CPI constants for {year} in macro")
    return {"cpi_year": int(m.group(1)), "cpilag": (int(m.group(2)), int(m.group(3)))}


def resolve_macro(year: int, cpibase: int, src: str | None = None) -> list[str]:
    """Return the active DATA-step statements of BULLIT for `year`, PUBLIC=YES."""
    src = src if src is not None else MACRO_PATH.read_text(errors="ignore")
    mconv = re.search(r"%MACRO MCONV\(F\);(.*?)%MEND MCONV;", src, re.S).group(1)
    mconv = " ".join(mconv.split())
    k = _macro_constants(src, year)
    mvars = {"YEAR": year, "PUBLIC": "YES", "REAL": "YES", "ADJINC": "YES", "CPIBASE": cpibase,
             "CPIADJ": f"({cpibase}/{k['cpi_year']})", "CPILAG": f"({k['cpilag'][0]}/{k['cpilag'][1]})",
             "ID": "Y1", "IID": "YY1", "PID": "Y1", "PIID": "YY1"}

    start = src.index("DATA %UNQUOTE(SCFB&YEAR)(KEEP=&KEEPLIST);")
    end = src.index("RUN;", src.index("INCCL2=5;", start))
    stmts = _split_statements(src[start:end])[1:]  # drop the DATA statement

    out: list[str] = []
    stack: list[dict] = []  # frames: active, taken
    last_closed = None

    def parent_active():
        return all(f["active"] for f in stack)

    for s in stmts:
        u = s.upper()
        m_if = re.fullmatch(r"%IF\s*(.*?)\s*%THEN\s*%DO", s, re.I | re.S)
        m_elif = re.fullmatch(r"%ELSE\s*%IF\s*(.*?)\s*%THEN\s*%DO", s, re.I | re.S)
        if m_if:
            pa = parent_active()
            c = pa and _eval_macro_condition(m_if.group(1), mvars)
            stack.append({"active": c, "taken": c})
            last_closed = None
        elif m_elif:
            if last_closed is None:
                raise SyntaxError(f"%ELSE without %IF: {s}")
            pa = parent_active()
            c = pa and not last_closed["taken"] and _eval_macro_condition(m_elif.group(1), mvars)
            stack.append({"active": c, "taken": last_closed["taken"] or c})
            last_closed = None
        elif re.fullmatch(r"%ELSE\s*%DO", s, re.I):
            if last_closed is None:
                raise SyntaxError("%ELSE %DO without %IF")
            c = parent_active() and not last_closed["taken"]
            stack.append({"active": c, "taken": True})
            last_closed = None
        elif u == "%END":
            last_closed = stack.pop()
        else:
            last_closed = None
            if not parent_active():
                continue
            if u.startswith("%LET"):
                name, val = s[4:].split("=", 1)
                mvars[name.strip().upper()] = _substitute(val.strip(), mvars)
                continue
            if u.startswith("%INCLUDE"):
                raise RuntimeError(f"%INCLUDE in active public-data path: {s}")
            if _is_ignored(u):
                continue
            s = _substitute(_expand_mconv(s, mconv), mvars)
            if "%" in re.sub(r"'[^']*'|\"[^\"]*\"", "", s):
                raise SyntaxError(f"Unresolved macro code: {s[:150]}")
            out.append(s)
    if stack:
        raise SyntaxError("Unbalanced %DO/%END")
    return out


# --------------------------------------------------------------------------
# DATA-step parser
# --------------------------------------------------------------------------

_TOKEN = re.compile(r"""
    (?P<num>\d+\.\d*(?:E[+-]?\d+)?|\.\d+(?:E[+-]?\d+)?|\d+(?:E[+-]?\d+)?)
  | (?P<miss>\.[A-Z_]?(?![A-Z0-9_]))
  | (?P<name>[A-Z_][A-Z0-9_]*)
  | (?P<str>'[^']*'|"[^"]*")
  | (?P<op>\*\*|<=|>=|\^=|~=|¬=|[=<>+\-*/(){},&|^~:])
""", re.X | re.I)

_CMP_WORDS = {"EQ": "=", "NE": "^=", "GT": ">", "LT": "<", "GE": ">=", "LE": "<=", "~=": "^=", "¬=": "^="}
_CMP_OPS = {"=", "^=", "<", ">", "<=", ">="}


def tokenize(s: str) -> list[tuple[str, str]]:
    toks, pos = [], 0
    s = s.strip()
    while pos < len(s):
        if s[pos].isspace():
            pos += 1
            continue
        m = _TOKEN.match(s, pos)
        if not m:
            raise SyntaxError(f"Cannot tokenize at: {s[pos:pos + 40]!r} in {s[:100]!r}")
        kind = m.lastgroup
        val = m.group(kind)
        if kind in ("name", "miss"):
            val = val.upper()
        toks.append((kind, val))
        pos = m.end()
    return toks


@dataclass
class Node:
    kind: str
    a: object = None
    b: object = None
    c: object = None
    args: list = field(default_factory=list)


class ExprParser:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self, k=0):
        j = self.i + k
        return self.t[j] if j < len(self.t) else (None, None)

    def take(self, val=None):
        tok = self.peek()
        if val is not None and (tok[1] is None or tok[1].upper() != val):
            raise SyntaxError(f"Expected {val}, got {tok} in {self.t}")
        self.i += 1
        return tok

    def at(self, *vals):
        v = self.peek()[1]
        return v is not None and v.upper() in vals

    def parse(self):
        e = self.or_()
        if self.i != len(self.t):
            raise SyntaxError(f"Trailing tokens {self.t[self.i:]} in {self.t}")
        return e

    def or_(self):
        e = self.and_()
        while self.at("|", "OR"):
            self.take()
            e = Node("or", e, self.and_())
        return e

    def and_(self):
        e = self.cmp()
        while self.at("&", "AND"):
            self.take()
            e = Node("and", e, self.cmp())
        return e

    def _cmp_op(self):
        v = self.peek()[1]
        if v is None:
            return None
        u = v.upper()
        u = _CMP_WORDS.get(u, u)
        return u if u in _CMP_OPS else None

    def cmp(self):
        first = self.add()
        if self.at("IN") or (self.at("NOT") and (self.peek(1)[1] or "").upper() == "IN"):
            neg = False
            if self.at("NOT"):
                self.take()
                neg = True
            self.take("IN")
            vals = self._in_list()
            node = Node("in", first, vals)
            return Node("not", node) if neg else node
        parts, ops = [first], []
        while (op := self._cmp_op()) is not None:
            self.take()
            ops.append(op)
            parts.append(self.add())
        if not ops:
            return first
        node = Node("cmp", ops[0], parts[0], parts[1])
        for k in range(1, len(ops)):
            node = Node("and", node, Node("cmp", ops[k], parts[k], parts[k + 1]))
        return node

    def _in_list(self):
        self.take("(")
        vals = []
        while not self.at(")"):
            if self.at(","):
                self.take()
                continue
            sign = 1.0
            if self.at("-"):
                self.take()
                sign = -1.0
            elif self.at("+"):
                self.take()
            kind, v = self.take()
            if kind == "num":
                vals.append(sign * float(v))
            elif kind == "miss":
                vals.append(-np.inf)
            else:
                raise SyntaxError(f"Unsupported IN element {v}")
        self.take(")")
        return vals

    def add(self):
        e = self.mul()
        while self.at("+", "-"):
            op = self.take()[1]
            e = Node("bin", op, e, self.mul())
        return e

    def mul(self):
        e = self.unary()
        while self.at("*", "/"):
            op = self.take()[1]
            e = Node("bin", op, e, self.unary())
        return e

    def unary(self):
        if self.at("-"):
            self.take()
            return Node("neg", self.unary())
        if self.at("+"):
            self.take()
            return self.unary()
        if self.at("NOT", "^", "~"):
            self.take()
            return Node("not", self.unary())
        return self.power()

    def power(self):
        base = self.primary()
        if self.at("**"):
            self.take()
            return Node("bin", "**", base, self.unary())
        return base

    def primary(self):
        kind, v = self.take()
        if kind == "num":
            return Node("num", float(v))
        if kind == "miss":
            return Node("num", np.nan)
        if v == "(":
            e = self.or_()
            self.take(")")
            return e
        if kind == "name":
            if self.at("("):
                self.take("(")
                args = []
                while not self.at(")"):
                    args.append(self.or_())
                    if self.at(","):
                        self.take()
                self.take(")")
                return Node("call", v, args=args)
            if self.at("{"):
                self.take("{")
                idx = self.or_()
                self.take("}")
                return Node("aref", v, idx)
            return Node("var", v)
        raise SyntaxError(f"Unexpected token {v!r} in {self.t}")


def parse_expr(s: str) -> Node:
    return ExprParser(tokenize(s)).parse()


_IGNORED = ("PUT", "CALL", "LENGTH", "DROP", "KEEP", "SET", "LABEL", "FORMAT", "RETAIN")


def _is_ignored(stmt_upper: str) -> bool:
    """True for non-computational statements (PUT ..., CALL VNAME(...)), but not for an
    assignment to a variable that shares the keyword's name (e.g. CALL=... for call accounts)."""
    m = re.match(r"([A-Z_]+)\b", stmt_upper)
    if not m or m.group(1) not in _IGNORED:
        return False
    return not re.match(r"[A-Z_]\w*\s*(\{[^}]*\})?\s*=(?!=)", stmt_upper)


def _split_then(s: str):
    """Split 'IF cond THEN rest' at the top-level THEN; returns (cond, rest) or (cond, None)."""
    depth, i, u = 0, 0, s.upper()
    while i < len(s):
        c = s[i]
        if c in "({":
            depth += 1
        elif c in ")}":
            depth -= 1
        elif depth == 0 and u.startswith("THEN", i) and (i == 0 or not (u[i - 1].isalnum() or u[i - 1] == "_")) \
                and (i + 4 == len(u) or not (u[i + 4].isalnum() or u[i + 4] == "_")):
            return s[2:i].strip(), s[i + 4:].strip()
        i += 1
    return s[2:].strip(), None


def _expand_varlist(spec: str) -> list[str]:
    names = []
    for tok in spec.split():
        m = re.fullmatch(r"([A-Z_]+?)(\d+)-([A-Z_]+?)(\d+)", tok.upper())
        if m and m.group(1) == m.group(3):
            names += [f"{m.group(1)}{k}" for k in range(int(m.group(2)), int(m.group(4)) + 1)]
        else:
            names.append(tok.upper())
    return names


class StepParser:
    def __init__(self, stmts):
        self.s, self.i = stmts, 0

    def parse_all(self):
        out = self.block(top=True)
        return out

    def block(self, top=False):
        out = []
        while self.i < len(self.s):
            st = self.s[self.i]
            if st.upper() == "END":
                if top:
                    raise SyntaxError("Unmatched END")
                self.i += 1
                return out
            self.i += 1
            node = self.statement(st)
            if node is not None:
                out.append(node)
        if not top:
            raise SyntaxError("Missing END")
        return out

    def statement(self, st: str):
        u = st.upper()
        first = re.match(r"[A-Z_%]+", u)
        first = first.group(0) if first else ""
        if _is_ignored(u):
            return None
        if first == "ELSE":
            raise SyntaxError(f"ELSE without IF: {st}")
        if first == "IF" and re.match(r"IF\b", u):
            cond, rest = _split_then(st)
            if rest is None:
                return Node("subset", parse_expr(cond))
            then = self._body(rest)
            els = None
            if self.i < len(self.s) and re.match(r"ELSE\b", self.s[self.i].upper()):
                rest_else = self.s[self.i][4:].strip()
                self.i += 1
                els = self._body(rest_else)
            return Node("if", parse_expr(cond), then, els)
        if first == "DO" and re.match(r"DO\b", u):
            if u == "DO":
                return Node("block", self.block())
            m = re.fullmatch(r"DO\s+([A-Z_]\w*)\s*=\s*(.+?)\s+TO\s+(.+)", st, re.I)
            if not m:
                raise SyntaxError(f"Unsupported DO: {st}")
            return Node("loop", m.group(1).upper(), (m.group(2), m.group(3)), self.block())
        if first == "ARRAY":
            m = re.fullmatch(r"ARRAY\s+([A-Z_]\w*)\s*[{(\[]\s*[*\d]+\s*[})\]]\s*(.*)", st, re.I)
            if not m:
                raise SyntaxError(f"Unsupported ARRAY: {st}")
            return Node("array", m.group(1).upper(), _expand_varlist(m.group(2)))
        m = re.fullmatch(r"([A-Z_]\w*)\s*(\{[^}]*\})?\s*=(?!=)\s*(.+)", st, re.I | re.S)
        if m:
            target = m.group(1).upper()
            idx = parse_expr(m.group(2)[1:-1]) if m.group(2) else None
            return Node("assign", target, idx, parse_expr(m.group(3)))
        raise SyntaxError(f"Unsupported statement: {st}")

    def _body(self, rest: str):
        if rest.upper() == "DO":
            return Node("block", self.block())
        node = self.statement(rest)
        return Node("block", [node] if node is not None else [])


# --------------------------------------------------------------------------
# Vectorized executor
# --------------------------------------------------------------------------

def _fill(x):
    """Missing sorts lowest in SAS comparisons."""
    x = np.asarray(x, dtype=float)
    return np.where(np.isnan(x), -np.inf, x)


def _truth(x):
    x = np.asarray(x, dtype=float)
    return (~np.isnan(x)) & (x != 0)


class Executor:
    def __init__(self, data: dict[str, np.ndarray], n: int):
        self.V = data
        self.n = n
        self.arrays: dict[str, list[str]] = {}
        self.loopvars: dict[str, int] = {}
        self.keep = np.ones(n, dtype=bool)
        self.uninitialized: set[str] = set()

    # ---- expressions
    def ev(self, e: Node):
        k = e.kind
        if k == "num":
            return e.a
        if k == "var":
            if e.a in self.loopvars:
                return float(self.loopvars[e.a])
            if e.a not in self.V:
                self.uninitialized.add(e.a)
                self.V[e.a] = np.full(self.n, np.nan)
            return self.V[e.a]
        if k == "aref":
            return self.ev(Node("var", self._resolve(e.a, e.b)))
        if k == "bin":
            x, y = self.ev(e.b), self.ev(e.c)
            op = e.a
            with np.errstate(all="ignore"):
                if op == "+":
                    return np.add(x, y)
                if op == "-":
                    return np.subtract(x, y)
                if op == "*":
                    return np.multiply(x, y)
                if op == "/":
                    y_arr = np.asarray(y, dtype=float)
                    r = np.divide(x, y_arr)
                    return np.where(y_arr == 0, np.nan, r)
                if op == "**":
                    return np.power(x, y)
        if k == "neg":
            return np.negative(self.ev(e.a))
        if k == "cmp":
            x, y = _fill(self.ev(e.b)), _fill(self.ev(e.c))
            op = e.a
            r = {"=": x == y, "^=": x != y, "<": x < y, ">": x > y, "<=": x <= y, ">=": x >= y}[op]
            return r.astype(float)
        if k == "in":
            x = _fill(self.ev(e.a))
            return np.isin(x, np.array(e.b)).astype(float)
        if k == "and":
            return (_truth(self.ev(e.a)) & _truth(self.ev(e.b))).astype(float)
        if k == "or":
            return (_truth(self.ev(e.a)) | _truth(self.ev(e.b))).astype(float)
        if k == "not":
            return (~_truth(self.ev(e.a))).astype(float)
        if k == "call":
            return self._call(e.a, e.args)
        raise ValueError(f"Bad node {k}")

    def _call(self, fn, args):
        if fn == "DIM":
            return float(len(self.arrays[args[0].a]))
        vals = [np.broadcast_to(np.asarray(self.ev(a), dtype=float), (self.n,)) for a in args]
        with np.errstate(all="ignore"):
            if fn in ("MAX", "MIN"):
                f = np.fmax if fn == "MAX" else np.fmin
                out = vals[0]
                for v in vals[1:]:
                    out = f(out, v)
                return out
            if fn == "SUM":
                stack = np.vstack(vals)
                allnan = np.isnan(stack).all(axis=0)
                return np.where(allnan, np.nan, np.nansum(stack, axis=0))
            if fn == "INT":
                return np.trunc(vals[0])
            if fn == "ROUND":
                unit = vals[1] if len(vals) > 1 else 1.0
                q = vals[0] / unit
                return np.sign(q) * np.floor(np.abs(q) + 0.5) * unit
        raise NotImplementedError(f"SAS function {fn}")

    def _resolve(self, arr, idx_node):
        idx = self.ev(idx_node)
        if not np.isscalar(idx):
            raise ValueError(f"Non-scalar subscript for {arr}")
        names = self.arrays[arr]
        return names[int(idx) - 1]

    # ---- statements
    def run(self, nodes, mask=None):
        mask = np.ones(self.n, dtype=bool) if mask is None else mask
        for nd in nodes:
            self.stmt(nd, mask)

    def stmt(self, nd: Node, mask):
        k = nd.kind
        if k == "assign":
            name = self._resolve(nd.a, nd.b) if nd.b is not None else nd.a
            val = np.broadcast_to(np.asarray(self.ev(nd.c), dtype=float), (self.n,))
            cur = self.V.get(name)
            if cur is None:
                cur = np.full(self.n, np.nan)
            self.V[name] = np.where(mask, val, cur)
        elif k == "if":
            t = _truth(self.ev(nd.a))
            self.run(nd.b.a, mask & t)
            if nd.c is not None:
                self.run(nd.c.a, mask & ~t)
        elif k == "block":
            self.run(nd.a, mask)
        elif k == "loop":
            lo, hi = (self.ev(parse_expr(x)) for x in nd.b)
            for i in range(int(lo), int(hi) + 1):
                self.loopvars[nd.a] = i
                self.run(nd.c, mask)
            self.loopvars.pop(nd.a, None)
        elif k == "array":
            self.arrays[nd.a] = nd.b
        elif k == "subset":
            self.keep &= ~(mask & ~_truth(self.ev(nd.a)))
        else:
            raise ValueError(k)


def referenced_names(nodes) -> set[str]:
    names: set[str] = set()

    def walk(x):
        if isinstance(x, Node):
            if x.kind == "var":
                names.add(x.a)
            if x.kind == "array":
                names.update(x.b)
            if x.kind == "assign":
                names.add(x.a)
            for ch in (x.a, x.b, x.c, *x.args):
                walk(ch)
        elif isinstance(x, (list, tuple)):
            for ch in x:
                walk(ch)

    walk(nodes)
    return names


@dataclass
class BullitResult:
    data: pd.DataFrame
    uninitialized: set
    n_statements: int


def compile_year(year: int, cpibase: int):
    stmts = resolve_macro(year, cpibase)
    return stmts, StepParser(stmts).parse_all()


def run_bullit(raw: pd.DataFrame, year: int, cpibase: int, program=None) -> BullitResult:
    """Execute the macro for one wave. `raw` holds the needed X-variables (any case)."""
    stmts, prog = program if program is not None else compile_year(year, cpibase)
    n = len(raw)
    data = {c.upper(): raw[c].to_numpy(dtype=float, copy=True) for c in raw.columns}
    ex = Executor(data, n)
    ex.run(prog)
    out = pd.DataFrame({k: v for k, v in ex.V.items() if isinstance(v, np.ndarray) and v.shape == (n,)})
    out = out.loc[ex.keep].reset_index(drop=True)
    return BullitResult(out, ex.uninitialized, len(stmts))
