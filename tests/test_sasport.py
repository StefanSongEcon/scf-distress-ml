"""SAS-semantics tests for the macro translator, plus a check against the Fed extract when data exist."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scf_distress import sasport

ROOT = Path(__file__).resolve().parents[1]


def run(code: str, **cols):
    stmts = sasport._split_statements(code)
    prog = sasport.StepParser(stmts).parse_all()
    n = len(next(iter(cols.values())))
    ex = sasport.Executor({k.upper(): np.asarray(v, dtype=float) for k, v in cols.items()}, n)
    ex.run(prog)
    return ex


def test_missing_sorts_low_and_propagates():
    ex = run("A = (X < 0); B = X + 1; C = MAX(0, X); D = SUM(X, 2); E = (X = .);", x=[np.nan, 3])
    assert ex.V["A"].tolist() == [1, 0]
    assert np.isnan(ex.V["B"][0]) and ex.V["B"][1] == 4
    assert ex.V["C"].tolist() == [0, 3]
    assert ex.V["D"].tolist() == [2, 5]
    assert ex.V["E"].tolist() == [1, 0]


def test_if_else_chain_and_new_vars_start_missing():
    ex = run("IF X<10 THEN K=1; ELSE IF 10<=X<25 THEN K=2; ELSE K=3; IF X>100 THEN Z=1;", x=[5, 10, 24, 25, 500])
    assert ex.V["K"].tolist() == [1, 2, 2, 3, 3]
    assert np.isnan(ex.V["Z"][:4]).all() and ex.V["Z"][4] == 1


def test_condition_uses_values_before_then_branch_runs():
    ex = run("IF X=1 THEN DO; X=2; Y=1; END; ELSE Y=0;", x=[1, 2])
    assert ex.V["X"].tolist() == [2, 2]
    assert ex.V["Y"].tolist() == [1, 0]


def test_arrays_loops_and_ranges():
    ex = run("ARRAY A{*} V1-V3; T=0; DO I=1 TO DIM(A); IF A{I}=14 THEN A{I}=18; T=T+MAX(0,A{I}); END;",
             v1=[14, 1], v2=[2, -1], v3=[3, 14])
    assert ex.V["V1"].tolist() == [18, 1]
    assert ex.V["T"].tolist() == [23, 19]


def test_in_not_in_and_division_by_zero():
    ex = run("A=(X IN (1, 2)); B=(X NOT IN (0 . -9)); C=1/X;", x=[1, 0, -9, np.nan, 3])
    assert ex.V["A"].tolist() == [1, 0, 0, 0, 0]
    assert ex.V["B"].tolist() == [1, 0, 0, 0, 1]
    assert np.isnan(ex.V["C"][1])


def test_precedence_and_power():
    ex = run("A = 2 + 3 * 4 ** 2; B = (1.08**(1/12)-1)*1200; C = -X + 1;", x=[2])
    assert ex.V["A"][0] == 50
    assert ex.V["B"][0] == pytest.approx((1.08 ** (1 / 12) - 1) * 1200)
    assert ex.V["C"][0] == -1


def test_keyword_named_variable_is_assigned():
    ex = run("CALL = 5 + X; PUT 'HELLO' X=;", x=[1])
    assert ex.V["CALL"][0] == 6


def test_macro_branches_resolve_by_year():
    src = """%MACRO MCONV(F); ((&F=4)) %MEND MCONV;
    %MACRO BULLIT(YEAR=,REAL=YES,CPIBASE=,PUBLIC=NO,ADJINC=YES);
    %IF (&YEAR=2016) %THEN %DO; %LET CPILAG=1/1; %END;
    %ELSE %IF (&YEAR=2019) %THEN %DO; %LET CPIADJ=&CPIBASE/3775; %LET CPILAG=3758/3691;
      %IF (&PUBLIC NE YES) %THEN %LET SCFDS=SCF19I6;
      %ELSE %LET SCFDS=P19I6;
    %END;
    %IF (&YEAR=2009) %THEN %DO; %LET V14=P14; %END;
    %ELSE %IF (&YEAR=2019) %THEN %DO; %LET V14=X14; %END;
    %IF (&REAL EQ YES AND &ADJINC NE YES) %THEN %PUT WARNING!;
    DATA %UNQUOTE(SCFB&YEAR)(KEEP=&KEEPLIST);
      %IF (&YEAR GE 2016) %THEN %DO; A=1; %END;
      %ELSE %IF (&YEAR GE 2010) %THEN %DO; A=2; %END;
      %ELSE %DO; A=3; %END;
      * comment with %IF inside;
      B=%MCONV(F=X) * &CPILAG;
      AGE=&V14;
      C=&SCFDS;
      INCCL2=5;
    RUN;"""
    stmts = sasport.resolve_macro(2019, 4789, src)
    assert stmts == ["A=1", "B=(((X)=4)) * 3758/3691", "AGE=X14", "C=P19I6", "INCCL2=5"]


@pytest.mark.parametrize("year", [2010, 2013, 2016, 2019, 2025])
def test_port_matches_fed_extract(year):
    from scf_distress import prepare, schema
    try:
        schema.wave_file(ROOT / "data/raw", year)
    except FileNotFoundError:
        pytest.skip("raw SCF data not available")
    ext_path = ROOT / "data/external/summary_extract" / str(year) / f"SCFP{year}.csv"
    if not ext_path.exists():
        pytest.skip("summary extract not available")
    built, _ = prepare.build_wave(ROOT / "data/raw", year)
    cmp = prepare.compare_to_extract(built, prepare.read_extract(ROOT / "data/external/summary_extract", year))
    bad = cmp[cmp["status"].isin(["mismatch", "missing_in_build"])]
    assert bad.empty, bad.to_string()
