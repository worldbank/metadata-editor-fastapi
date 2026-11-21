"""Quoted evidence must come from the text, without being thrown off by PDF extraction noise."""

from src.assistant.host.tools import evidence_is_grounded

PASSAGE = (
    "The first set of weights are those for the combined wave 4 sample. They can be used for cross-sectional "
    "analysis for the full GHS-Panel wave 4 sample (refresh plus long panel sample). The second set of weights "
    "are designed for longitudinal analysis using the long panel sample only."
)


def doc(*parts: str) -> str:
    return "\n\n".join(parts)


def test_exact_and_reformatted_quotes_are_grounded():
    text = doc("Intro.", PASSAGE, "Outro.")
    assert evidence_is_grounded("The second set of weights are designed for longitudinal analysis", text)
    assert evidence_is_grounded("the SECOND set  of weights, are designed for longitudinal analysis.", text)  # case, spacing, comma


def test_a_line_break_hyphen_and_a_page_header_inside_a_quote_do_not_reject_it():
    noisy = PASSAGE.replace("cross-sectional analysis", "cross-sec-\ntional analysis").replace(
        "(refresh plus", "\n\nNigeria GHS-Panel 2018-19 | Page 41\n\n(refresh plus"
    )
    text = doc("Intro.", noisy)
    assert evidence_is_grounded(PASSAGE, text)


def test_an_ellipsis_between_two_quoted_sentences_is_two_quotes():
    text = doc("The ESS4 sample is a two-stage stratified probability sample.", "Filler paragraph. " * 30,
               "The second stage of sampling was to use systematic random sampling to select households.")
    evidence = "The ESS4 sample is a two-stage stratified probability sample. ... The second stage of sampling was to use systematic random sampling to select households."
    assert evidence_is_grounded(evidence, text)


def test_a_paraphrase_or_invented_quote_is_refused():
    text = doc("Intro.", PASSAGE)
    invented = ("Two weighting schemes were produced by the statistics office, one for pooled analysis of all "
                "households interviewed and another for tracking the same households across visits over time.")
    assert not evidence_is_grounded(invented, text)
    assert not evidence_is_grounded("The weights were computed by the World Bank in 2021", text)  # short, not in text


def test_one_bad_fragment_still_rejects_the_whole_evidence():
    text = doc("Intro.", PASSAGE)
    assert not evidence_is_grounded("The second set of weights are designed for longitudinal analysis; the sample had 9,999 clusters", text)


def test_empty_evidence_is_not_grounded():
    assert not evidence_is_grounded("", "anything")
    assert not evidence_is_grounded(" ... ", "anything")
