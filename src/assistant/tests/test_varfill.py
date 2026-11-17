"""Variable labelling from a codebook: line reading, model fallback, guards and the job."""

import asyncio
import json

from src.assistant.core.config import parse_registry
from src.assistant.host.varfill import parse_codebook, parse_items, run_variable_fill

TABLE = """# HIES codebook

| Variable | Label | Type |
|---|---|---|
| `hhid` | Household identifier | numeric |
| `age` | Age of household member (years) | numeric |
| `sex` | Sex of household member | numeric |
| `income` | Monthly household income (XCD) | numeric |
"""

TABLE_WITH_QUESTIONS = """
| Name | Description | Question text |
|---|---|---|
| hhsize | Number of household members | How many people usually live in this household? |
| tenure | Housing tenure | Does your household own or rent this dwelling? |
"""

LINES = """Section 2: Education

edu_lvl: Highest education level completed
edu_yrs - Years of schooling completed
attend  Currently attending school
"""

PROSE = """The survey covers all private households.
Country: Saint Lucia
Version: 1.0
"""


def names(*n):
    return {x.casefold() for x in n}


def test_table_rows_are_read_with_the_header_columns():
    found = parse_codebook(TABLE, names("hhid", "age", "income", "weight"))
    assert found["hhid"][0].values == {"label": "Household identifier"}
    assert found["income"][0].values["label"] == "Monthly household income (XCD)"
    assert "weight" not in found


def test_header_maps_description_and_question_columns():
    found = parse_codebook(TABLE_WITH_QUESTIONS, names("hhsize", "tenure"))
    # No Label column: the Description column is the label.
    assert found["hhsize"][0].values == {
        "label": "Number of household members",
        "question": "How many people usually live in this household?",
    }


def test_name_label_lines_count_only_inside_a_run():
    found = parse_codebook(LINES, names("edu_lvl", "edu_yrs", "attend"))
    assert found["edu_lvl"][0].values["label"] == "Highest education level completed"
    assert found["edu_yrs"][0].values["label"] == "Years of schooling completed"
    assert found["attend"][0].values["label"] == "Currently attending school"


def test_a_lone_name_colon_value_in_prose_is_not_a_codebook_line():
    assert parse_codebook(PROSE, names("country", "version")) == {}


def test_case_of_the_name_does_not_matter():
    found = parse_codebook("| Variable | Label |\n|---|---|\n| HHID | Household identifier |\n| AGE | Age in years |\n", names("hhid", "age"))
    assert set(found) == {"hhid", "age"}


def test_parse_items_accepts_fences_prose_and_a_bare_list():
    assert parse_items('```json\n{"variables": [{"variable": 1}]}\n```') == [{"variable": 1}]
    assert parse_items('Here you go: {"variables": [{"variable": 2}]}') == [{"variable": 2}]
    assert parse_items('[{"variable": 3}]') == [{"variable": 3}]
    assert parse_items("no json") == []


class Scripted:
    """A gateway that answers each batch from a script; records what it was asked."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []
        self.failed = 0

    async def complete(self, spec, messages, **kw):
        self.prompts.append(messages[0]["content"])
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def mark_failed(self, spec):
        self.failed += 1


def spec():
    reg = parse_registry(
        {"default": "m", "models": [{"alias": "m", "local": True, "litellm_params": {"model": "openai/x", "api_base": "http://x/v1", "api_key": "k"}}]},
        {},
    )
    return reg.resolve("m")


def bundle(*vars_):
    return {"record_id": "1", "file_id": "F1", "datafile": {"file_id": "F1", "file_name": "hh.dta", "variables": list(vars_)}}


def var(uid, name, label="", **kw):
    return {"uid": uid, "name": name, "label": label, **kw}


def run(gateway, b, doc, attrs=("label",)):
    return asyncio.run(run_variable_fill(gateway, spec(), b, doc, attrs=list(attrs), slot=asyncio.Semaphore(1)))


def test_codebook_lines_fill_missing_labels_without_calling_the_model():
    gw = Scripted()
    out = run(gw, bundle(var(1, "hhid"), var(2, "age", "Age"), var(3, "sex", "Sex of household member")), TABLE)
    items = {i["name"]: i for i in out["variables"]["items"]}
    assert gw.prompts == []
    assert items["hhid"]["status"] == "empty" and items["hhid"]["via"] == "codebook"
    assert items["age"]["status"] == "differs" and items["age"]["current"] == "Age"
    assert "sex" not in items  # already matches
    assert out["variables"]["stats"]["agreed"] == 1


def test_the_model_reads_what_no_line_states_and_its_evidence_is_checked():
    doc = "Questionnaire\nQ7. In what year were you born? (variable birth_yr)\nQ8. Do you own this home? (variable own_home)\n"
    reply = json.dumps(
        {
            "variables": [
                {"variable": 1, "label": "Year of birth", "evidence": "Q7. In what year were you born? (variable birth_yr)"},
                {"variable": 2, "label": "Owns the home", "evidence": "Q99. Invented line about own_home"},
            ]
        }
    )
    out = run(Scripted(reply), bundle(var(1, "birth_yr"), var(2, "own_home"), var(3, "missing_in_doc")), doc)
    items = out["variables"]["items"]
    assert [(i["name"], i["value"], i["via"]) for i in items] == [("birth_yr", "Year of birth", "model")]
    assert out["variables"]["stats"]["not_found"] == ["missing_in_doc", "own_home"]


def test_model_evidence_about_another_variable_is_refused():
    doc = "Q1. Household income last month (variable income)\nQ2. Age of the respondent (variable age)\n"
    reply = json.dumps({"variables": [{"variable": 1, "label": "Respondent age", "evidence": "Q2. Age of the respondent (variable age)"}]})
    out = run(Scripted(reply), bundle(var(1, "income"), var(2, "age")), doc)
    assert [i["name"] for i in out["variables"]["items"]] == []  # the evidence names `age`, not `income`


def test_saved_values_are_not_overwritten_by_the_model():
    doc = "Q1. Household income last month (variable income)\n"
    reply = json.dumps({"variables": [{"variable": 1, "label": "Income", "evidence": "Q1. Household income last month (variable income)"}]})
    assert run(Scripted(reply), bundle(var(1, "income", "Monthly income")), doc)["variables"]["items"] == []


def test_invented_numbers_in_a_model_label_are_refused():
    doc = "Q1. Household income last month (variable income)\n"
    reply = json.dumps({"variables": [{"variable": 1, "label": "Household income 2031", "evidence": "Q1. Household income last month (variable income)"}]})
    assert run(Scripted(reply), bundle(var(1, "income")), doc)["variables"]["items"] == []


def test_a_failed_batch_is_reported_and_other_work_stays():
    doc = TABLE + "\nQ5. Number of rooms (variable rooms)\n"
    out = run(Scripted(RuntimeError("down")), bundle(var(1, "hhid"), var(2, "rooms")), doc)
    assert out["variables"]["stats"]["failed_batches"] == 1
    assert [i["name"] for i in out["variables"]["items"]] == ["hhid"]


def test_descriptions_and_questions_are_proposed_when_asked():
    out = run(Scripted(), bundle(var(1, "hhsize"), var(2, "tenure")), TABLE_WITH_QUESTIONS, attrs=("label", "question"))
    fields = {(i["name"], i["field"]) for i in out["variables"]["items"]}
    assert fields == {("hhsize", "label"), ("hhsize", "question"), ("tenure", "label"), ("tenure", "question")}


def test_a_file_without_variables_says_so():
    assert "no variables" in run(Scripted(), bundle(), TABLE)["text"]


def test_a_question_that_only_repeats_the_label_or_is_not_a_question_is_refused():
    doc = "edu_ever - Ever attended school\nedu_level - Highest education level\nNote for variable insured: record whether covered.\n"
    reply = json.dumps(
        {
            "variables": [
                {"variable": 1, "question": "Ever attended school", "evidence": "edu_ever - Ever attended school"},
                {"variable": 2, "question": "Highest education level completed?", "evidence": "edu_level - Highest education level"},
                {"variable": 3, "question": "record whether covered.", "evidence": "Note for variable insured: record whether covered."},
            ]
        }
    )
    out = run(Scripted(reply), bundle(var(1, "edu_ever"), var(2, "edu_level"), var(3, "insured")), doc, attrs=("question",))
    assert [(i["name"], i["field"]) for i in out["variables"]["items"]] == [("edu_level", "question")]


def test_not_found_means_the_document_says_nothing_about_the_variable():
    doc = TABLE
    out = run(Scripted(json.dumps({"variables": []})), bundle(var(1, "hhid"), var(2, "nowhere")), doc, attrs=("label", "question"))
    stats = out["variables"]["stats"]
    assert stats["not_found"] == ["nowhere"]  # hhid has a label line even though no question text
