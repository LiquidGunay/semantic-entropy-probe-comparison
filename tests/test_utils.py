import math

import numpy as np

from sep_marimo import utils as U


def test_extract_think_and_answer_prefers_last_block():
    text = (
        "Intro <think>first</think> middle "
        "<think>final reasoning</think> answer text"
    )
    think, answer = U.extract_think_and_answer(text, "<think>", "</think>")
    assert think == "final reasoning"
    assert answer == "answer text"


def test_extract_think_and_answer_fallback():
    text = "Prompt <think>only block</think> trailing"
    think, answer = U.extract_think_and_answer(text, "<think>", "</think>")
    assert think == "only block"
    assert answer == "trailing"


def test_extract_answer_text_prefers_boxed():
    full = "solution: \\boxed{42}"
    assert U.extract_answer_text(full) == "42"
    after_think = "prefix</think>\nAnswer: seventy"
    assert U.extract_answer_text(after_think) == "seventy"
    nested = "start \\boxed{1 + \\boxed{2}} end"
    assert U.extract_answer_text(nested) == "1 + \\boxed{2}"


def test_extract_answer_text_ignores_prompt_box_and_uses_tail():
    text = "Prompt box \\boxed{999}</think> tail \\boxed{123}"
    # Should pick the tail boxed after </think>
    assert U.extract_answer_text(text) == "123"


def test_extract_answer_text_nested_after_think():
    text = "</think> final Answer: here \\boxed{a + \\boxed{b} + c}"
    assert U.extract_answer_text(text) == "a + \\boxed{b} + c"


def test_extract_full_and_final_answer():
    text = "prefix </think> tail here \\boxed{42} and more"
    full = U.extract_full_answer(text)
    assert full.startswith("tail here")
    assert U.extract_final_answer(full) == "42"


def test_dfrac_equivalence():
    assert U.answers_equivalent("\\dfrac{9}{256}", "\\frac{9}{256}")


def test_find_subsequence():
    seq = [1, 2, 3, 2, 3, 4]
    assert U.find_subsequence(seq, [2, 3]) == 1
    assert U.find_subsequence(seq, [2, 3], start=2) == 3
    assert U.find_subsequence(seq, [9]) == -1


def test_token_entropy_from_logprobs_dict():
    # uniform over two tokens → entropy ≈ ln 2
    lp = {0: -0.5, 1: -0.5}
    ent = U.token_entropy_from_logprobs(lp)
    assert math.isclose(ent, math.log(2), rel_tol=1e-5)


def test_mean_entropy_in_span_skips_none():
    ent = [None, 0.1, None, 0.3]
    assert math.isclose(U.mean_entropy_in_span(ent, 0, 3), 0.2)


def test_compute_is_correct_numeric_and_string():
    assert U.compute_is_correct("5", "5.0", dataset="math")
    assert U.compute_is_correct("Paris", "paris", dataset="openqa")


def test_normalize_answer_for_cluster():
    assert U.normalize_answer_for_cluster(" \\boxed{7} ") == "7"
    assert U.normalize_answer_for_cluster(" 3 + 4 ") == "3 + 4"


def test_build_prompt_contains_think_tags():
    prompt = U.build_prompt("Q", "<think>", "</think>")
    assert "<think>" in prompt and "</think>" in prompt
