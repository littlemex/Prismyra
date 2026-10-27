"""`prismyra.learn.features`: one tokenizer for both a sentence and a JSON state, a hashed TF-IDF that is
exactly reproducible given the same corpus and the same `dims`, and the small registry (learn3,
DISTILL-RL-DESIGN-v2.md section 3) that lets a tag ask for a hand-written feature scheme instead."""

from __future__ import annotations

import pytest

from prismyra.learn.features import (
    DEFAULT_TEXT_TOKENIZER,
    FeatureError,
    HashingTfidf,
    Vectorizer,
    char_ngrams,
    get_feature_builder,
    is_valid_text_tokenizer,
    register_features,
    tokenize,
    tokenize_for,
)


def test_json_is_canonicalised_so_key_order_does_not_matter():
    assert tokenize('{"a": 1, "b": 2}') == tokenize('{"b": 2, "a": 1}')


def test_json_is_flattened_into_key_value_tokens():
    tokens = tokenize('{"snake_head_x": 3, "food": {"x": 7, "y": 2}}')
    assert "snake_head_x=3" in tokens
    assert "food.x=7" in tokens
    assert "food.y=2" in tokens


def test_a_list_is_flattened_by_index():
    tokens = tokenize('["a", "b"]')
    assert "0=a" in tokens
    assert "1=b" in tokens


def test_plain_text_is_lowercased_and_tokenized():
    assert tokenize("Buy NOW!!!") == ["buy", "now"]


def test_a_bare_json_scalar_is_treated_as_text():
    # `json.loads("3")` succeeds and returns an int, which is not a dict/list -- flattening it would be
    # meaningless, so it falls through to the plain-text path, same as any other short string.
    assert tokenize("3") == ["3"]


def test_hashing_tfidf_is_deterministic_across_separate_fits():
    corpus = [tokenize("buy now"), tokenize("hello friend"), tokenize("buy tickets now")]
    a = HashingTfidf.fit(corpus, dims=32)
    b = HashingTfidf.fit(corpus, dims=32)
    assert a == b
    assert a.transform(tokenize("buy now")) == b.transform(tokenize("buy now"))


def test_a_token_that_appears_in_every_document_gets_a_lower_idf_weight():
    corpus = [["common", "rare1"], ["common", "rare2"], ["common", "rare3"]]
    tf = HashingTfidf.fit(corpus, dims=4096)
    vec_common = dict(enumerate(tf.transform(["common"])))
    vec_rare = dict(enumerate(tf.transform(["rare1"])))
    # Both are single-token vectors (one nonzero bucket each, almost certainly in different buckets at this
    # width), but the common token's own weight is strictly smaller than a token seen in only one document.
    assert max(vec_common.values()) < max(vec_rare.values())


def test_transform_is_a_fixed_width_vector():
    tf = HashingTfidf.fit([["a"], ["b", "c"]], dims=16)
    assert len(tf.transform(["a", "b", "unseen-token"])) == 16
    assert len(tf.transform([])) == 16


# --------------------------------------------------------------------------- the feature-builder registry


def test_register_features_then_get_feature_builder_round_trips():
    def builder(context: str, question: str) -> dict[str, float]:
        return {"len": float(len(context))}

    register_features("test-roundtrip-v1", builder)
    assert get_feature_builder("test-roundtrip-v1") is builder


def test_registering_the_same_function_object_twice_is_a_no_op():
    def builder(context: str, question: str) -> dict[str, float]:
        return {"len": float(len(context))}

    register_features("test-idempotent-v1", builder)
    register_features("test-idempotent-v1", builder)  # must not raise
    assert get_feature_builder("test-idempotent-v1") is builder


def test_registering_a_different_function_under_a_taken_name_is_refused():
    def builder_a(context: str, question: str) -> dict[str, float]:
        return {"a": 1.0}

    def builder_b(context: str, question: str) -> dict[str, float]:
        return {"b": 2.0}

    register_features("test-collision-v1", builder_a)
    with pytest.raises(FeatureError, match="already registered"):
        register_features("test-collision-v1", builder_b)


def test_an_unregistered_name_is_refused():
    with pytest.raises(FeatureError, match="no feature builder named"):
        get_feature_builder("test-does-not-exist-v1")


# --------------------------------------------------------------------------- Vectorizer


def test_vectorizer_fit_with_no_features_name_behaves_like_hashing_tfidf_alone():
    records = [("buy now", ""), ("hello friend", ""), ("buy tickets now", "")]
    vec = Vectorizer.fit(records, features_name=None, dims=32)
    direct = HashingTfidf.fit([tokenize(c) for c, _ in records], 32)
    assert vec.transform("buy now") == direct.transform(tokenize("buy now"))


def test_vectorizer_structured_feature_columns_are_fixed_at_fit_time():
    def builder(context: str, question: str) -> dict[str, float]:
        return {"a": float(len(context)), "b": 1.0 if question else 0.0}

    register_features("test-structured-v1", builder)
    records = [("abc", "q"), ("de", "q")]
    vec = Vectorizer.fit(records, features_name="test-structured-v1", dims=8)
    assert vec.names == ("a", "b")
    assert vec.transform("abc", "q") == [3.0, 1.0]
    assert vec.transform("de", "") == [2.0, 0.0]


def test_vectorizer_structured_feature_defaults_a_missing_key_to_zero():
    def builder(context: str, question: str) -> dict[str, float]:
        # Returns an extra key ("c") only for this particular input -- transform must still only ever emit
        # the columns fixed at fit time (here: "a" only, from the first training record).
        if context == "has-extra":
            return {"a": 1.0, "c": 99.0}
        return {"a": 2.0}

    register_features("test-structured-missing-v1", builder)
    vec = Vectorizer.fit([("first", "")], features_name="test-structured-missing-v1", dims=4)
    assert vec.names == ("a",)
    assert vec.transform("has-extra") == [1.0]  # "c" is not a fixed column, so it is dropped, not appended


def test_vectorizer_with_an_unregistered_features_name_raises_at_fit():
    with pytest.raises(FeatureError, match="no feature builder named"):
        Vectorizer.fit([("x", "")], features_name="test-never-registered-v1", dims=4)


# --------------------------------------------------------------------------- char n-grams


def test_char_ngrams_slides_a_window_across_the_lowercased_squashed_text():
    assert char_ngrams("AB  CD", 3) == ["ab ", "b c", " cd"]


def test_char_ngrams_keeps_the_space_crossing_a_word_boundary():
    # A character scheme has no notion of "word" to respect -- the whole point of choosing it over bag-of-
    # words is catching sub-word and cross-word structure a word tokenizer throws away.
    assert " cd" in char_ngrams("ab cd", 3)
    assert "cd" not in char_ngrams("ab cd", 3)


def test_char_ngrams_on_text_shorter_than_n_returns_the_whole_squashed_text():
    assert char_ngrams("hi", 5) == ["hi"]


def test_char_ngrams_on_empty_text_returns_nothing():
    assert char_ngrams("   ", 3) == []


def test_is_valid_text_tokenizer_accepts_words_and_charn_only():
    assert is_valid_text_tokenizer("words")
    assert is_valid_text_tokenizer("char3")
    assert is_valid_text_tokenizer("char10")
    assert not is_valid_text_tokenizer("char1")  # below CHAR_NGRAM_MIN_N: degenerates to unordered letters
    assert not is_valid_text_tokenizer("char0")
    assert not is_valid_text_tokenizer("chars")
    assert not is_valid_text_tokenizer("word")
    assert not is_valid_text_tokenizer("")


def test_tokenize_for_words_is_exactly_tokenize():
    assert tokenize_for("words", "Buy Now") == tokenize("Buy Now")


def test_tokenize_for_charn_dispatches_to_char_ngrams():
    assert tokenize_for("char3", "abcd") == char_ngrams("abcd", 3)


def test_tokenize_for_refuses_an_invalid_name():
    with pytest.raises(FeatureError, match="not a valid text tokenizer name"):
        tokenize_for("char1", "abcd")


def test_vectorizer_fit_defaults_to_words_when_text_tokenizer_is_left_unset():
    records = [("buy now", ""), ("hello friend", "")]
    vec = Vectorizer.fit(records, features_name=None, dims=16)
    assert vec.text_tokenizer == DEFAULT_TEXT_TOKENIZER
    assert vec.transform("buy now") == Vectorizer.fit(
        records, features_name=None, dims=16, text_tokenizer="words"
    ).transform("buy now")


def test_vectorizer_fit_with_char_ngrams_uses_char_ngrams_not_words():
    records = [("buy now", ""), ("hello friend", ""), ("buy tickets now", "")]
    vec = Vectorizer.fit(records, features_name=None, dims=32, text_tokenizer="char3")
    direct = HashingTfidf.fit([char_ngrams(c, 3) for c, _ in records], 32)
    assert vec.transform("buy now") == direct.transform(char_ngrams("buy now", 3))
    # And it must disagree with the word scheme on the same corpus -- otherwise this would not be testing
    # that the char-n-gram path is actually wired in, only that two equal-dims hashers happen to agree.
    word_vec = Vectorizer.fit(records, features_name=None, dims=32, text_tokenizer="words")
    assert vec.transform("buy now") != word_vec.transform("buy now")


def test_vectorizer_structured_branch_ignores_text_tokenizer():
    def builder(context: str, question: str) -> dict[str, float]:
        return {"a": float(len(context))}

    register_features("test-structured-ignores-tokenizer-v1", builder)
    vec = Vectorizer.fit(
        [("abc", "")], features_name="test-structured-ignores-tokenizer-v1", dims=4, text_tokenizer="char3"
    )
    assert vec.transform("abc") == [3.0]  # the structured builder answered, not a char-n-gram hash
