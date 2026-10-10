"""`prismyra.learn.features`: one tokenizer for both a sentence and a JSON state, and a hashed TF-IDF that is
exactly reproducible given the same corpus and the same `dims`."""

from __future__ import annotations

from prismyra.learn.features import HashingTfidf, tokenize


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
