"""Length-sorted packing of branch groups, checked on the CPU against a stand-in tokenizer."""

from types import SimpleNamespace

from prismyra.engine import PACK_ALIGN, Prismyra


class Words:
    """One token per word, so a plan's length is its word count (the newline the engine prepends splits away)."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split() or [0]}


def groups_for(lengths, group, graphs=False, calibration=None, width=128):
    fake = SimpleNamespace(group=group, graphs=graphs, calibration=calibration, tokenizer=Words())
    plans = [SimpleNamespace(text=" ".join(["w"] * n)) for n in lengths]
    return Prismyra._packed_groups(fake, plans, width)


def test_every_question_is_in_exactly_one_group():
    lengths = [9, 40, 3, 77, 12, 12, 50, 1, 64, 30]
    groups = groups_for(lengths, group=4)
    members = sorted(i for g, _ in groups for i in g)
    assert members == list(range(len(lengths)))
    assert all(len(g) <= 4 for g, _ in groups)


def test_groups_are_sorted_by_length_and_sized_to_their_longest_row():
    lengths = [9, 40, 3, 77, 12, 12, 50, 1]
    for members, width in groups_for(lengths, group=4):
        longest = max(lengths[i] for i in members)
        assert width >= longest and width % PACK_ALIGN == 0 and width - longest < PACK_ALIGN
    first, second = groups_for(lengths, group=4)
    assert max(lengths[i] for i in first[0]) <= min(lengths[i] for i in second[0])


def test_packing_never_exceeds_the_shared_width():
    groups = groups_for([120, 121, 5], group=2, width=128)
    assert all(width <= 128 for _, width in groups)


def test_calibration_keeps_one_shared_width_in_arrival_order():
    groups = groups_for([30, 2, 90, 7, 7], group=2, width=96, calibration=object())
    assert [g for g, _ in groups] == [[0, 1], [2, 3], [4]]
    assert {w for _, w in groups} == {96}


def test_recordings_are_packed_too_and_the_same_questions_give_the_same_shapes():
    lengths = [30, 2, 90, 7, 7]
    first = groups_for(lengths, group=2, width=96, graphs=True)
    assert first == groups_for(lengths, group=2, width=96, graphs=True)
    assert len({w for _, w in first}) > 1
