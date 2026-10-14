import pytest

from match_cull.crops import AI_ASPECTS, SIZES, ai_crop_box, crop_box
from match_cull.select import Candidate, choose_keepers, group_bursts, parse_keep


@pytest.mark.parametrize("text,total,expected", [("30", 100, 30), ("25%", 10, 3), (" 50 % ", 7, 4), ("100%", 5, 5)])
def test_parse_keep(text, total, expected):
    assert parse_keep(text, total) == expected


@pytest.mark.parametrize("text", ["0", "-3", "abc", "150%", "0%"])
def test_parse_keep_rejects_nonsense(text):
    with pytest.raises(ValueError):
        parse_keep(text, 10)


def cand(name, score, t=None, h=0, sharp=1.0):
    return Candidate(name, score, sharp, h, t)


def test_burst_keeps_only_its_best_frame():
    shots = [cand("a", 6, 0.0), cand("b", 9, 0.1), cand("c", 7, 0.2), cand("d", 8, 10.0, h=2**64 - 1)]
    kept = choose_keepers(shots, keep=10)
    assert [c.path for c in kept] == ["b", "d"]


def test_similar_looking_frames_without_times_are_one_burst():
    shots = [cand("a", 5, h=0b1111), cand("b", 8, h=0b1110), cand("c", 7, h=2**64 - 1)]
    group_bursts(shots)
    assert [c.burst for c in shots] == [0, 0, 1]


def test_far_apart_in_time_is_never_a_burst_even_if_similar():
    shots = [cand("a", 5, 0.0), cand("b", 8, 60.0)]
    group_bursts(shots)
    assert shots[0].burst != shots[1].burst


def test_top_n_in_camera_order_and_rejects_never_fill():
    shots = [cand(n, s, t=i * 10.0, h=(2**64 - 1) * (i % 2)) for i, (n, s) in
             enumerate([("a", 5), ("b", 9), ("c", 3), ("d", 8), ("e", 7)])]
    assert [c.path for c in choose_keepers(shots, keep=3)] == ["b", "d", "e"]
    assert "c" not in [c.path for c in choose_keepers(shots, keep=10)]


def test_sharper_frame_wins_a_tie():
    shots = [cand("soft", 8, 0.0, sharp=10), cand("crisp", 8, 0.1, sharp=50)]
    assert [c.path for c in choose_keepers(shots, keep=1)] == ["crisp"]


@pytest.mark.parametrize("key", sorted(SIZES))
@pytest.mark.parametrize("tight", [True, False])
@pytest.mark.parametrize("subject", [(0.4, 0.4, 0.6, 0.6), (0.0, 0.0, 0.05, 0.05), (0.9, 0.8, 1.2, 1.1), (0.0, 0.0, 1.0, 1.0)])
@pytest.mark.parametrize("dims", [(6000, 4000), (4000, 6000), (1080, 1920)])
def test_crop_has_the_right_shape_and_stays_inside(key, tight, subject, dims):
    w, h = dims
    l, t, r, b = crop_box(w, h, subject, SIZES[key].aspect, tight)
    assert 0 <= l < r <= w and 0 <= t < b <= h
    aspect = SIZES[key].aspect or w / h
    assert (r - l) / (b - t) == pytest.approx(aspect, rel=0.01)


def test_tight_crop_is_closer_than_wide_and_centered_on_subject():
    subject = (0.55, 0.3, 0.65, 0.5)
    wide = crop_box(6000, 4000, subject, 4 / 5, tight=False)
    tight = crop_box(6000, 4000, subject, 4 / 5, tight=True)
    assert (tight[2] - tight[0]) < (wide[2] - wide[0])
    assert (tight[0] + tight[2]) / 2 == pytest.approx(0.6 * 6000, abs=2)


def test_all_keeps_every_photo_including_duplicates_and_rejects():
    from match_cull.select import KEEP_ALL
    assert parse_keep("all", 10) == KEEP_ALL
    assert parse_keep(" ALL ", 10) == KEEP_ALL
    shots = [cand("a", 6, 0.0), cand("b", 9, 0.1), cand("c", 2, 30.0, h=2**64 - 1)]
    assert [c.path for c in choose_keepers(shots, keep=KEEP_ALL)] == ["a", "b", "c"]
    assert [c.path for c in choose_keepers(shots, keep=100)] == ["b"]


@pytest.mark.parametrize("shape", sorted(AI_ASPECTS) + ["nonsense"])
@pytest.mark.parametrize("suggested", [(0.1, 0.1, 0.9, 0.9), (0.6, 0.2, 0.7, 0.3), (-0.2, 0.5, 1.4, 1.2), (0.8, 0.8, 0.2, 0.2)])
@pytest.mark.parametrize("dims", [(6000, 4000), (4000, 6000)])
def test_ai_crop_has_the_chosen_shape_and_stays_inside(shape, suggested, dims):
    w, h = dims
    l, t, r, b = ai_crop_box(w, h, suggested, shape)
    assert 0 <= l < r <= w and 0 <= t < b <= h
    assert (r - l) / (b - t) == pytest.approx(AI_ASPECTS.get(shape, w / h), rel=0.01)


def test_ai_crop_contains_the_suggestion_and_is_centered_on_it():
    l, t, r, b = ai_crop_box(6000, 4000, (0.3, 0.25, 0.6, 0.75), "4:5")
    assert l <= 0.3 * 6000 and r >= 0.6 * 6000 and t <= 0.25 * 4000 + 1 and b >= 0.75 * 4000 - 1
    assert (l + r) / 2 == pytest.approx(0.45 * 6000, abs=2)


def test_ai_crop_never_zooms_in_too_far():
    l, t, r, b = ai_crop_box(6000, 4000, (0.5, 0.5, 0.51, 0.51), "3:2")
    assert (b - t) >= 0.4 * 4000 - 1


def test_crops_never_zoom_below_the_minimum_size():
    from match_cull.crops import MIN_CROP_SHORT_SIDE
    # tiny subject on a big photo: still at least 1080 px on the short side
    l, t, r, b = ai_crop_box(6000, 4000, (0.5, 0.5, 0.52, 0.52), "4:5")
    assert min(r - l, b - t) >= MIN_CROP_SHORT_SIDE - 1
    l, t, r, b = crop_box(6000, 4000, (0.5, 0.5, 0.52, 0.52), 4 / 5, tight=True)
    assert min(r - l, b - t) >= MIN_CROP_SHORT_SIDE - 1
    # small photo (WhatsApp size): the crop is as large as the shape allows
    l, t, r, b = ai_crop_box(1600, 1066, (0.4, 0.3, 0.6, 0.8), "4:5")
    assert b - t == 1066
