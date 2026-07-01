from __future__ import annotations

from musicidx.tempo import perceived_tempo_bpm


def test_perceived_tempo_halves_mid_high_hip_hop_double_time():
    assert perceived_tempo_bpm(161.499, descriptors=["Rap", "hip hop---gangsta"]) == 80.7495
    assert perceived_tempo_bpm(152.0, descriptors=["hip hop---trip hop"]) == 76.0


def test_perceived_tempo_keeps_true_high_tempo_styles_high():
    assert perceived_tempo_bpm(174.0, descriptors=["electronic---drum n bass"]) == 174.0
    assert perceived_tempo_bpm(170.0, descriptors=["jungle"]) == 170.0
    assert perceived_tempo_bpm(150.0, descriptors=["techno"]) == 150.0
