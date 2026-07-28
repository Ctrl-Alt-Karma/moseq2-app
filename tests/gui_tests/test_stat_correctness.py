"""
Regression tests for reported-number correctness in the interactive stats tools.
"""
import numpy as np
import pandas as pd
from unittest import TestCase

from moseq2_app.stat.controller import fill_missing_syllable_stats, COUNT_COLUMNS
from moseq2_app.stat.view import (
    get_minmax_tp,
    EDGE_WIDTH_SCALE,
    DIFF_EDGE_WIDTH_SCALE,
)


class TestMissingSyllableStats(TestCase):
    def test_usage_filled_but_physical_scalars_left_missing(self):
        # A session that never performs a syllable used it zero times, but has no
        # height/velocity measurement for it. Filling those with 0 invents a
        # physically impossible observation that drags group means toward zero.
        df = pd.DataFrame(
            {
                "syllable": [0, 1],
                "usage": [0.5, np.nan],
                "height_ave_mm_mean": [50.0, np.nan],
                "velocity_2d_mm_mean": [12.0, np.nan],
            }
        )

        out = fill_missing_syllable_stats(df)

        assert "usage" in COUNT_COLUMNS
        assert out["usage"].iloc[1] == 0
        assert np.isnan(out["height_ave_mm_mean"].iloc[1])
        assert np.isnan(out["velocity_2d_mm_mean"].iloc[1])

    def test_does_not_mutate_the_input(self):
        df = pd.DataFrame({"syllable": [0], "usage": [np.nan]})
        fill_missing_syllable_stats(df)
        assert np.isnan(df["usage"].iloc[0])


class TestTransitionProbabilityLegend(TestCase):
    def test_reported_probability_matches_drawn_edges(self):
        # Edges are drawn at weight * EDGE_WIDTH_SCALE, so recovering the
        # probability must divide by that same constant. The legend previously
        # divided by 200 while edges were drawn at 250, inflating every reported
        # P(transition) by 25%.
        probabilities = {("a", "b"): 0.004, ("b", "c"): 0.012}
        edge_width = {k: v * EDGE_WIDTH_SCALE for k, v in probabilities.items()}

        min_tp, max_tp = get_minmax_tp(edge_width, diff=False)

        np.testing.assert_allclose([min_tp, max_tp], [0.004, 0.012])

    def test_down_regulated_transitions_are_reported(self):
        # edge_width is built with abs(), so selecting down-regulated edges by
        # `width < 0` matched nothing and the legend always displayed 0.0000.
        signed = {("a", "b"): -0.006, ("b", "c"): 0.009}
        edge_width = {k: abs(v * DIFF_EDGE_WIDTH_SCALE) for k, v in signed.items()}

        min_down, max_down, min_up, max_up = get_minmax_tp(
            edge_width, diff=True, signed_weights=signed
        )

        np.testing.assert_allclose([min_down, max_down], [0.006, 0.006])
        np.testing.assert_allclose([min_up, max_up], [0.009, 0.009])

        # the old approach found no negative widths at all
        assert [w for w in edge_width.values() if w < 0] == []

    def test_empty_graph_does_not_raise(self):
        assert get_minmax_tp({}, diff=False) == (0, 0)
        assert get_minmax_tp({}, diff=True, signed_weights={}) == (0, 0, 0, 0)


class TestBootstrapReproducibility(TestCase):
    def test_confidence_interval_is_deterministic(self):
        from moseq2_app.stat.view import get_ci_vect_vectorized

        values = pd.Series(np.linspace(0.1, 0.9, 25))

        first = get_ci_vect_vectorized(values, n_boots=500)
        second = get_ci_vect_vectorized(values, n_boots=500)

        np.testing.assert_allclose(first, second)

    def test_missing_observations_are_dropped(self):
        from moseq2_app.stat.view import get_ci_vect_vectorized

        clean = pd.Series([1.0, 2.0, 3.0, 4.0])
        with_nan = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan])

        np.testing.assert_allclose(
            get_ci_vect_vectorized(clean, n_boots=500),
            get_ci_vect_vectorized(with_nan, n_boots=500),
        )
