"""
Unit tests for the control law in `src/controller.py`.

These run without TORCS, without torch and without numpy -- the point of
extracting the control law into a pure module was that the part of the system
with the interesting behaviour could be tested at all.  Stdlib `unittest`, so
`python -m unittest discover -s tests` works on a bare interpreter; the same
classes are collected by pytest if you prefer it.

The reference values below come from `results/stage4_cma_8param_sector_s35.json`, the
parameter set the 108.692 s lap was measured with.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import controller as ctl  # noqa: E402

PARAMS_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "stage4_cma_8param_sector_s35.json",
)


def load_params():
    return ctl.Params.from_json(PARAMS_JSON)


class TestParams(unittest.TestCase):
    def test_loads_the_published_parameter_set(self):
        p = load_params()
        self.assertAlmostEqual(p.A, 25.72)
        self.assertAlmostEqual(p.C_s35, 71.40476732348836)
        self.assertAlmostEqual(p.switch_dist, 3032.0)
        self.assertAlmostEqual(p.back_dist, 3305.0)

    def test_missing_parameter_is_rejected(self):
        d = {f: 1.0 for f in ctl.Params.FIELDS}
        del d["C_s35"]
        with self.assertRaises(KeyError):
            ctl.Params(**d)

    def test_zone_bounds_are_ordered(self):
        p = load_params()
        self.assertLess(p.switch_dist, p.back_dist)
        self.assertLess(p.back_dist, ctl.TRACK_LAP_M)
        self.assertLessEqual(p.s35_start, ctl.D_S35_EXIT)


class TestRacingLineProfiles(unittest.TestCase):
    """Both profiles must be zero outside their zone and continuous inside it."""

    def test_corkscrew_zero_outside_zone(self):
        for d in (0.0, 500.0, ctl.D_CORK_RAMP, ctl.D_CORK_EXIT, 2000.0, 3600.0):
            self.assertEqual(ctl.target_trackpos_corkscrew(d, 0.5, -0.5), 0.0)

    def test_corkscrew_hits_its_control_points(self):
        self.assertAlmostEqual(ctl.target_trackpos_corkscrew(ctl.D_CORK_APP, 0.4, -0.3), 0.4)
        self.assertAlmostEqual(ctl.target_trackpos_corkscrew(ctl.D_CORK_APEX, 0.4, -0.3), -0.3)

    def test_corkscrew_ramp_is_linear(self):
        mid = (ctl.D_CORK_RAMP + ctl.D_CORK_APP) / 2.0
        self.assertAlmostEqual(ctl.target_trackpos_corkscrew(mid, 0.4, -0.3), 0.2)

    def test_corkscrew_unwinds_to_centre_at_exit(self):
        just_inside = ctl.D_CORK_EXIT - 1e-6
        self.assertAlmostEqual(
            ctl.target_trackpos_corkscrew(just_inside, 0.4, -0.3), 0.0, places=6
        )

    def test_s35_zero_outside_zone(self):
        for d in (0.0, ctl.D_S35_RAMP - 1.0, ctl.D_S35_EXIT, 3000.0):
            self.assertEqual(ctl.target_trackpos_s35(d, 0.16, -0.008), 0.0)

    def test_s35_hits_its_control_points(self):
        self.assertAlmostEqual(ctl.target_trackpos_s35(ctl.D_S35_APPROACH, 0.16, -0.008), 0.16)
        self.assertAlmostEqual(ctl.target_trackpos_s35(ctl.D_S35_APEX, 0.16, -0.008), -0.008)

    def test_profiles_never_overlap(self):
        """The combined target is only ever one profile at a time."""
        p = load_params()
        step = 0.5
        d = 0.0
        while d < ctl.TRACK_LAP_M:
            cork = ctl.target_trackpos_corkscrew(d, p.tp_approach, p.tp_apex)
            s35 = ctl.target_trackpos_s35(d, p.tp_s35_approach, p.tp_s35_apex)
            self.assertTrue(cork == 0.0 or s35 == 0.0, "overlap at %.1f m" % d)
            d += step

    def test_combined_target_matches_the_active_profile(self):
        p = load_params()
        d = ctl.D_S35_APEX
        self.assertAlmostEqual(
            ctl.target_trackpos(d, p),
            ctl.target_trackpos_s35(d, p.tp_s35_approach, p.tp_s35_apex),
        )


class TestSteering(unittest.TestCase):
    def test_deadband_suppresses_small_lateral_error(self):
        """Inside D the lateral term is dropped -- this is what stopped the
        straight-line zigzag that dominated the remaining time loss."""
        inside = ctl.steering(0.0, 0.05, 0.0, 0.0, A=25.0, B=1.6, D=0.072)
        self.assertEqual(inside, 0.0)

    def test_deadband_is_subtracted_not_thresholded(self):
        """Just outside D the correction must start from ~0, not jump to B*err."""
        D = 0.072
        just_outside = ctl.steering(0.0, D + 1e-4, 0.0, 0.0, A=25.0, B=1.6, D=D)
        self.assertAlmostEqual(just_outside, -1.6e-4, places=6)

    def test_correction_opposes_lateral_offset(self):
        right_of_line = ctl.steering(0.0, 0.5, 0.0, 0.0, A=25.0, B=1.6, D=0.072)
        left_of_line = ctl.steering(0.0, -0.5, 0.0, 0.0, A=25.0, B=1.6, D=0.072)
        self.assertLess(right_of_line, 0.0)
        self.assertGreater(left_of_line, 0.0)
        self.assertAlmostEqual(right_of_line, -left_of_line)

    def test_error_is_measured_against_the_racing_line_not_the_centre(self):
        """Sitting exactly on a non-zero target is zero error."""
        on_line = ctl.steering(0.0, 0.16, 0.0, 0.16, A=25.0, B=1.6, D=0.072)
        self.assertEqual(on_line, 0.0)

    def test_angle_term_signs_correctly(self):
        self.assertGreater(ctl.steering(0.2, 0.0, 0.0, 0.0, A=25.0, B=1.6, D=0.072), 0.0)
        self.assertLess(ctl.steering(-0.2, 0.0, 0.0, 0.0, A=25.0, B=1.6, D=0.072), 0.0)

    def test_rate_damping_opposes_lateral_motion(self):
        still = ctl.steering(0.0, 0.5, 0.0, 0.0, A=25.0, B=1.6, D=0.072)
        drifting_right = ctl.steering(0.0, 0.5, 0.01, 0.0, A=25.0, B=1.6, D=0.072)
        self.assertLess(drifting_right, still)

    def test_output_is_clipped(self):
        self.assertEqual(ctl.steering(3.0, 0.0, 0.0, 0.0, A=25.0, B=1.6, D=0.0), 1.0)
        self.assertEqual(ctl.steering(-3.0, 0.0, 0.0, 0.0, A=25.0, B=1.6, D=0.0), -1.0)


class TestSectorGain(unittest.TestCase):
    def test_main_sector_uses_k(self):
        p = load_params()
        self.assertEqual(ctl.effective_k(0.0, p), p.K)
        self.assertEqual(ctl.effective_k(p.switch_dist - 1e-6, p), p.K)

    def test_braking_sector_uses_k_final(self):
        p = load_params()
        self.assertEqual(ctl.effective_k(p.switch_dist, p), p.K_final)
        self.assertEqual(ctl.effective_k(p.back_dist - 1e-6, p), p.K_final)

    def test_finish_straight_returns_to_k(self):
        p = load_params()
        self.assertEqual(ctl.effective_k(p.back_dist, p), p.K)
        self.assertEqual(ctl.effective_k(ctl.TRACK_LAP_M - 1.0, p), p.K)

    def test_k_final_is_the_slower_gain(self):
        p = load_params()
        self.assertLess(p.K_final, p.K)


class TestSpeedTarget(unittest.TestCase):
    def test_lookahead_scales_the_target(self):
        p = load_params()
        near = ctl.target_speed(0.0, 20.0, p)
        far = ctl.target_speed(0.0, 90.0, p)
        self.assertLess(near, far)

    def test_target_is_clamped_to_the_speed_cap(self):
        p = load_params()
        self.assertAlmostEqual(ctl.target_speed(0.0, 200.0, p), p.C)

    def test_target_has_a_floor(self):
        """Off track the range sensor reads -1; the target must stay positive."""
        p = load_params()
        self.assertAlmostEqual(ctl.target_speed(0.0, -1.0, p), ctl.V_TARGET_MIN)

    def test_s35_cap_applies_inside_the_chicane(self):
        p = load_params()
        inside = ctl.target_speed(p.s35_start + 50.0, 200.0, p)
        self.assertAlmostEqual(inside, p.C_s35)

    def test_s35_cap_does_not_apply_outside_the_chicane(self):
        p = load_params()
        before = ctl.target_speed(p.s35_start - 1.0, 200.0, p)
        after = ctl.target_speed(ctl.D_S35_EXIT + 1.0, 200.0, p)
        self.assertAlmostEqual(before, p.C)
        self.assertAlmostEqual(after, p.C)
        self.assertGreater(before, p.C_s35)

    def test_s35_cap_is_a_ceiling_not_a_setpoint(self):
        """In a slow part of the chicane the lookahead target already sits below
        the cap, and the cap must not raise it."""
        p = load_params()
        v = ctl.target_speed(p.s35_start + 50.0, 10.0, p)
        self.assertLess(v, p.C_s35)

    def test_s35_cap_covers_the_documented_crash_zone(self):
        """The chicane crashes were located at 2339-2510 m; the cap must span it."""
        p = load_params()
        self.assertLessEqual(p.s35_start, 2339.0)
        self.assertGreaterEqual(ctl.D_S35_EXIT, 2510.0)


class TestThrottle(unittest.TestCase):
    def test_accelerates_when_below_target(self):
        self.assertGreater(ctl.rule_throttle(150.0, 100.0, 0.00938), 0.0)

    def test_brakes_when_above_target(self):
        self.assertLess(ctl.rule_throttle(100.0, 150.0, 0.00938), 0.0)

    def test_output_is_clipped(self):
        self.assertEqual(ctl.rule_throttle(1e6, 0.0, 0.00938), 1.0)
        self.assertEqual(ctl.rule_throttle(0.0, 1e6, 0.00938), -1.0)


class TestResidual(unittest.TestCase):
    def test_zero_network_output_is_the_rule_controller(self):
        """The property that makes the residual safe to start from."""
        for rule in (-1.0, -0.4, 0.0, 0.37, 1.0):
            self.assertAlmostEqual(ctl.residual_throttle(rule, 0.0), rule)

    def test_correction_is_bounded_by_the_scale(self):
        for nn in (-1.0, 1.0):
            delta = ctl.residual_throttle(0.0, nn) - 0.0
            self.assertLessEqual(abs(delta), ctl.RESIDUAL_SCALE + 1e-12)

    def test_output_is_clipped(self):
        self.assertEqual(ctl.residual_throttle(1.0, 1.0), 1.0)
        self.assertEqual(ctl.residual_throttle(-1.0, -1.0), -1.0)

    def test_ema_converges_towards_the_raw_signal(self):
        v = 0.0
        for _ in range(200):
            v = ctl.update_nn_ema(v, 1.0)
        self.assertAlmostEqual(v, 1.0, places=6)

    def test_ema_smooths_a_single_spike(self):
        v = ctl.update_nn_ema(0.0, 1.0)
        self.assertAlmostEqual(v, ctl.EMA_ALPHA)


class TestOverrideZones(unittest.TestCase):
    def test_sprint_zone_forces_full_throttle_over_the_network(self):
        p = load_params()
        t = ctl.throttle_command(p.back_dist + 10.0, 200.0, 5.0, p, nn_ema=-1.0)
        self.assertEqual(t, 1.0)

    def test_poststart_zone_forces_full_brake_over_the_network(self):
        p = load_params()
        t = ctl.throttle_command(100.0, 200.0, 200.0, p, nn_ema=1.0)
        self.assertEqual(t, -1.0)

    def test_poststart_brake_needs_both_conditions(self):
        self.assertFalse(ctl.in_poststart_brake(100.0, 100.0))
        self.assertFalse(ctl.in_poststart_brake(600.0, 200.0))
        self.assertTrue(ctl.in_poststart_brake(100.0, 200.0))

    def test_network_is_ignored_in_the_braking_sector(self):
        p = load_params()
        d = p.switch_dist + 10.0
        with_nn = ctl.throttle_command(d, 100.0, 80.0, p, nn_ema=1.0)
        without = ctl.throttle_command(d, 100.0, 80.0, p, nn_ema=None)
        self.assertEqual(with_nn, without)

    def test_network_acts_in_the_open_part_of_the_lap(self):
        p = load_params()
        d = 1800.0  # between the two racing-line zones, no override active
        with_nn = ctl.throttle_command(d, 100.0, 60.0, p, nn_ema=1.0)
        without = ctl.throttle_command(d, 100.0, 60.0, p, nn_ema=None)
        self.assertNotEqual(with_nn, without)

    def test_zones_are_mutually_exclusive(self):
        p = load_params()
        d = 0.0
        while d < ctl.TRACK_LAP_M:
            self.assertFalse(
                ctl.in_sprint_zone(d, p) and ctl.in_kfinal_zone(d, p),
                "sprint and braking sectors overlap at %.1f m" % d,
            )
            d += 1.0


class TestThrottleCommandRanges(unittest.TestCase):
    def test_command_stays_in_range_across_the_lap(self):
        p = load_params()
        d = 0.0
        while d < ctl.TRACK_LAP_M:
            for speed in (0.0, 60.0, 150.0, 250.0):
                for fwd in (-1.0, 5.0, 50.0, 200.0):
                    for nn in (None, -1.0, 0.0, 1.0):
                        t = ctl.throttle_command(d, speed, fwd, p, nn)
                        self.assertGreaterEqual(t, -1.0)
                        self.assertLessEqual(t, 1.0)
            d += 37.0


class TestObservation(unittest.TestCase):
    def test_observation_is_23_dimensional(self):
        obs = ctl.observation([1.0] * 19, 100.0, 0.1, 0.2, 1500.0)
        self.assertEqual(len(obs), 23)
        self.assertEqual(len(ctl.observation_scale()), 23)

    def test_normalised_observation_is_roughly_unit_scale(self):
        track = [200.0] * 19
        obs = ctl.observation(track, 200.0, ctl.PI, 2.0, ctl.TRACK_LAP_M)
        scale = ctl.observation_scale()
        for value, s in zip(obs, scale):
            self.assertAlmostEqual(value / s, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
