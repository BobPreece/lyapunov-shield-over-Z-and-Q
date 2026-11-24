from __future__ import annotations

import io
import json
import pathlib
import tempfile
import unittest
from contextlib import redirect_stdout

from lyapunov_shield import (
    Verdict,
    build_coupled_platoon_microgrid_4d,
    build_double_integrator_2d_corridor_diamond,
    build_double_integrator_4d_corridor_diamond,
    build_dubins_3sector_obstacle_3d,
    build_inverted_pendulum_sector_2d,
    build_path_following_car_2d,
    check_barrier_certificate_bundle,
    check_lyapunov_certificate_bundle,
    check_shield_step_result,
    simulate_shielded_4d_double_integrator,
    verify_barrier,
    verify_continuous_lyapunov,
)
from lyapunov_shield.cli import main as cli_main
from lyapunov_shield.io_schema import (
    export_signed_certificate_bundle,
    load_pytorch_state_dict_json,
    verify_signed_certificate_bundle,
)


class TestBenchmarksAndCLI(unittest.TestCase):
    def test_benchmark1_inverted_pendulum_sector_and_roa(self) -> None:
        v_p, p_p, d_p = build_inverted_pendulum_sector_2d(torque_limit=None)
        r_glob = verify_continuous_lyapunov(v_p, p_p, d_p, decay_lambda=(1, 2))
        self.assertEqual(r_glob.verdict, Verdict.CERTIFIED_EXACT)
        self.assertTrue(check_lyapunov_certificate_bundle(r_glob))

        v_s, p_s, d_s = build_inverted_pendulum_sector_2d(torque_limit=(3, 2))
        r_roa = verify_continuous_lyapunov(v_s, p_s, d_s, decay_lambda=(1, 2), compute_maximal_roa=True)
        self.assertEqual(r_roa.verdict, Verdict.CERTIFIED_EXACT)
        self.assertEqual(r_roa.rho_star, (1, 1))
        self.assertTrue(check_lyapunov_certificate_bundle(r_roa))

    def test_benchmark2_path_car_2d_and_dubins_3sector_3d(self) -> None:
        hc, pc, dc, ic, uc = build_path_following_car_2d()
        rc = verify_barrier(hc, pc, domain=dc, initial_set=ic, unsafe_set=uc)
        self.assertEqual(rc.verdict, Verdict.CERTIFIED_EXACT)
        self.assertTrue(check_barrier_certificate_bundle(rc))

        hd, pd, dd, id_, ud = build_dubins_3sector_obstacle_3d()
        rd = verify_barrier(hd, pd, domain=dd, initial_set=id_, unsafe_set=ud)
        self.assertEqual(rd.verdict, Verdict.CERTIFIED_EXACT)
        self.assertTrue(check_barrier_certificate_bundle(rd))

    def test_benchmark3_double_integrator_2d_and_4d_corridor_diamond(self) -> None:
        c2, dr2, p2, d2 = build_double_integrator_2d_corridor_diamond()
        r_c2 = verify_barrier(c2, p2, domain=d2)
        r_dr2 = verify_barrier(dr2, p2, domain=d2)
        self.assertEqual(r_c2.verdict, Verdict.CERTIFIED_EXACT)
        self.assertEqual(r_dr2.verdict, Verdict.CERTIFIED_EXACT)
        self.assertTrue(check_barrier_certificate_bundle(r_c2))
        self.assertTrue(check_barrier_certificate_bundle(r_dr2))

        c4, do4, pc4, pd4, dv4, i4, u4 = build_double_integrator_4d_corridor_diamond()
        r_c4 = verify_barrier(c4, pc4)
        r_do4 = verify_barrier(do4, pd4, domain=dv4, initial_set=i4, unsafe_set=u4)
        self.assertEqual(r_c4.verdict, Verdict.CERTIFIED_EXACT)
        self.assertEqual(r_do4.verdict, Verdict.CERTIFIED_EXACT)
        self.assertTrue(check_barrier_certificate_bundle(r_c4))
        self.assertTrue(check_barrier_certificate_bundle(r_do4))

        trace = simulate_shielded_4d_double_integrator(steps=8)
        self.assertEqual(len(trace), 8)
        for _, step_res, h_c, h_d in trace:
            self.assertEqual(step_res.verdict, Verdict.CERTIFIED_EXACT)
            self.assertTrue(check_shield_step_result(step_res))
            self.assertGreaterEqual(h_c[0], 0)
            self.assertGreaterEqual(h_d[0], 0)

    def test_benchmark4_coupled_platoon_microgrid_4d(self) -> None:
        vp4, pnet4, hb4, pbp4, dom4 = build_coupled_platoon_microgrid_4d()
        r_lyap = verify_continuous_lyapunov(vp4, pnet4, dom4, decay_lambda=(1, 2))
        self.assertEqual(r_lyap.verdict, Verdict.CERTIFIED_EXACT)
        self.assertTrue(check_lyapunov_certificate_bundle(r_lyap))

        r_bar = verify_barrier(hb4, pbp4, domain=dom4)
        self.assertEqual(r_bar.verdict, Verdict.CERTIFIED_EXACT)
        self.assertTrue(check_barrier_certificate_bundle(r_bar))

    def test_io_schema_pytorch_state_dict_and_sha256_tamper_detection(self) -> None:
        state_dict = {
            "layers.0.weight": [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
            "layers.0.bias": [0.1, 0.1, 0.0, 0.0],
            "layers.2.weight": [[1.0, 1.0, 1.0, 1.0]],
            "layers.2.bias": [-0.2],
        }
        net = load_pytorch_state_dict_json(
            state_dict, scale_denom=16, anchor_equilibrium=((0, 0), 1)
        )
        self.assertEqual(net.eval_scalar(((0, 0), 1)), (0, 1))

        v_p, p_p, d_p = build_inverted_pendulum_sector_2d(torque_limit=None)
        res = verify_continuous_lyapunov(v_p, p_p, d_p, decay_lambda=(1, 2))
        bundle = export_signed_certificate_bundle(res, "lyapunov_continuous")
        ok, msg = verify_signed_certificate_bundle(bundle)
        self.assertTrue(ok, msg)

        # Tamper with payload and verify SHA-256 rejection
        tampered = json.loads(json.dumps(bundle))
        tampered["payload"]["shards_certified"] += 1
        ok_bad, msg_bad = verify_signed_certificate_bundle(tampered)
        self.assertFalse(ok_bad)
        self.assertIn("SHA-256", msg_bad)

    def test_cli_all_seven_subcommands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = pathlib.Path(tmp) / "out_cert.json"
            commands = [
                ["verify-lyapunov", "--benchmark", "inverted_pendulum", "--output", str(out_path)],
                ["verify-barrier", "--benchmark", "path_car_2d", "--output", str(out_path)],
                ["audit-filippov", "--benchmark", "inverted_pendulum", "--output", str(out_path)],
                ["detect-orbits", "--benchmark", "trap3a", "--output", str(out_path)],
                ["shield-step", "--benchmark", "dubins_3d", "--output", str(out_path)],
                ["synthesize-cegis", "--max-iters", "10", "--output", str(out_path)],
            ]
            for argv in commands:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = cli_main(argv)
                self.assertEqual(rc, 0)
                parsed = json.loads(buf.getvalue())
                self.assertIn("sha256", parsed)
                ok_b, msg_b = verify_signed_certificate_bundle(out_path)
                self.assertTrue(ok_b, msg_b)

    def test_exported_examples_verify_cleanly(self) -> None:
        root = pathlib.Path(__file__).resolve().parent.parent
        ex_dir = root / "examples"
        cert_files = [
            "inverted_pendulum_sector_cert.json",
            "dubins_3sector_barrier_cert.json",
            "platoon_4d_lyapunov_cert.json",
            "trap1_filippov_counterexample_bundle.json",
        ]
        for fname in cert_files:
            fpath = ex_dir / fname
            self.assertTrue(fpath.exists(), f"Missing example {fpath}")
            ok, msg = verify_signed_certificate_bundle(fpath)
            self.assertTrue(ok, f"{fname} failed verification: {msg}")

        # Also test loading model JSON via CLI --input
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli_main(
                ["verify-lyapunov", "--input", str(ex_dir / "inverted_pendulum_sector_model.json")]
            )
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(buf.getvalue())["verdict"], "CERTIFIED_EXACT")


if __name__ == "__main__":
    unittest.main()
