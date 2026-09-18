# pyright: reportMissingTypeStubs=false, reportMissingImports=false
"""Standalone diagnostic: broadcast a stream of GameState packets to an
already-running rlviser and see whether the stadium/field ever spawns.
Not part of the real training path -- for debugging only."""

import time

import RocketSim as rsim
import rlviser_py as rlviser

BOOST_LOCATIONS = [(0.0, 0.0, 0.0)] * 34  # placeholder, count doesn't matter here

print("Sending HOOPS for 5s...")
deadline = time.perf_counter() + 5.0
tick = 0
while time.perf_counter() < deadline:
    tick += 1
    rlviser.render(
        tick_count=tick,
        tick_rate=60.0,
        game_mode=rsim.GameMode.HOOPS,
        boost_pad_states=[],
        ball=rsim.BallState(),
        cars=[],
    )
    time.sleep(1 / 60)

print("Sending SOCCAR for 10s...")
deadline = time.perf_counter() + 10.0
while time.perf_counter() < deadline:
    tick += 1
    rlviser.render(
        tick_count=tick,
        tick_rate=60.0,
        game_mode=rsim.GameMode.SOCCAR,
        boost_pad_states=[],
        ball=rsim.BallState(),
        cars=[],
    )
    time.sleep(1 / 60)

print("Done.")
