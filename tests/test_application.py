"""Exercise the actual play loop with SDL's dummy display and synthetic inputs."""
from types import SimpleNamespace
import csv
import json
import pygame
from taklatoon import TaklaToonEnv
from taklatoon import cli


def test_play_controls_and_clean_exit(monkeypatch, tmp_path):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    instances = []

    def factory(**kwargs):
        env = TaklaToonEnv(**kwargs)
        instances.append(env)
        return env

    monkeypatch.setattr(cli, "TaklaToonEnv", factory)

    class Clock:
        def tick(self, fps): return 34

    monkeypatch.setattr(pygame.time, "Clock", Clock)
    sequence = iter([
        [pygame.event.Event(pygame.KEYDOWN, key=k)] for k in (
            pygame.K_F2, pygame.K_p, pygame.K_p, pygame.K_g, pygame.K_j,
            pygame.K_f, pygame.K_h, pygame.K_c, pygame.K_t, pygame.K_1,
            pygame.K_b, pygame.K_n, pygame.K_r)
    ] + [[pygame.event.Event(pygame.QUIT)]])
    monkeypatch.setattr(pygame.event, "get", lambda: next(sequence, []))
    args = SimpleNamespace(level="gap", preset="normal", randomize=False, max_steps=600,
                           no_record=False, output=tmp_path/"manual", seed=42, autoplay=False, frames=20)
    cli.play(args)
    assert instances[0].config.level == "motion"
    assert instances[0].config.randomize
    assert instances[0]._renderer is None  # Window closed cleanly.
    with (args.output/"episodes.csv").open() as stream:
        summaries = list(csv.DictReader(stream))
    assert len(summaries) >= 8
    assert summaries[-1]["outcome"] == "interrupted"
    transitions = [json.loads(line) for line in (args.output/"transitions.jsonl").read_text().splitlines()]
    assert {t["control_source"] for t in transitions} == {"manual", "scripted_reference"}
    cli.inspect_run(SimpleNamespace(directory=args.output))


def test_click_coordinates_scale_with_window(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    env = TaklaToonEnv(render_mode="human")
    try:
        env.reset()
        r = env._renderer
        ww, wh = r.window.get_size()
        assert r.click((100*ww/1440, 145*wh/900)) == "level:motion"
        r.window = pygame.display.set_mode((720, 450), pygame.RESIZABLE)
        assert r.click((50, 72)) == "level:motion"
    finally:
        env.close()

