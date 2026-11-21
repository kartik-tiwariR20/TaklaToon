"""Pygame presentation layer. Drawing never modifies simulator state."""
import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import math
import numpy as np
import pygame
from .levels import WORLD_WIDTH, WORLD_HEIGHT, PLAYER_WIDTH, PLAYER_HEIGHT

WIDTH, HEIGHT = 1440, 900
C = {"bg": "#FFF7DE", "ink": "#493326", "muted": "#82664F", "tan": "#E8D3AF",
     "paper": "#FFFCF0", "gold": "#BD8434", "edge": "#CFB58D", "sage": "#74805C",
     "danger": "#AD5B45", "sky": "#F5E8BD", "stone": "#BB9870", "dark": "#3D2C23"}


class Renderer:
    def __init__(self, human=False):
        pygame.font.init()
        self.human = human
        self.surface = pygame.Surface((WIDTH, HEIGHT))
        self.window = None
        if human:
            pygame.display.init()
            desktop_w, desktop_h = pygame.display.get_desktop_sizes()[0]
            scale = min(1., max(640, desktop_w-80)/WIDTH, max(400, desktop_h-140)/HEIGHT)
            initial_size = (round(WIDTH*scale), round(HEIGHT*scale))
            self.window = pygame.display.set_mode(initial_size, pygame.RESIZABLE)
            pygame.display.set_caption("TaklaToon | Physics Sandbox")
        self.fonts = {}
        self.paused = False
        self.show_physics = True
        self.show_trail = True
        self.controller = "MANUAL CONTROL"
        self.record_path = "Recording disabled"
        self.seed = 42
        self.buttons = []
        self.viewport = pygame.Rect(40, 190, 1000, 500)

    def font(self, size, bold=False):
        key = (size, bold)
        if key not in self.fonts:
            self.fonts[key] = pygame.font.SysFont("segoeui", size, bold=bold)
        return self.fonts[key]

    def text(self, value, x, y, size=22, color="ink", bold=False, max_width=None):
        f = self.font(size, bold)
        value = str(value)
        if max_width:
            while f.size(value)[0] > max_width and len(value) > 3:
                value = value[:-4] + "..."
        self.surface.blit(f.render(value, True, C.get(color, color)), (x, y))

    def panel(self, rect, fill="paper", border=True):
        pygame.draw.rect(self.surface, C[fill], rect, border_radius=16)
        if border:
            pygame.draw.rect(self.surface, C["edge"], rect, width=1, border_radius=16)

    def point(self, x, y):
        scale = self.viewport.width / WORLD_WIDTH
        return (round(self.viewport.left + x * scale), round(self.viewport.bottom - y * scale))

    def button(self, label, rect, command, active=False):
        self.panel(rect, "tan" if active else "paper")
        text = self.font(20, active).render(label, True, C["ink"])
        self.surface.blit(text, text.get_rect(center=rect.center))
        self.buttons.append((rect, command))

    def click(self, pos):
        if not self.window:
            return None
        ww, wh = self.window.get_size()
        scaled = (pos[0] * WIDTH / ww, pos[1] * HEIGHT / wh)
        return next((command for rect, command in self.buttons if rect.collidepoint(scaled)), None)

    def draw(self, env):
        s = self.surface
        s.fill(C["bg"])
        self.buttons = []
        self.text("TAKLATOON", 40, 25, 45, bold=True)
        self.text("PHYSICS SANDBOX", 365, 42, 20, "gold", True)
        self.text("Environment foundation  /  no trained AI yet", 40, 84, 21, "muted")
        for i, (name, title) in enumerate((("motion", "1  MotionLab"), ("jump", "2  JumpLab"), ("gap", "3  GapRunner"))):
            self.button(title, pygame.Rect(40 + i * 224, 124, 210, 44), "level:" + name, env.config.level == name)
        self.button("R  Reset", pygame.Rect(752, 124, 132, 44), "reset")
        self.button("P  Resume" if self.paused else "P  Pause", pygame.Rect(896, 124, 144, 44), "pause", self.paused)
        self.text(self.controller, 1070, 42, 22, "sage", True, 330)
        self.text(f"Seed {self.seed}  /  {env.config.control_hz} Hz control", 1070, 81, 20, "muted")
        self.button("B  Reference controller", pygame.Rect(1064, 124, 336, 44), "reference", self.controller.startswith("SCRIPTED"))

        # The world is drawn in one fixed coordinate system, independent of display FPS.
        self.panel(self.viewport, "sky")
        old_clip = s.get_clip()
        s.set_clip(self.viewport)
        for i in range(6):
            pygame.draw.polygon(s, "#E8D5A7", [(40+i*205, 470), (165+i*205, 250+(i%3)*30), (330+i*205, 470)])
        for x in range(0, 1201, 100):
            a, b = self.point(x, 0), self.point(x, 600)
            pygame.draw.line(s, "#EBDDB7", a, b, 1)
        for y in range(0, 601, 100):
            pygame.draw.line(s, "#EBDDB7", self.point(0, y), self.point(1200, y), 1)
        for platform in env.level.platforms:
            left, top = self.point(platform.left, platform.top)
            right, bottom = self.point(platform.right, -150)
            pygame.draw.rect(s, C["stone"], (left, top, right-left, bottom-top))
            pygame.draw.rect(s, "#D4B991", (left, top, right-left, 12))
            pygame.draw.line(s, C["ink"], (left, top), (right, top), 2)
            for y in range(top+34, bottom, 40):
                pygame.draw.line(s, "#A7835E", (left, y), (right, y), 1)
                offset = 35 if ((y-top)//40)%2 else 0
                for x in range(left+offset, right, 75):
                    pygame.draw.line(s, "#A7835E", (x, y-35), (x, y), 1)
        gx, gy = self.point(*env.level.goal)
        if env.config.level == "motion":
            pygame.draw.rect(s, "#C9D0A4", (gx-35, gy-5, 70, 12), border_radius=4)
            pygame.draw.line(s, C["sage"], (gx-35, gy), (gx-35, gy-80), 3)
            pygame.draw.line(s, C["sage"], (gx+35, gy), (gx+35, gy-80), 3)
        else:
            pygame.draw.line(s, C["ink"], (gx, gy), (gx, gy-92), 4)
            pygame.draw.polygon(s, C["sage"], [(gx, gy-90), (gx+55, gy-76), (gx, gy-60)])
        if self.show_trail:
            for i, point in enumerate(env.trail[::3]):
                px, py = self.point(*point)
                pygame.draw.circle(s, "#B79259", (px, py), 2 if i < len(env.trail[::3])-8 else 3)
        x, y = self.point(*env.body.position)
        # Simple original robot graphics; the Pymunk collider remains a 32x44 box.
        pygame.draw.ellipse(s, "#A98B60", (x-21, self.point(0, env.body.position.y-23)[1]-2, 42, 8))
        pygame.draw.rect(s, "#BA7D29", (x-11, y+6, 9, 14), border_radius=3)
        pygame.draw.rect(s, "#BA7D29", (x+3, y+6, 9, 14), border_radius=3)
        pygame.draw.rect(s, "#DEAF54", (x-14, y-8, 28, 22), border_radius=5)
        pygame.draw.rect(s, C["ink"], (x-16, y-22, 32, 26), 0, border_radius=8)
        pygame.draw.rect(s, "#EDC669", (x-14, y-21, 28, 23), 0, border_radius=7)
        pygame.draw.rect(s, C["dark"], (x-10, y-16, 23, 15), border_radius=5)
        pygame.draw.rect(s, "#FFE4A0", (x-5, y-13, 4, 9), border_radius=2)
        pygame.draw.rect(s, "#FFE4A0", (x+5, y-13, 4, 9), border_radius=2)
        pygame.draw.line(s, C["ink"], (x-7, y-22), (x-11, y-31), 2)
        pygame.draw.circle(s, "#EFC76A", (x-11, y-32), 4)
        s.set_clip(old_clip)
        pygame.draw.rect(s, C["edge"], self.viewport, width=1, border_radius=16)

        # Evaluator panel. Numeric observations and Gymnasium info never include these values.
        self.panel(pygame.Rect(1064, 190, 336, 500))
        self.text("PHYSICS LAB", 1084, 209, 26, bold=True)
        self.text("Evaluator-only display  /  H to hide", 1084, 247, 17, "muted")
        rows = [("Gravity", env.physics.gravity, "G"), ("Jump speed", env.physics.jump_speed, "J"),
                ("Ground drag", env.physics.ground_drag, "F")]
        for i, (label, value, key) in enumerate(rows):
            yy = 290 + i*61
            self.text(f"{key}  {label}", 1084, yy, 20, "muted")
            self.text(f"{value:g}" if self.show_physics else "hidden", 1255, yy-3, 26, bold=True)
        for i, preset in enumerate(("normal", "heavy", "floaty", "slippery")):
            self.button(f"F{i+1}  {preset.title()}", pygame.Rect(1084+(i%2)*152, 489+(i//2)*46, 144, 38),
                        "preset:"+preset, env.config.preset == preset and env.config.physics_override is None)
        self.text("Randomization: " + ("ON" if env.config.randomize else "OFF"), 1084, 591, 20, "sage", True)
        self.text("T toggles and resets episode", 1084, 622, 18, "muted")
        self.text("Actual trajectory trail, not AI prediction", 1084, 656, 16, "muted")

        self.panel(pygame.Rect(40, 711, 1360, 77), "tan")
        vals = [("X / Y", f"{env.body.position.x:.1f} / {env.body.position.y:.1f}"),
                ("VX / VY", f"{env.body.velocity.x:.1f} / {env.body.velocity.y:.1f}"),
                ("GROUNDED", "yes" if env._grounded else "no"), ("ACTION", env.last_action.name),
                ("STEPS", f"{env.steps}/{env.config.max_steps}"), ("RETURN", f"{env.return_so_far:+.2f}")]
        for i,(label,value) in enumerate(vals):
            xx = 59+i*225
            self.text(label, xx, 722, 15, "muted", True)
            self.text(value, xx, 747, 21, bold=True, max_width=215)
        self.text("MOVE  A/D or arrows     JUMP  Space/W/up     R reset     N next seed     1/2/3 labs     F1-F4 presets", 42, 803, 20, "muted")
        self.text("G gravity   J jump speed   F drag   H hide values   T randomize   C trail   P pause   B reference   Esc exit", 42, 834, 19, "muted")
        self.text("LOG: "+self.record_path, 42, 871, 15, "muted", max_width=1355)
        self.text(env.level.title+"   /   "+env.level.instruction, 57, 204, 21, bold=True)
        if self.paused or env._done:
            overlay = pygame.Rect(302, 360, 480, 140)
            self.panel(overlay)
            message = "PAUSED" if self.paused else "GOAL REACHED" if env.last_info.get("success") else "TRY AGAIN"
            self.text(message, 333, 382, 35, "sage" if env.last_info.get("success") else "ink", True)
            self.text("P to resume" if self.paused else "R to reset   /   change physics to experiment", 333, 440, 20, "muted")
        if self.human:
            pygame.event.pump()
            if self.window.get_size() != (WIDTH, HEIGHT):
                self.window.blit(pygame.transform.smoothscale(s, self.window.get_size()), (0,0))
            else:
                self.window.blit(s, (0,0))
            pygame.display.flip()
            return None
        return np.transpose(pygame.surfarray.array3d(s), (1,0,2)).copy()

    def close(self):
        if self.human:
            pygame.display.quit()

