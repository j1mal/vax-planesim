"""
VAX PlaneSim — Step 2
Environment, circling fighter, lock-on box, and predicted velocity vector.

Python 3.12 recommended (Panda3D wheels):
    py -3.12 -m venv .venv
    .venv\\Scripts\\pip install -r requirements.txt
    .venv\\Scripts\\python main.py

Controls: hold RMB to orbit, RMB+WASD to move, scroll to zoom, MMB to pan.
"""

from __future__ import annotations

import math

from ursina import (
    EditorCamera,
    Entity,
    Grid,
    Mesh,
    Text,
    Ursina,
    Vec2,
    Vec3,
    camera,
    color,
    time,
    window,
)


PHOSPHOR = color.rgb32(0, 255, 92)
PHOSPHOR_DIM = color.rgba32(0, 255, 92, 90)
HULL = color.rgb32(26, 30, 36)
HULL_LIGHT = color.rgb32(40, 46, 54)
DECK = color.rgb32(18, 22, 28)


def wire_cube_mesh() -> Mesh:
    s = 0.5
    corners = [
        Vec3(-s, -s, -s),
        Vec3(s, -s, -s),
        Vec3(s, -s, s),
        Vec3(-s, -s, s),
        Vec3(-s, s, -s),
        Vec3(s, s, -s),
        Vec3(s, s, s),
        Vec3(-s, s, s),
    ]
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    return Mesh(vertices=corners, triangles=edges, mode="line", thickness=1, static=True)


def corner_bracket_mesh(arm: float = 0.22) -> Mesh:
    s = 0.5
    verts: list[Vec3] = []
    edges: list[tuple[int, int]] = []
    for x in (-s, s):
        for y in (-s, s):
            for z in (-s, s):
                origin = Vec3(x, y, z)
                for delta in (
                    Vec3(-math.copysign(arm, x), 0, 0),
                    Vec3(0, -math.copysign(arm, y), 0),
                    Vec3(0, 0, -math.copysign(arm, z)),
                ):
                    i = len(verts)
                    verts.extend((origin, origin + delta))
                    edges.append((i, i + 1))
    return Mesh(vertices=verts, triangles=edges, mode="line", thickness=2, static=True)


def axis_cross_mesh(size: float = 0.7) -> Mesh:
    s = size
    verts = [
        Vec3(-s, 0, 0),
        Vec3(s, 0, 0),
        Vec3(0, -s, 0),
        Vec3(0, s, 0),
        Vec3(0, 0, -s),
        Vec3(0, 0, s),
    ]
    return Mesh(vertices=verts, triangles=((0, 1), (2, 3), (4, 5)), mode="line", thickness=2, static=True)


class AircraftCarrier(Entity):
    """Simple block carrier sitting on the waterline."""

    def __init__(self, **kwargs):
        super().__init__(position=Vec3(0, 0, 0), **kwargs)

        hull = Entity(
            parent=self,
            model="cube",
            scale=(14, 2.2, 52),
            position=(0, 1.1, 0),
            color=HULL,
        )
        Entity(
            parent=hull,
            model="cube",
            scale=1.002,
            wireframe=True,
            color=color.rgb32(0, 70, 48),
            unlit=True,
        )
        Entity(
            parent=self,
            model="cube",
            scale=(13.2, 0.16, 50),
            position=(0, 2.25, 0),
            color=DECK,
        )
        Entity(
            parent=self,
            model="cube",
            scale=(0.12, 0.05, 44),
            position=(0, 2.36, 1),
            color=PHOSPHOR,
            unlit=True,
        )
        Entity(
            parent=self,
            model="cube",
            scale=(3.4, 4.6, 8),
            position=(4.4, 4.6, 6),
            color=HULL_LIGHT,
        )
        Entity(
            parent=self,
            model="cube",
            scale=(0.35, 6.2, 0.35),
            position=(4.4, 8.4, 6),
            color=color.rgb32(58, 64, 72),
        )


class FighterJet(Entity):
    """Low-poly jet flying a circular combat air patrol around the carrier."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orbit_radius = 46.0
        self.orbit_height = 18.0
        self.orbit_speed = 0.28
        self.theta = 0.0
        self.bank_angle = -20
        self.velocity = Vec3(0, 0, 0)
        # Local-space volume the tracking box wraps.
        self.lock_size = Vec3(11, 3.2, 12)

        Entity(parent=self, model="cube", scale=(1.15, 0.85, 8.2), color=HULL)
        Entity(
            parent=self,
            model="cube",
            scale=(0.7, 0.5, 2.0),
            position=(0, 0.05, 4.8),
            color=color.rgb32(0, 160, 70),
        )
        Entity(
            parent=self,
            model="cube",
            scale=(10.5, 0.16, 2.5),
            position=(0, -0.05, 0.4),
            color=HULL_LIGHT,
        )
        Entity(
            parent=self,
            model="cube",
            scale=(3.4, 0.12, 1.2),
            position=(0, 0.2, -3.4),
            color=HULL_LIGHT,
        )
        Entity(
            parent=self,
            model="cube",
            scale=(0.14, 1.7, 1.5),
            position=(0, 1.0, -3.5),
            color=HULL,
        )
        Entity(
            parent=self,
            model="cube",
            scale=(0.65, 0.32, 1.5),
            position=(0, 0.55, 1.6),
            color=color.rgb32(36, 70, 78),
        )

        self.position = self._place_on_orbit(self.theta)

    def _place_on_orbit(self, theta: float) -> Vec3:
        return Vec3(
            math.cos(theta) * self.orbit_radius,
            self.orbit_height,
            math.sin(theta) * self.orbit_radius,
        )

    def update(self):
        dt = time.dt
        self.theta += self.orbit_speed * dt
        pos = self._place_on_orbit(self.theta)
        ahead = self._place_on_orbit(self.theta + 0.12)
        speed = self.orbit_speed * self.orbit_radius
        self.velocity = (ahead - pos).normalized() * speed
        self.position = pos
        self.look_at(ahead)
        self.rotation_z = self.bank_angle


class TrackingBox(Entity):
    """3D phosphor wireframe that stays locked to a target's bounds."""

    def __init__(self, target: FighterJet, padding: float = 1.15, **kwargs):
        super().__init__(parent=target, unlit=True, **kwargs)
        self.target = target
        self.padding = padding

        Entity(
            parent=self,
            model=wire_cube_mesh(),
            color=PHOSPHOR_DIM,
            unlit=True,
        )
        Entity(
            parent=self,
            model=corner_bracket_mesh(),
            color=PHOSPHOR,
            unlit=True,
        )

        self.lock_label = Text(
            text="",
            color=PHOSPHOR,
            scale=0.85,
            origin=(-0.5, 0.5),
            font=Text.default_monospace_font,
        )
        self._sync_scale(1.0)

    def _sync_scale(self, pulse: float) -> None:
        size = getattr(self.target, "lock_size", Vec3(2, 2, 2))
        self.scale = size * self.padding * pulse

    def update(self):
        pulse = 1.0 + 0.035 * math.sin(time.time() * 5.0)
        self._sync_scale(pulse)

        to_target = self.target.world_position - camera.world_position
        visible = camera.forward.dot(to_target) > 0
        self.lock_label.enabled = visible
        if not visible:
            return

        screen = self.target.screen_position
        pos = self.target.world_position
        speed = self.target.velocity.length()
        self.lock_label.position = Vec2(screen.x + 0.035, screen.y + 0.05)
        self.lock_label.text = (
            f"TGT-01   LOCK\n"
            f"XYZ  {pos.x:6.1f}  {pos.y:6.1f}  {pos.z:6.1f}\n"
            f"SPD  {speed:5.1f}"
        )


class PredictedPath(Entity):
    """Linear lead vector from the target's instantaneous velocity."""

    def __init__(self, target: FighterJet, lookahead: float = 6.0, dashes: int = 12, **kwargs):
        super().__init__(unlit=True, **kwargs)
        self.target = target
        self.lookahead = lookahead
        self.dashes = dashes

        self.vector_line = Entity(
            parent=self,
            model=Mesh(
                vertices=[Vec3(0, 0, 0), Vec3(0, 0, 1)],
                triangles=[(0, 1)],
                mode="line",
                thickness=2,
                static=False,
            ),
            color=PHOSPHOR,
            unlit=True,
        )
        self.marker = Entity(
            parent=self,
            model=axis_cross_mesh(),
            color=PHOSPHOR,
            unlit=True,
        )
        self.pred_label = Text(
            text="",
            color=PHOSPHOR_DIM,
            scale=0.7,
            origin=(-0.5, 0.5),
            font=Text.default_monospace_font,
        )

    def update(self):
        origin = Vec3(self.target.world_position)
        vel = Vec3(self.target.velocity)
        speed = vel.length()
        if speed < 0.05:
            self.vector_line.enabled = False
            self.marker.enabled = False
            self.pred_label.enabled = False
            return

        direction = vel / speed
        lock_depth = getattr(self.target, "lock_size", Vec3(0, 0, 2)).z * 0.55
        start = origin + direction * lock_depth
        end = origin + vel * self.lookahead
        span = end - start

        verts: list[Vec3] = []
        tris: list[tuple[int, int]] = []
        filled = 0.62
        for i in range(self.dashes):
            t0 = i / self.dashes
            t1 = min(1.0, t0 + (1.0 / self.dashes) * filled)
            idx = len(verts)
            verts.extend((start + span * t0, start + span * t1))
            tris.append((idx, idx + 1))

        mesh = self.vector_line.model
        mesh.vertices = verts
        mesh.triangles = tris
        mesh.generate()

        self.vector_line.enabled = True
        self.marker.enabled = True
        self.marker.position = end

        to_marker = end - camera.world_position
        visible = camera.forward.dot(to_marker) > 0
        self.pred_label.enabled = visible
        if not visible:
            return
        screen = self.marker.screen_position
        self.pred_label.position = Vec2(screen.x + 0.025, screen.y + 0.03)
        self.pred_label.text = f"PRED  T+{self.lookahead:.1f}s"


def build_environment() -> None:
    Entity(model="plane", scale=420, color=color.rgb32(5, 10, 16), y=0)
    Entity(
        model=Grid(40, 40),
        scale=200,
        rotation_x=90,
        y=0.04,
        color=color.rgba32(0, 90, 62, 90),
        unlit=True,
    )


def build_hud() -> None:
    Text(
        text="VAX // AIR TRACK",
        origin=(-0.5, 0.5),
        position=window.top_left + Vec2(0.03, -0.03),
        color=PHOSPHOR,
        scale=1.0,
        font=Text.default_monospace_font,
    )
    Text(
        text="STEP 2  TARGET LOCK / PREDICTED VECTOR",
        origin=(-0.5, 0.5),
        position=window.top_left + Vec2(0.03, -0.07),
        color=color.rgb32(0, 140, 72),
        scale=0.75,
        font=Text.default_monospace_font,
    )
    Text(
        text="RMB orbit   RMB+WASD move   scroll zoom   MMB pan",
        origin=(0.5, -0.5),
        position=window.bottom_right + Vec2(-0.03, 0.03),
        color=color.rgb32(0, 110, 60),
        scale=0.7,
        font=Text.default_monospace_font,
    )


def main() -> None:
    app = Ursina(
        title="VAX // AIR TRACK",
        borderless=False,
        fullscreen=False,
        vsync=True,
        development_mode=False,
        editor_ui_enabled=False,
    )
    window.color = color.rgb32(4, 8, 12)

    build_environment()
    AircraftCarrier()
    jet = FighterJet()
    TrackingBox(jet)
    PredictedPath(jet)
    build_hud()

    editor = EditorCamera(move_speed=25)
    editor.position = Vec3(0, 6, 0)
    editor.rotation_x = 18
    editor.rotation_y = -42
    camera.fov = 55
    camera.clip_plane_far = 800
    camera.z = -78
    editor.target_z = -78

    app.run()


if __name__ == "__main__":
    main()
