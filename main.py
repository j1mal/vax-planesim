"""
VAX PlaneSim — Step 4
AA defense: intercept solution, PN-guided missile, existing track/camera systems.

Python 3.12 recommended (Panda3D wheels):
    py -3.12 -m venv .venv
    .venv\\Scripts\\pip install -r requirements.txt
    .venv\\Scripts\\python main.py

Cameras: [1] free look  [2] turret  [3] missile  [4] follow jet
AA: [G] engage/standby   [F] fire
Free look: hold RMB to orbit, RMB+WASD to move, scroll to zoom, MMB to pan.
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
    destroy,
    lerp,
    lerp_angle,
    lerp_exponential_decay,
    scene,
    time,
    window,
)


PHOSPHOR = color.rgb32(0, 255, 92)
PHOSPHOR_DIM = color.rgba32(0, 255, 92, 90)
AMBER = color.rgb32(255, 196, 64)
AMBER_DIM = color.rgba32(255, 196, 64, 110)
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


def solve_intercept(
    shooter_pos: Vec3,
    missile_speed: float,
    target_pos: Vec3,
    target_vel: Vec3,
) -> tuple[Vec3 | None, float | None]:
    """Linear lead: first positive time where a constant-speed shot meets the target."""
    r = target_pos - shooter_pos
    a = target_vel.dot(target_vel) - missile_speed * missile_speed
    b = 2.0 * r.dot(target_vel)
    c = r.dot(r)

    def _point(t: float) -> Vec3:
        return target_pos + target_vel * t

    if abs(a) < 1e-6:
        if abs(b) < 1e-6:
            t = r.length() / max(missile_speed, 0.1)
            return _point(t), t
        t = -c / b
        if t > 0.05:
            return _point(t), t
        return None, None

    disc = b * b - 4.0 * a * c
    if disc < 0:
        t = r.length() / max(missile_speed, 0.1)
        return _point(t), t

    root = math.sqrt(disc)
    candidates = [t for t in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)) if t > 0.05]
    if not candidates:
        t = r.length() / max(missile_speed, 0.1)
        return _point(t), t
    t = min(candidates)
    return _point(t), t


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
        self.turret_hardpoint = Entity(parent=self, position=(4.4, 12.2, 6))
        Entity(
            parent=self.turret_hardpoint,
            model="cube",
            scale=(1.4, 0.55, 2.2),
            color=HULL_LIGHT,
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

    def __init__(self, target: FighterJet, lookahead: float = 3.0, dashes: int = 10, **kwargs):
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


class CameraDirector(Entity):
    """Toggles free-look, turret, missile, and auto-follow POVs."""

    FREE = "free"
    TURRET = "turret"
    MISSILE = "missile"
    FOLLOW = "follow"

    _labels = {
        FREE: "FREE LOOK",
        TURRET: "AA TURRET",
        MISSILE: "MISSILE",
        FOLLOW: "FOLLOW TGT",
    }
    _fov = {
        FREE: 55,
        TURRET: 55,
        MISSILE: 70,
        FOLLOW: 60,
    }
    _base_pivot_height = 6
    _base_pitch = 18

    def __init__(self, editor: EditorCamera, target: Entity, turret_anchor: Entity, **kwargs):
        super().__init__(**kwargs)
        self.editor = editor
        self.target = target
        self.turret_anchor = turret_anchor
        self.missile = None
        self.mode = self.FREE
        self._aim = Entity(add_to_scene_entities=False)

        self.status = Text(
            text="",
            origin=(0.5, 0.5),
            position=window.top_right + Vec2(-0.03, -0.03),
            color=PHOSPHOR,
            scale=0.85,
            font=Text.default_monospace_font,
        )
        self.hint = Text(
            text="[1] FREE   [2] TURRET   [3] MISSILE   [4] FOLLOW",
            origin=(0.5, 0.5),
            position=window.top_right + Vec2(-0.03, -0.07),
            color=color.rgb32(0, 140, 72),
            scale=0.7,
            font=Text.default_monospace_font,
        )
        self._refresh_status()

    def attach_missile(self, missile: Entity | None) -> None:
        self.missile = missile

    def input(self, key):
        binds = {"1": self.FREE, "2": self.TURRET, "3": self.MISSILE, "4": self.FOLLOW}
        if key in binds:
            self.set_mode(binds[key])

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        camera.fov = self._fov[mode]
        if mode in (self.FREE, self.TURRET):
            self.editor.target_fov = self._fov[mode]
            self.editor.enabled = True
        else:
            if self.editor.enabled:
                self.editor.enabled = False
            camera.world_parent = scene
            camera.rotation_z = 0
            if mode == self.FOLLOW:
                camera.world_position = self._follow_point()
                camera.look_at(self.target.world_position)
            elif mode == self.MISSILE:
                if self._missile_live():
                    camera.world_position = self._missile_point()
                    camera.look_at(self._missile_look())
                else:
                    camera.world_position = self.turret_anchor.world_position
                    camera.look_at(self.target.world_position)
        self._refresh_status()

    def _missile_live(self) -> bool:
        return bool(self.missile and getattr(self.missile, "enabled", True))

    def _horizontal_forward(self, entity: Entity) -> Vec3:
        vel = Vec3(getattr(entity, "velocity", Vec3(0, 0, 0)))
        horizontal = Vec3(vel.x, 0, vel.z)
        if horizontal.length() > 0.1:
            return horizontal.normalized()
        fwd = Vec3(entity.forward.x, 0, entity.forward.z)
        if fwd.length() > 0.1:
            return fwd.normalized()
        return Vec3(0, 0, 1)

    def _base_pivot(self) -> Vec3:
        origin = Vec3(0, 0, 0)
        if self.turret_anchor.parent:
            origin = Vec3(self.turret_anchor.parent.world_position)
        return origin + Vec3(0, self._base_pivot_height, 0)

    def _pan_editor_toward_target(self, decay: float = 4.5) -> None:
        """Orbit the carrier as if RMB were held and dragged toward the jet."""
        self.editor.position = lerp_exponential_decay(
            self.editor.position, self._base_pivot(), time.dt, 5
        )
        self._aim.world_position = self.editor.world_position
        self._aim.look_at(self.target.world_position)
        t = 1 - math.exp(-decay * time.dt)
        self.editor.rotation_y = lerp_angle(self.editor.rotation_y, self._aim.rotation_y, t)
        self.editor.rotation_x = lerp_angle(self.editor.rotation_x, self._base_pitch, t * 0.35)
        self.editor.rotation_z = 0

    def _follow_point(self) -> Vec3:
        fwd = self._horizontal_forward(self.target)
        return self.target.world_position - fwd * 26 + Vec3(0, 11, 0)

    def _missile_point(self) -> Vec3:
        missile = self.missile
        return missile.world_position + missile.back * 8 + missile.up * 2

    def _missile_look(self) -> Vec3:
        missile = self.missile
        vel = Vec3(getattr(missile, "velocity", Vec3(0, 0, 0)))
        if vel.length() > 0.1:
            return missile.world_position + vel.normalized() * 40
        return missile.world_position + missile.forward * 40

    def _aim_at(self, world_pos: Vec3, decay: float = 8.0) -> None:
        self._aim.world_position = camera.world_position
        self._aim.look_at(world_pos)
        t = 1 - math.exp(-decay * time.dt)
        camera.rotation_x = lerp_angle(camera.rotation_x, self._aim.rotation_x, t)
        camera.rotation_y = lerp_angle(camera.rotation_y, self._aim.rotation_y, t)
        camera.rotation_z = 0

    def _refresh_status(self) -> None:
        extra = ""
        if self.mode == self.MISSILE and not self._missile_live():
            extra = "\nAWAITING LAUNCH"
        self.status.text = f"CAM  {self._labels[self.mode]}{extra}"

    def update(self):
        if self.mode == self.FREE:
            return
        if self.mode == self.TURRET:
            self._pan_editor_toward_target()
            return
        if self.mode == self.FOLLOW:
            camera.world_position = lerp_exponential_decay(
                camera.world_position, self._follow_point(), time.dt, 6
            )
            self._aim_at(self.target.world_position, decay=10)
            return
        if self.mode == self.MISSILE:
            if self._missile_live():
                camera.world_position = lerp_exponential_decay(
                    camera.world_position, self._missile_point(), time.dt, 10
                )
                self._aim_at(self._missile_look(), decay=12)
            else:
                camera.world_position = self.turret_anchor.world_position
                self._aim_at(self.target.world_position, decay=7)
            self._refresh_status()


class Missile(Entity):
    """Constant-speed SAM steered with proportional navigation + linear lead."""

    def __init__(
        self,
        origin: Entity,
        target: Entity,
        speed: float = 42.0,
        navigation_gain: float = 4.0,
        **kwargs,
    ):
        super().__init__(position=Vec3(origin.world_position), **kwargs)
        self.target = target
        self.speed = speed
        self.N = navigation_gain
        self.velocity = Vec3(0, 0, 0)
        self.alive = True
        self.age = 0.0
        self.max_age = 14.0
        self.hit_radius = 4.0
        self.on_splash = None

        Entity(parent=self, model="cube", scale=(0.28, 0.28, 2.4), color=HULL_LIGHT)
        Entity(
            parent=self,
            model="cube",
            scale=(0.2, 0.2, 0.55),
            position=(0, 0, 1.35),
            color=PHOSPHOR,
            unlit=True,
        )
        Entity(
            parent=self,
            model="cube",
            scale=(0.9, 0.08, 0.45),
            position=(0, 0, -0.7),
            color=HULL,
        )
        Entity(
            parent=self,
            model="cube",
            scale=(0.08, 0.7, 0.4),
            position=(0, 0, -0.7),
            color=HULL,
        )

        intercept, _ = solve_intercept(
            self.world_position, self.speed, target.world_position, Vec3(target.velocity)
        )
        if intercept is not None:
            direction = intercept - self.world_position
        else:
            direction = target.world_position - self.world_position
        if direction.length() < 0.1:
            direction = Vec3(0, 1, 0)
        self.velocity = direction.normalized() * self.speed
        self.look_at(self.world_position + self.velocity)

    def _steer(self, dt: float) -> bool:
        target_pos = self.target.world_position
        target_vel = Vec3(self.target.velocity)
        r = target_pos - self.world_position
        dist = r.length()
        if dist < self.hit_radius:
            return True
        r_hat = r / dist
        rel_vel = target_vel - self.velocity
        omega = r.cross(rel_vel) / (dist * dist)
        closing = -rel_vel.dot(r_hat)
        accel = omega.cross(r_hat) * (self.N * max(closing, 0.0))

        intercept, _ = solve_intercept(self.world_position, self.speed, target_pos, target_vel)
        if intercept is not None:
            to_int = intercept - self.world_position
            if to_int.length() > 0.1:
                desired = to_int.normalized() * self.speed
                self.velocity = lerp(self.velocity, desired, min(1.0, 5.5 * dt))

        self.velocity += accel * dt
        if self.velocity.length() < 0.05:
            self.velocity = r_hat * self.speed
        else:
            self.velocity = self.velocity.normalized() * self.speed
        return False

    def _splash(self, hit: bool) -> None:
        self.alive = False
        callback = self.on_splash
        self.on_splash = None
        if callback:
            callback(hit)
        destroy(self)

    def update(self):
        if not self.alive:
            return
        dt = max(time.dt, 1e-5)
        self.age += dt
        hit = self._steer(dt)
        if hit or self.age > self.max_age or self.y < 0.4:
            self._splash(hit=hit)
            return
        self.position += self.velocity * dt
        self.look_at(self.world_position + self.velocity)


class DefenseSystem(Entity):
    """Carrier AA: toggleable intercept solution and manual missile launch."""

    def __init__(self, carrier: AircraftCarrier, target: Entity, camera_director: CameraDirector, **kwargs):
        super().__init__(**kwargs)
        self.carrier = carrier
        self.target = target
        self.camera_director = camera_director
        self.launch_origin = carrier.turret_hardpoint
        self.missile_speed = 42.0
        self.engaged = False
        self.missile = None
        self.cooldown = 0.0
        self.last_hit = None
        self.status_hold = 0.0

        self.solution_line = Entity(
            model=Mesh(
                vertices=[Vec3(0, 0, 0), Vec3(0, 0, 1)],
                triangles=[(0, 1)],
                mode="line",
                thickness=2,
                static=False,
            ),
            color=AMBER,
            unlit=True,
            enabled=False,
        )
        self.solution_marker = Entity(
            model=axis_cross_mesh(0.85),
            color=AMBER,
            unlit=True,
            enabled=False,
        )
        self.int_label = Text(
            text="",
            color=AMBER,
            scale=0.7,
            origin=(-0.5, 0.5),
            font=Text.default_monospace_font,
            enabled=False,
        )
        self.status = Text(
            text="",
            origin=(-0.5, -0.5),
            position=window.bottom_left + Vec2(0.03, 0.03),
            color=PHOSPHOR,
            scale=0.75,
            font=Text.default_monospace_font,
        )
        self._refresh_status()

    def input(self, key):
        if key == "g":
            self.engaged = not self.engaged
            self._refresh_status()
        elif key == "f":
            self.fire()

    def fire(self) -> None:
        if self.cooldown > 0:
            return
        if self.missile and getattr(self.missile, "alive", False):
            old = self.missile
            old.on_splash = None
            old.alive = False
            destroy(old)
        missile = Missile(self.launch_origin, self.target, speed=self.missile_speed)
        missile.on_splash = self._on_splash
        self.missile = missile
        self.camera_director.attach_missile(missile)
        self.cooldown = 1.25
        self.last_hit = None
        self.status_hold = 0.0
        self._refresh_status()

    def _on_splash(self, hit: bool) -> None:
        self.missile = None
        self.camera_director.attach_missile(None)
        self.last_hit = hit
        self.status_hold = 2.4
        self._refresh_status()

    def _solution_origin(self) -> Vec3:
        if self.missile and getattr(self.missile, "alive", False):
            return Vec3(self.missile.world_position)
        return Vec3(self.launch_origin.world_position)

    def _refresh_status(self, intercept: Vec3 | None = None, tti: float | None = None) -> None:
        state = "ENGAGED" if self.engaged else "STANDBY"
        lines = [f"AA  {state}", "[G] TOGGLE   [F] FIRE"]
        if tti is not None:
            lines.insert(1, f"INT T+{tti:4.1f}s")
        if self.status_hold > 0 and self.last_hit is not None:
            lines.insert(1, "MSL  SPLASH" if self.last_hit else "MSL  MISS")
        elif self.missile and getattr(self.missile, "alive", False):
            lines.insert(1, "MSL  IN FLIGHT")
        self.status.text = "\n".join(lines)

    def update(self):
        dt = time.dt
        self.cooldown = max(0.0, self.cooldown - dt)
        self.status_hold = max(0.0, self.status_hold - dt)

        origin = self._solution_origin()
        intercept, tti = solve_intercept(
            origin, self.missile_speed, self.target.world_position, Vec3(self.target.velocity)
        )
        show = self.engaged and intercept is not None and tti is not None
        self.solution_line.enabled = show
        self.solution_marker.enabled = show
        self.int_label.enabled = False
        if show:
            self.solution_marker.position = intercept
            mesh = self.solution_line.model
            mesh.vertices = [origin, intercept]
            mesh.triangles = [(0, 1)]
            mesh.generate()
            to_marker = intercept - camera.world_position
            visible = camera.forward.dot(to_marker) > 0
            self.int_label.enabled = visible
            if visible:
                screen = self.solution_marker.screen_position
                self.int_label.position = Vec2(screen.x + 0.025, screen.y - 0.04)
                self.int_label.text = f"INT  T+{tti:.1f}s"
        self._refresh_status(intercept if show else None, tti if show else None)


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
        text="STEP 4  AA DEFENSE / INTERCEPT",
        origin=(-0.5, 0.5),
        position=window.top_left + Vec2(0.03, -0.07),
        color=color.rgb32(0, 140, 72),
        scale=0.75,
        font=Text.default_monospace_font,
    )
    Text(
        text="[G] AA   [F] fire   RMB orbit   WASD   scroll   MMB",
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
    carrier = AircraftCarrier()
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
    director = CameraDirector(editor=editor, target=jet, turret_anchor=carrier.turret_hardpoint)
    DefenseSystem(carrier, jet, director)

    app.run()


if __name__ == "__main__":
    main()
