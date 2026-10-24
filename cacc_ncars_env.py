import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv


def _bezier(p0, p1, p2, p3, steps=24):
    """Harmadfokú Bézier-görbe pontjainak generálása az áramvonalas karosszériaívekhez."""
    pts = []
    for i in range(steps + 1):
        t = i / float(steps)
        u = 1.0 - t
        x = (u**3) * p0[0] + 3 * (u**2) * t * p1[0] + 3 * u * (t**2) * p2[0] + (t**3) * p3[0]
        y = (u**3) * p0[1] + 3 * (u**2) * t * p1[1] + 3 * u * (t**2) * p2[1] + (t**3) * p3[1]
        pts.append((int(round(x)), int(round(y))))
    return pts


def _mirror_poly(top_pts, cy):
    """Felső profilpontok szimmetrikus tükrözése a középvonalra (cy), zárt poligont alkotva."""
    bot_pts = [(x, int(round(2 * cy - y))) for (x, y) in reversed(top_pts)]
    return top_pts + bot_pts


class CACCNCarsEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "cacc_ncars_v0", "render_fps": 30}

    def __init__(self,
                 num_cars=4,
                 target_speed=36.11,    # ~130 km/h
                 safe_dist=22.0,
                 danger_dist=7.0,
                 crash_dist=4.0,
                 max_dist=150.0,
                 track_length=1500.0,
                 weight_progress=0.45,
                 weight_dist=0.06,
                 weight_rel_vel=0.04,
                 weight_jerk=0.02,
                 penalty_crash=-500.0,
                 bonus_finish=150.0,
                 max_steps=1500,
                 render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.num_cars = num_cars
        self.possible_agents = [f"car_{i}" for i in range(self.num_cars)]
        self.agents = self.possible_agents[:]

        self.dt = 0.1
        self.max_speed = 41.67         # Max 150 km/h
        self.min_speed = 22.22         # Min 80 km/h
        self.max_steps = max_steps
        self.track_length = track_length

        self.target_speed = target_speed
        self.safe_dist = safe_dist
        self.danger_dist = danger_dist
        self.crash_dist = crash_dist
        self.max_dist = max_dist

        self.weight_progress = weight_progress
        self.weight_dist = weight_dist
        self.weight_rel_vel = weight_rel_vel
        self.weight_jerk = weight_jerk
        self.penalty_crash = penalty_crash
        self.bonus_finish = bonus_finish

        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None
        self.sprite_cache = {}
        self.camera_x = 0.0
        self.camera_scale = 5.5
        self.crash_happened = False
        self.crashed_agents = set()
        self.danger_agents = set()

        self.action_spaces = {}
        self.observation_spaces = {}

        for agent in self.possible_agents:
            self.action_spaces[agent] = spaces.Box(
                low=np.array([-6.0, -1.0], dtype=np.float32),
                high=np.array([4.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
            self.observation_spaces[agent] = spaces.Box(
                low=np.array([0.0, -self.safe_dist, -self.max_speed, 0.0, -self.safe_dist, 0.0], dtype=np.float32),
                high=np.array([self.max_speed, self.max_dist, self.max_speed, 1.0, self.max_dist, self.max_dist], dtype=np.float32)
            )

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.current_step = 0
        self.traveled_dist = 0.0
        self.crash_happened = False
        self.crashed_agents = set()
        self.danger_agents = set()
        self.state = {}

        self.lead_pos = 250.0
        self.lead_vel = float(np.random.uniform(30.0, 38.0))
        self.lead_acc = 0.0
        self.lead_lane = int(np.random.choice([0, 1]))

        lane_last_pos = {0: self.lead_pos, 1: self.lead_pos}
        if self.lead_lane in lane_last_pos:
            lane_last_pos[self.lead_lane] = self.lead_pos - 5.0

        lane_last_vel = {0: self.lead_vel, 1: self.lead_vel}

        for i, agent in enumerate(self.possible_agents):
            lane = i % 2 if np.random.rand() < 0.75 else int(np.random.choice([0, 1]))
            random_gap = float(np.random.uniform(22.0, 38.0))
            pos = lane_last_pos[lane] - random_gap

            front_v = lane_last_vel[lane]
            min_v = max(24.0, front_v - 6.0)
            max_v = min(40.0, front_v + 6.0)
            random_vel = float(np.random.uniform(min_v, max_v))

            lane_last_pos[lane] = pos
            lane_last_vel[lane] = random_vel

            self.state[agent] = {
                "pos": pos,
                "vel": random_vel,
                "acc": 0.0,
                "lane": lane,
                "y": float(lane),
                "vy": 0.0,
                "lane_cooldown": 15
            }

        all_x = [self.lead_pos] + [self.state[a]["pos"] for a in self.possible_agents]
        self.camera_x = float(np.min(all_x)) - 25.0

        if self.render_mode == "human":
            self.render()

        return self._get_observations(), {agent: {} for agent in self.agents}

    def _get_leader_in_lane(self, agent_pos, target_lane, exclude_agent=None):
        best_pos = self.lead_pos if self.lead_lane == target_lane else 99999.0
        best_vel = self.lead_vel if self.lead_lane == target_lane else self.target_speed
        best_name = "Lead" if self.lead_lane == target_lane else "Virtual_Lead"

        for other in self.possible_agents:
            if other == exclude_agent:
                continue
            other_pos = self.state[other]["pos"]
            other_lane = self.state[other]["lane"]
            if (other_lane == target_lane or abs(self.state[other]["y"] - target_lane) < 0.55) and other_pos > agent_pos:
                if other_pos < best_pos:
                    best_pos = other_pos
                    best_vel = self.state[other]["vel"]
                    best_name = other

        if best_pos == 99999.0:
            best_pos = max(self.lead_pos, agent_pos + self.safe_dist)
            best_vel = self.lead_vel
            best_name = "Lead"

        return best_pos, best_vel, best_name

    def _get_min_clearance_in_lane(self, agent_pos, target_lane, exclude_agent=None):
        min_d = abs(self.lead_pos - agent_pos) if self.lead_lane == target_lane else 999.0
        for other in self.possible_agents:
            if other == exclude_agent:
                continue
            if self.state[other]["lane"] == target_lane or abs(self.state[other]["y"] - target_lane) < 0.6:
                d = abs(self.state[other]["pos"] - agent_pos)
                if d < min_d:
                    min_d = d
        return float(np.clip(min_d, 0.0, self.max_dist))

    def step(self, actions):
        self.current_step += 1
        self.danger_agents = set()

        if 400.0 <= self.traveled_dist <= 800.0:
            target_lead_v = 26.4
        elif 1000.0 <= self.traveled_dist <= 1300.0:
            target_lead_v = 41.0
        else:
            target_lead_v = self.target_speed + 3.5 * np.sin(self.current_step * 0.03)

        self.lead_acc = float(np.clip((target_lead_v - self.lead_vel) * 1.2, -4.5, 2.5))
        self.lead_vel = float(np.clip(self.lead_vel + self.lead_acc * self.dt, self.min_speed, self.max_speed))
        self.lead_pos += self.lead_vel * self.dt
        self.traveled_dist += self.lead_vel * self.dt

        step_dists = {}
        jerks = {}
        lane_penalties = {agent: 0.0 for agent in self.agents}

        for agent in self.agents:
            act = actions[agent]
            prev_acc = self.state[agent]["acc"]
            acceleration = float(np.clip(act[0], -6.0, 4.0))
            lane_cmd = float(act[1]) if len(act) > 1 else 0.0

            jerks[agent] = abs(acceleration - prev_acc)
            self.state[agent]["acc"] = acceleration

            if self.state[agent]["lane_cooldown"] > 0:
                self.state[agent]["lane_cooldown"] -= 1
            else:
                cur_lane = self.state[agent]["lane"]
                des_lane = cur_lane
                if lane_cmd < -0.65:
                    des_lane = 0
                elif lane_cmd > 0.65:
                    des_lane = 1

                if des_lane != cur_lane:
                    pos = self.state[agent]["pos"]
                    clearance = self._get_min_clearance_in_lane(pos, des_lane, exclude_agent=agent)
                    other_front_pos, _, _ = self._get_leader_in_lane(pos, des_lane, exclude_agent=agent)
                    cur_front_pos, _, _ = self._get_leader_in_lane(pos, cur_lane, exclude_agent=agent)

                    if clearance > 14.0 and (other_front_pos - pos) > (cur_front_pos - pos) + 4.0:
                        self.state[agent]["lane"] = des_lane
                        self.state[agent]["lane_cooldown"] = 25
                        lane_penalties[agent] = 0.2
                    else:
                        lane_penalties[agent] = 1.2

            target_y = float(self.state[agent]["lane"])
            dy = float(np.clip(target_y - self.state[agent]["y"], -0.10, 0.10))
            self.state[agent]["vy"] = dy
            self.state[agent]["y"] += dy

            self.state[agent]["vel"] = float(
                np.clip(self.state[agent]["vel"] + acceleration * self.dt, self.min_speed, self.max_speed)
            )
            d_pos = self.state[agent]["vel"] * self.dt
            self.state[agent]["pos"] += d_pos
            step_dists[agent] = d_pos

        observations = self._get_observations()
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        split_happened = False

        for agent in self.agents:
            pos = self.state[agent]["pos"]
            lane = self.state[agent]["lane"]
            front_pos, front_vel, front_name = self._get_leader_in_lane(pos, lane, exclude_agent=agent)

            dist = front_pos - pos
            rel_vel = front_vel - self.state[agent]["vel"]

            if dist <= self.crash_dist:
                self.crash_happened = True
                self.crashed_agents.add(agent)
                if front_name != "Lead":
                    self.crashed_agents.add(front_name)
            elif dist <= self.danger_dist:
                self.danger_agents.add(agent)

            if dist >= self.max_dist or (self.lead_pos - pos) >= self.max_dist:
                split_happened = True

            progress_rew = step_dists[agent] * self.weight_progress
            dist_penalty = abs(self.safe_dist - dist) * self.weight_dist
            vel_penalty = abs(rel_vel) * self.weight_rel_vel
            jerk_penalty = jerks[agent] * self.weight_jerk

            proximity_penalty = 0.0
            if dist <= self.danger_dist:
                proximity_penalty = 40.0 + (self.danger_dist - dist) * 15.0
            elif dist <= 13.0:
                proximity_penalty = (13.0 - dist) * 2.5

            rewards[agent] = (
                progress_rew
                - dist_penalty
                - vel_penalty
                - jerk_penalty
                - proximity_penalty
                - lane_penalties[agent]
            )

        if self.crash_happened or split_happened:
            remaining_ratio = (self.max_steps - self.current_step) / self.max_steps
            heavy_crash_pen = self.penalty_crash - (remaining_ratio * 200.0)
            for agent in self.agents:
                if agent in self.crashed_agents or split_happened:
                    rewards[agent] = heavy_crash_pen
                else:
                    rewards[agent] = heavy_crash_pen * 0.4
                terminations[agent] = True

            if self.render_mode == "human":
                self.render()
                import pygame
                pygame.time.wait(2000)

            self.agents = []
            return observations, rewards, terminations, truncations, {a: {} for a in self.possible_agents}

        if self.traveled_dist >= self.track_length or self.current_step >= self.max_steps:
            for agent in self.agents:
                if self.traveled_dist >= self.track_length:
                    rewards[agent] += self.bonus_finish
                truncations[agent] = True
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, {a: {} for a in self.possible_agents}

    def _get_observations(self):
        obs = {}
        for agent in self.possible_agents:
            pos = self.state[agent]["pos"]
            vel = self.state[agent]["vel"]
            lane = self.state[agent]["lane"]
            other_lane = 1 - lane

            front_pos, front_vel, _ = self._get_leader_in_lane(pos, lane, exclude_agent=agent)
            dist_error = np.clip((front_pos - pos) - self.safe_dist, -self.safe_dist, self.max_dist)
            rel_vel = np.clip(front_vel - vel, -self.max_speed, self.max_speed)

            other_front_pos, _, _ = self._get_leader_in_lane(pos, other_lane, exclude_agent=agent)
            other_dist_err = np.clip((other_front_pos - pos) - self.safe_dist, -self.safe_dist, self.max_dist)
            other_clearance = self._get_min_clearance_in_lane(pos, other_lane, exclude_agent=agent)

            obs[agent] = np.array([
                vel,
                dist_error,
                rel_vel,
                self.state[agent]["y"],
                other_dist_err,
                other_clearance
            ], dtype=np.float32)
        return obs

    # =========================================================================
    # 8X SUPERSAMPLED (440x220) HD BÉZIER-VEKTOROS AUTÓMODELL-GENERÁTOR
    # =========================================================================
    def _aa_poly(self, surf, color, pts):
        import pygame
        import pygame.gfxdraw
        if len(pts) >= 3:
            pygame.gfxdraw.filled_polygon(surf, pts, color)
            pygame.gfxdraw.aapolygon(surf, pts, color)

    def _draw_f1_wheel(self, surf, cx, cy, w, h, stripe_color):
        import pygame
        rect = pygame.Rect(cx - w // 2, cy - h // 2, w, h)
        # Gumiabroncs alap + 3D futófelület árnyékolás
        pygame.draw.rect(surf, (18, 19, 22), rect, border_radius=9)
        pygame.draw.rect(surf, (38, 40, 45), (cx - w // 2 + 5, cy - h // 2 + 3, w - 10, h - 6), border_radius=6)
        pygame.draw.rect(surf, (26, 28, 32), (cx - w // 2 + 10, cy - h // 2 + 5, w - 20, h - 10), border_radius=4)
        # Pirelli színes oldalfal-csík
        pygame.draw.line(surf, stripe_color, (cx - w // 2 + 8, cy - h // 2 + 3), (cx + w // 2 - 8, cy - h // 2 + 3), 3)
        pygame.draw.line(surf, stripe_color, (cx - w // 2 + 8, cy + h // 2 - 4), (cx + w // 2 - 8, cy + h // 2 - 4), 3)

    def _build_f1_hd(self, variant="ferrari", braking=False):
        """Nagyfelbontású (440x220) F1-es modell: Ferrari SF-25 (#44) és Aston Martin AMR (#18)."""
        import pygame
        W, H = 440, 220
        cy = H // 2
        surf = pygame.Surface((W, H), pygame.SRCALPHA)

        if variant == "ferrari":
            c_dark = (135, 10, 18)
            c_main = (195, 20, 32)
            c_bright = (235, 42, 52)
            c_wing_main = (165, 18, 28)
            c_wing_flap = (242, 242, 246)
            c_stripe = (245, 245, 250)
            c_accent = (255, 210, 0)
            tire_stripe = (230, 30, 30)   # Pirelli Soft (Piros)
            helmet_col = (240, 215, 30)
            num_txt = "44"
        else:
            c_dark = (0, 68, 58)
            c_main = (0, 115, 98)
            c_bright = (18, 162, 138)
            c_wing_main = (18, 22, 26)
            c_wing_flap = (230, 235, 240)
            c_stripe = (195, 235, 15)     # AMR Lime-zöld
            c_accent = (205, 245, 20)
            tire_stripe = (245, 205, 20)  # Pirelli Medium (Sárga)
            helmet_col = (0, 175, 145)
            num_txt = "18"

        # 1. Vetített lágy árnyék az autó alatt
        shadow_pts = _mirror_poly([
            (35, cy - 68), (140, cy - 84), (310, cy - 82), (418, cy - 72), (425, cy)
        ], cy)
        self._aa_poly(surf, (0, 0, 0, 95), shadow_pts)

        # 2. Karbon padlólemez (Floor & Diffuser + Venturi-csatornák)
        floor_top = [
            (55, cy - 36), (75, cy - 62), (165, cy - 66),
            (250, cy - 64), (295, cy - 48), (320, cy - 18)
        ]
        self._aa_poly(surf, (22, 24, 28), _mirror_poly(floor_top, cy))
        # Padlólemez szélén aerodinamikai él (floor edge wing)
        pygame.draw.lines(surf, (58, 62, 70), False, [(85, cy - 64), (245, cy - 63), (290, cy - 48)], 3)
        pygame.draw.lines(surf, (58, 62, 70), False, [(85, cy + 64), (245, cy + 63), (290, cy + 48)], 3)

        # 3. Dupla keresztlengőkaros futómű (Double-wishbone & Pushrod rudak)
        susp_col = (35, 38, 44)
        # Hátsó felfüggesztés
        for s in (-1, 1):
            pygame.draw.line(surf, susp_col, (118, cy + s * 18), (92, cy + s * 72), 6)
            pygame.draw.line(surf, susp_col, (76, cy + s * 16), (92, cy + s * 72), 6)
            pygame.draw.line(surf, (60, 65, 75), (132, cy + s * 14), (96, cy + s * 68), 4)
            # Első felfüggesztés
            pygame.draw.line(surf, susp_col, (345, cy + s * 12), (322, cy + s * 70), 6)
            pygame.draw.line(surf, susp_col, (302, cy + s * 14), (322, cy + s * 70), 6)
            pygame.draw.line(surf, (70, 75, 85), (288, cy + s * 14), (318, cy + s * 66), 4)

        # 4. Kerekek (Hátsó szélesebb: 68x36, Első: 62x32) + Fékcsatornák (Brake ducts)
        self._draw_f1_wheel(surf, 94, cy - 74, 68, 36, tire_stripe)
        self._draw_f1_wheel(surf, 94, cy + 74, 68, 36, tire_stripe)
        self._draw_f1_wheel(surf, 322, cy - 72, 62, 32, tire_stripe)
        self._draw_f1_wheel(surf, 322, cy + 72, 62, 32, tire_stripe)

        # 5. Oldaldobozok (Coke-bottle Sidepods) Bézier-görbével
        sidepod_curve = (
            [(72, cy - 15)]
            + _bezier((72, cy - 15), (115, cy - 22), (140, cy - 58), (205, cy - 58), 18)
            + _bezier((205, cy - 58), (245, cy - 58), (268, cy - 52), (278, cy - 24), 12)
            + [(412, cy - 7), (416, cy)]
        )
        self._aa_poly(surf, c_dark, _mirror_poly(sidepod_curve, cy))

        # Belső, világosabb karosszériaréteg (3D domborulat)
        inner_curve = (
            [(78, cy - 11)]
            + _bezier((78, cy - 11), (125, cy - 18), (150, cy - 48), (205, cy - 48), 16)
            + _bezier((205, cy - 48), (240, cy - 48), (260, cy - 42), (272, cy - 18), 10)
            + [(408, cy - 5), (412, cy)]
        )
        self._aa_poly(surf, c_main, _mirror_poly(inner_curve, cy))

        # Felső csúcsfény a monokokon és az orrkúpon
        highlight_curve = (
            [(95, cy - 6)]
            + _bezier((95, cy - 6), (160, cy - 16), (220, cy - 20), (285, cy - 14), 14)
            + [(402, cy - 3), (406, cy)]
        )
        self._aa_poly(surf, c_bright, _mirror_poly(highlight_curve, cy))

        # Mély fekete légcsatorna (Sidepod undercut / water-slide) és hűtőkopoltyúk
        for s in (-1, 1):
            undercut = [
                (125, cy + s * 22), (175, cy + s * 46),
                (245, cy + s * 46), (264, cy + s * 26), (195, cy + s * 30)
            ]
            self._aa_poly(surf, (18, 20, 24), undercut)
            # Hűtőnyílás-kopoltyúk (Cooling louvers)
            for gx in range(165, 225, 10):
                pygame.draw.line(surf, (25, 25, 30), (gx, cy + s * 26), (gx - 5, cy + s * 38), 3)

        # 6. Motorborítás dekorcsík (SF-25 fehér cápauszony / AMR lime gerinc)
        spine_poly = _mirror_poly([
            (75, cy - 3), (135, cy - 13), (195, cy - 14), (210, cy - 4)
        ], cy)
        self._aa_poly(surf, c_stripe, spine_poly)
        pygame.draw.line(surf, (20, 20, 25), (72, cy), (190, cy), 2)

        # 7. Többelemes Első Szárny (Front Wing + Endplates + Nose tip)
        fw_back = _mirror_poly([
            (372, cy - 12), (382, cy - 82), (408, cy - 84), (404, cy - 48), (418, cy - 8)
        ], cy)
        self._aa_poly(surf, c_wing_main, fw_back)
        # Második és harmadik fehér/világos szárnylap (flaps)
        for offset in (0, 9):
            flap = _mirror_poly([
                (386 + offset, cy - 16), (392 + offset, cy - 80),
                (398 + offset, cy - 80), (394 + offset, cy - 16)
            ], cy)
            self._aa_poly(surf, c_wing_flap, flap)
        # Első szárny véglapok (Endplates)
        pygame.draw.rect(surf, c_stripe, (376, cy - 86, 36, 6), border_radius=3)
        pygame.draw.rect(surf, c_stripe, (376, cy + 80, 36, 6), border_radius=3)

        # Orrkúp hegye (nyújtottan ráhajol az első szárnyra)
        nose_tip = _mirror_poly([(350, cy - 10), (416, cy - 5), (422, cy)], cy)
        self._aa_poly(surf, c_main, nose_tip)
        self._aa_poly(surf, c_stripe, _mirror_poly([(404, cy - 5), (420, cy - 3), (422, cy)], cy))

        # 8. Pilótafülke (Cockpit), Sisak, Kormány és Titanium HALO
        cockpit_pts = _mirror_poly([
            (202, cy - 12), (222, cy - 18), (262, cy - 16), (275, cy - 8)
        ], cy)
        self._aa_poly(surf, (14, 16, 20), cockpit_pts)

        # Pilóta sisakja + sötétített plexi (visor)
        pygame.draw.circle(surf, helmet_col, (236, cy), 12)
        pygame.draw.arc(surf, (15, 20, 30), (226, cy - 11, 22, 22), -1.1, 1.1, 6)
        # Kormánykerék
        pygame.draw.rect(surf, (45, 50, 58), (254, cy - 9, 6, 18), border_radius=2)

        # Airbox (Felső légbeömlő + T-Cam kamera a pilóta mögött)
        airbox_pts = _mirror_poly([(178, cy - 4), (204, cy - 8), (208, cy)], cy)
        self._aa_poly(surf, (18, 18, 22), airbox_pts)
        pygame.draw.rect(surf, c_accent, (194, cy - 14, 6, 28), border_radius=2)

        # HALO bukókeret
        halo_outer = _mirror_poly([
            (218, cy - 16), (248, cy - 15), (278, cy - 3), (278, cy)
        ], cy)
        halo_inner = _mirror_poly([
            (222, cy - 11), (246, cy - 10), (268, cy - 2), (268, cy)
        ], cy)
        self._aa_poly(surf, (30, 32, 36), halo_outer)
        self._aa_poly(surf, (14, 16, 20), halo_inner)
        pygame.draw.line(surf, c_main, (220, cy - 13), (274, cy), 4)
        pygame.draw.line(surf, c_main, (220, cy + 13), (274, cy), 4)

        # 9. Hátsó Szárny (Rear Wing + DRS + Véglapok)
        rw_rect = pygame.Rect(30, cy - 56, 30, 112)
        pygame.draw.rect(surf, c_wing_main, rw_rect, border_radius=5)
        pygame.draw.rect(surf, c_wing_flap, (36, cy - 52, 10, 104), border_radius=3)
        pygame.draw.rect(surf, c_stripe, (48, cy - 52, 8, 104), border_radius=2)
        pygame.draw.rect(surf, (20, 20, 24), (26, cy - 60, 38, 6), border_radius=2)
        pygame.draw.rect(surf, (20, 20, 24), (26, cy + 54, 38, 6), border_radius=2)

        # 10. Rajtszám az orron (#44 / #18)
        num_font = pygame.font.SysFont("sans", 22, bold=True)
        txt_surf = num_font.render(num_txt, True, c_accent)
        txt_rot = pygame.transform.rotate(txt_surf, -90)
        surf.blit(txt_rot, (318, cy - txt_rot.get_height() // 2))

        # 11. FIA Középső Féklámpa / Véglap LED-ek (fékezéskor izzó glóriával!)
        if braking:
            glow = pygame.Surface((90, 150), pygame.SRCALPHA)
            pygame.draw.circle(glow, (255, 20, 20, 140), (45, 75), 32)
            pygame.draw.circle(glow, (255, 40, 20, 110), (45, 22), 20)
            pygame.draw.circle(glow, (255, 40, 20, 110), (45, 128), 20)
            surf.blit(glow, (0, cy - 75))
            pygame.draw.rect(surf, (255, 50, 50), (24, cy - 8, 8, 16), border_radius=3)
            pygame.draw.rect(surf, (255, 40, 40), (24, cy - 58, 7, 12), border_radius=2)
            pygame.draw.rect(surf, (255, 40, 40), (24, cy + 46, 7, 12), border_radius=2)
        else:
            pygame.draw.rect(surf, (130, 20, 25), (26, cy - 5, 5, 10), border_radius=2)

        return surf

    def _build_supercar_hd(self, model="mclaren_orange", braking=False):
        """Nagyfelbontású (440x220) szupersportautók: McLaren 720S (Orange/Blue) és Porsche Mission X."""
        import pygame
        W, H = 440, 220
        cy = H // 2
        surf = pygame.Surface((W, H), pygame.SRCALPHA)

        if model == "mclaren_orange":
            c_edge = (175, 65, 5)
            c_main = (242, 112, 18)
            c_high = (255, 158, 45)
        elif model == "mclaren_blue":
            c_edge = (5, 95, 118)
            c_main = (12, 158, 182)
            c_high = (70, 212, 232)
        else:  # porsche_mission
            c_edge = (182, 188, 198)
            c_main = (236, 240, 246)
            c_high = (252, 253, 255)

        # 1. Árnyonalas szupersportautó sziluett (Bézier-görbékkel: széles első-hátsó sárvédők, darázsderék)
        outer_top = (
            [(42, cy)]
            + _bezier((42, cy), (42, cy - 58), (58, cy - 74), (105, cy - 78), 16)
            + _bezier((105, cy - 78), (150, cy - 80), (175, cy - 66), (218, cy - 66), 14)
            + _bezier((218, cy - 66), (262, cy - 66), (290, cy - 78), (338, cy - 76), 14)
            + _bezier((338, cy - 76), (382, cy - 72), (414, cy - 42), (416, cy), 16)
        )
        body_poly = _mirror_poly(outer_top, cy)

        # Vetített árnyék
        shadow_poly = [(x - 4, y + 4) for (x, y) in body_poly]
        self._aa_poly(surf, (0, 0, 0, 105), shadow_poly)

        # Sötétebb alap perem (3D keret)
        self._aa_poly(surf, c_edge, body_poly)

        # Fő fényezés (4 pixellel beljebb)
        mid_top = [(x, int(cy + (y - cy) * 0.93)) for (x, y) in outer_top]
        self._aa_poly(surf, c_main, _mirror_poly(mid_top, cy))

        # Sárvédő-domborulatok csúcsfénye (Specular highlights)
        for s in (-1, 1):
            fender_front = _bezier((275, cy + s * 52), (315, cy + s * 66), (355, cy + s * 64), (388, cy + s * 38), 14)
            fender_rear = _bezier((68, cy + s * 56), (105, cy + s * 68), (150, cy + s * 68), (188, cy + s * 52), 14)
            pygame.draw.lines(surf, c_high, False, fender_front, 7)
            pygame.draw.lines(surf, c_high, False, fender_rear, 7)

        # Oldalsó visszapillantó tükrök (szárakon)
        for s in (-1, 1):
            pygame.draw.line(surf, (30, 32, 36), (262, cy + s * 54), (254, cy + s * 84), 5)
            mirror_pts = [
                (246, cy + s * 82), (262, cy + s * 80),
                (258, cy + s * 90), (242, cy + s * 88)
            ]
            self._aa_poly(surf, c_main, mirror_pts)

        if model in ("mclaren_orange", "mclaren_blue"):
            # --- MCLAREN 720S SPECIFIKUS RÉSZLETEK ---
            # Motorháztető dupla légbeömlő (hood nostrils) és oldalsó ajtó-légcsatornák
            for s in (-1, 1):
                nostril = [(332, cy + s * 20), (368, cy + s * 32), (358, cy + s * 39), (326, cy + s * 25)]
                self._aa_poly(surf, (22, 24, 28), nostril)
                door_blade = _bezier((165, cy + s * 66), (205, cy + s * 56), (245, cy + s * 56), (278, cy + s * 64), 12)
                pygame.draw.lines(surf, (20, 22, 26), False, door_blade, 6)

            # Csepp alakú (Teardrop) fekete üvegkupola + Hátsó sötét motortér
            canopy_top = (
                [(72, cy)]
                + _bezier((72, cy), (85, cy - 36), (155, cy - 52), (225, cy - 52), 16)
                + _bezier((225, cy - 52), (275, cy - 52), (308, cy - 36), (314, cy), 14)
            )
            self._aa_poly(surf, (18, 22, 28), _mirror_poly(canopy_top, cy))

            # Első szélvédő íve (kékes-szürke üvegcsillanással)
            windshield = _mirror_poly(
                [(232, cy - 42)]
                + _bezier((232, cy - 42), (272, cy - 46), (298, cy - 32), (304, cy), 12)
                + [(242, cy)], cy
            )
            self._aa_poly(surf, (38, 48, 60), windshield)

            if model == "mclaren_blue":
                # Nyitott Spider utastér: 2 db részletes sport-kagylóülés és kormánykerék
                for sy in (cy - 22, cy + 22):
                    pygame.draw.rect(surf, (48, 54, 62), (186, sy - 14, 44, 28), border_radius=7)
                    pygame.draw.rect(surf, (32, 36, 42), (194, sy - 10, 28, 20), border_radius=5)
                    pygame.draw.rect(surf, (68, 74, 82), (180, sy - 9, 12, 18), border_radius=4)
                pygame.draw.circle(surf, (25, 28, 32), (238, cy - 22), 9, 3)
                # Vörösen izzó motortér-betét a hátsó rácson át
                self._aa_poly(surf, (165, 22, 28), _mirror_poly([(96, cy), (112, cy - 18), (132, cy - 12), (132, cy)], cy))
            else:
                # Coupé zárt üvegtető + ezüst ikerturbó-hűtőrácsok hátul
                roof_glass = _mirror_poly([(168, cy - 34), (226, cy - 38), (228, cy)], cy)
                self._aa_poly(surf, (28, 34, 44), roof_glass)
                for lx in range(84, 128, 9):
                    pygame.draw.line(surf, (150, 158, 168), (lx, cy - 22), (lx + 4, cy - 8), 3)
                    pygame.draw.line(surf, (150, 158, 168), (lx, cy + 22), (lx + 4, cy + 8), 3)

            # Széles, ívelt fekete aktív hátsó szárny (Active Rear Spoiler)
            spoiler_pts = _mirror_poly(
                [(42, cy)]
                + _bezier((42, cy), (42, cy - 54), (54, cy - 72), (76, cy - 68), 14)
                + [(64, cy - 42), (62, cy)], cy
            )
            self._aa_poly(surf, (22, 24, 28), spoiler_pts)

            # McLaren „Eye-socket” sötétített LED fényszórók elöl
            for s in (-1, 1):
                hl_pts = [(364, cy + s * 62), (398, cy + s * 44), (382, cy + s * 36), (354, cy + s * 52)]
                self._aa_poly(surf, (24, 28, 34), hl_pts)
                pygame.draw.line(surf, (240, 248, 255), (370, cy + s * 58), (394, cy + s * 44), 3)

        else:
            # --- PORSCHE MISSION X SPECIFIKUS RÉSZLETEK ---
            # Első csomagtértető sötét légkivezető nyílások (Dual hood vents)
            for s in (-1, 1):
                vent = [(308, cy + s * 14), (344, cy + s * 20), (340, cy + s * 36), (304, cy + s * 28)]
                self._aa_poly(surf, (24, 26, 30), vent)

            # Le Mans üvegkupola és hátsó sötét üveg motortér
            dome_top = (
                [(78, cy)]
                + _bezier((78, cy), (84, cy - 42), (150, cy - 48), (225, cy - 48), 16)
                + _bezier((225, cy - 48), (275, cy - 48), (302, cy - 34), (306, cy), 14)
            )
            self._aa_poly(surf, (16, 20, 26), _mirror_poly(dome_top, cy))

            # Üvegen át látszó világosszürke sportülések
            for sy in (cy - 20, cy + 20):
                pygame.draw.rect(surf, (110, 116, 125), (195, sy - 11, 36, 22), border_radius=5)

            # Jellegzetes fehér középső tetőgerinc és tető-légbeömlő (White central roof spine)
            spine = _mirror_poly([(185, cy - 8), (255, cy - 11), (262, cy - 16), (276, cy - 14), (278, cy)], cy)
            self._aa_poly(surf, c_main, spine)

            # Hátsó üveg alatt látható kerek bronz/arany turbina-ventilátor
            pygame.draw.circle(surf, (145, 118, 72), (112, cy), 14)
            pygame.draw.circle(surf, (30, 28, 25), (112, cy), 10)
            for ang in range(0, 360, 45):
                rad = math.radians(ang)
                pygame.draw.line(surf, (180, 150, 95), (112, cy), (int(112 + 10 * math.cos(rad)), int(cy + 10 * math.sin(rad))), 2)

            # Arany/piros Porsche-címer az orron
            pygame.draw.rect(surf, (215, 175, 55), (392, cy - 4, 7, 8), border_radius=2)
            pygame.draw.line(surf, (190, 30, 30), (395, cy - 3), (395, cy + 3), 2)

            # Porsche vertikális LED fényszórók
            for s in (-1, 1):
                hl = [(346, cy + s * 54), (378, cy + s * 50), (372, cy + s * 62), (342, cy + s * 64)]
                self._aa_poly(surf, (25, 30, 38), hl)
                pygame.draw.line(surf, (245, 250, 255), (352, cy + s * 56), (374, cy + s * 53), 3)

        # Hátsó LED féklámpa-csík (fékezéskor intenzív vörös ragyogással!)
        rear_arc = (
            _bezier((58, cy - 64), (44, cy - 42), (42, cy - 20), (42, cy), 12)
            + _bezier((42, cy), (42, cy + 20), (44, cy + 42), (58, cy + 64), 12)
        )
        if braking:
            glow = pygame.Surface((110, 180), pygame.SRCALPHA)
            pygame.draw.ellipse(glow, (255, 20, 20, 145), (10, 15, 75, 150))
            surf.blit(glow, (0, cy - 90))
            pygame.draw.lines(surf, (255, 35, 35), False, rear_arc, 9)
            pygame.draw.lines(surf, (255, 200, 180), False, rear_arc, 3)
        else:
            pygame.draw.lines(surf, (165, 22, 28), False, rear_arc, 5)

        return surf

    def _init_hd_sprites(self):
        """A 10 nagyfelbontású (5 autó × normál/fékező) sprite előgenerálása és gyorsítótárazása."""
        if self.sprite_cache:
            return
        mapping = {
            "Lead": ("f1", "ferrari"),
            "car_0": ("super", "mclaren_orange"),
            "car_1": ("super", "mclaren_blue"),
            "car_2": ("super", "porsche_mission"),
            "car_3": ("f1", "aston")
        }
        for name, (kind, style) in mapping.items():
            for braking in (False, True):
                if kind == "f1":
                    self.sprite_cache[(name, braking)] = self._build_f1_hd(style, braking)
                else:
                    self.sprite_cache[(name, braking)] = self._build_supercar_hd(style, braking)

    def _blit_hd_car(self, screen, x, center_y, length, width, car_name, acc, vy=0.0):
        import pygame
        self._init_hd_sprites()

        key_name = car_name if car_name in ("Lead", "car_0", "car_1", "car_2", "car_3") else f"car_{ hash(car_name) % 4 }"
        braking = (acc < -0.5)
        hd_surf = self.sprite_cache[(key_name, braking)]

        # 8x Supersampled leskálázás (smoothscale bilinear filterrel a tűéles kontúrokért)
        scaled_surf = pygame.transform.smoothscale(hd_surf, (length, width))

        # Sávváltáskor valódi bekormányzási szög (dinamikus elforgatás menetirányba!)
        if abs(vy) > 0.003:
            angle_deg = float(np.clip(-vy * 135.0, -14.0, 14.0))
            scaled_surf = pygame.transform.rotozoom(scaled_surf, angle_deg, 1.0)

        rect = scaled_surf.get_rect(center=(x, center_y))
        screen.blit(scaled_surf, rect)

    def render(self):
        if self.render_mode != "human":
            return

        import pygame

        width, height = 1380, 410
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("2-Sávos CACC Autópálya — 8x SSAA HD F1 & Supercar Szimuláció")
            self.screen = pygame.display.set_mode((width, height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("monospace", 13, bold=True)
            self.small_font = pygame.font.SysFont("monospace", 11, bold=True)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        all_positions = [self.lead_pos] + [self.state[a]["pos"] for a in self.possible_agents]
        min_pos = min(all_positions)
        max_pos = max(all_positions)
        spread = max(70.0, (max_pos - min_pos) + 36.0)

        target_scale = np.clip((width - 180) / spread, 4.8, 9.0)
        self.camera_scale += (target_scale - self.camera_scale) * 0.15
        scale = self.camera_scale

        target_camera_left = min_pos - 16.0
        self.camera_x += (target_camera_left - self.camera_x) * 0.25
        camera_left_world = self.camera_x

        self.screen.fill((24, 28, 36))
        road_y = 140
        lane_height = 64
        total_road_height = lane_height * 2

        # Aszfalt és szegélyvonalak
        pygame.draw.rect(self.screen, (46, 50, 58), (0, road_y, width, total_road_height))
        pygame.draw.line(self.screen, (235, 235, 235), (0, road_y), (width, road_y), 3)
        pygame.draw.line(self.screen, (235, 235, 235), (0, road_y + total_road_height), (width, road_y + total_road_height), 3)

        visible_world_width = int(width / scale) + 40
        first_stripe = int(camera_left_world // 12) * 12
        for world_x in range(first_stripe, first_stripe + visible_world_width, 12):
            sx = int((world_x - camera_left_world) * scale)
            dash_len = max(12, int(5.0 * scale))
            pygame.draw.line(
                self.screen, (240, 200, 60),
                (sx, road_y + lane_height),
                (sx + dash_len, road_y + lane_height), 2
            )

        first_post = int(camera_left_world // 50) * 50
        for world_x in range(first_post, first_post + visible_world_width, 50):
            sx = int((world_x - camera_left_world) * scale)
            pygame.draw.line(self.screen, (120, 140, 160), (sx, road_y - 16), (sx, road_y), 2)
            pygame.draw.rect(self.screen, (45, 85, 130), (sx - 22, road_y - 34, 44, 18), border_radius=3)
            post_lbl = self.small_font.render(f"{world_x}m", True, (255, 255, 255))
            self.screen.blit(post_lbl, (sx - 18, road_y - 32))

        for agent in self.possible_agents:
            pos = self.state[agent]["pos"]
            lane = self.state[agent]["lane"]
            front_pos, _, _ = self._get_leader_in_lane(pos, lane, exclude_agent=agent)
            dist = front_pos - pos
            x_back = int((pos - camera_left_world) * scale)
            x_front = int((front_pos - camera_left_world) * scale)
            line_y = road_y + int((self.state[agent]["y"] + 0.5) * lane_height) + 22

            if dist <= self.danger_dist:
                line_color = (255, 60, 60)
                txt = f"VESZÉLY! {dist:.1f}m"
            elif dist <= 13.0:
                line_color = (243, 156, 18)
                txt = f"{dist:.1f}m"
            else:
                line_color = (46, 204, 113)
                txt = f"{dist:.1f}m"

            pygame.draw.line(self.screen, line_color, (x_back, line_y), (x_front, line_y), 2)
            lbl = self.small_font.render(txt, True, line_color)
            self.screen.blit(lbl, ((x_back + x_front) // 2 - 18, line_y + 2))

        vehicles = [("Lead", self.lead_pos, self.lead_vel, self.lead_acc, float(self.lead_lane), 0.0)]
        for agent in self.possible_agents:
            vehicles.append((
                agent,
                self.state[agent]["pos"],
                self.state[agent]["vel"],
                self.state[agent]["acc"],
                self.state[agent]["y"],
                self.state[agent].get("vy", 0.0)
            ))

        # HD Autó-méretek (2:1 képarány, hogy az összes aerodinamikai részlet élesen látsszon)
        car_w = max(84, int(11.5 * scale))
        car_h = max(42, int(car_w * 0.50))

        for name, pos, vel, acc, y_val, vy_val in vehicles:
            screen_x = int((pos - camera_left_world) * scale)
            center_y = road_y + int((y_val + 0.5) * lane_height)

            self._blit_hd_car(self.screen, screen_x, center_y, car_w, car_h, name, acc, vy_val)

            vel_kmh = vel * 3.6
            lbl_y = center_y - car_h // 2 - 14 if y_val < 0.5 else center_y + car_h // 2 + 4
            name_lbl = self.small_font.render(f"{name}:{vel_kmh:.0f}km/h", True, (255, 255, 255))
            self.screen.blit(name_lbl, (screen_x - 38, lbl_y))

        info_text = (
            f"Autók: {len(self.possible_agents)}  |  "
            f"Út: {self.traveled_dist:.0f}/{self.track_length:.0f} m  |  "
            f"Idő: {self.current_step * self.dt:.1f} s  |  "
            f"Lead (SF-25): {self.lead_vel * 3.6:.0f} km/h"
        )
        header = self.font.render(info_text, True, (240, 240, 240))
        self.screen.blit(header, (20, 20))

        if self.crash_happened:
            banner = self.font.render("!!! ÜTKÖZÉS TÖRTÉNT (-500 BÜNTETÉS) !!!", True, (255, 70, 70))
            self.screen.blit(banner, (width // 2 - 160, 55))
        elif self.danger_agents:
            warn_names = ", ".join(sorted(self.danger_agents))
            banner = self.font.render(f"VESZÉLYZÓNA (<7m): {warn_names} erős büntetést kap!", True, (255, 165, 50))
            self.screen.blit(banner, (width // 2 - 200, 55))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]