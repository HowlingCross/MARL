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
    metadata = {"render_modes": ["human"], "name": "cacc_merging_v0", "render_fps": 30}

    def __init__(self,
                 num_cars=4,
                 target_speed=36.11,       # ~130 km/h
                 safe_dist=22.0,
                 danger_dist=7.0,
                 crash_dist=4.2,
                 max_dist=260.0,           # Megemelve, hogy a 4+1 autós konvoj ne szakadjon szét tévesen
                 track_length=1500.0,
                 merge_zone_start=220.0,   # Itt kezdődik a szaggatott vonal (besorolási zóna)
                 ramp_end=750.0,           # Itt ér véget fizikailag a felhajtósáv (1-es sáv)
                 weight_progress=0.45,
                 weight_dist=0.06,
                 weight_rel_vel=0.04,
                 weight_jerk=0.02,
                 penalty_crash=-500.0,
                 bonus_merge=120.0,        # Bónusz a sikeres, kooperatív besorolásért
                 bonus_finish=150.0,
                 max_steps=600,
                 render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.num_cars = num_cars
        self.possible_agents = [f"car_{i}" for i in range(self.num_cars)]
        self.agents = self.possible_agents[:]
        self.merging_agent = f"car_{self.num_cars - 1}"  # Az utolsó ágens (car_3) érkezik a felhajtón

        self.dt = 0.1
        self.max_speed = 41.67         # Max 150 km/h
        self.min_speed = 18.00         # Min ~65 km/h (hogy a felhajtón tudjon alkalmazkodni)
        self.max_steps = max_steps
        self.track_length = track_length
        self.merge_zone_start = merge_zone_start
        self.ramp_end = ramp_end

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
        self.bonus_merge = bonus_merge
        self.bonus_finish = bonus_finish

        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None
        self.sprite_cache = {}
        self.camera_x = 0.0
        self.camera_scale = 5.5
        self.crash_happened = False
        self.ramp_overrun = False
        self.merge_completed = False
        self.crashed_agents = set()
        self.danger_agents = set()

        self.action_spaces = {}
        self.observation_spaces = {}

        # 7 elemű megfigyelési tér minden ágensnek (Parameter Sharing kompatibilis):
        # [vel, dist_error, rel_vel, y_lane, main_front_gap, main_rear_gap, rem_ramp_dist]
        obs_low = np.array([
            0.0, -self.safe_dist, -self.max_speed, 0.0,
            -self.max_dist, -self.max_dist, 0.0
        ], dtype=np.float32)
        obs_high = np.array([
            self.max_speed, self.max_dist, self.max_speed, 1.0,
            self.max_dist, self.max_dist, self.ramp_end
        ], dtype=np.float32)

        for agent in self.possible_agents:
            self.action_spaces[agent] = spaces.Box(
                low=np.array([-6.0, -1.0], dtype=np.float32),
                high=np.array([4.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
            self.observation_spaces[agent] = spaces.Box(
                low=obs_low,
                high=obs_high,
                dtype=np.float32
            )

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.current_step = 0
        self.traveled_dist = 0.0
        self.crash_happened = False
        self.ramp_overrun = False
        self.merge_completed = False
        self.crashed_agents = set()
        self.danger_agents = set()
        self.state = {}

        # Vezérjármű (Lead) mindig a főpályán (0-s sáv) halad
        self.lead_pos = 210.0
        self.lead_vel = float(np.random.uniform(31.0, 35.5))
        self.lead_acc = 0.0
        self.lead_lane = 0

        # Főpályás konvoj (car_0, car_1, car_2) felépítése a 0-s sávban
        last_main_pos = self.lead_pos
        last_main_vel = self.lead_vel

        mainline_agents = [a for a in self.possible_agents if a != self.merging_agent]
        for agent in mainline_agents:
            gap = float(np.random.uniform(22.0, 30.0))
            pos = last_main_pos - gap
            vel = float(np.clip(last_main_vel + np.random.uniform(-2.0, 2.0), 26.0, 37.0))
            last_main_pos = pos
            last_main_vel = vel

            self.state[agent] = {
                "pos": pos,
                "vel": vel,
                "acc": 0.0,
                "lane": 0,
                "y": 0.0,
                "vy": 0.0,
                "merged": True
            }

        # Besoroló jármű (car_3) elhelyezése a felhajtósávon (1-es sáv)
        # Pontosan a car_0 (P1) és car_1 (F1) közötti longitudinális tartomány mellé indul,
        # hogy kooperatív résnyitásra és besorolásra legyen szükség!
        p1_pos = self.state["car_0"]["pos"]
        f1_pos = self.state["car_1"]["pos"] if len(mainline_agents) > 1 else p1_pos - 25.0
        merge_init_pos = float(np.random.uniform(f1_pos + 2.0, p1_pos - 4.0))
        merge_init_vel = float(np.clip(self.lead_vel + np.random.uniform(-3.0, 1.5), 25.0, 36.0))

        self.state[self.merging_agent] = {
            "pos": merge_init_pos,
            "vel": merge_init_vel,
            "acc": 0.0,
            "lane": 1,
            "y": 1.0,
            "vy": 0.0,
            "merged": False
        }

        all_x = [self.lead_pos] + [self.state[a]["pos"] for a in self.possible_agents]
        self.camera_x = float(np.min(all_x)) - 25.0

        if self.render_mode == "human":
            self.render()

        return self._get_observations(), {agent: {} for agent in self.agents}

    def _get_leader_in_lane(self, agent_pos, target_lane, exclude_agent=None, include_virtual=True):
        """
        Megkeresi az adott sávban közvetlenül előttünk haladó járművet.
        Ha include_virtual=True, akkor a főpályán (0-s sáv) haladó követő autó (F1)
        érzékeli a felhajtón előtte lévő besoroló autó vetületét (Virtuális jármű V_m).
        """
        best_pos = self.lead_pos if self.lead_lane == target_lane else 99999.0
        best_vel = self.lead_vel if self.lead_lane == target_lane else self.target_speed
        best_name = "Lead" if self.lead_lane == target_lane else "Virtual_Lead"

        for other in self.possible_agents:
            if other == exclude_agent:
                continue
            other_pos = self.state[other]["pos"]
            other_lane = self.state[other]["lane"]
            other_y = self.state[other]["y"]

            same_lane = (other_lane == target_lane) or (abs(other_y - target_lane) < 0.55)

            # Virtuális jármű (V_m) leképezés a főpályára, ha a besoroló autó a vezérlési zónában jár
            virtual_projection = (
                include_virtual
                and target_lane == 0
                and other == self.merging_agent
                and other_lane == 1
                and other_pos >= (self.merge_zone_start - 70.0)
            )

            if (same_lane or virtual_projection) and other_pos > agent_pos:
                if other_pos < best_pos:
                    best_pos = other_pos
                    best_vel = self.state[other]["vel"]
                    best_name = other

        if best_pos == 99999.0:
            best_pos = max(self.lead_pos, agent_pos + self.safe_dist)
            best_vel = self.lead_vel
            best_name = "Lead"

        return best_pos, best_vel, best_name

    def _get_mainline_gaps(self, pos, exclude_agent=None):
        """Visszaadja a főpályán (0-s sáv) előttünk és mögöttünk lévő legközelebbi autó távolságát és sebességét."""
        front_dist = self.lead_pos - pos if self.lead_pos >= pos else self.max_dist
        front_vel = self.lead_vel
        rear_dist = self.max_dist
        rear_vel = self.target_speed

        for other in self.possible_agents:
            if other == exclude_agent:
                continue
            if self.state[other]["lane"] == 0 or self.state[other]["y"] < 0.55:
                d = self.state[other]["pos"] - pos
                if d >= 0.0 and d < front_dist:
                    front_dist = d
                    front_vel = self.state[other]["vel"]
                elif d < 0.0 and (-d) < rear_dist:
                    rear_dist = -d
                    rear_vel = self.state[other]["vel"]

        return (
            float(np.clip(front_dist, 0.0, self.max_dist)),
            front_vel,
            float(np.clip(rear_dist, 0.0, self.max_dist)),
            rear_vel
        )

    def step(self, actions):
        self.current_step += 1
        self.danger_agents = set()

        # ---------------------------------------------------------------------
        # 1. VEZÉRJÁRMŰ (LEAD) MOZGÁSA
        # ---------------------------------------------------------------------
        if 550.0 <= self.traveled_dist <= 850.0:
            target_lead_v = 29.5
        elif 1050.0 <= self.traveled_dist <= 1350.0:
            target_lead_v = 39.0
        else:
            target_lead_v = self.target_speed + 2.5 * np.sin(self.current_step * 0.03)

        self.lead_acc = float(np.clip((target_lead_v - self.lead_vel) * 1.2, -4.0, 2.5))
        self.lead_vel = float(np.clip(self.lead_vel + self.lead_acc * self.dt, self.min_speed, self.max_speed))
        self.lead_pos += self.lead_vel * self.dt
        self.traveled_dist += self.lead_vel * self.dt

        step_dists = {}
        jerks = {}
        merge_bonuses = {agent: 0.0 for agent in self.agents}

        # ---------------------------------------------------------------------
        # 2. ÁGENSEK KINEMATIKÁJA ÉS SÁVVÁLTÁSA
        # ---------------------------------------------------------------------
        for agent in self.agents:
            act = actions[agent]
            prev_acc = self.state[agent]["acc"]
            acceleration = float(np.clip(act[0], -6.0, 4.0))
            lane_cmd = float(act[1]) if len(act) > 1 else 0.0

            jerks[agent] = abs(acceleration - prev_acc)
            self.state[agent]["acc"] = acceleration

            pos = self.state[agent]["pos"]

            # Besorolási döntés: a felhajtón (1-es sáv) lévő autó átléphet a 0-s sávba
            if self.state[agent]["lane"] == 1 and pos >= self.merge_zone_start:
                front_gap, _, rear_gap, _ = self._get_mainline_gaps(pos, exclude_agent=agent)
                if lane_cmd < 0.0:
                    self.state[agent]["lane"] = 0
                    if front_gap > 7.0 and rear_gap > 7.0:
                        self.state[agent]["merged"] = True
                        self.merge_completed = True
                        safety_factor = min(front_gap, rear_gap) / self.safe_dist
                        merge_bonuses[agent] = self.bonus_merge * float(np.clip(safety_factor, 0.5, 1.3))
                        for coop_a in self.agents:
                            if coop_a != agent and self.state[coop_a]["pos"] < pos:
                                merge_bonuses[coop_a] += 35.0

            # Oldalirányú (y) mozgás folyamatos átmenettel
            target_y = float(self.state[agent]["lane"])
            dy = float(np.clip(target_y - self.state[agent]["y"], -0.10, 0.10))
            self.state[agent]["vy"] = dy
            self.state[agent]["y"] += dy

            # Hosszirányú sebesség és pozíció frissítése
            self.state[agent]["vel"] = float(
                np.clip(self.state[agent]["vel"] + acceleration * self.dt, self.min_speed, self.max_speed)
            )
            d_pos = self.state[agent]["vel"] * self.dt
            self.state[agent]["pos"] += d_pos
            step_dists[agent] = d_pos

            # Felhajtósáv vége ellenőrzése
            if self.state[agent]["y"] > 0.45 and self.state[agent]["pos"] >= self.ramp_end:
                self.ramp_overrun = True
                self.crashed_agents.add(agent)

        observations = self._get_observations()
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        split_happened = False

        # ---------------------------------------------------------------------
        # 3. VALÓDI 2D HITBOX ÜTKÖZÉSVIZSGÁLAT (X és Y egyidejű metszete)
        # ---------------------------------------------------------------------
        HITBOX_X = self.crash_dist   # 4.2 méter hosszirányban
        HITBOX_Y = 0.52              # 0.52 sávszélesség oldalirányban (ha külön sávban vannak, dy ~ 1.0 -> nem ütköznek!)

        for agent in self.agents:
            pos_i = self.state[agent]["pos"]
            y_i = self.state[agent]["y"]

            # Ellenőrzés a Lead ellen (Lead mindig a 0-s sávban halad, y = 0.0)
            dx_lead = abs(self.lead_pos - pos_i)
            dy_lead = abs(float(self.lead_lane) - y_i)
            if dx_lead <= HITBOX_X and dy_lead <= HITBOX_Y:
                self.crash_happened = True
                self.crashed_agents.add(agent)
            elif dx_lead <= self.danger_dist and dy_lead <= HITBOX_Y:
                self.danger_agents.add(agent)

            # Ellenőrzés a többi tanuló jármű ellen
            for other in self.agents:
                if other == agent:
                    continue
                dx = abs(self.state[other]["pos"] - pos_i)
                dy = abs(self.state[other]["y"] - y_i)

                if dx <= HITBOX_X and dy <= HITBOX_Y:
                    self.crash_happened = True
                    self.crashed_agents.add(agent)
                    self.crashed_agents.add(other)
                elif dx <= self.danger_dist and dy <= HITBOX_Y:
                    self.danger_agents.add(agent)

        # ---------------------------------------------------------------------
        # 4. JUTALOMFÜGGVÉNYEK SZÁMÍTÁSA
        # ---------------------------------------------------------------------
        for agent in self.agents:
            pos = self.state[agent]["pos"]
            lane = self.state[agent]["lane"]

            # Valós elöl haladó a fizikai sávban
            phys_front_pos, phys_front_vel, _ = self._get_leader_in_lane(
                pos, lane, exclude_agent=agent, include_virtual=False
            )
            phys_dist = phys_front_pos - pos

            # Szabályozási referencia (a virtuális vetület figyelembevételével a résnyitáshoz)
            ref_front_pos, ref_front_vel, _ = self._get_leader_in_lane(
                pos, lane, exclude_agent=agent, include_virtual=True
            )
            ref_dist = ref_front_pos - pos
            rel_vel = ref_front_vel - self.state[agent]["vel"]

            if phys_dist >= self.max_dist or (self.lead_pos - pos) >= self.max_dist:
                split_happened = True

            progress_rew = step_dists[agent] * self.weight_progress
            dist_penalty = abs(self.safe_dist - ref_dist) * self.weight_dist
            vel_penalty = abs(rel_vel) * self.weight_rel_vel
            jerk_penalty = jerks[agent] * self.weight_jerk

            ramp_urgency_pen = 0.0
            if self.state[agent]["y"] > 0.35 and pos > self.merge_zone_start:
                progress_on_ramp = (pos - self.merge_zone_start) / (self.ramp_end - self.merge_zone_start)
                ramp_urgency_pen = float(np.clip(progress_on_ramp, 0.0, 1.0)) * 0.8

            proximity_penalty = 0.0
            if agent in self.danger_agents:
                proximity_penalty = 40.0 + max(0.0, self.danger_dist - phys_dist) * 15.0
            elif phys_dist <= 13.0 and self.state[agent]["y"] < 0.35:
                proximity_penalty = (13.0 - phys_dist) * 2.5

            rewards[agent] = (
                progress_rew
                - dist_penalty
                - vel_penalty
                - jerk_penalty
                - proximity_penalty
                - ramp_urgency_pen
                + merge_bonuses[agent]
            )

        # ---------------------------------------------------------------------
        # 5. EPIZÓDVÉGI FELTÉTELEK (ÜTKÖZÉS / PÁLYAVÉG)
        # ---------------------------------------------------------------------
        if self.crash_happened or self.ramp_overrun or split_happened:
            remaining_ratio = (self.max_steps - self.current_step) / self.max_steps
            heavy_crash_pen = self.penalty_crash - (remaining_ratio * 200.0)
            for agent in self.agents:
                if agent in self.crashed_agents or split_happened or self.ramp_overrun:
                    rewards[agent] = heavy_crash_pen
                else:
                    rewards[agent] = heavy_crash_pen * 0.4
                terminations[agent] = True

            if self.render_mode == "human":
                self.render()
                import pygame
                pygame.time.wait(1500)

            self.agents = []
            return observations, rewards, terminations, truncations, {a: {} for a in self.possible_agents}

        if self.traveled_dist >= self.track_length or self.current_step >= self.max_steps:
            for agent in self.agents:
                if self.traveled_dist >= self.track_length and self.merge_completed:
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

            # Követési hiba és relatív sebesség (Virtuális jármű V_m figyelembevételével)
            front_pos, front_vel, _ = self._get_leader_in_lane(
                pos, lane, exclude_agent=agent, include_virtual=True
            )
            dist_error = float(np.clip((front_pos - pos) - self.safe_dist, -self.safe_dist, self.max_dist))
            rel_vel = float(np.clip(front_vel - vel, -self.max_speed, self.max_speed))

            # Főpályás (0-s sáv) első és hátsó rések
            main_front_gap, _, main_rear_gap, _ = self._get_mainline_gaps(pos, exclude_agent=agent)

            # Hátralévő távolság a felhajtósáv végéig (ha már a főpályán van, akkor ramp_end)
            if lane == 1:
                rem_ramp = float(np.clip(self.ramp_end - pos, 0.0, self.ramp_end))
            else:
                rem_ramp = float(self.ramp_end)

            obs[agent] = np.array([
                vel,
                dist_error,
                rel_vel,
                self.state[agent]["y"],
                main_front_gap,
                main_rear_gap,
                rem_ramp
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
        pygame.draw.rect(surf, (18, 19, 22), rect, border_radius=9)
        pygame.draw.rect(surf, (38, 40, 45), (cx - w // 2 + 5, cy - h // 2 + 3, w - 10, h - 6), border_radius=6)
        pygame.draw.rect(surf, (26, 28, 32), (cx - w // 2 + 10, cy - h // 2 + 5, w - 20, h - 10), border_radius=4)
        pygame.draw.line(surf, stripe_color, (cx - w // 2 + 8, cy - h // 2 + 3), (cx + w // 2 - 8, cy - h // 2 + 3), 3)
        pygame.draw.line(surf, stripe_color, (cx - w // 2 + 8, cy + h // 2 - 4), (cx + w // 2 - 8, cy + h // 2 - 4), 3)

    def _build_f1_hd(self, variant="ferrari", braking=False):
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
            tire_stripe = (230, 30, 30)
            helmet_col = (240, 215, 30)
            num_txt = "44"
        else:
            c_dark = (0, 68, 58)
            c_main = (0, 115, 98)
            c_bright = (18, 162, 138)
            c_wing_main = (18, 22, 26)
            c_wing_flap = (230, 235, 240)
            c_stripe = (195, 235, 15)
            c_accent = (205, 245, 20)
            tire_stripe = (245, 205, 20)
            helmet_col = (0, 175, 145)
            num_txt = "18"

        shadow_pts = _mirror_poly([
            (35, cy - 68), (140, cy - 84), (310, cy - 82), (418, cy - 72), (425, cy)
        ], cy)
        self._aa_poly(surf, (0, 0, 0, 95), shadow_pts)

        floor_top = [
            (55, cy - 36), (75, cy - 62), (165, cy - 66),
            (250, cy - 64), (295, cy - 48), (320, cy - 18)
        ]
        self._aa_poly(surf, (22, 24, 28), _mirror_poly(floor_top, cy))
        pygame.draw.lines(surf, (58, 62, 70), False, [(85, cy - 64), (245, cy - 63), (290, cy - 48)], 3)
        pygame.draw.lines(surf, (58, 62, 70), False, [(85, cy + 64), (245, cy + 63), (290, cy + 48)], 3)

        susp_col = (35, 38, 44)
        for s in (-1, 1):
            pygame.draw.line(surf, susp_col, (118, cy + s * 18), (92, cy + s * 72), 6)
            pygame.draw.line(surf, susp_col, (76, cy + s * 16), (92, cy + s * 72), 6)
            pygame.draw.line(surf, (60, 65, 75), (132, cy + s * 14), (96, cy + s * 68), 4)
            pygame.draw.line(surf, susp_col, (345, cy + s * 12), (322, cy + s * 70), 6)
            pygame.draw.line(surf, susp_col, (302, cy + s * 14), (322, cy + s * 70), 6)
            pygame.draw.line(surf, (70, 75, 85), (288, cy + s * 14), (318, cy + s * 66), 4)

        self._draw_f1_wheel(surf, 94, cy - 74, 68, 36, tire_stripe)
        self._draw_f1_wheel(surf, 94, cy + 74, 68, 36, tire_stripe)
        self._draw_f1_wheel(surf, 322, cy - 72, 62, 32, tire_stripe)
        self._draw_f1_wheel(surf, 322, cy + 72, 62, 32, tire_stripe)

        sidepod_curve = (
            [(72, cy - 15)]
            + _bezier((72, cy - 15), (115, cy - 22), (140, cy - 58), (205, cy - 58), 18)
            + _bezier((205, cy - 58), (245, cy - 58), (268, cy - 52), (278, cy - 24), 12)
            + [(412, cy - 7), (416, cy)]
        )
        self._aa_poly(surf, c_dark, _mirror_poly(sidepod_curve, cy))

        inner_curve = (
            [(78, cy - 11)]
            + _bezier((78, cy - 11), (125, cy - 18), (150, cy - 48), (205, cy - 48), 16)
            + _bezier((205, cy - 48), (240, cy - 48), (260, cy - 42), (272, cy - 18), 10)
            + [(408, cy - 5), (412, cy)]
        )
        self._aa_poly(surf, c_main, _mirror_poly(inner_curve, cy))

        highlight_curve = (
            [(95, cy - 6)]
            + _bezier((95, cy - 6), (160, cy - 16), (220, cy - 20), (285, cy - 14), 14)
            + [(402, cy - 3), (406, cy)]
        )
        self._aa_poly(surf, c_bright, _mirror_poly(highlight_curve, cy))

        for s in (-1, 1):
            undercut = [
                (125, cy + s * 22), (175, cy + s * 46),
                (245, cy + s * 46), (264, cy + s * 26), (195, cy + s * 30)
            ]
            self._aa_poly(surf, (18, 20, 24), undercut)
            for gx in range(165, 225, 10):
                pygame.draw.line(surf, (25, 25, 30), (gx, cy + s * 26), (gx - 5, cy + s * 38), 3)

        spine_poly = _mirror_poly([
            (75, cy - 3), (135, cy - 13), (195, cy - 14), (210, cy - 4)
        ], cy)
        self._aa_poly(surf, c_stripe, spine_poly)
        pygame.draw.line(surf, (20, 20, 25), (72, cy), (190, cy), 2)

        fw_back = _mirror_poly([
            (372, cy - 12), (382, cy - 82), (408, cy - 84), (404, cy - 48), (418, cy - 8)
        ], cy)
        self._aa_poly(surf, c_wing_main, fw_back)
        for offset in (0, 9):
            flap = _mirror_poly([
                (386 + offset, cy - 16), (392 + offset, cy - 80),
                (398 + offset, cy - 80), (394 + offset, cy - 16)
            ], cy)
            self._aa_poly(surf, c_wing_flap, flap)
        pygame.draw.rect(surf, c_stripe, (376, cy - 86, 36, 6), border_radius=3)
        pygame.draw.rect(surf, c_stripe, (376, cy + 80, 36, 6), border_radius=3)

        nose_tip = _mirror_poly([(350, cy - 10), (416, cy - 5), (422, cy)], cy)
        self._aa_poly(surf, c_main, nose_tip)
        self._aa_poly(surf, c_stripe, _mirror_poly([(404, cy - 5), (420, cy - 3), (422, cy)], cy))

        cockpit_pts = _mirror_poly([
            (202, cy - 12), (222, cy - 18), (262, cy - 16), (275, cy - 8)
        ], cy)
        self._aa_poly(surf, (14, 16, 20), cockpit_pts)

        pygame.draw.circle(surf, helmet_col, (236, cy), 12)
        pygame.draw.arc(surf, (15, 20, 30), (226, cy - 11, 22, 22), -1.1, 1.1, 6)
        pygame.draw.rect(surf, (45, 50, 58), (254, cy - 9, 6, 18), border_radius=2)

        airbox_pts = _mirror_poly([(178, cy - 4), (204, cy - 8), (208, cy)], cy)
        self._aa_poly(surf, (18, 18, 22), airbox_pts)
        pygame.draw.rect(surf, c_accent, (194, cy - 14, 6, 28), border_radius=2)

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

        rw_rect = pygame.Rect(30, cy - 56, 30, 112)
        pygame.draw.rect(surf, c_wing_main, rw_rect, border_radius=5)
        pygame.draw.rect(surf, c_wing_flap, (36, cy - 52, 10, 104), border_radius=3)
        pygame.draw.rect(surf, c_stripe, (48, cy - 52, 8, 104), border_radius=2)
        pygame.draw.rect(surf, (20, 20, 24), (26, cy - 60, 38, 6), border_radius=2)
        pygame.draw.rect(surf, (20, 20, 24), (26, cy + 54, 38, 6), border_radius=2)

        num_font = pygame.font.SysFont("sans", 22, bold=True)
        txt_surf = num_font.render(num_txt, True, c_accent)
        txt_rot = pygame.transform.rotate(txt_surf, -90)
        surf.blit(txt_rot, (318, cy - txt_rot.get_height() // 2))

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
        else:
            c_edge = (182, 188, 198)
            c_main = (236, 240, 246)
            c_high = (252, 253, 255)

        outer_top = (
            [(42, cy)]
            + _bezier((42, cy), (42, cy - 58), (58, cy - 74), (105, cy - 78), 16)
            + _bezier((105, cy - 78), (150, cy - 80), (175, cy - 66), (218, cy - 66), 14)
            + _bezier((218, cy - 66), (262, cy - 66), (290, cy - 78), (338, cy - 76), 14)
            + _bezier((338, cy - 76), (382, cy - 72), (414, cy - 42), (416, cy), 16)
        )
        body_poly = _mirror_poly(outer_top, cy)

        shadow_poly = [(x - 4, y + 4) for (x, y) in body_poly]
        self._aa_poly(surf, (0, 0, 0, 105), shadow_poly)
        self._aa_poly(surf, c_edge, body_poly)

        mid_top = [(x, int(cy + (y - cy) * 0.93)) for (x, y) in outer_top]
        self._aa_poly(surf, c_main, _mirror_poly(mid_top, cy))

        for s in (-1, 1):
            fender_front = _bezier((275, cy + s * 52), (315, cy + s * 66), (355, cy + s * 64), (388, cy + s * 38), 14)
            fender_rear = _bezier((68, cy + s * 56), (105, cy + s * 68), (150, cy + s * 68), (188, cy + s * 52), 14)
            pygame.draw.lines(surf, c_high, False, fender_front, 7)
            pygame.draw.lines(surf, c_high, False, fender_rear, 7)

        for s in (-1, 1):
            pygame.draw.line(surf, (30, 32, 36), (262, cy + s * 54), (254, cy + s * 84), 5)
            mirror_pts = [
                (246, cy + s * 82), (262, cy + s * 80),
                (258, cy + s * 90), (242, cy + s * 88)
            ]
            self._aa_poly(surf, c_main, mirror_pts)

        if model in ("mclaren_orange", "mclaren_blue"):
            for s in (-1, 1):
                nostril = [(332, cy + s * 20), (368, cy + s * 32), (358, cy + s * 39), (326, cy + s * 25)]
                self._aa_poly(surf, (22, 24, 28), nostril)
                door_blade = _bezier((165, cy + s * 66), (205, cy + s * 56), (245, cy + s * 56), (278, cy + s * 64), 12)
                pygame.draw.lines(surf, (20, 22, 26), False, door_blade, 6)

            canopy_top = (
                [(72, cy)]
                + _bezier((72, cy), (85, cy - 36), (155, cy - 52), (225, cy - 52), 16)
                + _bezier((225, cy - 52), (275, cy - 52), (308, cy - 36), (314, cy), 14)
            )
            self._aa_poly(surf, (18, 22, 28), _mirror_poly(canopy_top, cy))

            windshield = _mirror_poly(
                [(232, cy - 42)]
                + _bezier((232, cy - 42), (272, cy - 46), (298, cy - 32), (304, cy), 12)
                + [(242, cy)], cy
            )
            self._aa_poly(surf, (38, 48, 60), windshield)

            if model == "mclaren_blue":
                for sy in (cy - 22, cy + 22):
                    pygame.draw.rect(surf, (48, 54, 62), (186, sy - 14, 44, 28), border_radius=7)
                    pygame.draw.rect(surf, (32, 36, 42), (194, sy - 10, 28, 20), border_radius=5)
                    pygame.draw.rect(surf, (68, 74, 82), (180, sy - 9, 12, 18), border_radius=4)
                pygame.draw.circle(surf, (25, 28, 32), (238, cy - 22), 9, 3)
                self._aa_poly(surf, (165, 22, 28), _mirror_poly([(96, cy), (112, cy - 18), (132, cy - 12), (132, cy)], cy))
            else:
                roof_glass = _mirror_poly([(168, cy - 34), (226, cy - 38), (228, cy)], cy)
                self._aa_poly(surf, (28, 34, 44), roof_glass)
                for lx in range(84, 128, 9):
                    pygame.draw.line(surf, (150, 158, 168), (lx, cy - 22), (lx + 4, cy - 8), 3)
                    pygame.draw.line(surf, (150, 158, 168), (lx, cy + 22), (lx + 4, cy + 8), 3)

            spoiler_pts = _mirror_poly(
                [(42, cy)]
                + _bezier((42, cy), (42, cy - 54), (54, cy - 72), (76, cy - 68), 14)
                + [(64, cy - 42), (62, cy)], cy
            )
            self._aa_poly(surf, (22, 24, 28), spoiler_pts)

            for s in (-1, 1):
                hl_pts = [(364, cy + s * 62), (398, cy + s * 44), (382, cy + s * 36), (354, cy + s * 52)]
                self._aa_poly(surf, (24, 28, 34), hl_pts)
                pygame.draw.line(surf, (240, 248, 255), (370, cy + s * 58), (394, cy + s * 44), 3)

        else:
            for s in (-1, 1):
                vent = [(308, cy + s * 14), (344, cy + s * 20), (340, cy + s * 36), (304, cy + s * 28)]
                self._aa_poly(surf, (24, 26, 30), vent)

            dome_top = (
                [(78, cy)]
                + _bezier((78, cy), (84, cy - 42), (150, cy - 48), (225, cy - 48), 16)
                + _bezier((225, cy - 48), (275, cy - 48), (302, cy - 34), (306, cy), 14)
            )
            self._aa_poly(surf, (16, 20, 26), _mirror_poly(dome_top, cy))

            for sy in (cy - 20, cy + 20):
                pygame.draw.rect(surf, (110, 116, 125), (195, sy - 11, 36, 22), border_radius=5)

            spine = _mirror_poly([(185, cy - 8), (255, cy - 11), (262, cy - 16), (276, cy - 14), (278, cy)], cy)
            self._aa_poly(surf, c_main, spine)

            pygame.draw.circle(surf, (145, 118, 72), (112, cy), 14)
            pygame.draw.circle(surf, (30, 28, 25), (112, cy), 10)
            for ang in range(0, 360, 45):
                rad = math.radians(ang)
                pygame.draw.line(surf, (180, 150, 95), (112, cy), (int(112 + 10 * math.cos(rad)), int(cy + 10 * math.sin(rad))), 2)

            pygame.draw.rect(surf, (215, 175, 55), (392, cy - 4, 7, 8), border_radius=2)
            pygame.draw.line(surf, (190, 30, 30), (395, cy - 3), (395, cy + 3), 2)

            for s in (-1, 1):
                hl = [(346, cy + s * 54), (378, cy + s * 50), (372, cy + s * 62), (342, cy + s * 64)]
                self._aa_poly(surf, (25, 30, 38), hl)
                pygame.draw.line(surf, (245, 250, 255), (352, cy + s * 56), (374, cy + s * 53), 3)

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

        scaled_surf = pygame.transform.smoothscale(hd_surf, (length, width))

        if abs(vy) > 0.003:
            angle_deg = float(np.clip(-vy * 135.0, -14.0, 14.0))
            scaled_surf = pygame.transform.rotozoom(scaled_surf, angle_deg, 1.0)

        rect = scaled_surf.get_rect(center=(x, center_y))
        screen.blit(scaled_surf, rect)

    def render(self):
        if self.render_mode != "human":
            return

        import pygame

        width, height = 1380, 420
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("CACC + Kooperatív Autópályás Besorolás (On-Ramp Merging) — 8x SSAA HD")
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
        spread = max(75.0, (max_pos - min_pos) + 42.0)

        target_scale = np.clip((width - 180) / spread, 4.5, 8.8)
        self.camera_scale += (target_scale - self.camera_scale) * 0.15
        scale = self.camera_scale

        target_camera_left = min_pos - 18.0
        self.camera_x += (target_camera_left - self.camera_x) * 0.25
        camera_left_world = self.camera_x

        self.screen.fill((24, 28, 36))
        road_y = 140
        lane_height = 64

        # 0-s sáv (Főpálya - végig fut)
        pygame.draw.rect(self.screen, (46, 50, 58), (0, road_y, width, lane_height))
        pygame.draw.line(self.screen, (235, 235, 235), (0, road_y), (width, road_y), 3)

        # 1-es sáv (Besoroló sáv - ramp_end-nél véget ér!)
        ramp_end_sx = int((self.ramp_end - camera_left_world) * scale)
        taper_len = int(28.0 * scale)
        ramp_start_taper_sx = ramp_end_sx - taper_len

        if ramp_end_sx > 0:
            flat_end = max(0, min(width, ramp_start_taper_sx))
            if flat_end > 0:
                pygame.draw.rect(self.screen, (52, 56, 66), (0, road_y + lane_height, flat_end, lane_height))
                pygame.draw.line(self.screen, (235, 235, 235), (0, road_y + lane_height * 2), (flat_end, road_y + lane_height * 2), 3)

            # Beszűkülő terelőszakasz a besorolósáv végén
            taper_poly = [
                (ramp_start_taper_sx, road_y + lane_height),
                (ramp_end_sx, road_y + lane_height),
                (ramp_start_taper_sx, road_y + lane_height * 2)
            ]
            pygame.draw.polygon(self.screen, (52, 56, 66), taper_poly)
            pygame.draw.line(
                self.screen, (230, 60, 60),
                (ramp_start_taper_sx, road_y + lane_height * 2),
                (ramp_end_sx, road_y + lane_height), 4
            )
            lbl_end = self.small_font.render("BESOROLÓ SÁV VÉGE (750m)", True, (255, 110, 110))
            self.screen.blit(lbl_end, (ramp_start_taper_sx - 20, road_y + lane_height * 2 + 8))

        if ramp_end_sx < width:
            pygame.draw.line(
                self.screen, (235, 235, 235),
                (max(0, ramp_end_sx), road_y + lane_height),
                (width, road_y + lane_height), 3
            )

        # Sávelválasztó vonal (merge_zone_start-ig folytonos záróvonal, utána szaggatott)
        merge_start_sx = int((self.merge_zone_start - camera_left_world) * scale)
        if merge_start_sx > 0:
            pygame.draw.line(
                self.screen, (240, 240, 240),
                (0, road_y + lane_height),
                (min(width, merge_start_sx), road_y + lane_height), 3
            )

        visible_world_width = int(width / scale) + 40
        first_stripe = int(camera_left_world // 12) * 12
        for world_x in range(first_stripe, first_stripe + visible_world_width, 12):
            if self.merge_zone_start <= world_x <= self.ramp_end:
                sx = int((world_x - camera_left_world) * scale)
                dash_len = max(12, int(5.0 * scale))
                pygame.draw.line(
                    self.screen, (240, 200, 60),
                    (sx, road_y + lane_height),
                    (sx + dash_len, road_y + lane_height), 2
                )

        # Kilométer-/méterkövek
        first_post = int(camera_left_world // 50) * 50
        for world_x in range(first_post, first_post + visible_world_width, 50):
            sx = int((world_x - camera_left_world) * scale)
            pygame.draw.line(self.screen, (120, 140, 160), (sx, road_y - 16), (sx, road_y), 2)
            pygame.draw.rect(self.screen, (45, 85, 130), (sx - 22, road_y - 34, 44, 18), border_radius=3)
            post_lbl = self.small_font.render(f"{world_x}m", True, (255, 255, 255))
            self.screen.blit(post_lbl, (sx - 18, road_y - 32))

        # Virtuális jármű (V_m) vetület megjelenítése a főpályán, amíg a car_3 a felhajtón van
        m_state = self.state[self.merging_agent]
        if m_state["lane"] == 1:
            vm_sx = int((m_state["pos"] - camera_left_world) * scale)
            vm_cy = road_y + int(0.5 * lane_height)
            ramp_cy = road_y + int((m_state["y"] + 0.5) * lane_height)
            pygame.draw.rect(self.screen, (46, 204, 113), (vm_sx - 36, vm_cy - 16, 72, 32), 2, border_radius=5)
            pygame.draw.line(self.screen, (46, 204, 113), (vm_sx, ramp_cy - 18), (vm_sx, vm_cy + 16), 2)
            vm_lbl = self.small_font.render("V_m vetület", True, (46, 204, 113))
            self.screen.blit(vm_lbl, (vm_sx - 34, vm_cy - 30))

        # Távolságvonalak kirajzolása
        for agent in self.possible_agents:
            pos = self.state[agent]["pos"]
            lane = self.state[agent]["lane"]
            front_pos, _, _ = self._get_leader_in_lane(pos, lane, exclude_agent=agent, include_virtual=True)
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

        car_w = max(84, int(11.5 * scale))
        car_h = max(42, int(car_w * 0.50))

        for name, pos, vel, acc, y_val, vy_val in vehicles:
            screen_x = int((pos - camera_left_world) * scale)
            center_y = road_y + int((y_val + 0.5) * lane_height)

            self._blit_hd_car(self.screen, screen_x, center_y, car_w, car_h, name, acc, vy_val)

            vel_kmh = vel * 3.6
            lbl_y = center_y - car_h // 2 - 14 if y_val < 0.5 else center_y + car_h // 2 + 4
            role_tag = " (Besoroló)" if name == self.merging_agent else ""
            name_lbl = self.small_font.render(f"{name}{role_tag}:{vel_kmh:.0f}km/h", True, (255, 255, 255))
            self.screen.blit(name_lbl, (screen_x - 42, lbl_y))

        merge_status = "SIKERES" if self.merge_completed else "FOLYAMATBAN"
        info_text = (
            f"Út: {self.traveled_dist:.0f}/{self.track_length:.0f} m  |  "
            f"Idő: {self.current_step * self.dt:.1f} s  |  "
            f"Besorolás ({self.merging_agent}): {merge_status}  |  "
            f"Lead: {self.lead_vel * 3.6:.0f} km/h"
        )
        header = self.font.render(info_text, True, (240, 240, 240))
        self.screen.blit(header, (20, 20))

        if self.crash_happened:
            banner = self.font.render("!!! ÜTKÖZÉS TÖRTÉNT (-500 BÜNTETÉS) !!!", True, (255, 70, 70))
            self.screen.blit(banner, (width // 2 - 160, 55))
        elif self.ramp_overrun:
            banner = self.font.render("!!! ELFOGYOTT A BESOROLÓSÁV (-500 BÜNTETÉS) !!!", True, (255, 70, 70))
            self.screen.blit(banner, (width // 2 - 190, 55))
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