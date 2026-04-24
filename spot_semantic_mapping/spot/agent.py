import pickle as pkl
import numpy as np
import bosdyn.client
import asyncio
import json
import cv2
from pathlib import Path
import threading
import time

from bosdyn.client.frame_helpers import (
    get_a_tform_b,
    ODOM_FRAME_NAME,
    BODY_FRAME_NAME,
)
from bosdyn.client.local_grid import LocalGridClient

from spot_semantic_mapping.models.embedding import DinoModel
from spot_semantic_mapping.spot.decoding import decode_rgb, decode_depth
from spot_semantic_mapping.mapping.io import load_scene_graph
from spot_semantic_mapping.localization.localizer import localize, retrieve_subgraphs, print_subgraph
from spot_semantic_mapping.localization.encoder import ImageEncoder


def authenticate_from_file(robot, cred_path: str):
    cred_path = Path(cred_path).expanduser()
    creds = json.loads(cred_path.read_text())
    robot.authenticate(creds["username"], creds["password"])


class SpotAgent:
    def __init__(self, hostname):
        self.cameras = {
            "hand": ["hand_color_image", "hand_depth_in_hand_color_frame"],
            "frontleft": ["frontleft_depth_in_visual_frame", "frontleft_fisheye_image"],
            "frontright": ["frontright_depth_in_visual_frame", "frontright_fisheye_image"],
            "left": ["left_depth_in_visual_frame", "left_fisheye_image"],
            "right": ["right_depth_in_visual_frame", "right_fisheye_image"],
            "back": ["back_depth_in_visual_frame", "back_fisheye_image"],
        }

        self.observation = {
            'battery': None,
            'cameras': {},
            'imu': {
                'angular_velocity': None,
                'linear_velocity': None
            },
            'pose': {
                'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'rotation': {'w': 1.0, 'x': 0.0, 'y': 0.0, 'z': 0.0}
            }
        }

        self.sdk = bosdyn.client.create_standard_sdk("spot_agent")
        self.robot = self.sdk.create_robot(hostname)

        # Credentials path relative to package configs/
        cred_path = Path(__file__).parent.parent / "configs" / "spot_credentials.json"
        authenticate_from_file(self.robot, cred_path)

        self.image_client = self.robot.ensure_client('image')
        self.robot_state_client = self.robot.ensure_client('robot-state')

    def _get_observation_sync(self):
        obs = {
            "battery": None,
            "imu": {"angular_velocity": None, "linear_velocity": None},
            "pose": {"position": None, "rotation": None},
            "cameras": {},
        }

        flatten = []
        for cams in self.cameras.values():
            flatten.extend(cams)

        image_fut = self.image_client.get_image_from_sources_async(flatten)
        state_fut = self.robot_state_client.get_robot_state_async()

        camera_captures = image_fut.result()
        state = state_fut.result()

        obs["battery"] = state.power_state.locomotion_charge_percentage.value

        twist = state.kinematic_state.velocity_of_body_in_vision
        obs["imu"]["angular_velocity"] = {"x": twist.angular.x, "y": twist.angular.y, "z": twist.angular.z}
        obs["imu"]["linear_velocity"] = {"x": twist.linear.x, "y": twist.linear.y, "z": twist.linear.z}

        vision_tform_body = get_a_tform_b(state.kinematic_state.transforms_snapshot, "vision", "body")
        if vision_tform_body is not None:
            obs["pose"]["position"] = {"x": vision_tform_body.x, "y": vision_tform_body.y, "z": vision_tform_body.z}
            obs["pose"]["rotation"] = {
                "w": vision_tform_body.rot.w,
                "x": vision_tform_body.rot.x,
                "y": vision_tform_body.rot.y,
                "z": vision_tform_body.rot.z,
            }

        for camera in camera_captures:
            src = camera.source.name
            location = src.split("_")[0]
            obs["cameras"].setdefault(location, {})
            if camera.shot.image.pixel_format == camera.shot.image.PIXEL_FORMAT_DEPTH_U16:
                obs["cameras"][location]["depth"] = decode_depth(camera)
            else:
                obs["cameras"][location]["color"] = decode_rgb(camera)

        return obs

    async def get_observation_async(self):
        return await asyncio.to_thread(self._get_observation_sync)


class SpotAgentGraph(SpotAgent):
    """SpotAgent extended with scene graph localization."""

    def __init__(self, cfg, graph_path='data/graph/graph_dataset/spot_aligned_graph.json', *args, **kwargs):
        super().__init__(*args, **kwargs)
        if cfg['vpr']['encoder'] == "DINOv2":
            self.encoder = ImageEncoder(DinoModel(cfg))
        else:
            raise NotImplementedError("Only DINOv2 is currently supported.")

        with open(cfg['vpr']['db_path'], 'rb') as fp:
            self.dataset = pkl.load(fp)

        self.fixed_g = load_scene_graph(graph_path)
        self.emb_images_db = self.encoder.embed(
            images=self.dataset["db_images"],
            patches=cfg['vpr']['patches'],
            agg_method=cfg['vpr']['agg_method'],
            num_clusters=cfg['vpr']['num_clusters'],
            num_domains=cfg['vpr']['num_domains'],
            seed=cfg['vpr']['seed'],
            save_path=cfg['vpr']['save_path'],
            save=cfg['vpr']['save'],
        )
        self.cfg = cfg

    def _get_images_from_obs(self, obs, fig_size=(640, 480)):
        images = []
        for cam in obs['cameras']:
            if 'color' in obs['cameras'][cam]:
                frame = obs['cameras'][cam]['color']
                resized_frame = cv2.resize(frame, fig_size, interpolation=cv2.INTER_AREA)
                images.append(resized_frame)
        return np.array(images)

    async def get_observation_async(self, top_k=5, window=3):
        base_obs = await asyncio.to_thread(self._get_observation_sync)

        X = self._get_images_from_obs(base_obs)

        if len(X) == 0:
            print("[WARNING] No valid color images received from Spot this step.")
            subgraph = {'nodes': [], 'edges': []}
        else:
            X_emb = self.encoder.embed(
                images=X,
                patches=self.cfg['vpr']['patches'],
                agg_method=self.cfg['vpr']['agg_method'],
                num_clusters=self.cfg['vpr']['num_clusters'],
                num_domains=self.cfg['vpr']['num_domains'],
                seed=self.cfg['vpr']['seed'],
                save_path=None,
                save=False,
            ).sum(0)

            _, sorted_ind = localize(X_emb, self.emb_images_db)
            subgraph = retrieve_subgraphs(self.dataset, sorted_ind, self.fixed_g, top_k=top_k, window=window)

        if isinstance(base_obs, dict):
            base_obs['graph'] = subgraph
            base_obs['g_text_desc'] = print_subgraph(subgraph)
            return base_obs
        else:
            return {"physical_obs": base_obs, "graph": subgraph}


class FastSpotGraphAgent(SpotAgent):
    """High-rate Spot agent with background threads for cameras, battery, and graph."""

    def __init__(self, cfg, graph_path='data/graph/graph_dataset/spot_aligned_graph.json', delays=(0.05, 1.0, 2.0), *args, **kwargs):
        super().__init__(*args, **kwargs)
        if cfg['vpr']['encoder'] == "DINOv2":
            self.encoder = ImageEncoder(DinoModel(cfg))
        else:
            raise NotImplementedError("Only DINOv2 is currently supported.")

        with open(cfg['vpr']['db_path'], 'rb') as fp:
            self.dataset = pkl.load(fp)

        self.fixed_g = load_scene_graph(graph_path)
        self.emb_images_db = self.encoder.embed(
            images=self.dataset["db_images"],
            patches=cfg['vpr']['patches'],
            agg_method=cfg['vpr']['agg_method'],
            num_clusters=cfg['vpr']['num_clusters'],
            num_domains=cfg['vpr']['num_domains'],
            seed=cfg['vpr']['seed'],
            save_path=cfg['vpr']['save_path'],
            save=cfg['vpr']['save'],
        )
        self.cfg = cfg

        try:
            self.local_grid_client = self.robot.ensure_client(LocalGridClient.default_service_name)
        except Exception as e:
            print(f"[WARNING] Could not initialize LocalGridClient: {e}")
            self.local_grid_client = None

        self.fast_lane = {'cameras': {}, 'pose': {}, 'imu': {}}
        self.med_lane = {'battery': None, 'local_grid': None}
        self.slow_lane = {'graph': {'nodes': [], 'edges': []}, 'g_text_desc': "Initializing graph..."}

        self._lock_fast = threading.Lock()
        self._lock_med = threading.Lock()
        self._lock_slow = threading.Lock()

        self.fast_delay, self.med_delay, self.slow_delay = delays

        threading.Thread(target=self._update_fast_lane, args=(self.fast_delay,), daemon=True).start()
        threading.Thread(target=self._update_medium_lane, args=(self.med_delay,), daemon=True).start()
        threading.Thread(target=self._update_slow_lane, args=(self.slow_delay,), daemon=True).start()

    def _get_images_from_obs(self, obs, fig_size=(640, 480)):
        images = []
        for cam in obs.get('cameras', {}):
            if 'color' in obs['cameras'][cam]:
                frame = obs['cameras'][cam]['color']
                resized_frame = cv2.resize(frame, fig_size, interpolation=cv2.INTER_AREA)
                images.append(resized_frame)
        return np.array(images)

    def _update_fast_lane(self, delay=0):
        flatten = []
        for cams in self.cameras.values():
            flatten.extend(cams)

        while True:
            try:
                camera_captures = self.image_client.get_image_from_sources(flatten)
                state = self.robot_state_client.get_robot_state()

                fast_data = {"cameras": {}, "pose": {}, "imu": {}}

                twist = state.kinematic_state.velocity_of_body_in_vision
                fast_data["imu"]["angular_velocity"] = {"x": twist.angular.x, "y": twist.angular.y, "z": twist.angular.z}
                fast_data["imu"]["linear_velocity"] = {"x": twist.linear.x, "y": twist.linear.y, "z": twist.linear.z}

                vision_tform_body = get_a_tform_b(state.kinematic_state.transforms_snapshot, "vision", "body")
                if vision_tform_body is not None:
                    fast_data["pose"]["position"] = {"x": vision_tform_body.x, "y": vision_tform_body.y, "z": vision_tform_body.z}
                    fast_data["pose"]["rotation"] = {
                        "w": vision_tform_body.rot.w, "x": vision_tform_body.rot.x,
                        "y": vision_tform_body.rot.y, "z": vision_tform_body.rot.z,
                    }

                for camera in camera_captures:
                    src = camera.source.name
                    location = src.split("_")[0]
                    fast_data["cameras"].setdefault(location, {})
                    if camera.shot.image.pixel_format == camera.shot.image.PIXEL_FORMAT_DEPTH_U16:
                        fast_data["cameras"][location]["depth"] = decode_depth(camera)
                    else:
                        fast_data["cameras"][location]["color"] = decode_rgb(camera)

                with self._lock_fast:
                    self.fast_lane = fast_data

            except Exception as e:
                print(f"[Fast Lane Error] {e}")

            time.sleep(delay)

    def _update_medium_lane(self, delay=10):
        while True:
            try:
                state = self.robot_state_client.get_robot_state()
                batt = state.power_state.locomotion_charge_percentage.value

                grid_data = None
                if self.local_grid_client:
                    grids = self.local_grid_client.get_local_grids(
                        local_grid_type_names=["terrain", "obstacle_distance"]
                    )
                    grid_data = grids

                with self._lock_med:
                    self.med_lane["battery"] = batt
                    self.med_lane["local_grid"] = grid_data

            except Exception as e:
                print(f"[Medium Lane Error] {e}")

            time.sleep(delay)

    def _update_slow_lane(self, delay, top_k=5, window=3):
        while True:
            try:
                with self._lock_fast:
                    cams_copy = self.fast_lane.get("cameras", {}).copy()

                X = self._get_images_from_obs({"cameras": cams_copy})

                if len(X) == 0:
                    subgraph = {'nodes': [], 'edges': []}
                    text_desc = "No valid color images available for graph localization."
                else:
                    X_emb = self.encoder.embed(
                        images=X,
                        patches=self.cfg['vpr']['patches'],
                        agg_method=self.cfg['vpr']['agg_method'],
                        num_clusters=self.cfg['vpr']['num_clusters'],
                        num_domains=self.cfg['vpr']['num_domains'],
                        seed=self.cfg['vpr']['seed'],
                        save_path=None, save=False,
                    ).sum(0)

                    _, sorted_ind = localize(X_emb, self.emb_images_db)
                    subgraph = retrieve_subgraphs(self.dataset, sorted_ind, self.fixed_g, top_k=top_k, window=window)
                    text_desc = print_subgraph(subgraph)

                with self._lock_slow:
                    self.slow_lane["graph"] = subgraph
                    self.slow_lane["g_text_desc"] = text_desc

            except Exception as e:
                print(f"[Slow Lane Error] {e}")

    def get_observation(self):
        """Return a synchronized snapshot across all data lanes."""
        with self._lock_fast, self._lock_med, self._lock_slow:
            obs = {
                "battery": self.med_lane.get("battery"),
                "local_grid": self.med_lane.get("local_grid"),
                "imu": self.fast_lane.get("imu"),
                "pose": self.fast_lane.get("pose"),
                "cameras": self.fast_lane.get("cameras"),
                "graph": self.slow_lane.get("graph"),
                "g_text_desc": self.slow_lane.get("g_text_desc"),
            }
        return obs
