"""M2 tests — extended /control API, SimConfig, WebSocket, presets."""
import json
import socket
import struct
import threading
import time
import urllib.request

import pytest

from engine.config import SimConfig
from engine.network import Edge, Network, Node
from engine.network_simulation import NetworkSimulation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _two_node_net(length: float = 800.0, speed: float = 14.0) -> Network:
    net = Network()
    net.add_node(Node(id="A", x=0.0,      y=0.0))
    net.add_node(Node(id="B", x=length,   y=0.0))
    net.add_edge(Edge(id="AB", from_node="A", to_node="B",
                      num_lanes=1, speed_limit=speed,
                      geometry=[[0.0, 0.0], [length, 0.0]]))
    return net


# ---------------------------------------------------------------------------
# 1. SimConfig unit tests
# ---------------------------------------------------------------------------

class TestSimConfig:

    def test_defaults(self):
        cfg = SimConfig()
        assert cfg.idm_a_max    == pytest.approx(1.4)
        assert cfg.idm_T        == pytest.approx(1.5)
        assert cfg.demand_scale == pytest.approx(1.0)
        assert cfg.speed_mult   == pytest.approx(1.0)

    def test_update_valid_keys(self):
        cfg = SimConfig()
        cfg.update(idm_a_max=3.0, idm_T=0.8)
        assert cfg.idm_a_max == pytest.approx(3.0)
        assert cfg.idm_T     == pytest.approx(0.8)

    def test_ignores_unknown_keys(self):
        cfg = SimConfig()
        cfg.update(nonexistent_param=999)   # must not raise
        assert not hasattr(cfg, "nonexistent_param")

    def test_type_coercion_string_to_float(self):
        cfg = SimConfig()
        cfg.update(idm_a_max="2.5")         # string → float
        assert isinstance(cfg.idm_a_max, float)
        assert cfg.idm_a_max == pytest.approx(2.5)

    def test_thread_safety(self):
        cfg = SimConfig()
        errors = []

        def writer():
            for _ in range(1000):
                try:
                    cfg.update(idm_a_max=1.4)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Thread-safety errors: {errors}"

    def test_snapshot_returns_dict(self):
        cfg = SimConfig()
        snap = cfg.snapshot()
        assert isinstance(snap, dict)
        assert "idm_a_max" in snap
        assert "speed_mult" in snap

    def test_update_weather_multipliers(self):
        cfg = SimConfig()
        cfg.update(weather_v0_mult=0.9, weather_s0_mult=1.2, weather_T_mult=1.15)
        assert cfg.weather_v0_mult == pytest.approx(0.9)
        assert cfg.weather_s0_mult == pytest.approx(1.2)
        assert cfg.weather_T_mult  == pytest.approx(1.15)


# ---------------------------------------------------------------------------
# 2. Config applied inside simulation
# ---------------------------------------------------------------------------

class TestConfigInSim:

    def test_config_a_max_propagates_to_spawned_vehicles(self):
        """SimConfig.idm_a_max must be applied to every spawned vehicle.

        Vehicles spawn at free-flow speed so avg-speed comparisons are noisy
        (no acceleration delta visible until vehicles recover from a stop).
        Instead we directly verify veh.a_max matches the configured value.
        """
        CONFIGURED_A = 3.5
        cfg = SimConfig(idm_a_max=CONFIGURED_A)
        sim = NetworkSimulation(
            _two_node_net(length=2000.0),
            demand={"A": {"B": 3600.0}},
            duration=120, seed=0, config=cfg,
        )
        for _ in range(20):   # spawn a few vehicles
            sim.step()
        assert sim.vehicles, "Expected vehicles to have spawned"
        for v in sim.vehicles:
            assert v.a_max == pytest.approx(CONFIGURED_A), \
                f"Vehicle a_max should be {CONFIGURED_A}, got {v.a_max}"

    def test_config_weather_slows_vehicles(self):
        """weather_v0_mult < 1 should reduce average speed."""
        def _avg_speed(v0_mult: float) -> float:
            cfg = SimConfig(weather_v0_mult=v0_mult)
            sim = NetworkSimulation(
                _two_node_net(), demand={"A": {"B": 120.0}},
                duration=60, seed=0, config=cfg,
            )
            for _ in range(300):
                sim.step()
            speeds = [v.speed for v in sim.vehicles] if sim.vehicles else [0.0]
            return sum(speeds) / len(speeds)

        assert _avg_speed(0.7) < _avg_speed(1.0), \
            "weather_v0_mult=0.7 should slow vehicles vs 1.0"

    def test_config_passed_to_sim(self):
        """SimConfig object attached to sim as .config attribute."""
        cfg = SimConfig()
        sim = NetworkSimulation(
            _two_node_net(), demand={}, duration=10, config=cfg,
        )
        assert sim.config is cfg


# ---------------------------------------------------------------------------
# 3. Preset key validation (no server needed)
# ---------------------------------------------------------------------------

class TestPresets:

    PRESETS = {
        "Rush Hour":  {"idm_T": 1.0, "idm_s0": 1.5,
                       "demand_scale": 2.0, "signal_green_ratio": 0.6},
        "Aggressive": {"idm_a_max": 3.0, "idm_T": 0.8, "mobil_politeness": 0.0},
        "Rainy Day":  {"weather_v0_mult": 0.9, "weather_s0_mult": 1.2,
                       "weather_T_mult": 1.15},
        "Night":      {"demand_scale": 0.15, "idm_T": 1.8},
    }

    def test_all_preset_keys_exist_in_simconfig(self):
        """Every key used by a preset must be a valid SimConfig field."""
        cfg = SimConfig()
        for preset_name, params in self.PRESETS.items():
            for key in params:
                assert hasattr(cfg, key), \
                    f"Preset '{preset_name}' references unknown key '{key}'"

    def test_presets_apply_without_error(self):
        """Applying each preset via SimConfig.update() must not raise."""
        for preset_name, params in self.PRESETS.items():
            cfg = SimConfig()
            cfg.update(**params)   # should not raise


# ---------------------------------------------------------------------------
# 4. Live /control API integration tests (embedded server fixture)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def api_server():
    """Spin up the run.py HTTP server in-process for integration tests."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from run import _serve, _build_network_sim

    scenario = {
        "nodes": [
            {"id": "A", "x": 0.0,    "y": 0.0},
            {"id": "B", "x": 1000.0, "y": 0.0},
        ],
        "edges": [
            {"id": "AB", "from_node": "A", "to_node": "B",
             "num_lanes": 1, "speed_limit": 14.0},
        ],
        "demand":   {"A": {"B": 120.0}},
        "duration": 60.0,
        "seed":     0,
    }

    port   = 19999
    thread = threading.Thread(
        target=_serve, args=(scenario, port, "TestCity"),
        daemon=True,
    )
    thread.start()
    # Wait for server to bind
    for _ in range(20):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=1)
            break
        except Exception:
            time.sleep(0.3)
    yield f"http://127.0.0.1:{port}"


def _post(base, path, body):
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        f"{base}{path}", data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def _get(base, path):
    with urllib.request.urlopen(f"{base}{path}", timeout=5) as r:
        return json.loads(r.read())


class TestControlAPI:

    def test_state_endpoint_returns_vehicles_key(self, api_server):
        state = _get(api_server, "/state")
        assert "vehicles" in state
        assert "time" in state

    def test_control_pause(self, api_server):
        resp = _post(api_server, "/control", {"paused": True})
        assert resp.get("paused") is True
        # restore
        _post(api_server, "/control", {"paused": False})

    def test_control_speed_mult_top_level(self, api_server):
        resp = _post(api_server, "/control", {"speed_mult": 8.0})
        assert resp.get("speed_mult") == pytest.approx(8.0)
        _post(api_server, "/control", {"speed_mult": 1.0})

    def test_control_params_idm(self, api_server):
        """params dict forwarded to SimConfig without crashing."""
        resp = _post(api_server, "/control", {"params": {"idm_a_max": 2.5}})
        # response is the ctrl dict — just check it came back valid
        assert isinstance(resp, dict)

    def test_control_params_weather(self, api_server):
        resp = _post(api_server, "/control",
                     {"params": {"weather_v0_mult": 0.85, "weather_s0_mult": 1.2}})
        assert isinstance(resp, dict)

    def test_control_params_unknown_keys_ignored(self, api_server):
        """Unknown keys in params dict must not crash the server."""
        resp = _post(api_server, "/control",
                     {"params": {"not_a_real_param": 99}})
        assert isinstance(resp, dict)

    def test_control_params_speed_mult(self, api_server):
        resp = _post(api_server, "/control", {"params": {"speed_mult": 32}})
        assert resp.get("speed_mult") == pytest.approx(32.0)
        _post(api_server, "/control", {"speed_mult": 1.0})


# ---------------------------------------------------------------------------
# 5. WebSocket integration test
# ---------------------------------------------------------------------------

class TestWebSocket:

    def _ws_connect(self, base: str):
        """Open a raw WebSocket connection to /ws and return the socket."""
        import base64 as _b64
        host = base.replace("http://", "").split(":")[0]
        port = int(base.split(":")[-1])
        sock = socket.create_connection((host, port), timeout=5)
        key  = _b64.b64encode(b"testkey1234567890").decode()
        req  = (
            f"GET /ws HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n"
        )
        sock.sendall(req.encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            resp += sock.recv(256)
        return sock, resp.decode()

    def _read_ws_frame(self, sock) -> dict:
        """Read one WebSocket text frame and return parsed JSON."""
        hdr = b""
        while len(hdr) < 2:
            hdr += sock.recv(2 - len(hdr))
        plen = hdr[1] & 0x7F
        if plen == 126:
            ext = b""
            while len(ext) < 2: ext += sock.recv(2 - len(ext))
            plen = struct.unpack(">H", ext)[0]
        elif plen == 127:
            ext = b""
            while len(ext) < 8: ext += sock.recv(8 - len(ext))
            plen = struct.unpack(">Q", ext)[0]
        data = b""
        while len(data) < plen:
            chunk = sock.recv(min(4096, plen - len(data)))
            if not chunk:
                break
            data += chunk
        return json.loads(data)

    def test_ws_handshake_101(self, api_server):
        sock, resp = self._ws_connect(api_server)
        assert "101" in resp
        assert "websocket" in resp.lower()
        sock.close()

    def test_ws_receives_state_frame(self, api_server):
        sock, _ = self._ws_connect(api_server)
        sock.settimeout(3.0)
        try:
            msg = self._read_ws_frame(sock)
            assert "vehicles" in msg
            assert "time" in msg
            assert "signals" in msg
        finally:
            sock.close()

    def test_ws_multiple_frames(self, api_server):
        """Server should push at least 3 frames within 1 second."""
        sock, _ = self._ws_connect(api_server)
        sock.settimeout(2.0)
        frames = []
        try:
            while len(frames) < 3:
                frames.append(self._read_ws_frame(sock))
        finally:
            sock.close()
        assert len(frames) >= 3
        # Time should advance between frames
        times = [f["time"] for f in frames]
        assert times[-1] > times[0], "Simulation time should advance between frames"

    def test_ws_reconnect(self, api_server):
        """Opening a second connection while one is open should work."""
        sock1, _ = self._ws_connect(api_server)
        sock2, _ = self._ws_connect(api_server)
        sock1.settimeout(3.0)
        sock2.settimeout(3.0)
        try:
            msg1 = self._read_ws_frame(sock1)
            msg2 = self._read_ws_frame(sock2)
            assert "time" in msg1
            assert "time" in msg2
        finally:
            sock1.close()
            sock2.close()
