# animation.py
from typing import Dict, Tuple, List, Optional, Set
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import colorsys

# -----------------------------------------------------------------------------
# 自動為 AGV 指派顏色（避開背景色），不夠用時以 HSV 補色
# -----------------------------------------------------------------------------
def _auto_agv_color_map(agv_ids: List[int], forbidden: Set[str]) -> Dict[int, str]:
    base = [
        "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#1f77b4",
        "#8c564b", "#e377c2", "#17becf", "#7f7f7f", "#bcbd22",
    ]
    forb = {str(c).lower() for c in forbidden}

    picked: List[str] = []
    for c in base:
        if c.lower() not in forb:
            picked.append(c)
        if len(picked) >= len(agv_ids):
            break

    i = 0
    while len(picked) < len(agv_ids):
        h = (i / max(1, len(agv_ids))) % 1.0
        s, v = 0.85, 0.95
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        hexc = "#%02x%02x%02x" % (int(r*255), int(g*255), int(b*255))
        if hexc.lower() not in forb and hexc.lower() not in (c.lower() for c in picked):
            picked.append(hexc)
        i += 1

    return {tid: picked[j] for j, tid in enumerate(agv_ids)}


# -----------------------------------------------------------------------------
# 1) schedule -> steps（加上 t_departure）
# -----------------------------------------------------------------------------
# def make_steps_from_schedule(
#     schedule_df: pd.DataFrame,
#     service_time_map: Optional[Dict[str, float]] = None,
# ) -> pd.DataFrame:
#     need = {"truck_id", "seq_index", "waypoint", "waypoint_type", "arrival_time"}
#     if not need.issubset(schedule_df.columns):
#         raise ValueError(f"schedule_df 欄位不足，缺：{need - set(schedule_df.columns)}")

#     if service_time_map is None:
#         service_time_map = {"Pickup": 0.0, "Delivery": 0.0, "Depot": 0.0, "w": 0.0}

#     df = schedule_df.copy()
#     df = df[df["arrival_time"].notna()]
#     df["truck_id"] = df["truck_id"].astype(int)
#     df["seq_index"] = df["seq_index"].astype(int)
#     df = df.sort_values(["truck_id", "seq_index"]).reset_index(drop=True)

#     def _svc(t: str) -> float:
#         return float(service_time_map.get(str(t), service_time_map.get("w", 0.0)))

#     df["t_arrival"] = df["arrival_time"].astype(float)
#     df["t_departure"] = df.apply(lambda r: r["t_arrival"] + _svc(str(r["waypoint_type"])), axis=1)

#     return df[["truck_id", "seq_index", "waypoint", "waypoint_type", "t_arrival", "t_departure"]]

# 1) schedule -> steps（加上 t_departure / t_arrival，支援 Break）
def make_steps_from_schedule(
    schedule_df: pd.DataFrame,
    service_time_map: Optional[Dict[str, float]] = None,
    arrival_is_departure: bool = True,   # ★ 你的 schedule_df["arrival_time"] 是「離站時間」
) -> pd.DataFrame:
    need = {"truck_id", "seq_index", "waypoint", "waypoint_type", "arrival_time"}
    if not need.issubset(schedule_df.columns):
        raise ValueError(f"schedule_df 欄位不足，缺：{need - set(schedule_df.columns)}")

    # 預設各事件的停留（可以在呼叫端覆寫，特別是 Break）
    base_map = {"Pickup": 0.0, "Delivery": 0.0, "Break": 0.0, "Depot": 0.0, "w": 0.0}
    if service_time_map:
        base_map.update({str(k): float(v) for k, v in service_time_map.items()})

    df = schedule_df.copy()
    df = df[df["arrival_time"].notna()].copy()
    df["truck_id"] = df["truck_id"].astype(int)
    df["seq_index"] = df["seq_index"].astype(int)
    df = df.sort_values(["truck_id", "seq_index"]).reset_index(drop=True)

    # per-row 停留時間（優先用 break_duration 欄位，其次用 map）
    df["dwell"] = df["waypoint_type"].map(base_map).astype(float)
    if "break_duration" in df.columns:
        m = (df["waypoint_type"].astype(str) == "Break") & df["break_duration"].notna()
        df.loc[m, "dwell"] = df.loc[m, "break_duration"].astype(float)

    # 你的 schedule_df["arrival_time"] 是「離站時間」，所以先回推出到站時間
    if arrival_is_departure:
        df["t_departure"] = df["arrival_time"].astype(float)
        df["t_arrival"]   = df["t_departure"] - df["dwell"].astype(float)
    else:
        df["t_arrival"]   = df["arrival_time"].astype(float)
        df["t_departure"] = df["t_arrival"] + df["dwell"].astype(float)

    return df[["truck_id", "seq_index", "waypoint", "waypoint_type", "t_arrival", "t_departure"]]


# -----------------------------------------------------------------------------
# 2) 背景圖層（有向邊 + 節點）；不出現在圖例
# -----------------------------------------------------------------------------
def make_background_traces(
    coords: Dict[int, Tuple[float, float]],
    graph: Dict[int, Dict[str, List[int]]],
    edge_color: str = "lightblue",
    node_color: str = "lightblue",
    node_size: int = 18,
) -> List[go.Scatter]:
    traces: List[go.Scatter] = []
    # edges
    for u, rec in graph.items():
        if int(u) not in coords:
            raise ValueError(f"coords 缺少節點 {u}")
        xu, yu = coords[int(u)]
        for v in rec.get("edges", []):
            if int(v) not in coords:
                raise ValueError(f"coords 缺少節點 {v}")
            xv, yv = coords[int(v)]
            traces.append(
                go.Scatter(
                    x=[xu, xv], y=[yu, yv],
                    mode="lines",
                    line=dict(width=2, color=edge_color),
                    showlegend=False, hoverinfo="skip",
                )
            )
    # nodes
    keys_sorted = sorted(coords.keys())
    traces.append(
        go.Scatter(
            x=[coords[k][0] for k in keys_sorted],
            y=[coords[k][1] for k in keys_sorted],
            mode="markers+text",
            marker_symbol="square",
            marker=dict(size=node_size, color=node_color),
            text=[str(k) for k in keys_sorted],
            textposition="middle center",
            showlegend=False, hoverinfo="skip",
        )
    )
    return traces


# -----------------------------------------------------------------------------
# 3) 時間網格
# -----------------------------------------------------------------------------
# def build_time_grid(steps_df: pd.DataFrame, dt: float) -> List[float]:
#     t_min = float(steps_df["t_arrival"].min())
#     t_max = float(max(steps_df["t_arrival"].max(), steps_df["t_departure"].max()))
#     return list(np.arange(t_min, t_max + 1e-9, float(dt)))

def build_time_grid_event_aware(steps_df: pd.DataFrame, dt: float, eps: float = 1e-9) -> List[float]:
    arr = steps_df["t_arrival"].astype(float).to_numpy()
    dep = steps_df["t_departure"].astype(float).to_numpy()

    t_min = float(np.nanmin(arr))
    t_max = float(np.nanmax(np.maximum(arr, dep)))

    # 基礎等距格
    grid = set(np.round(np.arange(t_min, t_max + eps, float(dt)), 6).tolist())

    # 一定包含所有事件時間（到站/離站）
    grid.update(np.round(arr, 6).tolist())
    grid.update(np.round(dep, 6).tolist())

    # 確保每段移動區間內至少有 1 個取樣點
    for _, g in steps_df.groupby("truck_id"):
        g = g.sort_values("seq_index").reset_index(drop=True)
        deps = g["t_departure"].astype(float).to_numpy()
        arrs = g["t_arrival"].astype(float).to_numpy()
        for i in range(len(g) - 1):
            a, b = deps[i], arrs[i + 1]  # 由 wp[i] 移動到下一個 wp[i+1] 的時間區間
            if b - a > eps:
                has_inner = any((t > a + eps) and (t < b - eps) for t in grid)
                if not has_inner:
                    grid.add(round((a + b) / 2.0, 6))  # 補一個中點，確保能看到「24→23」或「23→29」

    return sorted(grid)



# -----------------------------------------------------------------------------
# 4) 每台車的時間線（便於查詢）
# -----------------------------------------------------------------------------
def prepare_vehicle_timelines(steps_df: pd.DataFrame) -> Dict[int, Dict[str, List[float]]]:
    by_tid: Dict[int, Dict[str, List[float]]] = {}
    for tid, g in steps_df.groupby("truck_id"):
        gg = g.sort_values("seq_index").reset_index(drop=True)
        by_tid[int(tid)] = {
            "wp": gg["waypoint"].astype(int).tolist(),
            "arr": gg["t_arrival"].astype(float).tolist(),
            "dep": gg["t_departure"].astype(float).tolist(),
        }
    return by_tid


# -----------------------------------------------------------------------------
# 5) 線性插值：給定時刻，回傳車輛座標
# -----------------------------------------------------------------------------
def pos_at_time_for_truck(
    rec: Dict[str, List[float]],
    coords: Dict[int, Tuple[float, float]],
    t: float,
) -> Tuple[Optional[float], Optional[float], Optional[int], str]:
    wp = rec["wp"]; arr = rec["arr"]; dep = rec["dep"]
    n = len(wp)
    if n == 0:
        return (None, None, None, "none")

    if t <= arr[0]:
        x, y = coords[int(wp[0])]
        return (x, y, int(wp[0]), "node")

    for i in range(n):
        if arr[i] <= t <= dep[i]:
            x, y = coords[int(wp[i])]
            return (x, y, int(wp[i]), "node")
        if i < n - 1 and dep[i] < t < arr[i + 1]:
            (xu, yu) = coords[int(wp[i])]
            (xv, yv) = coords[int(wp[i + 1])]
            denom = arr[i + 1] - dep[i]
            r = 0.0 if denom <= 0 else (t - dep[i]) / denom
            x = xu + r * (xv - xu)
            y = yu + r * (yv - yu)
            return (x, y, None, "edge")

    x, y = coords[int(wp[-1])]
    return (x, y, int(wp[-1]), "node")


# -----------------------------------------------------------------------------
# 6) 建 legend traces（只在圖例顯示，不畫在畫布）與 animated placeholders
# -----------------------------------------------------------------------------
def build_traces_for_cars(
    agv_ids: List[int],
    agv_color_map: Dict[int, str],
    marker_size: int = 14,
) -> Tuple[List[go.Scatter], List[go.Scatter]]:
    """
    回傳 (legend_traces, animated_placeholders)
    - legend_traces：只顯示在圖例（visible="legendonly"），不畫在畫布
    - animated_placeholders：動畫專用空 trace（frames 會覆蓋）
    """
    legend_traces: List[go.Scatter] = []
    anim_placeholders: List[go.Scatter] = []
    for tid in agv_ids:
        name = f"AGV{int(tid)}"
        color = agv_color_map[int(tid)]
        # 圖例專用（不畫在畫布，只顯示於 legend）
        legend_traces.append(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker_symbol="circle",
            marker=dict(size=marker_size, color=color, line=dict(width=2, color="#ffffff")),
            name=name, showlegend=True,
            visible="legendonly",  # ★ 只在圖例顯示，避免左上角幽靈點
        ))
        # 動畫專用（會被 frames 更新）
        anim_placeholders.append(go.Scatter(
            x=[None], y=[None],
            mode="markers+text",
            marker_symbol="circle",
            marker=dict(size=marker_size, color=color, line=dict(width=2, color="#ffffff")),
            text=[name], textposition="top center",
            name=name, showlegend=False,
        ))
    return legend_traces, anim_placeholders


# -----------------------------------------------------------------------------
# 7) 建 frames（只更新 animated traces）
# -----------------------------------------------------------------------------
def build_frames(
    agv_ids: List[int],
    time_steps: List[float],
    timelines: Dict[int, Dict[str, List[float]]],
    coords: Dict[int, Tuple[float, float]],
    agv_color_map: Dict[int, str],
    marker_size: int = 14,
) -> List[go.Frame]:
    frames: List[go.Frame] = []
    for t in time_steps:
        fr_traces = []
        for tid in agv_ids:
            rec = timelines.get(int(tid), {"wp": [], "arr": [], "dep": []})
            x, y, wp_here, where = pos_at_time_for_truck(rec, coords, float(t))
            label = f"AGV{int(tid)}"
            if x is None:
                tr = go.Scatter(
                    x=[None], y=[None], mode="markers+text",
                    marker_symbol="circle",
                    marker=dict(size=marker_size, color=agv_color_map[int(tid)],
                                line=dict(width=2, color="#ffffff")),
                    text=[label], textposition="top center",
                    name=label, showlegend=False
                )
            else:
                hover = f"{label}<br>t={t:.2f}" + (f"<br>wp={wp_here}<br>at node" if wp_here is not None else "<br>on edge")
                tr = go.Scatter(
                    x=[x], y=[y], mode="markers+text",
                    marker_symbol="circle",
                    marker=dict(size=marker_size, color=agv_color_map[int(tid)],
                                line=dict(width=2, color="#ffffff")),
                    text=[label], textposition="top center",
                    name=label, showlegend=False,
                    hovertext=[hover], hoverinfo="text",
                )
            fr_traces.append(tr)
        frames.append(go.Frame(data=fr_traces, name=f"{t:.3f}", traces=list(range(len(agv_ids)))))
    return frames


# -----------------------------------------------------------------------------
# 8) 高層封裝（圖例穩定不消失，且避免左上角閃爍）
# -----------------------------------------------------------------------------
# def build_animation_figure(
#     *,
#     schedule_df: pd.DataFrame,                 # wp_times / wp_times_noconf
#     graph: Dict[int, Dict[str, List[int]]],   # GRAPH
#     coords: Dict[int, Tuple[float, float]],   # 手動座標（必填）
#     service_time_map: Optional[Dict[str, float]] = None,
#     dt: float = 0.5,
#     frame_ms: int = 80,
#     slider_stride: int = 1,
#     title: str = "AGV 動態路徑動畫（等距時間軸 + 線性插值）",
#     edge_color: str = "lightblue",
#     node_color: str = "lightblue",
#     node_size: int = 18,
#     agv_color_map: Optional[Dict[int, str]] = None,  # 不提供則自動配色
#     marker_size: int = 14,
#     width: int = 900, height: int = 700,
#     legend_x: float = 1.02, legend_y: float = 1.0,   # 圖例位置（右側）
# ) -> go.Figure:

#     # 1) steps
#     steps_df = make_steps_from_schedule(schedule_df, service_time_map)

def build_animation_figure(
    *,
    schedule_df: pd.DataFrame,
    graph: Dict[int, Dict[str, List[int]]],
    coords: Dict[int, Tuple[float, float]],
    service_time_map: Optional[Dict[str, float]] = None,
    dt: float = 0.5,
    frame_ms: int = 80,
    slider_stride: int = 1,
    title: str = "AGV 動態路徑動畫（等距時間軸 + 線性插值）",
    edge_color: str = "lightblue",
    node_color: str = "lightblue",
    node_size: int = 18,
    agv_color_map: Optional[Dict[int, str]] = None,
    marker_size: int = 14,
    width: int = 900, height: int = 700,
    legend_x: float = 1.02, legend_y: float = 1.0,
) -> go.Figure:

    # 1) steps（★ Break 停留要能看得出來）
    steps_df = make_steps_from_schedule(
        schedule_df,
        service_time_map=service_time_map,
        arrival_is_departure=True,   # 你的 schedule_df["arrival_time"] 是離站時間
    )

    # 2) 驗證 coords 覆蓋
    graph_nodes = set(int(k) for k in graph.keys())
    coords_nodes = set(int(k) for k in coords.keys())
    if not graph_nodes.issubset(coords_nodes):
        missing = sorted(graph_nodes - coords_nodes)
        raise ValueError(f"coords 缺少以下 graph 節點：{missing}")

    used_nodes = set(int(x) for x in steps_df["waypoint"].unique())
    if not used_nodes.issubset(coords_nodes):
        missing = sorted(used_nodes - coords_nodes)
        raise ValueError(f"coords 缺少以下 schedule 會用到的節點：{missing}")

    # 3) 背景
    bg_traces = make_background_traces(coords, graph, edge_color=edge_color, node_color=node_color, node_size=node_size)

    # 4) 時間網格 / 時間線 / 車輛清單
    # time_steps = build_time_grid(steps_df, dt)
    time_steps = build_time_grid_event_aware(steps_df, dt)
    timelines = prepare_vehicle_timelines(steps_df)
    agv_ids = sorted(int(x) for x in steps_df["truck_id"].unique())

    # 5) 顏色（自動配色，避開背景）
    if agv_color_map is None:
        forbidden = {str(node_color), str(edge_color)}
        agv_color_map = _auto_agv_color_map(agv_ids, forbidden)

    # 6) legend(靜態僅供圖例) + animated placeholders
    legend_traces, anim_placeholders = build_traces_for_cars(agv_ids, agv_color_map, marker_size)

    # 7) 建 frames（只更新 animated traces）
    frames = build_frames(agv_ids, time_steps, timelines, coords, agv_color_map, marker_size)

    # 8) 初始圖層 = 背景 + legend(靜態) + animated placeholders
    init_traces = list(bg_traces) + list(legend_traces) + list(anim_placeholders)
    fig = go.Figure(data=init_traces, frames=frames)

    # 9) frames 只應更新「animated placeholders」那一段 → 索引偏移 = 背景數 + legend數
    offset = len(bg_traces) + len(legend_traces)
    for fr in fig.frames:
        fr.traces = [offset + i for i in fr.traces]

    # 10) ★用第一幀的位置覆蓋 animated placeholders，避免 None→有值 的閃爍（左上角幽靈點）
    if len(frames) > 0 and len(frames[0].data) == len(anim_placeholders):
        for i, tr in enumerate(frames[0].data):
            fig.data[offset + i].update(x=tr.x, y=tr.y, text=tr.text, hovertext=getattr(tr, "hovertext", None))

    # 11) slider（可稀疏）
    slider_steps = []
    for i, t in enumerate(time_steps):
        if (i % max(1, int(slider_stride))) != 0 and i != len(time_steps) - 1:
            continue
        slider_steps.append(dict(
            method="animate",
            args=[[f"{t:.3f}"], {"frame": {"duration": frame_ms, "redraw": True}, "mode": "immediate", "fromcurrent": True}],
            label=f"{t:.2f}",
        ))

    # 12) 畫布範圍 + 圖例位置
    xs = [xy[0] for xy in coords.values()]
    ys = [xy[1] for xy in coords.values()]
    xpad = (max(xs) - min(xs)) * 0.05 if len(xs) > 1 else 1.0
    ypad = (max(ys) - min(ys)) * 0.05 if len(ys) > 1 else 1.0

    fig.update_layout(
        title=title,
        xaxis=dict(range=[min(xs)-xpad, max(xs)+xpad], constrain="domain"),
        yaxis=dict(range=[min(ys)-ypad, max(ys)+ypad], scaleanchor="x"),
        width=width, height=height,
        legend=dict(
            x=legend_x, y=legend_y,
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="rgba(0,0,0,0.15)", borderwidth=1
        ),
        updatemenus=[dict(
            type="buttons", showactive=False, y=1.05,
            buttons=[
                dict(label="播放", method="animate",
                     args=[None, {"frame": {"duration": frame_ms, "redraw": True},
                                  "mode": "immediate", "fromcurrent": True}]),
                dict(label="暫停", method="animate", args=[[None], {"mode": "immediate"}]),
            ]
        )],
        sliders=[dict(
            steps=slider_steps,
            transition={"duration": 0}, x=0.1, y=0,
            currentvalue={"prefix": "Time: "}
        )]
    )
    return fig


