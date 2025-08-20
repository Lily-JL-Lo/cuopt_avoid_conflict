# solve.py
from typing import Optional, Dict, List, Tuple
import numpy as np
import cudf
from cuopt import routing, distance_engine

# =========================
# Graph / CSR / WaypointMatrix
# =========================
def graph_to_csr_arrays(graph: Dict[int, Dict[str, List[int]]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    將 {u: {"edges":[...], "weights":[...]}} 轉成 CSR 三元組 (offsets, edges, weights)
    - 允許鍵不連續；缺點當作無邊
    - 嚴格檢查 edges/weights 長度一致
    """
    if not graph:
        raise ValueError("graph 為空")

    V = max(int(k) for k in graph.keys()) + 1
    offsets, edges, weights = [0], [], []
    cur = 0
    for u in range(V):
        rec = graph.get(u)
        e = rec.get("edges", []) if rec else []
        w = rec.get("weights", []) if rec else []
        if len(e) != len(w):
            raise ValueError(f"節點 {u} 的 edges({len(e)}) 與 weights({len(w)}) 長度不一致")
        edges.extend(int(v) for v in e)
        weights.extend(float(x) for x in w)
        cur += len(e)
        offsets.append(cur)

    return (
        np.asarray(offsets, dtype=np.int32),
        np.asarray(edges,   dtype=np.int32),
        np.asarray(weights, dtype=np.float32),
    )

def build_waypoint_graph(graph_dict: Dict[int, Dict[str, List[int]]]) -> distance_engine.WaypointMatrix:
    """由 CSR 建立 WaypointMatrix（官方 API 入口）。"""
    offsets, edges, weights = graph_to_csr_arrays(graph_dict)
    return distance_engine.WaypointMatrix(offsets, edges, weights)




# =========================
# （你的規則）就近平均分配 deliveries
# =========================
def choose_deliveries_near_first(
    *,
    stations: List[int],             # 原始 waypoint id（例：[10,12,...]）
    num_items: int,                  # 貨物數
    pickup_node: int,                # 取貨點（原始 id）
    target_locations: np.ndarray,    # 用來決定矩陣索引
    cost_matrix: cudf.DataFrame = None,
    waypoint_matrix: Optional[distance_engine.WaypointMatrix] = None,
) -> Tuple[List[int], Dict[int, float]]:
    """
    規則：
      1) 計算 pickup_node -> station 的成本（用 cost_matrix）
      2) 依成本由近到遠排序
      3) 平均分配；餘數分配給最近的前 r 個站
    回傳：
      deliveries: list[int]（每筆訂單的 delivery_location，原始 id）
      eta_map: {station原始id: eta}
    """
    if cost_matrix is None:
        if waypoint_matrix is None:
            raise ValueError("未提供 cost_matrix，也未提供 waypoint_matrix 以便計算")
        cost_matrix = waypoint_matrix.compute_cost_matrix(target_locations)

    # 確保 target_locations 含 pickup 與所有 stations
    id2idx = {int(v): i for i, v in enumerate(target_locations)}
    if int(pickup_node) not in id2idx:
        raise ValueError(f"pickup_node {pickup_node} 不在 target_locations")
    missing = [int(s) for s in stations if int(s) not in id2idx]
    if missing:
        raise ValueError(f"以下工作站不在 target_locations：{missing}")

    pick_idx = id2idx[int(pickup_node)]
    # cuDF 讀值用 .iloc
    eta_map = {int(s): float(cost_matrix.iloc[pick_idx, id2idx[int(s)]]) for s in stations}

    stations_sorted = sorted(stations, key=lambda s: eta_map[int(s)])
    S = len(stations_sorted)
    if S == 0:
        return [], {}

    k, r = divmod(int(num_items), S)
    deliveries: List[int] = []
    # 先平均
    for s in stations_sorted:
        deliveries.extend([int(s)] * k)
    # 餘數給最近的前 r 個
    deliveries.extend([int(s) for s in stations_sorted[:r]])

    assert len(deliveries) == int(num_items)
    return deliveries, eta_map

# =========================
# DataModel（完全照官方：id→index 在 set_order_locations 前完成）
# =========================
def setup_data_model(
    *,
    target_locations: np.ndarray,         # 原始 waypoint id 子集合，決定 NxN 順序
    transport_df: cudf.DataFrame,         # 欄位使用原始 id：pickup_location / delivery_location / ...
    robot_df: cudf.DataFrame,             # index=robot_ids, 欄位 carrying_capacity
    cost_matrix: cudf.DataFrame,          # NxN
    transit_matrix: Optional[cudf.DataFrame] = None,  # 若 None 表示時間=成本
    factory_open: int,
    factory_close: int,
    must_return_to_depot: bool = True,
    min_vehicles: Optional[int] = None,
) -> routing.DataModel:
    N = int(len(target_locations))
    n_veh = int(len(robot_df))
    n_ord = int(len(transport_df) * 2)

    # 原始 id -> 索引
    target_map = {int(v): i for i, v in enumerate(target_locations)}
    if len(target_map) != N:
        raise ValueError("target_locations 含重複值")

    dm = routing.DataModel(N, n_veh, n_ord)
    dm.add_cost_matrix(cost_matrix)
    if transit_matrix is not None:
        dm.add_transit_time_matrix(transit_matrix)

    # 訂單位置：原始 id → 索引
    pu_ids = transport_df["pickup_location"].to_arrow().to_pylist()
    dl_ids = transport_df["delivery_location"].to_arrow().to_pylist()
    try:
        pu_idx = [target_map[int(x)] for x in pu_ids]
        dl_idx = [target_map[int(x)] for x in dl_ids]
    except KeyError as e:
        raise ValueError(f"有訂單位置不在 target_locations：{e}")

    order_locations = cudf.Series(pu_idx + dl_idx, dtype="int32")
    _assert_is_index_range(order_locations, N, "order_locations")
    dm.set_order_locations(order_locations)

    # 取送配對
    npairs = len(transport_df)
    dm.set_pickup_delivery_pairs(
        cudf.Series(list(range(npairs)), dtype="int32"),
        cudf.Series([i + npairs for i in range(npairs)], dtype="int32"),
    )

    # === 容量維度（搬到這裡，且顯式對齊 veh_cap） ===
    raw = transport_df["order_demand"].astype("int32")
    order_demand_vec = cudf.concat([raw, -raw], ignore_index=True).astype("int32")
    veh_cap = (
        robot_df.sort_index()["carrying_capacity"]
                .astype("int32")
                .reset_index(drop=True)
    )
    dm.add_capacity_dimension("demand", order_demand_vec, veh_cap)

    # 時窗與服務時間
    order_tw_earliest = cudf.concat(
        [transport_df["earliest_pickup"], transport_df["earliest_delivery"]],
        ignore_index=True,
    ).astype("int32")
    order_tw_latest = cudf.concat(
        [transport_df["latest_pickup"], transport_df["latest_delivery"]],
        ignore_index=True,
    ).astype("int32")
    order_service = cudf.concat(
        [transport_df["pickup_service_time"], transport_df["delivery_service_time"]],
        ignore_index=True,
    ).astype("int32")
    dm.set_order_time_windows(order_tw_earliest, order_tw_latest)
    dm.set_order_service_times(order_service)

    # 車輛時窗 / 回程

    veh_df = robot_df.sort_index().reset_index(drop=True)

    veh_earliest = veh_df["earliest_start"].astype("int32")
    veh_latest   = veh_df["latest_end"].astype("int32")

    dm.set_vehicle_time_windows(veh_earliest, veh_latest)
    
    # dm.set_vehicle_time_windows(
    #     cudf.Series([factory_open]  * n_veh, dtype="int32"),
    #     cudf.Series([factory_close] * n_veh, dtype="int32"),
    # )
    # dm.set_drop_return_trips(cudf.Series([not must_return_to_depot] * n_veh))

    if min_vehicles is not None:
        dm.set_min_vehicles(int(min_vehicles))

    return dm

def solve_routing(dm: routing.DataModel, *, time_limit: int = 5) -> routing.Assignment:
    s = routing.SolverSettings()
    s.set_time_limit(int(time_limit))
    return routing.Solve(dm, s)

# =========================
# 展開前守門（不修改資料，只檢查）
# =========================
def assert_route_indexed(route_df: cudf.DataFrame, *, n_locations: int) -> None:
    if "location" not in route_df.columns:
        raise ValueError("route_df 缺少 'location' 欄位")
    _assert_is_index_range(route_df["location"], n_locations, "route['location']")

def _assert_is_index_range(col: cudf.Series, N: int, name: str) -> None:
    mn = int(col.min()); mx = int(col.max())
    if not (0 <= mn and mx < N):
        raise AssertionError(f"{name} 應為 0..{N-1} 的索引，實得範圍 {mn}..{mx}")
    


def apply_vehicle_breaks(dm, target_locations, breaks):
    """
    dm: routing.DataModel
    target_locations: np.ndarray（原始 waypoint id 序列，長度 = N）
    breaks: List[Dict]，每一筆格式：
      {
        "vehicle_id": 3,
        "earliest": 80,   # 最早開始
        "latest": 100,    # 最晚必須開始
        "duration": 10,   # 休息/充電持續時間
        # 可選，限定可休息之地點（用『原始 waypoint id』寫）：
        # "allowed_waypoints": [0, 10, 14]
      }
    """
    import cudf

    # 原始 waypoint id -> 矩陣索引（0..N-1）
    id2idx = {int(v): i for i, v in enumerate(target_locations)}

    for b in breaks:
        vid  = int(b["vehicle_id"])
        ear  = int(b["earliest"])
        lat  = int(b["latest"])
        dur  = int(b["duration"])
        locs = b.get("allowed_waypoints", None)

        if locs is not None and len(locs) > 0:
            try:
                loc_idx = [id2idx[int(x)] for x in locs]
            except KeyError as e:
                raise ValueError(f"break 的 allowed_waypoints 含不在 target_locations 的節點: {e}")
            loc_series = cudf.Series(loc_idx, dtype="int32")
            dm.add_vehicle_break(
                vehicle_id=vid,
                earliest=ear,
                latest=lat,
                duration=dur,
                locations=loc_series,   # 限定可休息地點（索引空間）
            )
        else:
            # 不限地點
            dm.add_vehicle_break(
                vehicle_id=vid,
                earliest=ear,
                latest=lat,
                duration=dur,
            )


# =========================
# 官方不足：Waypoint 到達時間（選用）
# =========================


# def compute_waypoint_arrival_times(*, assignment, waypoint_matrix, target_locations, offsets, edges, weights,
#                                    pickup_service_time=0, delivery_service_time=0):
#     """改良版：正確累加 Pickup 和 Delivery 的服務時間到 Arrival Time"""
#     import pandas as pd
    
#     N = len(target_locations)
#     rt_cu = assignment.get_route()
#     assert_route_indexed(rt_cu, n_locations=N)  # 確保路徑合法性
    
#     # 計算 Waypoint Sequence
#     wseq = waypoint_matrix.compute_waypoint_sequence(target_locations, rt_cu)
#     rt = rt_cu.to_pandas().sort_values(["truck_id", "route"]).reset_index(drop=True)
#     ws = wseq.to_pandas() if hasattr(wseq, "to_pandas") else pd.DataFrame(wseq)
    
#     if "sequence_offset" not in rt.columns:
#         raise RuntimeError("route 缺少 sequence_offset；請先呼叫 compute_waypoint_sequence")
    
#     # 構建 CSR Adjacency 結構
#     V = int(len(offsets) - 1)
#     adj = [{} for _ in range(V)]
#     for u in range(V):
#         a, b = int(offsets[u]), int(offsets[u + 1])
#         for k in range(a, b):
#             v = int(edges[k])
#             adj[u][v] = float(weights[k])

#     # 用於儲存計算結果
#     rows = []
#     for tid, g in rt.groupby("truck_id"):
#         g = g.sort_values("route").reset_index(drop=True)
#         a = int(g["sequence_offset"].min())
#         b = int(g["sequence_offset"].max())
        
#         # 取得更新的 waypoints 與類型
#         seq_wp = ws.loc[a:b, "waypoint_sequence"].astype(int).tolist()  # Waypoint IDs
#         seq_typ = ["w"] * len(seq_wp)  # 初始類型
        
#         for _, ev in g.iterrows():
#             k = int(ev["sequence_offset"]) - a
#             if 0 <= k < len(seq_typ):
#                 seq_typ[k] = str(ev["type"])  # 設定正確的 waypoint 類型
        
#         # 初始化到達時間（從基礎到達時間開始）
#         arr = [0.0] * len(seq_wp)
#         base = float(g["arrival_stamp"].iloc[0]) if len(g) else 0.0
#         arr[0] = base
        
#         # 計算每個 Waypoint 的到達時間
#         for i in range(1, len(seq_wp)):
#             u, v = int(seq_wp[i - 1]), int(seq_wp[i])
#             w = adj[u].get(v) if u != v else 0.0  # 當 u == v 時，路徑權值為 0
#             if w is None:
#                 raise ValueError(f"CSR 缺邊 {u}->{v}")
            
#             # 根據 type 決定 Service Time
#             service_time = 0.0
#             if seq_typ[i] == "Pickup":
#                 service_time = pickup_service_time
#             elif seq_typ[i] == "Delivery":
#                 service_time = delivery_service_time
            
#             # 計算到達時間
#             arr[i] = arr[i - 1] + float(w) + float(service_time)
        
#         # 收集計算結果
#         for i in range(len(seq_wp)):
#             rows.append({
#                 "truck_id": int(tid),
#                 "seq_index": i,
#                 "waypoint": int(seq_wp[i]),
#                 "waypoint_type": seq_typ[i],
#                 "arrival_time": float(arr[i]),
#             })

#     # 篩除冗餘結果並整理格式
#     df = pd.DataFrame(rows).sort_values(["truck_id", "seq_index"]).reset_index(drop=True)

#     def _clean_once(x: pd.DataFrame) -> pd.DataFrame:
#         x = x.copy().sort_values(["arrival_time", "seq_index"]).reset_index(drop=True)
#         keep = []
#         for _, sub in x.groupby(["arrival_time", "waypoint"], sort=False):
#             real = sub[sub["waypoint_type"].isin(["Pickup", "Delivery", "Depot"])]
#             keep.extend((real.index if len(real) > 0 else sub.index).tolist())
#         out = x.loc[sorted(set(keep))].reset_index(drop=True)
#         out["seq_index"] = range(len(out))
#         return out

#     return df.groupby("truck_id", group_keys=False).apply(_clean_once).reset_index(drop=True)


def compute_waypoint_arrival_times(
    *, assignment, waypoint_matrix, target_locations, offsets, edges, weights,
    pickup_service_time=0, delivery_service_time=0,
    vehicle_breaks=None,   # ← 新增：直接吃你 apply_vehicle_breaks 的那個 list
):
    """改良版：正確累加 Pickup / Delivery / Break 服務時間到 arrival_time（此為離站時間）。"""
    import pandas as pd
    import math

    # --- 建 break 規格：每車一個 FIFO 佇列（按 earliest 排），供 route 缺字段時使用 ---
    brq = {}  # {truck_id: [dur1, dur2, ...]}
    if vehicle_breaks:
        tmp = {}
        for b in vehicle_breaks:
            vid = int(b["vehicle_id"])
            tmp.setdefault(vid, []).append(
                (int(b.get("earliest", 0)), float(b["duration"]))
            )
        # 依 earliest 排序；僅保留 duration，後面遇到 Break 就依序取用
        brq = {vid: [dur for _, dur in sorted(pairs, key=lambda t: t[0])]
               for vid, pairs in tmp.items()}

    N = len(target_locations)
    rt_cu = assignment.get_route()
    assert_route_indexed(rt_cu, n_locations=N)

    # Waypoint sequence 與 route 事件
    wseq = waypoint_matrix.compute_waypoint_sequence(target_locations, rt_cu)
    rt = rt_cu.to_pandas().sort_values(["truck_id", "route"]).reset_index(drop=True)
    ws = wseq.to_pandas() if hasattr(wseq, "to_pandas") else pd.DataFrame(wseq)

    if "sequence_offset" not in rt.columns:
        raise RuntimeError("route 缺少 sequence_offset；請先呼叫 compute_waypoint_sequence")

    # CSR adjacency
    V = int(len(offsets) - 1)
    adj = [{} for _ in range(V)]
    for u in range(V):
        a, b = int(offsets[u]), int(offsets[u + 1])
        for k in range(a, b):
            v = int(edges[k]); adj[u][v] = float(weights[k])

    precedence = {"Pickup": 4, "Delivery": 3, "Break": 2, "Depot": 1, "Start": 1, "End": 1, "w": 0}

    def _extract_break_duration(ev_row, g_df, tid):
        """優先讀 route 欄位；沒有就用 vehicle_breaks 佇列；再不行用 dep-arr；最後 0。"""
        for key in ("break_duration", "duration", "service_time", "break_time"):
            if key in ev_row and ev_row[key] is not None and not (isinstance(ev_row[key], float) and math.isnan(ev_row[key])):
                try:
                    d = float(ev_row[key])
                    if d > 0:
                        return d
                except Exception:
                    pass
        # 用你傳進來的 vehicle_breaks
        q = brq.get(int(tid), [])
        if q:
            return float(q.pop(0))
        # 以 departure_stamp - arrival_stamp 估
        if "departure_stamp" in g_df.columns and "arrival_stamp" in g_df.columns:
            try:
                d = float(ev_row["departure_stamp"]) - float(ev_row["arrival_stamp"])
                return max(0.0, d)
            except Exception:
                pass
        return 0.0

    rows = []
    for tid, g in rt.groupby("truck_id"):
        g = g.sort_values("route").reset_index(drop=True)
        a = int(g["sequence_offset"].min())
        b = int(g["sequence_offset"].max())

        seq_wp  = ws.loc[a:b, "waypoint_sequence"].astype(int).tolist()
        seq_typ = ["w"] * len(seq_wp)
        seq_svc = [0.0] * len(seq_wp)

        # 將 route 事件灌回序列：服務時間「累加」、顯示型別依優先序覆蓋
        for _, ev in g.iterrows():
            k = int(ev["sequence_offset"]) - a
            if 0 <= k < len(seq_wp):
                t = str(ev.get("type", "w"))
                if t == "Pickup":
                    seq_svc[k] += float(pickup_service_time)
                elif t == "Delivery":
                    seq_svc[k] += float(delivery_service_time)
                elif t == "Break":
                    seq_svc[k] += _extract_break_duration(ev, g, tid)
                if precedence.get(t, 0) >= precedence.get(seq_typ[k], 0):
                    seq_typ[k] = ("Depot" if t in ("Start", "End") else t)

        # 累積離站時間
        arr = [0.0] * len(seq_wp)
        base = float(g["arrival_stamp"].iloc[0]) if len(g) else 0.0
        arr[0] = base + float(seq_svc[0])
        for i in range(1, len(seq_wp)):
            u, v = int(seq_wp[i - 1]), int(seq_wp[i])
            w = adj[u].get(v) if u != v else 0.0
            if w is None:
                raise ValueError(f"CSR 缺邊 {u}->{v}")
            arr[i] = arr[i - 1] + float(w) + float(seq_svc[i])

        for i in range(len(seq_wp)):
            rows.append({
                "truck_id": int(tid),
                "seq_index": i,
                "waypoint": int(seq_wp[i]),
                "waypoint_type": seq_typ[i],
                "arrival_time": float(arr[i]),
            })

    df = pd.DataFrame(rows).sort_values(["truck_id", "seq_index"]).reset_index(drop=True)

    def _clean_once(x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy().sort_values(["arrival_time", "seq_index"]).reset_index(drop=True)
        keep = []
        for _, sub in x.groupby(["arrival_time", "waypoint"], sort=False):
            real = sub[sub["waypoint_type"].isin(["Pickup", "Delivery", "Depot", "Break"])]
            keep.extend((real.index if len(real) > 0 else sub.index).tolist())
        out = x.loc[sorted(set(keep))].reset_index(drop=True)
        out["seq_index"] = range(len(out))
        return out

    return df.groupby("truck_id", group_keys=False).apply(_clean_once).reset_index(drop=True)



# =========================
# 純文字輸出（替代 display）
# =========================
def print_all(df_or_series) -> None:
    """
    穩定輸出 cuDF / pandas DataFrame（或 Series），不使用 display。
    """
    if hasattr(df_or_series, "to_pandas"):
        obj = df_or_series.to_pandas()
    else:
        obj = df_or_series
    try:
        print(obj.to_string(max_rows=None, max_cols=None))
    except Exception:
        print(obj)

