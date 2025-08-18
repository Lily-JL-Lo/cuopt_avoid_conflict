# dynamic_avoidance.py

import numpy as np
import pandas as pd
import cudf
from cuopt import routing, distance_engine

import importlib
import routing_utils
importlib.reload(routing_utils)

from routing_utils import (
    build_waypoint_graph,
    setup_data_model,
    solve_routing,
    compute_reservation_table
)

def plan_and_reserve(graph, transport_df, robot_df, target_locations,
                     factory_open=0, factory_close=100, scale=1):
    """
    為單一 AGV 做路徑規劃並產生 reservation table。
    transport_df: pandas or cudf DataFrame 運單
    robot_df: pandas or cudf DataFrame 車輛資料
    target_locations: np.array pickup/delivery nodes
    scale: 時間放大倍率（用於小數精度）
    回傳 (sol, reservation_df, offsets, edges, weights, waypoint_graph, index_map)
    """
    # 1. 建圖 CSR
    # 假設 graph 已經在外面 build_waypoint_graph
    waypoint_graph, offsets, edges, weights = build_waypoint_graph(graph)

    # 2. cost & time 矩陣
    cost = waypoint_graph.compute_cost_matrix(target_locations)
    time = cost.copy(deep=True)
    index_map = {i: loc for i, loc in enumerate(target_locations)}

    # 3. Model & Solve
    dm = setup_data_model(
        target_locations,
        transport_df,
        robot_df,
        cost, time,
        factory_open, factory_close
    )
    sol = solve_routing(dm)
    
    # 4. Reservation table
    reservation_df = compute_reservation_table(
        sol,
        waypoint_graph,
        target_locations,
        index_map,
        offsets, edges, weights
    )
    return sol, reservation_df, (offsets, edges, weights, waypoint_graph, index_map)

def detect_conflicts(res0, res1):
    """
    比對兩張 reservation table，回傳 conflicts list of dict
    """
    df0 = res0.reset_index(drop=True)
    df1 = res1.reset_index(drop=True)
    conf = df0.merge(df1, on=["waypoint", "step_time"])
    return conf[["waypoint", "step_time"]].to_dict("records")

def avoid_and_replan(graph, transport_df1, robot_df1, target_loc1,
                     reservation0, avoid_wp, scale=1):
    """
    在 transport_df1 中，根據 reservation0 假設只取第一個衝突點，
    自動插入虛擬訂單到 avoid_wp，並重跑規劃。
    回傳新的 sol1, reservation1
    """
    # 1. 偵測 conflicts
    sol1, res1, extras = plan_and_reserve(
        graph, transport_df1, robot_df1, target_loc1
    )
    conflicts = detect_conflicts(reservation0, res1)
    if not conflicts:
        return sol1, res1, target_loc1  # no need to avoid

    # 2. 取第一筆衝突時間
    conflict = conflicts[0]
    t_conflict = conflict["step_time"]
    # 3. 建虛擬訂單
    virtual = cudf.DataFrame({
        "pickup_location":       [avoid_wp],
        "delivery_location":     [avoid_wp],
        "order_demand":          [0],
        "earliest_pickup":       [t_conflict],
        "latest_pickup":         [t_conflict],
        "pickup_service_time":   [0],
        "earliest_delivery":     [t_conflict],
        "latest_delivery":       [t_conflict ] ,
        "delivery_service_time":[0]
    })
    transport_df2 = cudf.concat([transport_df1, virtual], ignore_index=True)

    # 4. 更新 target_locations1 + index_map + recompute cost/time
    if avoid_wp not in target_loc1:
        target_loc1 = np.append(target_loc1, avoid_wp)

    # 5. 重跑
    sol1b, res1b, _ = plan_and_reserve(
        graph, transport_df2, robot_df1, target_loc1
    )
    return sol1b, res1b, target_loc1


def avoid_and_wait(graph, transport_df1, robot_df1, target_loc1,
                     reservation0, scale=1):
    """
    如果與 reservation0 衝突，就：
      1. 找到衝突點 conflict_wp 及衝突時間 t_conflict
      2. 在 res1 中定位它前一個 waypoint prev_wp 及到達時間 t_arr
      3. wait = t_conflict - t_arr
      4. 插入「在 prev_wp 等待 wait 秒」的虛擬訂單
      5. 重跑 cuOpt，回傳新解
    """
    # 1) 首次排程 & 衝突檢測
    sol1, res1, extras = plan_and_reserve(
        graph, transport_df1, robot_df1, target_loc1
    )
    conflicts = detect_conflicts(reservation0, res1)
    if not conflicts:
        return sol1, res1, target_loc1

    # 2) 取第一筆衝突
    conflict = conflicts[0]
    t_conflict = conflict["step_time"]
    conflict_wp = conflict["waypoint"]

    # 3) 定位 prev_wp 及其到達時間 t_arr
    seq   = res1["waypoint"].to_list()
    idx   = seq.index(conflict_wp)
    prev_wp = seq[idx - 1]
    t_arr   = float(res1.query(f"waypoint == {prev_wp}")["step_time"].min())

    # 4) 計算等待秒數
    wait = max(0.0, t_conflict - t_arr)

    # 5) 構造「等待」虛擬訂單
    virtual = cudf.DataFrame({
        "pickup_location":       [prev_wp],
        "delivery_location":     [prev_wp],
        "order_demand":          [0],
        "earliest_pickup":       [t_arr],
        "latest_pickup":         [t_arr],
        "pickup_service_time":   [wait],         # ← 關鍵：停留 wait 秒
        "earliest_delivery":     [t_arr + wait],
        "latest_delivery":       [t_arr + wait],
        "delivery_service_time":[0]
    })

    # 6) append 並確保 target list 包含 prev_wp
    transport2 = cudf.concat([transport_df1, virtual], ignore_index=True)
    if prev_wp not in target_loc1:
        target_loc1 = np.append(target_loc1, prev_wp)

    # 7) 重跑
    sol2, res2, extras2 = plan_and_reserve(
        graph, transport2, robot_df1, target_loc1
    )
    return sol2, res2, target_loc1

