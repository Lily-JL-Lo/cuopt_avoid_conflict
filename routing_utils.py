# routing_utils.py

import cudf
import pandas as pd
from cuopt import routing, distance_engine

def build_waypoint_graph(graph_dict):
    """用 dict 生成 CSR 並回傳 WaypointMatrix 與 CSR 三元組"""
    import numpy as np
    offsets, edges, weights = [], [], []
    cur = 0
    for node in range(len(graph_dict)):
        offsets.append(cur)
        edges.extend(graph_dict[node]["edges"])
        weights.extend(graph_dict[node]["weights"])
        cur += len(graph_dict[node]["edges"])
    offsets.append(cur)

    matrix = distance_engine.WaypointMatrix(
        np.array(offsets), np.array(edges), np.array(weights)
    )
    return matrix, np.array(offsets), np.array(edges), np.array(weights)

def setup_data_model(target_locations, transport_df, robot_df,
                     cost_matrix, transit_matrix,
                     factory_open=0, factory_close=100,
                     drop_return=True):
    """建立並回傳一個已設定好所有約束的 cuOpt DataModel"""
    # mappings
    target_map = {v:k for k,v in enumerate(target_locations)}

    # 初始化
    n_loc = len(target_locations)
    n_veh = len(robot_df)
    n_ord = len(transport_df)*2
    dm = routing.DataModel(n_loc, n_veh, n_ord)
    dm.add_cost_matrix(cost_matrix)
    dm.add_transit_time_matrix(transit_matrix)

    # capacity
    raw = transport_df["order_demand"]
    drops = raw * -1
    dm.add_capacity_dimension(
        "demand",
        cudf.concat([raw, drops], ignore_index=True),
        robot_df["carrying_capacity"]
    )

    # order locations + pairs
    pu = transport_df["pickup_location"].to_arrow().to_pylist()
    dl = transport_df["delivery_location"].to_arrow().to_pylist()
    pu_idx = cudf.Series([target_map[p] for p in pu])
    dl_idx = cudf.Series([target_map[d] for d in dl])
    dm.set_order_locations(cudf.concat([pu_idx, dl_idx], ignore_index=True))
    n_pair = len(transport_df)
    dm.set_pickup_delivery_pairs(
        cudf.Series(range(n_pair)),
        cudf.Series(range(n_pair, 2*n_pair))
    )

    # time windows & service times
    earliest = cudf.concat(
        [transport_df["earliest_pickup"], transport_df["earliest_delivery"]],
        ignore_index=True
    )
    latest   = cudf.concat(
        [transport_df["latest_pickup"], transport_df["latest_delivery"]],
        ignore_index=True
    )
    service  = cudf.concat(
        [transport_df["pickup_service_time"], transport_df["delivery_service_time"]],
        ignore_index=True
    )
    earliest = earliest.astype("int32")  # cuOpt 的時間窗、服務時間都只能是整數（int32）
    latest   = latest.astype("int32")
    service  = service.astype("int32")

    dm.set_order_time_windows(earliest, latest)
    dm.set_order_service_times(service)

    dm.set_vehicle_time_windows(
        cudf.Series([factory_open]*n_veh),
        cudf.Series([factory_close]*n_veh)
    )

    # drop return trip if desired
    if drop_return:
        dm.set_drop_return_trips(cudf.Series([True]*n_veh))

    return dm

def solve_routing(dm, time_limit=5):
    """呼叫 cuOpt Solve 並回傳 solution"""
    s = routing.SolverSettings()
    s.set_time_limit(time_limit)
    sol = routing.Solve(dm, s)
    if sol.get_status() != 0:
        raise RuntimeError(f"cuOpt failed with status {sol.get_status()}")
    return sol

def expand_and_reserve(sol, waypoint_graph, target_locations,
                       index_map, offsets, edges, weights):
    """把 target-level solution 展開到 waypoint-level，並生成 reservation table"""
    # -- 1. target-level 轉 pandas，抽 times & positions (同前面邏輯) --
    route_pd = sol.route.to_pandas().sort_values("order_array_index")
    route_pd["wp"] = route_pd["location"].map(lambda i: index_map[int(i)])
    target_times = dict(zip(route_pd["order_array_index"], route_pd["arrival_stamp"]))
    order_positions = {
        int(r.order_array_index): waypoint_graph
            .compute_waypoint_sequence(target_locations,
                                       sol.get_route()[sol.get_route()["truck_id"]==r.truck_id])
            .to_pandas()["waypoint_sequence"]
            .tolist()
            .index(int(r.wp))
        for _, r in route_pd.iterrows()
    }

    # -- 2. waypoint-level seq --
    all_r = sol.get_route()
    truck = sol.route["truck_id"].unique().to_arrow().to_pylist()[0]
    wl = waypoint_graph.compute_waypoint_sequence(
        target_locations,
        all_r[all_r["truck_id"]==truck]
    ).to_pandas()
    seq = wl["waypoint_sequence"].tolist()

    # -- 3. 累計時間 --
    rows = []
    cur = None
    for i, wp in enumerate(seq):
        if i in order_positions.values():
            idx = next(k for k,v in order_positions.items() if v==i)
            cur = float(target_times[idx])
        else:
            prev = seq[i-1]
            w = next(
                (float(weights[j]) for j in range(offsets[prev],offsets[prev+1]) if int(edges[j])==wp),
                1.0
            ) if wp!=prev else 0.0
            cur += w
        rows.append({"waypoint":wp, "truck_id":truck, "step_time":cur})

    return pd.DataFrame(rows)



# def compute_reservation_table(
#     sol,               # cuOpt routing_solution
#     waypoint_graph,    # cuOpt distance_engine.WaypointMatrix
#     target_locations,  # np.array，該車要取送的 waypoint 列表
#     index_map,         # dict：matrix 索引 → 原始 waypoint
#     offsets, edges, weights  # CSR 三元組
# ):
#     """
#     從 cuOpt 的 target‐level 解 (sol)，產生完整的 waypoint‐level
#     reservation table: 每個 waypoint 並標示該車在此的佔用時間(step_time)。
#     """

#     # 1. 取出 AGV 的編號
#     truck_id = int(sol.route["truck_id"].unique().to_arrow().to_pylist()[0])

#     # 2. 展開完整的 waypoint‐level 路徑序列
#     all_routes = sol.get_route()
#     mask = all_routes["truck_id"] == truck_id
#     wl_cudf = waypoint_graph.compute_waypoint_sequence(
#         target_locations,
#         all_routes[mask]
#     )
#     seq = wl_cudf["waypoint_sequence"].to_arrow().to_pylist()

#     # 3. 讀出 pickup 和 delivery 在 target‐level 的到達時間
#     route_pd = (
#         sol.route
#            .to_pandas()
#            .query("type != 'Depot'")
#            .sort_values("arrival_stamp")
#            .reset_index(drop=True)
#     )
#     # 第一筆為 pickup，第二筆為 delivery
#     pickup_time   = float(route_pd.loc[0, "arrival_stamp"])
#     delivery_time = float(route_pd.loc[1, "arrival_stamp"])
#     pu_idx = int(route_pd.loc[0, "location"])
#     dl_idx = int(route_pd.loc[1, "location"])
#     pu_wp = index_map[pu_idx]
#     dl_wp = index_map[dl_idx]

#     # 4. 找出 pickup 和 delivery 在 seq 中出現的位置
#     pu_pos = seq.index(pu_wp)
#     dl_pos = seq.index(dl_wp)

#     # 5. 掃描 seq，累計 step_time
#     reservation_rows = []
#     current_time = None

#     for i, wp in enumerate(seq):
#         if i == pu_pos:
#             # pickup 停靠：直接使用 cuOpt 算出的時間
#             current_time = pickup_time
#         elif i == dl_pos:
#             # delivery 停靠：直接使用 cuOpt 算出的時間
#             current_time = delivery_time
#         else:
#             # 中繼節點：從前一節點累加邊的權重
#             prev_wp = seq[i-1]
#             if wp == prev_wp:
#                 # 同一節點連續出現，不算移動
#                 w = 0.0
#             else:
#                 # 在 CSR 中查 prev_wp→wp 的邊權重
#                 w = next(
#                     (
#                         float(weights[j])
#                         for j in range(offsets[prev_wp], offsets[prev_wp+1])
#                         if int(edges[j]) == wp
#                     ),
#                     1.0  # 萬一找不到就預設 1.0
#                 )
#             current_time += w

#         # 記錄該 waypoint 的佔用時間
#         reservation_rows.append({
#             "waypoint": wp,
#             "truck_id": truck_id,
#             "step_time": current_time
#         })

#     # 6. 回傳 DataFrame
#     return pd.DataFrame(reservation_rows)


# def compute_reservation_table(
#     sol,
#     waypoint_graph,
#     target_locations,
#     index_map,
#     offsets, edges, weights
# ):
#     import pandas as pd

#     # 1. 車輛編號
#     truck_id = int(sol.route["truck_id"].unique().to_arrow().to_pylist()[0])

#     # 2. waypoint-level 路徑
#     all_routes = sol.get_route()
#     mask = all_routes["truck_id"] == truck_id
#     wl = waypoint_graph.compute_waypoint_sequence(
#         target_locations,
#         all_routes[mask]
#     ).to_pandas()
#     seq   = wl["waypoint_sequence"].tolist()
#     types = wl["waypoint_type"].tolist()

#     # 3. target-level 停靠時間（取 scalar）
#     route_pd      = sol.route.to_pandas()
#     pickup_time   = float(route_pd.query("type=='Pickup'")["arrival_stamp"].iloc[0])
#     delivery_time = float(route_pd.query("type=='Delivery'")["arrival_stamp"].iloc[0])

#     pu_idx = int(route_pd.query("type=='Pickup'")["location"].iloc[0])
#     dl_idx = int(route_pd.query("type=='Delivery'")["location"].iloc[0])
#     pu_wp  = index_map[pu_idx]
#     dl_wp  = index_map[dl_idx]

#     pu_pos = seq.index(pu_wp)
#     dl_pos = seq.index(dl_wp)

#     # 4. 累計每一段 step_time
#     reservation_rows = []
#     current_time = None

#     for i, wp in enumerate(seq):
#         if i == pu_pos:
#             current_time = pickup_time
#         elif i == dl_pos:
#             current_time = delivery_time
#         else:
#             prev = seq[i-1]
#             if wp == prev:
#                 w = 0.0
#             else:
#                 # 查 CSR 裡的邊權重
#                 w = next(
#                     (
#                         float(weights[j])
#                         for j in range(offsets[prev], offsets[prev+1])
#                         if int(edges[j]) == wp
#                     ),
#                     1.0
#                 )
#             current_time += w

#         reservation_rows.append({
#             "waypoint":  wp,
#             "truck_id":  truck_id,
#             "step_time": current_time
#         })

#     return pd.DataFrame(reservation_rows)



def compute_reservation_table(
    sol,                 # cuOpt 回傳的 routing_solution
    waypoint_graph,      # cuOpt distance_engine.WaypointMatrix
    target_locations,    # numpy array：此車的 pickup/delivery waypoint
    index_map,           # dict：matrix 索引 → 原始 waypoint
    offsets, edges, weights  # CSR 三元組
):
    import pandas as pd

    # 1. 取出車輛編號
    truck_id = int(sol.route["truck_id"]
                   .unique()
                   .to_arrow()
                   .to_pylist()[0])

    # 2. 取得完整的 waypoint-level 序列及對應的類型
    all_routes = sol.get_route()
    mask       = all_routes["truck_id"] == truck_id
    wl = waypoint_graph.compute_waypoint_sequence(
        target_locations,
        all_routes[mask]
    ).to_pandas()
    seq   = wl["waypoint_sequence"].tolist()  # 實際走過的節點
    types = wl["waypoint_type"].tolist()      # 每個節點是 Pickup, Delivery, 還是 w

    # 3. 從 target-level 結果擷取真正的 pick-up / delivery 時刻與節點
    rt = sol.route.to_pandas()
    # a) pick-up
    pu_row   = rt.query("type=='Pickup'").iloc[0]
    pu_time  = float(pu_row["arrival_stamp"])
    pu_wp    = index_map[int(pu_row["location"])]
    # b) delivery
    dl_row   = rt.query("type=='Delivery'").iloc[0]
    dl_time  = float(dl_row["arrival_stamp"])
    dl_wp    = index_map[int(dl_row["location"])]

    # 4. 用 types 去定位 pick-up / delivery 在 seq 的 index
    pu_pos = next(i for i,t in enumerate(types) if t == "Pickup")
    dl_pos = next(i for i,t in enumerate(types) if t == "Delivery")

    # 5. 扫描整條 seq，累計每段 hop 的時間
    reservation_rows = []
    current_time = None
    for i, wp in enumerate(seq):
        if i == pu_pos:
            # 真正的取貨時刻
            current_time = pu_time
        elif i == dl_pos:
            # 真正的送貨時刻
            current_time = dl_time
        else:
            # 中繼節點，累計 prev→wp 的邊權重
            prev = seq[i-1]
            if wp == prev:
                # 同一節點連續出現（可能是等待服務），不移動
                w = 0.0
            else:
                # 在 CSR 裡找 prev→wp 的邊的 weight
                w = next(
                    (
                        float(weights[j])
                        for j in range(offsets[prev], offsets[prev+1])
                        if int(edges[j]) == wp
                    ),
                    1.0  # 若找不到，預設 1.0
                )
            current_time += w

        # 記錄此時刻該節點的佔用
        reservation_rows.append({
            "waypoint":  wp,
            "truck_id":  truck_id,
            "step_time": current_time
        })

    return pd.DataFrame(reservation_rows)


