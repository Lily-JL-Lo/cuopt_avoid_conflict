# streamlit run wait_task_streamlit.py

# app.py
import os
import sys
import importlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# cuOpt / cuDF（宣告依賴）
import cudf
from cuopt import routing, distance_engine  # noqa: F401

# 找得到你的本地模組
sys.path.insert(0, os.getcwd())
import dynamic_avoidance
import routing_utils
from routing_utils import build_waypoint_graph
from dynamic_avoidance import (
    plan_and_reserve,
    detect_conflicts,
    avoid_and_replan,  # 指定避讓
    avoid_and_wait     # 自動避讓
)

# 熱重載
importlib.reload(dynamic_avoidance)
importlib.reload(routing_utils)

# ===== 工具：cuDF/pandas 相容、型別保險、顯示對齊 =====
def ensure_pandas(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else df

def ensure_int(x):
    try:
        return int(x)
    except Exception:
        return x

def align_for_display(res_df, depart_time):
    pdf = ensure_pandas(res_df).copy()
    if pdf.empty or "step_time" not in pdf:
        return pdf
    t0 = int(pdf["step_time"].min())
    if t0 < depart_time:
        pdf["step_time"] = pdf["step_time"] + (depart_time - t0)
    return pdf

# ===== 預設地圖（可替換）=====
DEFAULT_GRAPH = {
    0:  {"edges":[2],           "weights":[1]},
    1:  {"edges":[2],           "weights":[1]},
    2:  {"edges":[0,1,3,5],     "weights":[1,1,1,1]},
    3:  {"edges":[2],           "weights":[1]},
    4:  {"edges":[5],           "weights":[1]},
    5:  {"edges":[2,6,8],       "weights":[1,1,1]},
    6:  {"edges":[],            "weights":[]},
    7:  {"edges":[8],           "weights":[1]},
    8:  {"edges":[5,7,9,11],    "weights":[1,1,1,1]},
    9:  {"edges":[8],           "weights":[1]},
    10: {"edges":[11],          "weights":[1]},
    11: {"edges":[8,10,12],     "weights":[1,1,1]},
    12: {"edges":[11],          "weights":[1]},
}
DEFAULT_COORDS = {
    0:(0,4), 1:(-1,3), 2:(0,3), 3:(1,3), 4:(-1,2), 5:(0,2),
    6:(1,2), 7:(-1,1), 8:(0,1), 9:(1,1), 10:(-1,0), 11:(0,0), 12:(1,0),
}

# ===== Streamlit UI =====
st.set_page_config(page_title="cuOpt AGV 動態避讓示範", layout="wide")
st.title("cuOpt AGV 動態避讓（AGV0 + AGV1）")

with st.sidebar:
    st.header("地圖與時間設定")
    factory_open  = st.number_input("Factory Open", 0, 10000, 0, step=1)
    factory_close = st.number_input("Factory Close", 1, 10000, 100, step=1)

    st.markdown("---")
    st.header("AGV0 任務")
    agv0_start  = st.number_input("AGV0 Pickup Node", 0, 12, 12)
    agv0_end    = st.number_input("AGV0 Delivery Node", 0, 12, 6)
    agv0_depart = st.number_input("AGV0 Depart Time", 0, 1000, 0)

    st.markdown("---")
    st.header("AGV1 任務")
    agv1_start  = st.number_input("AGV1 Pickup Node", 0, 12, 0)
    agv1_end    = st.number_input("AGV1 Delivery Node", 0, 12, 10)
    agv1_depart = st.number_input("AGV1 Depart Time", 0, 1000, 1)

    st.markdown("---")
    st.header("避讓模式")
    avoid_mode = st.radio(
        "選擇避讓方式",
        ["等待避讓（avoid_and_wait）", "指定避讓（avoid_and_replan）"],
        index=0
    )
    avoid_wp_ui = st.number_input("指定避讓節點（選指定模式時使用）", 0, 12, 1)

    run_btn = st.button("Run Routing", type="primary", use_container_width=True)

# 構建 waypoint graph
graph  = DEFAULT_GRAPH
coords = DEFAULT_COORDS
waypoint_graph, offsets, edges, weights = build_waypoint_graph(graph)

with st.expander("查看 CSR 結構", expanded=False):
    st.write("Offsets:", offsets)
    st.write("Edges:", edges)
    st.write("Weights:", weights)

# ===== 主流程 =====
if run_btn:
    # --- AGV0 ---
    transport_order_data_0 = cudf.DataFrame({
        "pickup_location":[agv0_start],
        "delivery_location":[agv0_end],
        "order_demand":[1],
        "earliest_pickup":[agv0_depart],
        "latest_pickup":[agv0_depart],
        "pickup_service_time":[0],
        "earliest_delivery":[agv0_depart],
        "latest_delivery":[factory_close],
        "delivery_service_time":[0],
    })
    robot_data_0 = cudf.DataFrame({"robot_ids":[0], "carrying_capacity":[1]}).set_index("robot_ids")
    target_locations_0 = np.array([agv0_start, agv0_end], dtype=int)

    sol0, reservation0_df, extras0 = plan_and_reserve(
        graph, transport_order_data_0, robot_data_0, target_locations_0,
        factory_open=factory_open, factory_close=factory_close,
    )

    c0, c1 = st.columns(2)
    with c0:
        st.subheader("AGV0：Target-level Route")
        st.write(sol0.route)
    with c1:
        st.subheader("AGV0：Reservation Table")
        st.dataframe(ensure_pandas(reservation0_df), use_container_width=True)

    # --- AGV1（earliest_delivery 自動計算，不顯示 UI）---
    # shortest_path_time = cost_matrix[start,end]
    cm = waypoint_graph.compute_cost_matrix(np.array([agv1_start, agv1_end], dtype=int))
    cm_pd = ensure_pandas(cm)
    shortest_cost = int(cm_pd.iloc[0, 1])
    earliest_del = agv1_depart + shortest_cost
    latest_del   = factory_close

    transport_order_data_1 = cudf.DataFrame({
        "pickup_location":[agv1_start],
        "delivery_location":[agv1_end],
        "order_demand":[1],
        "earliest_pickup":[agv1_depart],
        "latest_pickup":[agv1_depart],
        "pickup_service_time":[0],
        "earliest_delivery":[earliest_del],
        "latest_delivery":[latest_del],
        "delivery_service_time":[0],
    })
    robot_data_1 = cudf.DataFrame({"robot_ids":[1], "carrying_capacity":[1]}).set_index("robot_ids")
    target_locations_1 = np.array([agv1_start, agv1_end], dtype=int)

    sol1, reservation1_df, extras1 = plan_and_reserve(
        graph, transport_order_data_1, robot_data_1, target_locations_1,
        factory_open=factory_open, factory_close=factory_close,
    )

    c2, c3 = st.columns(2)
    with c2:
        st.subheader("AGV1（初始）：Target-level Route")
        st.write(sol1.route)
    with c3:
        st.subheader("AGV1（初始）：Reservation Table")
        st.dataframe(ensure_pandas(reservation1_df), use_container_width=True)

    # --- 衝突偵測 ---
    conflicts = detect_conflicts(reservation0_df, reservation1_df)
    st.subheader("衝突偵測結果")
    st.write(conflicts if conflicts else "沒有衝突。")

    # --- 有衝突才避讓重排（AGV1）---
    res1_for_anim = reservation1_df
    if conflicts:
        if avoid_mode == "等待避讓（avoid_and_wait）":
            st.info("偵測到衝突 → 使用 avoid_and_wait 等待避讓重規劃 AGV1")
            sol1b, reservation1b_df, new_targets = avoid_and_wait(
                graph, transport_order_data_1, robot_data_1, target_locations_1, reservation0_df
            )
        else:
            chosen_avoid = int(avoid_wp_ui)
            st.info(f"偵測到衝突 → 使用 avoid_and_replan（指定避讓點 {chosen_avoid}）重規劃 AGV1")
            sol1b, reservation1b_df, new_targets = avoid_and_replan(
                graph, transport_order_data_1, robot_data_1, target_locations_1,
                reservation0_df, avoid_wp=chosen_avoid
            )

        c4, c5 = st.columns(2)
        with c4:
            st.subheader("AGV1（避讓後）：Target-level Route")
            st.write(sol1b.route)
        with c5:
            st.subheader("AGV1（避讓後）：Reservation Table")
            st.dataframe(ensure_pandas(reservation1b_df), use_container_width=True)

        post_conflicts = detect_conflicts(reservation0_df, reservation1b_df)
        solved_msg = "已解決衝突。" if not post_conflicts else f"仍有衝突：{post_conflicts}"
        if not post_conflicts:
            st.success(solved_msg)
        else:
            st.warning(solved_msg)

        res1_for_anim = reservation1b_df

    # --- 動畫（用出發時間對齊，並 gate 可見性）---
    res0_pd = align_for_display(reservation0_df, agv0_depart)
    res1_pd = align_for_display(res1_for_anim, agv1_depart)

    for c in ("waypoint", "step_time"):
        if c in res0_pd.columns:
            res0_pd[c] = res0_pd[c].map(ensure_int)
        if c in res1_pd.columns:
            res1_pd[c] = res1_pd[c].map(ensure_int)

    res0_pd["agv"] = "AGV0"
    res1_pd["agv"] = "AGV1"
    df = pd.concat([res0_pd, res1_pd], ignore_index=True)

    df["x"] = df["waypoint"].map(lambda w: DEFAULT_COORDS[w][0] if w in DEFAULT_COORDS else None)
    df["y"] = df["waypoint"].map(lambda w: DEFAULT_COORDS[w][1] if w in DEFAULT_COORDS else None)

    init_traces = []
    for u, (xu, yu) in DEFAULT_COORDS.items():
        for v in DEFAULT_GRAPH[u]["edges"]:
            xv, yv = DEFAULT_COORDS[v]
            init_traces.append(go.Scatter(
                x=[xu, xv], y=[yu, yv],
                mode="lines",
                line=dict(color="lightblue", width=2),
                showlegend=False
            ))
    init_traces.append(go.Scatter(
        x=[c[0] for c in DEFAULT_COORDS.values()],
        y=[c[1] for c in DEFAULT_COORDS.values()],
        mode="markers+text",
        marker_symbol="square",
        marker=dict(size=24, color="lightblue"),
        text=[str(n) for n in DEFAULT_COORDS.keys()],
        textposition="middle center",
        showlegend=False
    ))
    init_traces.append(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="circle", color="green", size=16),
        name="AGV0", showlegend=True
    ))
    init_traces.append(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="circle", color="orange", size=16),
        name="AGV1", showlegend=True
    ))
    dummy_indices = {"AGV0": len(init_traces)-2, "AGV1": len(init_traces)-1}

    agv_depart_time = {"AGV0": agv0_depart, "AGV1": agv1_depart}
    time_steps = sorted(set(df["step_time"].dropna().tolist()))
    frames = []
    for t in time_steps:
        agv_traces = []
        trace_idxs = []
        for agv in ["AGV0", "AGV1"]:
            if t < agv_depart_time[agv]:
                agv_traces.append(go.Scatter(x=[None], y=[None]))
                trace_idxs.append(dummy_indices[agv])
                continue
            sub = df[(df["agv"] == agv) & (df["step_time"] <= t)].sort_values("step_time")
            if sub.empty:
                agv_traces.append(go.Scatter(x=[None], y=[None]))
            else:
                last = sub.iloc[-1]
                color = "green" if agv == "AGV0" else "orange"
                agv_traces.append(go.Scatter(
                    x=[last["x"]], y=[last["y"]],
                    mode="markers+text",
                    marker=dict(symbol="circle", size=16, color=color),
                    text=[agv], textposition="top center", showlegend=False
                ))
            trace_idxs.append(dummy_indices[agv])
        frames.append(go.Frame(data=agv_traces, name=str(int(t)), traces=trace_idxs))

    fig = go.Figure(data=init_traces, frames=frames)
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", showactive=False, y=1.05,
            buttons=[dict(
                label="播放動畫", method="animate",
                args=[None, {"frame": {"duration": 800, "redraw": True}, "fromcurrent": True}]
            )]
        )],
        sliders=[dict(
            steps=[dict(
                method="animate",
                args=[[str(int(t))], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                label=str(int(t))
            ) for t in time_steps],
            transition={"duration": 0}, x=0.1, y=0,
            currentvalue={"prefix": "Time: "}
        )],
        xaxis=dict(
            range=[min(v[0] for v in DEFAULT_COORDS.values()) - 1,
                   max(v[0] for v in DEFAULT_COORDS.values()) + 1],
            constrain="domain"
        ),
        yaxis=dict(
            range=[min(v[1] for v in DEFAULT_COORDS.values()) - 1,
                   max(v[1] for v in DEFAULT_COORDS.values()) + 1],
            scaleanchor="x"
        ),
        width=900, height=700,
        title="AGV 動態避讓路徑動畫"
    )

    st.subheader("路徑動畫")
    st.plotly_chart(fig, use_container_width=True)