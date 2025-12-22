import pandas as pd
import streamlit as st
import plotly.express as px

from rules import add_impact_severity, add_phenomenon_type, add_user_journey_view

st.set_page_config(page_title="竞品洞察 & MVP 决策看板", layout="wide")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]

    # 基础清洗
    for col in ["comment_id","text","Interaction_Phase","Object","Issue",
                "Root_Cause_Context","Feedback_Type","Sentiment",
                "Subject","Context","Household"]:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown")

    df = add_impact_severity(df)
    df = add_phenomenon_type(df)
    df = add_user_journey_view(df)

    # 标准化为字符串（避免筛选器出错）
    for col in ["Interaction_Phase","Object","Issue","Root_Cause_Context","Feedback_Type",
                "Impact_Severity","Phenomenon_Type","Subject","Context","Household","User_Journey_View"]:
        df[col] = df[col].astype(str)

    return df

df = load_data("cat_litter_labelled_comments_merged.xlsx")

st.title("竞品洞察与 MoSCoW 战略决策看板（一次性猫砂盆）")
st.caption("原则：任何结论必须能回溯到原评论（Evidence）。")

# -----------------------
# 全局筛选器
# -----------------------
with st.sidebar:
    st.header("全局筛选器")

    subject = st.multiselect("Subject", sorted(df["Subject"].unique()))
    context = st.multiselect("Context", sorted(df["Context"].unique()))
    household = st.multiselect("Household", sorted(df["Household"].unique()))
    phase = st.multiselect("Interaction_Phase", sorted(df["Interaction_Phase"].unique()))

    sev = st.multiselect(
        "Impact_Severity",
        ["Critical","Major","Minor","Positive","Delighter","Unknown"],
        default=["Critical","Major","Positive","Delighter"]
    )
    ptype = st.multiselect("Phenomenon_Type", ["Defect","Constraint","Unclear"], default=["Defect","Constraint","Unclear"])

    fdf = df.copy()
    if subject: fdf = fdf[fdf["Subject"].isin(subject)]
    if context: fdf = fdf[fdf["Context"].isin(context)]
    if household: fdf = fdf[fdf["Household"].isin(household)]
    if phase: fdf = fdf[fdf["Interaction_Phase"].isin(phase)]
    if sev: fdf = fdf[fdf["Impact_Severity"].isin(sev)]
    if ptype: fdf = fdf[fdf["Phenomenon_Type"].isin(ptype)]

st.write(f"当前筛选后：**N={len(fdf)}** 行 insight（全量 N={len(df)}）")

# -----------------------
# Tabs
# -----------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["A 生死线", "B 机会点", "C 细分战场", "D 丢弃体验", "E 证据面板"])

# 用 session_state 保存“点击后要看的 evidence 条件”
if "evidence_filter" not in st.session_state:
    st.session_state.evidence_filter = {}

def set_evidence(obj=None, issue=None, extra=None):
    filt = {}
    if obj is not None: filt["Object"] = obj
    if issue is not None: filt["Issue"] = issue
    if extra:
        filt.update(extra)
    st.session_state.evidence_filter = filt

with tab1:
    st.subheader("A｜生死线视图（Critical/Major + Defect）")
    sdf = fdf[(fdf["Phenomenon_Type"]=="Defect") & (fdf["Impact_Severity"].isin(["Critical","Major"]))]

    c1, c2 = st.columns([1,1])

    with c1:
        killers = (sdf.groupby(["Object","Issue"]).size().reset_index(name="N")
                   .sort_values("N", ascending=False).head(20))
        killers["Key"] = killers["Object"].astype(str) + " | " + killers["Issue"].astype(str)

        fig = px.bar(killers, x="N", y="Key", orientation="h", title="Top Killers（Object | Issue）")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("点击下方表格任意行 → 跳到证据面板")
        st.dataframe(killers, use_container_width=True, hide_index=True)
        # 选择一行做 evidence（小白友好：用下拉）
        if len(killers) > 0:
            idx = st.selectbox("选择一个 Killer 查看证据", killers.index.tolist(), format_func=lambda i: killers.loc[i,"Key"])
            if st.button("查看证据（Killer）"):
                set_evidence(obj=killers.loc[idx,"Object"], issue=killers.loc[idx,"Issue"], extra={"View":"Survival"})
                st.success("已设置证据筛选，请切换到 Tab E 证据面板。")

    with c2:
        by_phase = sdf.groupby("Interaction_Phase").size().reset_index(name="N").sort_values("N", ascending=False)
        fig2 = px.bar(by_phase, x="Interaction_Phase", y="N", title="缺陷发生阶段分布")
        st.plotly_chart(fig2, use_container_width=True)


with tab2:
    st.subheader("B｜机会点视图（Delighter/Positive）")
    odf = fdf[fdf["Impact_Severity"].isin(["Delighter","Positive"])]

    c1, c2 = st.columns([1,1])

    with c1:
        hooks = (odf.groupby(["Object","Issue"]).size().reset_index(name="N")
                 .sort_values("N", ascending=False).head(20))
        hooks["Key"] = hooks["Object"].astype(str) + " | " + hooks["Issue"].astype(str)
        fig = px.bar(hooks, x="N", y="Key", orientation="h", title="Top Hooks（用户最夸赞）")
        st.plotly_chart(fig, use_container_width=True)

        if len(hooks) > 0:
            idx = st.selectbox("选择一个 Hook 查看证据", hooks.index.tolist(), format_func=lambda i: hooks.loc[i,"Key"], key="hook_select")
            if st.button("查看证据（Hook）"):
                set_evidence(obj=hooks.loc[idx,"Object"], issue=hooks.loc[idx,"Issue"], extra={"View":"Opportunity"})
                st.success("已设置证据筛选，请切换到 Tab E 证据面板。")

    with c2:
        by_phase = odf.groupby("Interaction_Phase").size().reset_index(name="N").sort_values("N", ascending=False)
        fig2 = px.bar(by_phase, x="Interaction_Phase", y="N", title="正向体验发生阶段")
        st.plotly_chart(fig2, use_container_width=True)


with tab3:
    st.subheader("C｜细分战场视图（S-C-H 放大）")
    issue_list = sorted(fdf["Issue"].unique())
    focus_issue = st.selectbox("选择要观察的 Issue", issue_list)

    base = fdf.copy()
    base["is_focus"] = (base["Issue"] == focus_issue).astype(int)

    # 全局占比
    global_share = base["is_focus"].mean() if len(base) else 0.0
    st.metric("全局占比（该 Issue）", f"{global_share:.2%}")

    # Subject 放大
    pivot = (base.groupby("Subject")["is_focus"].mean().reset_index(name="Share"))
    pivot["Lift"] = pivot["Share"] / (global_share if global_share > 0 else 1.0)

    fig = px.bar(pivot, x="Subject", y="Share", title=f"{focus_issue} 在不同 Subject 的占比")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(pivot.sort_values("Lift", ascending=False), use_container_width=True, hide_index=True)

    # 选择一个 Subject 查看证据
    if len(pivot) > 0:
        subj = st.selectbox("选择一个 Subject 查看该 Issue 的证据", pivot["Subject"].tolist())
        if st.button("查看证据（Niche）"):
            set_evidence(obj=None, issue=focus_issue, extra={"Subject": subj, "View":"Niche"})
            st.success("已设置证据筛选，请切换到 Tab E 证据面板。")


with tab4:
    st.subheader("D｜丢弃体验视图（派生：User_Journey_View = Disposal）")
    ddf = fdf[fdf["User_Journey_View"]=="Disposal"]

    c1, c2 = st.columns([1,1])

    with c1:
        st.write("Disposal 规模")
        share = len(ddf) / len(fdf) if len(fdf) else 0
        st.metric("Disposal 占比（在当前筛选内）", f"{share:.2%}", help="Maintenance/Durability 中含丢弃/搬运/倒砂语义的 insight")
        by_phase = ddf.groupby("Interaction_Phase").size().reset_index(name="N").sort_values("N", ascending=False)
        st.plotly_chart(px.bar(by_phase, x="Interaction_Phase", y="N", title="Disposal 分布（原始 Phase）"), use_container_width=True)

    with c2:
        top = (ddf.groupby(["Object","Issue"]).size().reset_index(name="N")
               .sort_values("N", ascending=False).head(20))
        top["Key"] = top["Object"].astype(str) + " | " + top["Issue"].astype(str)
        st.plotly_chart(px.bar(top, x="N", y="Key", orientation="h", title="Disposal Top Issues（Object | Issue）"),
                        use_container_width=True)

        if len(top) > 0:
            idx = st.selectbox("选择一个 Disposal 问题查看证据", top.index.tolist(), format_func=lambda i: top.loc[i,"Key"], key="disposal_select")
            if st.button("查看证据（Disposal）"):
                set_evidence(obj=top.loc[idx,"Object"], issue=top.loc[idx,"Issue"], extra={"View":"Disposal"})
                st.success("已设置证据筛选，请切换到 Tab E 证据面板。")


with tab5:
    st.subheader("E｜证据面板（原评论回溯）")
    ef = fdf.copy()
    filt = st.session_state.evidence_filter or {}

    st.write("当前证据筛选条件：", filt if filt else "（未选择，使用当前全局筛选）")

    # 应用证据条件
    if "Object" in filt:
        ef = ef[ef["Object"] == filt["Object"]]
    if "Issue" in filt:
        ef = ef[ef["Issue"] == filt["Issue"]]
    if "Subject" in filt:
        ef = ef[ef["Subject"] == filt["Subject"]]

    # 去重：同 comment_id 的多行 insight 合并展示（避免刷屏）
    cols = ["comment_id","text","Interaction_Phase","Object","Issue",
            "Root_Cause_Context","Feedback_Type",
            "Impact_Severity","Phenomenon_Type",
            "Subject","Context","Household","User_Journey_View"]
    ef_show = ef[cols].drop_duplicates(subset=["comment_id"])

    st.write(f"证据评论数（去重后）：**{len(ef_show)}**")
    st.dataframe(ef_show.head(50), use_container_width=True, hide_index=True)

    st.markdown("**提示**：如果你想快速做“故事地图”，就在这里复制 Top 10 条原评论。")
