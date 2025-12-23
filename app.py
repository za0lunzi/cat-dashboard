import os
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="MVP Definer (MoSCoW)", layout="wide")

# =========================
# Config / Column Map
# =========================
REPO_DEFAULT_XLSX = "cat_litter_labelled_comments_merged.xlsx"

# 你数据里如果列名不同，在这里改映射即可
COL = {
    "issue": "Issue_L2",
    "severity": "Impact_Severity",         # Critical / Failure / Annoyance / Minor
    "phenom": "Phenomenon_Type",           # Defect / Friction / Constraint (你也可以有 Delighter/Positive)
    "review_id": None,                     # 如果有 Review_ID，填列名；没有就用每行当一条记录
    "quote": "Review_Text",                # 可选：原文/摘要列名（没有也能跑）
    "subject": "Subject",                  # 可选：分群字段
    "context": "Context",                  # 可选
    "household": "Household",              # 可选
    "phase": "Interaction_Phase",          # 可选
}

SEVERITY_WEIGHT = {
    "Critical": 10,
    "Failure": 5,
    "Annoyance": 1,
    "Minor": 0,
}

P0_THRESHOLD = 0.50
NOISE_MINOR_THRESHOLD = 0.90
NOISE_CRITICAL_THRESHOLD = 0.05

# =========================
# Helpers
# =========================
def _norm_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 标准化字符串列
    for k in ["issue", "severity", "phenom", "quote", "subject", "context", "household", "phase"]:
        col = COL.get(k)
        if col and col in df.columns:
            df[col] = df[col].map(_norm_str)

    # 必需列检查
    required = [COL["issue"], COL["severity"], COL["phenom"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(
            "缺少必需列，当前缺少：\n"
            + "\n".join([f"- {c}" for c in missing])
            + "\n\n请确认你的表里至少有：Issue_L2 / Impact_Severity / Phenomenon_Type（或在 COL 映射里改成你的列名）"
        )
        st.stop()

    # severity 统一大小写
    df[COL["severity"]] = df[COL["severity"]].str.title()

    # 只保留我们认识的 severity；其余标为 Minor(0) 或直接丢弃，这里选择：丢弃
    df = df[df[COL["severity"]].isin(SEVERITY_WEIGHT.keys())].copy()

    return df

def _get_total_reviews(df: pd.DataFrame) -> int:
    # 如果有 review_id 列，用唯一 review 数；否则默认一行=一条 insight/review
    rid = COL.get("review_id")
    if rid and rid in df.columns:
        return df[rid].nunique()
    return len(df)

def _severity_dist(df: pd.DataFrame) -> pd.DataFrame:
    # per issue severity distribution
    issue = COL["issue"]
    sev = COL["severity"]
    pivot = (
        df.pivot_table(index=issue, columns=sev, values=sev, aggfunc="count", fill_value=0)
        .reset_index()
    )
    for s in SEVERITY_WEIGHT.keys():
        if s not in pivot.columns:
            pivot[s] = 0
    pivot["Total_Issue_Occurrences"] = pivot[list(SEVERITY_WEIGHT.keys())].sum(axis=1)
    # shares
    for s in SEVERITY_WEIGHT.keys():
        pivot[f"{s}_share"] = pivot[s] / pivot["Total_Issue_Occurrences"].clip(lower=1)
    pivot["CriticalFailure_share"] = (pivot["Critical"] + pivot["Failure"]) / pivot["Total_Issue_Occurrences"].clip(lower=1)
    return pivot

def _compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    issue = COL["issue"]
    sev = COL["severity"]
    phen = COL["phenom"]

    total_reviews = _get_total_reviews(df)

    df = df.copy()
    df["sev_weight"] = df[sev].map(SEVERITY_WEIGHT).fillna(0)

    # prevalence: issue count / total reviews
    issue_counts = df.groupby(issue, as_index=False).size().rename(columns={"size": "Issue_Count"})
    issue_counts["Prevalence"] = issue_counts["Issue_Count"] / max(total_reviews, 1)

    # pain index: avg weighted severity
    pain = df.groupby(issue, as_index=False)["sev_weight"].mean().rename(columns={"sev_weight": "Pain_Index"})

    # top phenomenon type for issue (mode)
    def _mode(s):
        if len(s) == 0:
            return ""
        return s.value_counts().index[0]

    phen_mode = df.groupby(issue)[phen].apply(_mode).reset_index().rename(columns={phen: "Phenomenon_Mode"})

    # severity dist for rules
    dist = _severity_dist(df)

    out = issue_counts.merge(pain, on=issue, how="left").merge(phen_mode, on=issue, how="left").merge(dist, on=issue, how="left")
    out["Pain_Index"] = out["Pain_Index"].fillna(0)

    # Decision flags
    out["P0_Blocker"] = out["CriticalFailure_share"] >= P0_THRESHOLD
    out["Noise"] = (out["Minor_share"] >= NOISE_MINOR_THRESHOLD) & (out["Critical_share"] <= NOISE_CRITICAL_THRESHOLD)

    # P1 heuristic: annoyance 高但 critical 很低
    # 这里给默认阈值，可在 UI 调整
    out["P1_Optimize"] = (out["Annoyance_share"] >= 0.50) & (out["Critical_share"] <= 0.05)

    # Display label
    def _label(row):
        if row["Noise"]:
            return "Noise"
        if row["P0_Blocker"]:
            return "P0"
        if row["P1_Optimize"]:
            return "P1"
        return "P2"

    out["Priority"] = out.apply(_label, axis=1)

    return out, total_reviews

def _filter_df(df_raw: pd.DataFrame, filters: dict) -> pd.DataFrame:
    df = df_raw.copy()
    # optional segment filters
    for key in ["subject", "context", "household", "phase"]:
        col = COL.get(key)
        if col and col in df.columns:
            chosen = filters.get(key, [])
            if chosen:
                df = df[df[col].isin(chosen)].copy()
    return df

def _build_markdown_prd(df_raw: pd.DataFrame, chosen_issues: list, title: str) -> str:
    issue_col = COL["issue"]
    sev_col = COL["severity"]
    phen_col = COL["phenom"]
    quote_col = COL.get("quote")

    lines = []
    lines.append(f"# {title}\n")
    lines.append("## Must-have / Selected Issues\n")

    for i, iss in enumerate(chosen_issues, start=1):
        sub = df_raw[df_raw[issue_col] == iss].copy()
        if sub.empty:
            continue
        # frequency
        freq = len(sub)
        # severity dist
        sev_counts = sub[sev_col].value_counts().to_dict()
        cf_share = (sev_counts.get("Critical", 0) + sev_counts.get("Failure", 0)) / max(freq, 1)
        # phenomenon
        phen = sub[phen_col].value_counts().index[0] if len(sub) else ""

        lines.append(f"### {i}. {iss}")
        lines.append(f"- Phenomenon_Type: **{phen}**")
        lines.append(f"- Frequency (rows): **{freq}**")
        lines.append(f"- Critical+Failure share: **{cf_share:.0%}**")
        lines.append(f"- Severity breakdown: {', '.join([f'{k}:{v}' for k,v in sev_counts.items()])}")

        # representative quotes (optional)
        if quote_col and quote_col in sub.columns:
            quotes = [q for q in sub[quote_col].dropna().astype(str).tolist() if q.strip()]
            quotes = quotes[:5]
            if quotes:
                lines.append("- Representative quotes:")
                for q in quotes:
                    lines.append(f"  - {q}")

        lines.append("")  # blank line

    return "\n".join(lines)

# =========================
# UI
# =========================
st.title("MoSCoW 动态定义阵列（个人决策外挂）")
st.caption("目标：用最少的图回答：MVP 必须包含什么 / 资源投向哪里 / 哪些问题属于噪音或可接受的 trade-off")

with st.sidebar:
    st.header("数据源")
    uploaded = st.file_uploader("上传 Excel（可选）", type=["xlsx"])
    use_default = st.toggle(f"使用仓库内默认文件：{REPO_DEFAULT_XLSX}", value=True if not uploaded else False)

    st.divider()
    st.header("筛选器（可选）")

@st.cache_data(show_spinner=False)
def load_df_from_xlsx(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

@st.cache_data(show_spinner=False)
def load_df_from_uploaded(file) -> pd.DataFrame:
    return pd.read_excel(file)

# Load data
if uploaded is not None and not use_default:
    df0 = load_df_from_uploaded(uploaded)
else:
    default_path = Path(__file__).parent / REPO_DEFAULT_XLSX
    if not default_path.exists():
        st.error(f"找不到默认文件：{REPO_DEFAULT_XLSX}。请上传 Excel 或把文件放到 repo 根目录。")
        st.stop()
    df0 = load_df_from_xlsx(str(default_path))

df0 = _ensure_columns(df0)

# Build filter choices
filters = {}
with st.sidebar:
    for key, label in [("subject", "Subject"), ("context", "Context"), ("household", "Household"), ("phase", "Interaction_Phase")]:
        col = COL.get(key)
        if col and col in df0.columns:
            options = sorted([x for x in df0[col].dropna().unique().tolist() if str(x).strip()])
            chosen = st.multiselect(label, options, default=[])
            filters[key] = chosen

df = _filter_df(df0, filters)

metrics, total_reviews = _compute_metrics(df)

# Controls
with st.sidebar:
    st.divider()
    st.header("阈值（Must-Have 切割）")

    min_prev = st.slider("Min_Frequency（Prevalence）", 0.0, 0.50, 0.02, 0.005)
    min_pain = st.slider("Min_Severity（Pain Index）", 0.0, 10.0, 2.0, 0.1)

    st.caption("解释：Prevalence=该 Issue 出现次数 / 总评论（或总行数）；Pain Index=按 Critical/Failure/Annoyance/Minor 加权的平均痛苦指数")

# Apply rule-based filtering
must_have = metrics[
    (~metrics["Noise"]) &
    (metrics["Prevalence"] >= min_prev) &
    (metrics["Pain_Index"] >= min_pain)
].copy()

# Main layout
colA, colB = st.columns([2, 1], gap="large")

with colA:
    st.subheader("MoSCoW 散点（Issue 粒度）")

    plot_df = metrics.copy()
    plot_df["Phenomenon_Mode"] = plot_df["Phenomenon_Mode"].replace("", "Unknown")

    # 优先级用符号区分
    symbol_map = {"P0": "diamond", "P1": "circle", "P2": "square", "Noise": "x"}
    plot_df["symbol"] = plot_df["Priority"].map(symbol_map).fillna("circle")

    fig = px.scatter(
        plot_df,
        x="Prevalence",
        y="Pain_Index",
        color="Phenomenon_Mode",
        symbol="Priority",
        hover_data={
            COL["issue"]: True,
            "Issue_Count": True,
            "Prevalence": ":.2%",
            "Pain_Index": ":.2f",
            "CriticalFailure_share": ":.0%",
            "Priority": True,
            "Noise": True,
        },
        title="X=普遍性(Prevalence)，Y=痛苦指数(Pain Index)，颜色=Phenomenon_Type(Mode)，符号=优先级(P0/P1/P2/Noise)"
    )
    # Must-have 区域辅助线
    fig.add_vline(x=min_prev, line_dash="dash")
    fig.add_hline(y=min_pain, line_dash="dash")

    st.plotly_chart(fig, use_container_width=True)

    st.info(f"总评论基数（分母）：{total_reviews} ；当前 Issue 数：{len(metrics)} ；Must-Have（按阈值）={len(must_have)}")

with colB:
    st.subheader("Must-Have 列表（可导出 PRD）")

    # 给你一个“个人用”最稳的选择方式：多选 Issue（不依赖 plotly 选择事件）
    must_issue_options = must_have[COL["issue"]].tolist()
    chosen = st.multiselect("选择要导出的 Issues（默认：全部 Must-Have）", options=must_issue_options, default=must_issue_options)

    st.markdown("**快速结论（用于你自己决策）**")
    p0_cnt = int((must_have["Priority"] == "P0").sum())
    p1_cnt = int((must_have["Priority"] == "P1").sum())
    st.write(f"- Must-Have 中 P0（Critical+Failure≥50%）: **{p0_cnt}**")
    st.write(f"- Must-Have 中 P1（Annoyance 高、Critical 低）: **{p1_cnt}**")
    st.write(f"- 已识别 Noise（默认不纳入 Must-Have）: **{int(metrics['Noise'].sum())}**")

    # 展示表格
    show_cols = [
        COL["issue"], "Prevalence", "Pain_Index", "Phenomenon_Mode",
        "CriticalFailure_share", "Priority", "Issue_Count"
    ]
    st.dataframe(
        must_have[show_cols].sort_values(["Priority", "Pain_Index", "Prevalence"], ascending=[True, False, False]),
        use_container_width=True,
        height=420
    )

    st.divider()
    title = st.text_input("PRD 标题", value="MVP Must-Have 清单（从 MoSCoW 阈值导出）")
    md = _build_markdown_prd(df_raw=df, chosen_issues=chosen, title=title)

    st.download_button(
        "下载 PRD（Markdown）",
        data=md.encode("utf-8"),
        file_name="mvp_prd.md",
        mime="text/markdown"
    )

# Data QA section
with st.expander("数据 QA / 列名检查（点开看看，避免之后讨论跑偏）", expanded=False):
    st.write("当前列名：", list(df0.columns))
    st.write("当前映射 COL：", COL)
    st.write("Severity 权重：", SEVERITY_WEIGHT)
    st.write("P0 阈值 Critical+Failure≥：", P0_THRESHOLD)
    st.write("Noise 规则：Minor≥90% 且 Critical≤5%（默认）")
