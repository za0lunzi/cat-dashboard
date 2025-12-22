import pandas as pd

# --- 1) 临时 Severity：由 Sentiment 推断（你未来若有 Impact_Severity 列，会自动跳过）
def add_impact_severity(df: pd.DataFrame) -> pd.DataFrame:
    if "Impact_Severity" in df.columns:
        return df

    if "Sentiment" in df.columns:
        def map_sev(x):
            try:
                x = float(x)
            except:
                return "Unknown"
            if x <= -4: return "Critical"
            if x <= -2: return "Major"
            if x <= 0:  return "Minor"
            if x <= 2:  return "Positive"
            return "Delighter"
        df["Impact_Severity"] = df["Sentiment"].apply(map_sev)
    else:
        df["Impact_Severity"] = "Unknown"

    return df


# --- 2) Defect vs Constraint（先启发式，后续你可以替换为映射表）
DEFECT_ISSUE_WORDS = {
    "leak", "leaking", "spill", "spilled", "drip",
    "broken", "crack", "cracked", "collapse", "collapsed",
    "unstable", "tear", "torn",
    "smell", "odor", "stink"
}
CONSTRAINT_ISSUE_WORDS = {
    "too small", "too big", "too soft", "too thin",
    "too heavy", "too tall", "too short"
}

def add_phenomenon_type(df: pd.DataFrame) -> pd.DataFrame:
    if "Phenomenon_Type" in df.columns:
        return df

    def decide(row):
        issue = str(row.get("Issue", "")).lower()
        sev = str(row.get("Impact_Severity", ""))
        ft = str(row.get("Feedback_Type", ""))

        severe_neg = sev in {"Critical", "Major", "Failure"}

        # 已定责 + 严重负面：优先 Defect（可执行整改）
        if severe_neg and ft in {"Quality_Defect", "Design_Flaw"}:
            return "Defect"

        # 严重负面 + 缺陷语义
        if severe_neg and any(w in issue for w in DEFECT_ISSUE_WORDS):
            return "Defect"

        # trade-off 语义 => Constraint
        if any(w in issue for w in CONSTRAINT_ISSUE_WORDS):
            return "Constraint"

        return "Unclear"

    df["Phenomenon_Type"] = df.apply(decide, axis=1)
    return df


# --- 3) 派生：Disposal 视角（一次性猫砂盆关键）
DISPOSAL_WORDS = {
    "throw", "threw", "trash", "dispose", "dump", "empty",
    "carry", "lift", "bag", "wrap",
    "spill", "spilled", "drip", "leak"
}

def add_user_journey_view(df: pd.DataFrame) -> pd.DataFrame:
    if "User_Journey_View" in df.columns:
        return df

    def is_disposal(row):
        phase = str(row.get("Interaction_Phase", "")).lower()
        txt = (str(row.get("text", "")) + " " + str(row.get("Issue", ""))).lower()

        phase_ok = phase in {"maintenance", "durability"}
        word_ok = any(w in txt for w in DISPOSAL_WORDS)

        return phase_ok and word_ok

    df["User_Journey_View"] = df.apply(lambda r: "Disposal" if is_disposal(r) else "Non-Disposal", axis=1)
    return df
