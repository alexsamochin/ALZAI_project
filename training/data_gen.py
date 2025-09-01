
"""
Generates synthetic dataset and saves in parquet format for memory savings
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

np.random.seed(42)

@dataclass
class SimConfig:
    n_patients: int = 1000000 
    years: Tuple[int, ...] = (2019, 2020, 2021, 2022, 2023)
    target_prevalence: float = 0.07
    chunk_size: int = 100_000
    out_csv_gz: str = "data/synth_clinical_patient_year.parquet"

cfg = SimConfig()

def logistic(x):
    return 1 / (1 + np.exp(-x))

def sample_categories(n, categories, probs):
    return np.random.choice(categories, size=n, p=probs)

def generate_patient_base(n_patients: int):
    sex = sample_categories(n_patients, ["F", "M"], [0.52, 0.48])
    race = sample_categories(n_patients, ["White", "Black", "Asian", "Hispanic", "Other"],
                             [0.6, 0.12, 0.07, 0.17, 0.04])
    region = sample_categories(n_patients, ["NE", "MW", "S", "W"], [0.18, 0.22, 0.37, 0.23])
    payer = sample_categories(n_patients, ["Commercial", "Medicare", "Medicaid", "SelfPay"], [0.55, 0.25, 0.18, 0.02])
    birth_year = np.random.randint(1935, 2005, size=n_patients)
    patient_id = np.arange(1, n_patients + 1, dtype=np.int64)
    risk_trait = np.random.normal(0, 1, size=n_patients)
    adherence_trait = np.random.normal(0, 1, size=n_patients)
    socioeconomic = np.random.normal(0, 1, size=n_patients)
    base = pd.DataFrame({
        "patient_id": patient_id,
        "sex": sex,
        "race": race,
        "region": region,
        "payer": payer,
        "birth_year": birth_year,
        "risk_trait": risk_trait,
        "adherence_trait": adherence_trait,
        "socioeconomic_z": socioeconomic,
    })
    return base

def generate_yearly(base: pd.DataFrame, years: List[int], chunk_rows: int):
    n_patients = base.shape[0]
    n_years = len(years)

    enroll_base = logistic(
        0.5
        - 0.1 * (base["socioeconomic_z"].values < -1)
        + 0.15 * (base["payer"].values == "Commercial")
        + 0.10 * (base["payer"].values == "Medicaid")
    )

    lab_random = np.random.normal(0, 1, size=(n_patients,))
    comorb_base = logistic( -0.6 + 0.8 * base["risk_trait"].values )

    all_pairs = [(pid, y) for pid in base["patient_id"].values for y in years]
    id_to_idx = dict(zip(base["patient_id"].values.tolist(), range(n_patients)))

    def build_chunk(pairs):
        m = len(pairs)
        pid = np.fromiter((p for p,_ in pairs), dtype=np.int64, count=m)
        yr = np.fromiter((y for _,y in pairs), dtype=np.int32, count=m)
        idx = np.fromiter((id_to_idx[i] for i in pid), dtype=np.int64, count=m)

        year_effect = (yr - years[0]) * 0.05
        enroll_prob = np.clip(enroll_base[idx] + year_effect, 0.05, 0.98)
        enrolled = np.random.binomial(1, enroll_prob, size=m).astype(bool)

        age = yr - base["birth_year"].values[idx]
        age = np.clip(age, 0, 120)

        p_diab = np.clip(comorb_base[idx] + 0.01*(age-50) + 0.05*(base["socioeconomic_z"].values[idx] < -1), 0, 1)
        p_htn  = np.clip(0.35 + 0.012*(age-45) + 0.4*base["risk_trait"].values[idx], 0, 1)
        p_ckd  = np.clip(0.05 + 0.008*(age-60) + 0.2*base["risk_trait"].values[idx], 0, 1)

        has_diabetes = np.random.binomial(1, logistic(p_diab - 0.5)).astype(bool)
        has_htn = np.random.binomial(1, logistic(p_htn - 0.5)).astype(bool)
        has_ckd = np.random.binomial(1, logistic(p_ckd - 0.5)).astype(bool)

        a1c = np.random.normal(5.4 + 1.4*has_diabetes + 0.2*base["risk_trait"].values[idx] + 0.1*lab_random[idx], 0.6, size=m)
        ldl = np.random.normal(120 - 8*(base["adherence_trait"].values[idx]>0) + 10*has_diabetes, 20, size=m)
        sbp = np.random.normal(122 + 10*has_htn + 0.2*(age-50), 12, size=m)
        bmi = np.random.normal(27 + 2.0*(base["socioeconomic_z"].values[idx]<-0.5) + 1.2*has_diabetes, 4, size=m)

        outpatient_visits = np.random.poisson(lam=np.clip(2.5 + 0.6*has_htn + 0.8*has_diabetes + 0.3*base["risk_trait"].values[idx], 0.2, 10), size=m)
        er_visits = np.random.poisson(lam=np.clip(0.2 + 0.15*has_diabetes + 0.1*has_ckd, 0.01, 3), size=m)
        inpatient_admits_prev = np.random.binomial(1, p=np.clip(0.05 + 0.03*base["risk_trait"].values[idx], 0.01, 0.6), size=m)
        meds_count = np.random.poisson(lam=np.clip(3 + 2*has_htn + 2*has_diabetes + 1*has_ckd - 0.5*base["adherence_trait"].values[idx], 0.5, 20), size=m)

        cost_scale = np.clip(800 + 120*outpatient_visits + 600*er_visits + 3000*inpatient_admits_prev + 150*meds_count, 200, None)
        annual_cost = np.random.gamma(shape=2.2, scale=cost_scale/2.2)

        smoker = np.random.binomial(1, p=np.clip(0.18 + 0.06*(base["socioeconomic_z"].values[idx]<-0.7) - 0.05*(base["adherence_trait"].values[idx]>0.5), 0.02, 0.6)).astype(bool)
        exercise_min_week = np.clip(np.random.normal(90 + 25*base["adherence_trait"].values[idx] - 15*has_diabetes - 10*smoker, 40, size=m), 0, 600).astype(int)

        lp = (
            -3.0
            + 0.03*(age-50)
            + 0.6*has_diabetes
            + 0.5*has_htn
            + 0.7*has_ckd
            + 0.35*smoker
            + 0.02*np.clip(a1c-5.5, 0, None)
            + 0.015*np.clip(sbp-120, 0, None)
            + 0.25*(er_visits>=1)
            + 0.35*inpatient_admits_prev
            + 0.004*np.clip(annual_cost/100.0, 0, 500)
            + 0.2*base["risk_trait"].values[idx]
            - 0.15*base["adherence_trait"].values[idx]
        )

        df = pd.DataFrame({
            "patient_id": pid,
            "calendar_year": yr,
            "enrolled": enrolled,
            "age": age.astype(np.int16),
            "sex": base["sex"].values[idx],
            "race": base["race"].values[idx],
            "region": base["region"].values[idx],
            "payer": base["payer"].values[idx],
            "risk_trait": base["risk_trait"].values[idx].round(3),
            "adherence_trait": base["adherence_trait"].values[idx].round(3),
            "socioeconomic_z": base["socioeconomic_z"].values[idx].round(3),
            "has_diabetes": has_diabetes,
            "has_htn": has_htn,
            "has_ckd": has_ckd,
            "a1c": a1c.round(2),
            "ldl": ldl.round(1),
            "sbp": sbp.round(1),
            "bmi": bmi.round(1),
            "outpatient_visits": outpatient_visits.astype(np.int16),
            "er_visits": er_visits.astype(np.int16),
            "inpatient_prev": inpatient_admits_prev.astype(bool),
            "meds_count": meds_count.astype(np.int16),
            "annual_cost": annual_cost.round(2),
            "smoker": smoker,
            "exercise_min_week": exercise_min_week.astype(np.int16),
            "_lp": lp,
        })

        df = df[df["enrolled"]].reset_index(drop=True)
        return df

    chunk_rows = chunk_rows
    df_chunks = []
    for start in range(0, len(all_pairs), chunk_rows):
        chunk_pairs = all_pairs[start:start+chunk_rows]
        df_chunk = build_chunk(chunk_pairs)
        df_chunks.append(df_chunk)

    df_all = pd.concat(df_chunks, ignore_index=True)
    return df_all

base = generate_patient_base(cfg.n_patients)
df = generate_yearly(base, list(cfg.years), cfg.chunk_size)

def calibrate_intercept(lp: np.ndarray, target: float, tol: float = 1e-4, max_iter: int = 100):
    lo, hi = -10.0, 10.0
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        p = logistic(lp + mid).mean()
        if abs(p - target) < tol:
            return mid
        if p < target:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo+hi)

c = calibrate_intercept(df["_lp"].values, cfg.target_prevalence)
df["acute_event"] = (np.random.rand(len(df)) < logistic(df["_lp"].values + c)).astype(bool)
df = df.drop(columns=["_lp"])
cols = [c for c in df.columns if c != "acute_event"]
df = df[cols + ["acute_event"]]

# Save parquet
df.to_parquet(cfg.out_csv_gz, engine="fastparquet", index=False)

# Report a quick summary and preview
summary = {
    "rows": len(df),
    "n_patients": cfg.n_patients,
    "years": cfg.years,
    "positive_prevalence": float(df["acute_event"].mean()),
    "n_columns": df.shape[1],
    "csv_gz_path": cfg.out_csv_gz
}

summary
