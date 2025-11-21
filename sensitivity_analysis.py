import pandas as pd

df = pd.read_csv("opt_results_revised\all_trials.csv")

regimes = {
    "0–5": df[df["user_attrs_n_comm"] <= 5],
    "6–10": df[(df["user_attrs_n_comm"] >= 6) & (df["user_attrs_n_comm"] <= 9)],
    "11–15": df[(df["user_attrs_n_comm"] >= 10) & (df["user_attrs_n_comm"] <= 15)],
}

results = []

for name, subset in regimes.items():
    if len(subset) < 5:
        continue

    corr = subset[[
        "user_attrs_alpha",
        "user_attrs_beta",
        "user_attrs_gamma",
        "user_attrs_modularity"
    ]].corr()["user_attrs_modularity"].drop("user_attrs_modularity")

    results.append([
        name,
        corr["user_attrs_alpha"],
        corr["user_attrs_beta"],
        corr["user_attrs_gamma"],
        len(subset)
    ])

df_out = pd.DataFrame(
    results,
    columns=["Regime", "Alpha_corr", "Beta_corr", "Gamma_corr", "Count"]
)

print(df_out)
