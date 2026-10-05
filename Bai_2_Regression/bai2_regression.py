from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor


# ============================================================
# BAI 2: PHAN TICH NHAN TO TAC DONG DEN GIA NHA
# ============================================================

# 1. Import libraries va khai bao duong dan
BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "2. Hồi quy tuyến tính.xlsx"
OUTPUT_EXCEL = BASE_DIR / "bai2_regression_results.xlsx"
PLOT_CORR = BASE_DIR / "bai2_correlation_heatmap.png"
PLOT_ACTUAL_PRED = BASE_DIR / "bai2_actual_vs_predicted.png"
PLOT_RESIDUALS = BASE_DIR / "bai2_residuals.png"
PLOT_COEFFICIENTS = BASE_DIR / "bai2_coefficients.png"

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
sns.set_theme(style="whitegrid")


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def explain(title, text):
    print(f"\n--- {title} ---")
    print(text)


def inspect_workbook(file_path):
    print_section("BUOC 1 - KIEM TRA WORKBOOK")
    print("Thu muc bai 2:", BASE_DIR)
    print("File Excel:", file_path.name)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay file: {file_path}\n"
            "Hay kiem tra lai ten file va vi tri file Excel."
        )

    excel = pd.ExcelFile(file_path)
    print("Cac sheet trong workbook:", excel.sheet_names)

    for sheet in excel.sheet_names:
        preview = pd.read_excel(file_path, sheet_name=sheet)
        print(f"\nSheet: {sheet}")
        print("So dong, so cot:", preview.shape)
        print("Ten cot:", list(preview.columns))
        print(preview.head().to_string(index=False))

    return excel.sheet_names[0]


def load_data(file_path, sheet_name):
    print_section("BUOC 2 - DOC DU LIEU")
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Doi ten cot de code de viet hon. Khong thay doi file Excel goc.
    df = df.rename(
        columns={
            "LIST.PRICE": "LIST_PRICE",
            "HOME.TYPE": "HOME_TYPE",
            "LOT.SIZE": "LOT_SIZE",
            "YEAR.BUILT": "YEAR_BUILT",
            "DAYS.ON.MARKET": "DAYS_ON_MARKET",
            "PARKING.TYPE": "PARKING_TYPE",
        }
    )

    print(df.head().to_string(index=False))
    explain(
        "Bien phu thuoc Y",
        "Y duoc chon la LIST_PRICE vi muc tieu cua bai la phan tich cac yeu "
        "to tac dong den gia nha. Cac bien con lai la ung vien bien doc lap X.",
    )
    return df


def inspect_data(df):
    print_section("BUOC 3 - KHAM PHA DU LIEU")
    print("Kich thuoc du lieu:", df.shape)
    print("\nTen cot:")
    print(list(df.columns))
    print("\nKieu du lieu:")
    print(df.dtypes.to_string())
    print("\nGia tri thieu:")
    print(df.isna().sum().to_string())
    print("\nSo dong trung lap:", df.duplicated().sum())
    print("\nThong ke mo ta:")
    print(df.describe().round(3).to_string())

    for col in ["HOME_TYPE", "PARKING_TYPE", "ZIP"]:
        print(f"\nTan suat cua {col}:")
        print(df[col].value_counts(dropna=False).to_string())

    explain(
        "Y nghia bien",
        "- LIST_PRICE: gia niem yet cua can nha, la bien can giai thich.\n"
        "- SQFT: dien tich nha.\n"
        "- HOME_TYPE: loai nha, vi du Condo, Townhouse, Single Family Residential.\n"
        "- BEDS, BATHS: so phong ngu va phong tam.\n"
        "- LOT_SIZE: dien tich dat. Bien nay co thieu du lieu.\n"
        "- YEAR_BUILT: nam xay dung.\n"
        "- DAYS_ON_MARKET: so ngay bat dong san nam tren thi truong.\n"
        "- PARKING_TYPE: co/khong co loai bai dau xe theo ma hoa trong file.\n"
        "- ZIP: khu vuc ma buu chinh, nen xem nhu bien phan loai.",
    )


def prepare_data(df):
    print_section("BUOC 4 - CHUAN BI DU LIEU CHO HOI QUY")

    model_df = df.copy()

    # LOT_SIZE thieu 94 dong. Neu xoa het, ta mat kha nhieu quan sat.
    # Cach lam don gian: dien median va them bien co LOT_SIZE_MISSING de
    # mo hinh biet dong nao tung bi thieu LOT_SIZE.
    model_df["LOT_SIZE_MISSING"] = model_df["LOT_SIZE"].isna().astype(int)
    model_df["LOT_SIZE_FILLED"] = model_df["LOT_SIZE"].fillna(model_df["LOT_SIZE"].median())

    y = model_df["LIST_PRICE"]

    numeric_features = [
        "SQFT",
        "BEDS",
        "BATHS",
        "LOT_SIZE_FILLED",
        "LOT_SIZE_MISSING",
        "YEAR_BUILT",
        "DAYS_ON_MARKET",
    ]
    categorical_features = ["HOME_TYPE", "PARKING_TYPE", "ZIP"]

    X_numeric = model_df[numeric_features]
    X_categorical = pd.get_dummies(
        model_df[categorical_features].astype("category"),
        drop_first=True,
        dtype=float,
    )
    X = pd.concat([X_numeric, X_categorical], axis=1)
    X = sm.add_constant(X)

    print("Bien Y:", "LIST_PRICE")
    print("Bien X dang so:", numeric_features)
    print("Bien categorical tao dummy:", categorical_features)
    print("\nCac cot X sau khi tao dummy:")
    print(list(X.columns))

    explain(
        "Tai sao phai tao dummy?",
        "Hoi quy tuyen tinh can dau vao la so. HOME_TYPE va ZIP la nhom/loai, "
        "khong nen dua truc tiep vao nhu so lien tuc. Dummy variables bien moi "
        "nhom thanh cot 0/1. drop_first=True giu lai mot nhom lam nhom co so "
        "de tranh trung lap thong tin.",
    )

    explain(
        "Reference category",
        "Voi dummy variables, nhom bi bo ra la nhom co so. He so cua cac dummy "
        "duoc doc la muc chenh lech trung binh so voi nhom co so, khi cac yeu "
        "to khac khong doi.",
    )

    return model_df, y, X, numeric_features


def run_simple_regression(df):
    print_section("BUOC 5 - HOI QUY TUYEN TINH DON")

    y = df["LIST_PRICE"]
    X_simple = sm.add_constant(df[["SQFT"]])
    model = sm.OLS(y, X_simple).fit()

    print(model.summary())
    explain(
        "Cach doc hoi quy don",
        "Mo hinh don chi xem quan he giua gia nha va SQFT. He so SQFT cho biet "
        "khi dien tich tang 1 don vi, gia niem yet thay doi trung binh bao nhieu "
        "don vi. Tuy nhien mo hinh don chua kiem soat cac yeu to khac.",
    )
    return model


def run_multiple_regression(y, X):
    print_section("BUOC 6 - HOI QUY TUYEN TINH DA BIEN")

    model = sm.OLS(y, X).fit()
    print(model.summary())

    equation = "LIST_PRICE = beta0 + beta1*SQFT + beta2*BEDS + ... + epsilon"
    print("\nDang mo hinh tong quat:")
    print(equation)

    explain(
        "Cach doc coefficient",
        "Voi bien so: Khi X tang 1 don vi, LIST_PRICE thay doi trung binh beta "
        "don vi, giu cac yeu to khac khong doi. Voi bien dummy: beta la muc "
        "chenh lech trung binh so voi nhom co so.",
    )
    return model


def coefficient_table(model):
    table = pd.DataFrame(
        {
            "Coefficient": model.params,
            "Std_Error": model.bse,
            "t_value": model.tvalues,
            "p_value": model.pvalues,
            "CI_Lower": model.conf_int()[0],
            "CI_Upper": model.conf_int()[1],
        }
    )
    return table.round(4)


def explain_model_results(model, coef_table):
    print_section("BUOC 7 - DOC KET QUA MO HINH")

    print("R-squared:", round(model.rsquared, 4))
    print("Adjusted R-squared:", round(model.rsquared_adj, 4))

    significant = coef_table[
        (coef_table.index != "const") & (coef_table["p_value"] < 0.05)
    ].sort_values("p_value")

    print("\nBien co y nghia thong ke o muc 5%:")
    if significant.empty:
        print("Khong co bien nao co p-value < 0.05.")
    else:
        print(significant.to_string())

    print("\nBang he so:")
    print(coef_table.to_string())

    explain(
        "p-value va muc y nghia 5%",
        "p-value nho hon 0.05 thuong duoc xem la co y nghia thong ke o muc 5%. "
        "Nhung khong nen ket luan chi dua vao p-value; can xem dau he so, do lon "
        "he so, y nghia kinh doanh va han che cua du lieu.",
    )

    explain(
        "R-squared va Adjusted R-squared",
        "R-squared cho biet mo hinh giai thich duoc bao nhieu phan tram bien "
        "dong cua LIST_PRICE. Adjusted R-squared dieu chinh theo so bien doc lap, "
        "nen huu ich hon khi mo hinh co nhieu bien.",
    )


def diagnostics(model, X):
    print_section("BUOC 8 - KIEM TRA VAN DE MO HINH")

    residuals = model.resid
    fitted = model.fittedvalues

    bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, model.model.exog)
    print("Breusch-Pagan p-value:", round(bp_pvalue, 4))
    if bp_pvalue < 0.05:
        print("Co dau hieu heteroskedasticity: phuong sai sai so co the khong dong nhat.")
    else:
        print("Chua thay bang chung ro ve heteroskedasticity o muc 5%.")

    vif_rows = []
    X_for_vif = X.drop(columns=["const"], errors="ignore")
    for i, col in enumerate(X_for_vif.columns):
        vif_rows.append(
            {
                "Variable": col,
                "VIF": variance_inflation_factor(X_for_vif.values, i),
            }
        )
    vif = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
    print("\nVIF:")
    print(vif.round(3).to_string(index=False))

    influence = OLSInfluence(model)
    cooks = pd.Series(influence.cooks_distance[0], name="Cooks_Distance")
    top_influence = cooks.sort_values(ascending=False).head(5)
    print("\n5 quan sat co Cook's distance cao nhat:")
    print(top_influence.round(4).to_string())

    explain(
        "VIF la gi?",
        "VIF dung de kiem tra da cong tuyen. VIF cao nghia la mot bien X co "
        "the dang duoc giai thich manh boi cac bien X khac, lam he so hoi quy "
        "kem on dinh hon. Nguong hay dung la 5 hoac 10, tuy boi canh.",
    )

    return fitted, residuals, vif, top_influence


def visualize(df, model, fitted, residuals, coef_table):
    print_section("BUOC 9 - TRUC QUAN HOA")

    plt.figure(figsize=(9, 7))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=False)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOT_CORR, dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=model.model.endog, y=fitted)
    plt.xlabel("Actual LIST_PRICE")
    plt.ylabel("Predicted LIST_PRICE")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(PLOT_ACTUAL_PRED, dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=fitted, y=residuals)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(PLOT_RESIDUALS, dpi=150)
    plt.close()

    coef_for_plot = coef_table.drop(index="const", errors="ignore").copy()
    coef_for_plot = coef_for_plot.reindex(
        coef_for_plot["Coefficient"].abs().sort_values(ascending=False).head(12).index
    )
    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=coef_for_plot.reset_index().rename(columns={"index": "Variable"}),
        x="Coefficient",
        y="Variable",
        orient="h",
    )
    plt.title("Top Coefficients by Absolute Size")
    plt.tight_layout()
    plt.savefig(PLOT_COEFFICIENTS, dpi=150)
    plt.close()

    print("Da luu bieu do:", PLOT_CORR.name)
    print("Da luu bieu do:", PLOT_ACTUAL_PRED.name)
    print("Da luu bieu do:", PLOT_RESIDUALS.name)
    print("Da luu bieu do:", PLOT_COEFFICIENTS.name)


def save_outputs(df, coef_table, vif, top_influence, simple_model, multiple_model):
    print_section("BUOC 10 - LUU KET QUA")

    summary_metrics = pd.DataFrame(
        [
            {
                "Model": "Simple: LIST_PRICE ~ SQFT",
                "R_squared": simple_model.rsquared,
                "Adj_R_squared": simple_model.rsquared_adj,
                "AIC": simple_model.aic,
                "BIC": simple_model.bic,
            },
            {
                "Model": "Multiple regression",
                "R_squared": multiple_model.rsquared,
                "Adj_R_squared": multiple_model.rsquared_adj,
                "AIC": multiple_model.aic,
                "BIC": multiple_model.bic,
            },
        ]
    )

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Prepared_Data", index=False)
        coef_table.to_excel(writer, sheet_name="Coefficients")
        vif.to_excel(writer, sheet_name="VIF", index=False)
        top_influence.to_frame().to_excel(writer, sheet_name="Influential_Rows")
        summary_metrics.to_excel(writer, sheet_name="Model_Metrics", index=False)

    print("Da xuat file ket qua:", OUTPUT_EXCEL.name)
    print("Luu y: file Excel goc khong bi ghi de.")


def final_answer(model, coef_table):
    print_section("DAP AN/NHAN XET BAI 2")

    significant = coef_table[
        (coef_table.index != "const") & (coef_table["p_value"] < 0.05)
    ].copy()

    print("Mo hinh hoi quy:")
    print("LIST_PRICE = beta0 + beta1*SQFT + beta2*BEDS + ... + epsilon")
    print("\nR-squared:", round(model.rsquared, 4))
    print("Adjusted R-squared:", round(model.rsquared_adj, 4))

    print("\nCac yeu to co y nghia thong ke o muc 5%:")
    if significant.empty:
        print("Khong co bien nao co y nghia thong ke o muc 5%.")
    else:
        for variable, row in significant.iterrows():
            sign = "tang" if row["Coefficient"] > 0 else "giam"
            print(
                f"- {variable}: he so {row['Coefficient']:.4f}, p-value "
                f"{row['p_value']:.4f}, tac dong {sign} LIST_PRICE khi giu "
                "cac yeu to khac khong doi."
            )

    explain(
        "Nhan xet kinh doanh mau",
        "Mo hinh cho thay gia nha co lien he manh voi cac dac diem nhu dien "
        "tich, so phong, loai nha va khu vuc neu cac bien nay co y nghia thong "
        "ke. Can phan biet lien he trong mo hinh voi quan he nhan qua: du lieu "
        "hien co chua du de khang dinh mot yeu to gay ra thay doi gia nha.",
    )

    explain(
        "Han che",
        "Mo hinh hoi quy tuyen tinh gia dinh quan he tuyen tinh va phuong sai "
        "sai so kha on dinh. LOT_SIZE co nhieu gia tri thieu nen ket qua lien "
        "quan den bien nay can doc can than. Ngoai ra gia nha co the con phu "
        "thuoc vao vi tri chi tiet, tinh trang nha, truong hoc, noi that va "
        "nhieu yeu to khac khong co trong file.",
    )

    print("\n5 cau hoi tu kiem tra Bai 2:")
    print("1. Tai sao LIST_PRICE duoc chon lam bien Y?")
    print("2. Tai sao HOME_TYPE va ZIP phai tao dummy variables?")
    print("3. Cau 'giu cac yeu to khac khong doi' co nghia la gi?")
    print("4. p-value < 0.05 noi len dieu gi va khong noi len dieu gi?")
    print("5. VIF dung de kiem tra van de nao trong hoi quy?")


def main():
    sheet_name = inspect_workbook(EXCEL_FILE)
    df = load_data(EXCEL_FILE, sheet_name)
    inspect_data(df)
    model_df, y, X, _ = prepare_data(df)
    simple_model = run_simple_regression(model_df)
    multiple_model = run_multiple_regression(y, X)
    coef_table = coefficient_table(multiple_model)
    explain_model_results(multiple_model, coef_table)
    fitted, residuals, vif, top_influence = diagnostics(multiple_model, X)
    visualize(model_df, multiple_model, fitted, residuals, coef_table)
    save_outputs(model_df, coef_table, vif, top_influence, simple_model, multiple_model)
    final_answer(multiple_model, coef_table)


if __name__ == "__main__":
    main()
