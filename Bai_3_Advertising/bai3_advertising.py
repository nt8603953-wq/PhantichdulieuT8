from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linprog


# ============================================================
# BAI 3: PHAN BO NGAN SACH QUANG CAO TOI UU
# ============================================================

# 1. Import libraries va khai bao duong dan
BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "1. Digital-Advertising.xlsx"
OUTPUT_EXCEL = BASE_DIR / "bai3_advertising_results.xlsx"

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def explain(title, text):
    print(f"\n--- {title} ---")
    print(text)


def inspect_workbook(file_path):
    print_section("BUOC 1 - KIEM TRA WORKBOOK")
    print("Thu muc bai 3:", BASE_DIR)
    print("File Excel:", file_path.name)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay file: {file_path}\n"
            "Hay kiem tra lai ten file va vi tri file Excel."
        )

    excel = pd.ExcelFile(file_path)
    print("Cac sheet trong workbook:", excel.sheet_names)

    for sheet in excel.sheet_names:
        preview = pd.read_excel(file_path, sheet_name=sheet, header=None)
        print(f"\nSheet: {sheet}")
        print("So dong, so cot:", preview.shape)
        print(preview.head(12).to_string(index=True, header=False))

    explain(
        "Nhan xet workbook",
        "Sheet Model chua bang ROI, bien quyet dinh, objective va constraints. "
        "Sheet Conversion Data chua du lieu conversion theo ngay. Sheet "
        "Sensitivity Report chua ket qua Solver va thong tin shadow price/"
        "reduced cost.",
    )


def read_model_sheet(file_path):
    print_section("BUOC 2 - DOC MO HINH TU SHEET MODEL")

    model_raw = pd.read_excel(file_path, sheet_name="Model", header=None)

    # Bang ROI nam o dong Excel 6-9, cot B-G.
    channel_table = model_raw.iloc[5:9, 1:7].copy()
    channel_table.columns = [
        "Ad_Type",
        "Cost",
        "Conversions",
        "Conversions_per_Cost",
        "Revenue",
        "ROI",
    ]
    channel_table = channel_table.reset_index(drop=True)

    # Bang decision variables nam o dong Excel 16-19, cot B-C.
    excel_solution = model_raw.iloc[15:19, 1:3].copy()
    excel_solution.columns = ["Ad_Type", "Excel_Budget_Allocation"]
    excel_solution = excel_solution.reset_index(drop=True)

    total_budget = float(model_raw.iat[14, 10])
    min_display = float(model_raw.iat[17, 10])
    min_youtube = float(model_raw.iat[20, 10])
    min_display_search = float(model_raw.iat[23, 10])
    gmail_max = float(model_raw.iat[29, 10])
    min_conversions = float(model_raw.iat[32, 10])
    excel_objective = float(model_raw.iat[14, 5])

    print("Bang kenh quang cao:")
    print(channel_table.to_string(index=False))
    print("\nNghiem Solver trong Excel:")
    print(excel_solution.to_string(index=False))
    print("\nCac tham so rang buoc doc tu file:")
    print("Tong ngan sach:", total_budget)
    print("Display toi thieu:", min_display)
    print("YouTube toi thieu:", min_youtube)
    print("Search + Display toi thieu:", min_display_search)
    print("Gmail toi da:", gmail_max)
    print("Conversions toi thieu:", min_conversions)
    print("Objective Excel:", excel_objective)

    explain(
        "Decision variables",
        "x_Search, x_Display, x_YouTube, x_Gmail la so tien phan bo cho tung "
        "kenh quang cao. Don vi tien theo file Excel.",
    )

    return {
        "channel_table": channel_table,
        "excel_solution": excel_solution,
        "total_budget": total_budget,
        "min_display": min_display,
        "min_youtube": min_youtube,
        "min_display_search": min_display_search,
        "gmail_max": gmail_max,
        "min_conversions": min_conversions,
        "excel_objective": excel_objective,
    }


def read_conversion_data(file_path):
    print_section("BUOC 3 - KIEM TRA CONVERSION DATA")

    # Dong Excel 3 la header that cua bang conversion.
    conversion_df = pd.read_excel(
        file_path,
        sheet_name="Conversion Data",
        header=2,
    )

    print("Kich thuoc Conversion Data:", conversion_df.shape)
    print(conversion_df.head().to_string(index=False))
    print("\nTong conversions theo kenh:")
    print(conversion_df.drop(columns=["Date"]).sum().to_string())

    explain(
        "Y nghia",
        "Sheet nay cho thay so conversion lich su theo ngay. Trong sheet Model, "
        "cac he so Conversions / Cost da duoc tinh san va duoc dung lam rang "
        "buoc toi thieu 1,500 conversions.",
    )

    return conversion_df


def build_math_model(params):
    print_section("BUOC 4 - VIET MO HINH TOAN HOC")

    table = params["channel_table"]
    channels = table["Ad_Type"].tolist()
    roi = table["ROI"].astype(float).to_numpy()
    conversion_rate = table["Conversions_per_Cost"].astype(float).to_numpy()

    channel_index = {name: i for i, name in enumerate(channels)}
    s = channel_index["Search"]
    d = channel_index["Display"]
    y = channel_index["YouTube"]
    g = channel_index["Gmail"]

    print("Decision variables:")
    for name in channels:
        print(f"x_{name} = ngan sach phan bo cho {name}")

    print("\nObjective:")
    print(
        "Max Z = "
        + " + ".join([f"{roi[i]:.6f}*x_{channels[i]}" for i in range(len(channels))])
    )

    print("\nSubject to:")
    print("x_Search + x_Display + x_YouTube + x_Gmail <= 5000")
    print("x_Display >= 1000")
    print("x_YouTube >= 200")
    print("x_Search + x_Display >= 3000")
    print("x_YouTube <= x_Gmail")
    print("x_Gmail <= 500")
    print(
        " + ".join(
            [f"{conversion_rate[i]:.6f}*x_{channels[i]}" for i in range(len(channels))]
        )
        + " >= 1500"
    )
    print("x_i >= 0")

    explain(
        "Day co phai Linear Programming khong?",
        "Co. Objective la tong he so nhan voi bien quyet dinh, cac constraints "
        "deu la bat dang thuc tuyen tinh theo ngan sach tung kenh. Vi vay co "
        "the dung scipy.optimize.linprog.",
    )

    return channels, roi, conversion_rate, (s, d, y, g)


def solve_optimization(params, channels, roi, conversion_rate, indexes):
    print_section("BUOC 5 - GIAI BAI TOAN BANG SCIPY")

    s, d, y, g = indexes
    n = len(channels)

    # scipy linprog la ham minimize. Muon maximize Net Revenue thi minimize -ROI.
    c = -roi

    A_ub = []
    b_ub = []
    constraint_names = []
    senses = []

    # 1. Tong ngan sach <= budget
    A_ub.append([1, 1, 1, 1])
    b_ub.append(params["total_budget"])
    constraint_names.append("Total ad cost should be within budget")
    senses.append("<=")

    # 2. Display >= minimum  -> -Display <= -minimum
    row = [0] * n
    row[d] = -1
    A_ub.append(row)
    b_ub.append(-params["min_display"])
    constraint_names.append("Minimum of $1000 in Display ads")
    senses.append(">=")

    # 3. YouTube >= minimum
    row = [0] * n
    row[y] = -1
    A_ub.append(row)
    b_ub.append(-params["min_youtube"])
    constraint_names.append("Minimum of $200 in YouTube ads")
    senses.append(">=")

    # 4. Search + Display >= 60% budget
    row = [0] * n
    row[s] = -1
    row[d] = -1
    A_ub.append(row)
    b_ub.append(-params["min_display_search"])
    constraint_names.append("At least 60% of budget in Display and Search")
    senses.append(">=")

    # 5. YouTube <= Gmail -> YouTube - Gmail <= 0
    row = [0] * n
    row[y] = 1
    row[g] = -1
    A_ub.append(row)
    b_ub.append(0)
    constraint_names.append("YouTube ads should not be more than Gmail ads")
    senses.append("<=")

    # 6. Gmail <= 10% budget
    row = [0] * n
    row[g] = 1
    A_ub.append(row)
    b_ub.append(params["gmail_max"])
    constraint_names.append("Gmail ads should not be more than 10% of budget")
    senses.append("<=")

    # 7. Conversions >= 1500
    A_ub.append((-conversion_rate).tolist())
    b_ub.append(-params["min_conversions"])
    constraint_names.append("Minimum of 1,500 conversions")
    senses.append(">=")

    result = linprog(
        c=c,
        A_ub=np.array(A_ub, dtype=float),
        b_ub=np.array(b_ub, dtype=float),
        bounds=[(0, None)] * n,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(f"Khong tim duoc nghiem toi uu: {result.message}")

    solution = pd.DataFrame(
        {
            "Ad_Type": channels,
            "Python_Budget_Allocation": result.x,
            "ROI": roi,
            "Conversions_per_Cost": conversion_rate,
            "Net_Revenue_Contribution": result.x * roi,
            "Conversions": result.x * conversion_rate,
        }
    )

    objective = float(solution["Net_Revenue_Contribution"].sum())
    total_budget = float(solution["Python_Budget_Allocation"].sum())
    total_conversions = float(solution["Conversions"].sum())

    print("Nghiem toi uu Python:")
    print(solution.round(6).to_string(index=False))
    print("\nTong ngan sach:", round(total_budget, 6))
    print("Tong conversions:", round(total_conversions, 6))
    print("Objective - Net Revenue:", round(objective, 6))

    explain(
        "Cach doc nghiem toi uu",
        "Kenh nao duoc phan bo nhieu hon la kenh giup objective tot hon trong "
        "bo rang buoc hien co. Khong co nghia la ngoai doi nen chi chay dung "
        "nhu vay mai mai; day la ket qua theo he so ROI/conversion trong file.",
    )

    model_info = {
        "A_ub": np.array(A_ub, dtype=float),
        "b_ub": np.array(b_ub, dtype=float),
        "constraint_names": constraint_names,
        "senses": senses,
        "objective": objective,
        "total_budget": total_budget,
        "total_conversions": total_conversions,
        "linprog_result": result,
    }

    return solution, model_info


def evaluate_constraints(solution, model_info, conversion_rate):
    print_section("BUOC 6 - KIEM TRA RANG BUOC BINDING")

    x = solution["Python_Budget_Allocation"].to_numpy()
    rows = []

    for name, sense, row, rhs in zip(
        model_info["constraint_names"],
        model_info["senses"],
        model_info["A_ub"],
        model_info["b_ub"],
    ):
        transformed_lhs = float(row @ x)

        if sense == "<=":
            original_lhs = transformed_lhs
            original_rhs = rhs
            slack = original_rhs - original_lhs
        else:
            # Constraints >= were converted by multiplying by -1.
            original_lhs = -transformed_lhs
            original_rhs = -rhs
            slack = original_lhs - original_rhs

        rows.append(
            {
                "Constraint": name,
                "Sense": sense,
                "LHS": original_lhs,
                "RHS": original_rhs,
                "Slack": slack,
                "Binding": abs(slack) < 1e-5,
            }
        )

    constraints = pd.DataFrame(rows)
    print(constraints.round(6).to_string(index=False))

    explain(
        "Binding constraint",
        "Binding nghia la rang buoc dang sat nut: LHS bang RHS, slack gan 0. "
        "Neu thay doi rang buoc binding, objective toi uu co the thay doi.",
    )

    return constraints


def read_sensitivity_report(file_path):
    print_section("BUOC 7 - DOC SENSITIVITY REPORT")

    raw = pd.read_excel(file_path, sheet_name="Sensitivity Report", header=None)

    decision = raw.iloc[12:16, 1:8].copy()
    decision.columns = [
        "Cell",
        "Name",
        "Final_Value",
        "Reduced_Cost",
        "Objective_Coefficient",
        "Allowable_Increase",
        "Allowable_Decrease",
    ]
    decision = decision.reset_index(drop=True)

    constraints = raw.iloc[20:27, 1:8].copy()
    constraints.columns = [
        "Cell",
        "Name",
        "Final_Value",
        "Shadow_Price",
        "RHS",
        "Allowable_Increase",
        "Allowable_Decrease",
    ]
    constraints = constraints.reset_index(drop=True)

    print("Decision Variable Cells:")
    print(decision.to_string(index=False))
    print("\nConstraints:")
    print(constraints.to_string(index=False))

    explain(
        "Reduced cost",
        "Reduced cost cho biet voi bien quyet dinh dang o bien/thap hon, he so "
        "objective can cai thien bao nhieu de bien do co the tham gia vao "
        "nghiem toi uu. Trong file nay reduced cost cua cac bien la 0 vi cac "
        "kenh deu duoc phan bo ngan sach duong.",
    )

    explain(
        "Shadow price",
        "Shadow price cho biet objective toi uu thay doi bao nhieu neu RHS cua "
        "mot rang buoc tang them 1 don vi, trong khoang allowable increase/"
        "decrease. Gia tri nay chi nen doc cho cac rang buoc binding.",
    )

    return decision, constraints


def compare_with_excel(solution, params, model_info):
    print_section("BUOC 8 - SO SANH PYTHON VOI EXCEL SOLVER")

    comparison = params["excel_solution"].merge(solution, on="Ad_Type", how="left")
    comparison["Difference"] = (
        comparison["Python_Budget_Allocation"]
        - comparison["Excel_Budget_Allocation"]
    )

    print(comparison.round(6).to_string(index=False))

    objective_diff = model_info["objective"] - params["excel_objective"]
    print("\nObjective Python:", round(model_info["objective"], 6))
    print("Objective Excel:", round(params["excel_objective"], 6))
    print("Chenh lech objective:", round(objective_diff, 6))

    if comparison["Difference"].abs().max() < 1e-5 and abs(objective_diff) < 1e-5:
        print("Ket qua Python trung voi ket qua Solver trong Excel.")
    else:
        print(
            "Ket qua co chenh lech. Can kiem tra lai objective, constraints, "
            "don vi, dau bat dang thuc va bien quyet dinh."
        )

    return comparison


def save_outputs(
    params,
    solution,
    constraints,
    sensitivity_decision,
    sensitivity_constraints,
    comparison,
):
    print_section("BUOC 9 - LUU KET QUA")

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        params["channel_table"].to_excel(writer, sheet_name="Channel_Data", index=False)
        solution.to_excel(writer, sheet_name="Python_Optimal", index=False)
        constraints.to_excel(writer, sheet_name="Constraint_Check", index=False)
        sensitivity_decision.to_excel(writer, sheet_name="Sensitivity_Decision", index=False)
        sensitivity_constraints.to_excel(writer, sheet_name="Sensitivity_Constraints", index=False)
        comparison.to_excel(writer, sheet_name="Python_vs_Excel", index=False)

    print("Da xuat file ket qua:", OUTPUT_EXCEL.name)
    print("Luu y: file Excel goc khong bi ghi de.")


def final_answer(solution, constraints, model_info):
    print_section("DAP AN/NHAN XET BAI 3")

    print("Mo hinh toi uu: Linear Programming - toi da hoa Net Revenue.")
    print("\nNgan sach toi uu tung kenh:")
    print(
        solution[["Ad_Type", "Python_Budget_Allocation"]]
        .round(2)
        .to_string(index=False)
    )
    print("\nTong ngan sach:", round(model_info["total_budget"], 2))
    print("Tong conversions:", round(model_info["total_conversions"], 2))
    print("Net Revenue toi uu:", round(model_info["objective"], 2))

    print("\nRang buoc binding:")
    binding = constraints[constraints["Binding"]]
    if binding.empty:
        print("Khong co rang buoc binding.")
    else:
        print(binding[["Constraint", "LHS", "RHS"]].round(4).to_string(index=False))

    explain(
        "Y nghia kinh doanh",
        "Ket qua de xuat dung het ngan sach 5,000 va phan bo manh vao cac kenh "
        "co ROI cao trong khi van thoa man cac yeu cau ve Display, YouTube, "
        "Gmail, ty trong Search+Display va tong conversions. Cac rang buoc "
        "binding la nhung dieu kien dang gioi han nghiem toi uu.",
    )

    explain(
        "Han che",
        "Mo hinh gia dinh ROI va conversion rate khong doi theo ngan sach. Trong "
        "thuc te, khi tang ngan sach, hieu qua bien co the giam dan. Vi vay ket "
        "qua nen duoc xem la de xuat theo du lieu lich su va rang buoc trong "
        "file, khong phai quy tac co dinh cho moi chien dich.",
    )

    print("\n5 cau hoi tu kiem tra Bai 3:")
    print("1. Bien quyet dinh trong bai nay la gi?")
    print("2. Objective function dang toi da hoa dieu gi?")
    print("3. Tai sao bai nay la Linear Programming?")
    print("4. Binding constraint co nghia la gi?")
    print("5. Shadow price nen duoc doc nhu the nao?")


def main():
    inspect_workbook(EXCEL_FILE)
    params = read_model_sheet(EXCEL_FILE)
    read_conversion_data(EXCEL_FILE)
    channels, roi, conversion_rate, indexes = build_math_model(params)
    solution, model_info = solve_optimization(
        params,
        channels,
        roi,
        conversion_rate,
        indexes,
    )
    constraints = evaluate_constraints(solution, model_info, conversion_rate)
    sensitivity_decision, sensitivity_constraints = read_sensitivity_report(EXCEL_FILE)
    comparison = compare_with_excel(solution, params, model_info)
    save_outputs(
        params,
        solution,
        constraints,
        sensitivity_decision,
        sensitivity_constraints,
        comparison,
    )
    final_answer(solution, constraints, model_info)


if __name__ == "__main__":
    main()
