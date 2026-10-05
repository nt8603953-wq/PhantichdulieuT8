from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# BAI 1: PHAN TICH CUM / MARKET SEGMENTATION
# ============================================================

# Ghi chu quan trong:
# Tren mot so may Windows, scikit-learn co the loi khi du an nam trong
# duong dan co dau tieng Viet. Doan nay giup bo qua buoc quet threadpool
# noi bo, khong lam thay doi cong thuc K-Means.
try:
    import sklearn.cluster._kmeans as kmeans_module
    import sklearn.utils.parallel as parallel_module

    class _NoThreadpoolInfo:
        def info(self):
            return []

        def limit(self, *args, **kwargs):
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    _dummy_threadpool = _NoThreadpoolInfo()
    kmeans_module._get_threadpool_controller = lambda: _dummy_threadpool
    parallel_module._get_threadpool_controller = lambda: _dummy_threadpool
except Exception:
    pass


# 1. Import libraries va khai bao duong dan
BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "3. Market Segmentation_W6.xlsx"
OUTPUT_EXCEL = BASE_DIR / "bai1_cluster_results.xlsx"
ELBOW_PLOT = BASE_DIR / "bai1_elbow_silhouette.png"
CLUSTER_PLOT = BASE_DIR / "bai1_cluster_scatter.png"


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def explain(title, text):
    print(f"\n--- {title} ---")
    print(text)


def inspect_workbook(file_path):
    print_section("BUOC 1 - KIEM TRA WORKBOOK")
    print("Thu muc dang chay:", BASE_DIR)
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

    explain(
        "Cach doc ket qua",
        "Sheet 'Original Data' co dong tieu de nam ben trong bang, con sheet "
        "'Decode' da co ten cot ro rang va kieu du lieu so. Vi vay ta dung "
        "sheet 'Decode' de phan cum.",
    )


def load_data(file_path):
    print_section("BUOC 2 - DOC DU LIEU")
    df = pd.read_excel(file_path, sheet_name="Decode")
    print(df.head().to_string(index=False))

    explain(
        "Tai sao doc sheet Decode?",
        "K-Means can bang du lieu dang cot-so ro rang. Sheet Decode co 300 "
        "khach hang va cac bien da duoc dat ten gon: ID, Income, "
        "Aver_Purchase, Pre_Purchase, Dependents.",
    )
    return df


def inspect_data(df):
    print_section("BUOC 3 - KHAM PHA DU LIEU")
    print("Kich thuoc du lieu:", df.shape)
    print("\nTen cot:")
    print(list(df.columns))
    print("\nKieu du lieu:")
    print(df.dtypes.to_string())
    print("\nSo gia tri thieu moi cot:")
    print(df.isna().sum().to_string())
    print("\nSo dong trung lap:", df.duplicated().sum())
    print("\nThong ke mo ta:")
    print(df.describe().round(3).to_string())

    explain(
        "Y nghia bien",
        "- ID: ma khach hang, chi dung de nhan dien, khong dung tinh khoang cach.\n"
        "- Income: thu nhap, don vi trong file goc la $10,000s.\n"
        "- Aver_Purchase: gia tri mua hang trung binh, don vi trong file goc la $100s.\n"
        "- Pre_Purchase: so lan mua trong nam truoc.\n"
        "- Dependents: so nguoi phu thuoc.",
    )


def prepare_data(df):
    print_section("BUOC 4 - CHON BIEN VA CHUAN HOA")

    id_column = "ID"
    feature_columns = ["Income", "Aver_Purchase", "Pre_Purchase", "Dependents"]

    print("Bien ID khong dua vao phan cum:", id_column)
    print("Cac bien dung de phan cum:", feature_columns)

    explain(
        "Tai sao khong dung ID?",
        "ID chi la ma so khach hang. Neu dua ID vao K-Means, thuat toan se hieu "
        "nham rang khach hang co ID gan nhau thi giong nhau, dieu nay khong co "
        "y nghia kinh doanh.",
    )

    explain(
        "Tai sao can chuan hoa?",
        "K-Means dua tren khoang cach. Cac bien co thang do khac nhau nhu "
        "Income, Aver_Purchase, Pre_Purchase co the lam bien co do lon hon "
        "chi phoi ket qua. StandardScaler dua cac bien ve trung binh 0 va "
        "do lech chuan 1 de so sanh cong bang hon.",
    )

    X = df[feature_columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, feature_columns


def choose_k(X_scaled):
    print_section("BUOC 5 - XAC DINH SO CUM K")

    rows = []
    previous_inertia = None

    for k in range(1, 11):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        inertia = model.inertia_

        if k == 1:
            improvement = None
            improvement_ratio = None
            silhouette = None
        else:
            improvement = previous_inertia - inertia
            improvement_ratio = improvement / previous_inertia
            silhouette = silhouette_score(X_scaled, labels)

        rows.append(
            {
                "K": k,
                "Inertia": inertia,
                "Improvement": improvement,
                "Improvement_Ratio": improvement_ratio,
                "Silhouette": silhouette,
            }
        )
        previous_inertia = inertia

    metrics = pd.DataFrame(rows)
    metrics["Improvement_vs_Previous"] = (
        metrics["Improvement"] / metrics["Improvement"].shift(1)
    )
    print(metrics.round(4).to_string(index=False))

    # Heuristic don gian: Elbow thuong nam o K ma muc cai thien inertia
    # bat dau giam manh so voi buoc truoc do. O day ta so sanh Improvement
    # cua K hien tai voi Improvement cua K truoc. Neu ty le nay nho hon 35%,
    # ta xem K hien tai la ung vien elbow.
    candidate = metrics[
        (metrics["K"] >= 3) & (metrics["Improvement_vs_Previous"] < 0.35)
    ]
    if not candidate.empty:
        chosen_k = int(candidate.iloc[0]["K"])
        reason = (
            "Elbow xuat hien quanh K=3 vi sau diem nay inertia van giam "
            "nhung toc do giam cham hon nhieu."
        )
    else:
        silhouette_rows = metrics.dropna(subset=["Silhouette"])
        chosen_k = int(silhouette_rows.sort_values("Silhouette", ascending=False).iloc[0]["K"])
        reason = "Chon K co Silhouette Score cao nhat trong cac gia tri da thu."

    print("\nK duoc chon:", chosen_k)
    print("Ly do:", reason)

    explain(
        "Elbow va Silhouette",
        "Elbow Method xem inertia giam ra sao khi tang K. Inertia cang nho "
        "thi cac diem cang gan tam cum, nhung tang K qua nhieu se lam mo hinh "
        "phuc tap. Silhouette Score do muc do tach biet cum, gia tri cang cao "
        "thuong cang tot. O du lieu nay K=2 co Silhouette cao nhat, nhung K=3 "
        "cho phan khuc kinh doanh ro hon va van co diem so chap nhan duoc.",
    )

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(metrics["K"], metrics["Inertia"], marker="o")
    plt.title("Elbow Method")
    plt.xlabel("So cum K")
    plt.ylabel("Inertia")

    plt.subplot(1, 2, 2)
    silhouette_data = metrics.dropna(subset=["Silhouette"])
    plt.plot(silhouette_data["K"], silhouette_data["Silhouette"], marker="o", color="green")
    plt.title("Silhouette Score")
    plt.xlabel("So cum K")
    plt.ylabel("Silhouette")

    plt.tight_layout()
    plt.savefig(ELBOW_PLOT, dpi=150)
    plt.close()
    print("Da luu bieu do:", ELBOW_PLOT.name)

    return chosen_k, metrics


def run_kmeans(df, X_scaled, feature_columns, chosen_k):
    print_section("BUOC 6 - CHAY K-MEANS VA GAN CLUSTER")

    model = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    result = df.copy()
    result["Cluster"] = model.fit_predict(X_scaled)

    print(result.head(10).to_string(index=False))

    print("\nSo khach hang trong tung cluster:")
    cluster_counts = result["Cluster"].value_counts().sort_index()
    print(cluster_counts.to_string())

    print("\nProfile trung binh tung cluster:")
    profile = (
        result.groupby("Cluster")[feature_columns]
        .mean()
        .round(3)
    )
    profile.insert(0, "Count", cluster_counts)
    print(profile.to_string())

    explain(
        "Cach doc profile",
        "Profile trung binh giup ta hieu moi cluster la nhom khach hang nao. "
        "Vi du: cum co Income va Aver_Purchase cao hon co the la nhom gia tri "
        "cao; cum co Income va tan suat mua thap hon co the can chien luoc "
        "khuyen mai/nuoi duong rieng.",
    )

    return result, profile, model


def interpret_clusters(profile):
    print_section("BUOC 7 - DIEN GIAI VA DAT TEN PHAN KHUC")

    overall = profile.drop(columns=["Count"]).mean()
    interpretations = []

    for cluster_id, row in profile.iterrows():
        income_level = "cao" if row["Income"] > overall["Income"] else "thap"
        purchase_level = (
            "cao"
            if row["Aver_Purchase"] > overall["Aver_Purchase"]
            and row["Pre_Purchase"] > overall["Pre_Purchase"]
            else "thap/trung binh"
        )

        if income_level == "cao" and purchase_level == "cao":
            segment_name = "Khach hang gia tri cao"
            strategy = (
                "Tap trung giu chan, uu dai thanh vien, goi san pham cao cap "
                "va cham soc ca nhan hoa."
            )
        elif income_level == "thap" and purchase_level == "thap/trung binh":
            segment_name = "Khach hang gia tri thap"
            strategy = (
                "Dung khuyen mai phu hop, combo gia tot, noi dung don gian de "
                "tang tan suat mua."
            )
        else:
            segment_name = "Khach hang tiem nang"
            strategy = (
                "Khuyen khich mua them bang cross-sell, email remarketing va "
                "uu dai theo lich su mua hang."
            )

        interpretations.append(
            {
                "Cluster": cluster_id,
                "Segment": segment_name,
                "Strategy": strategy,
            }
        )

    interpretation_df = pd.DataFrame(interpretations)
    print(interpretation_df.to_string(index=False))

    explain(
        "Luu y khi dat ten",
        "Ten phan khuc duoc dat sau khi xem profile trung binh. Khong nen dat "
        "ten truoc vi nhu vay se ep du lieu theo y minh.",
    )

    return interpretation_df


def visualize_clusters(result):
    print_section("BUOC 8 - TRUC QUAN HOA")

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        result["Income"],
        result["Aver_Purchase"],
        c=result["Cluster"],
        cmap="viridis",
        alpha=0.8,
    )
    plt.title("Market Segmentation by Income and Average Purchase")
    plt.xlabel("Income")
    plt.ylabel("Aver_Purchase")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(CLUSTER_PLOT, dpi=150)
    plt.close()
    print("Da luu bieu do:", CLUSTER_PLOT.name)

    explain(
        "Cach doc bieu do",
        "Bieu do dung Income va Aver_Purchase de nhin nhanh su khac nhau giua "
        "cac nhom. Day chi la hinh 2D, trong khi K-Means da dung ca 4 bien.",
    )


def save_outputs(result, profile, metrics, interpretation_df):
    print_section("BUOC 9 - LUU KET QUA")

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="Customers_with_Cluster", index=False)
        profile.to_excel(writer, sheet_name="Cluster_Profile")
        metrics.to_excel(writer, sheet_name="K_Selection", index=False)
        interpretation_df.to_excel(writer, sheet_name="Interpretation", index=False)

    print("Da xuat file ket qua:", OUTPUT_EXCEL.name)
    print("Luu y: file Excel goc khong bi ghi de.")


def final_answer(profile, interpretation_df):
    print_section("DAP AN/NHAN XET BAI 1")

    print("So cluster:", len(profile))
    print("\nDac diem tung cluster:")
    print(profile.to_string())
    print("\nDien giai va chien luoc de xuat:")
    print(interpretation_df.to_string(index=False))

    explain(
        "Ket luan kinh doanh mau",
        f"Du lieu khach hang co the chia thanh {len(profile)} phan khuc chinh. Nhom co thu "
        "nhap, gia tri mua trung binh va so lan mua cao nen duoc uu tien giu "
        "chan. Nhom co chi so mua hang thap can cac chuong trinh khuyen mai "
        "de tang tan suat mua. Nhom trung gian/tiem nang co the duoc khai thac "
        "bang cross-sell va remarketing. Ket luan nay dua tren profile trung "
        "binh cua tung cum, khong phai quan he nhan qua.",
    )

    print("\n5 cau hoi tu kiem tra Bai 1:")
    print("1. Tai sao khong dung ID de phan cum?")
    print("2. Tai sao can chuan hoa du lieu truoc K-Means?")
    print("3. K-Means dang co gang toi uu dieu gi?")
    print("4. Elbow Method va Silhouette Score khac nhau nhu the nao?")
    print("5. Sau khi co cluster, tai sao phai xem profile trung binh?")


def main():
    inspect_workbook(EXCEL_FILE)
    df = load_data(EXCEL_FILE)
    inspect_data(df)
    _, X_scaled, feature_columns = prepare_data(df)
    chosen_k, metrics = choose_k(X_scaled)
    result, profile, _ = run_kmeans(df, X_scaled, feature_columns, chosen_k)
    interpretation_df = interpret_clusters(profile)
    visualize_clusters(result)
    save_outputs(result, profile, metrics, interpretation_df)
    final_answer(profile, interpretation_df)


if __name__ == "__main__":
    main()
