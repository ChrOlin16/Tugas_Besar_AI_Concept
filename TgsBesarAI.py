"""
Sistem Pakar Penjadwalan Shift Praktikum Otomatis
Pendekatan : Rule-Based Reasoning + Constraint Satisfaction Problem (CSP)
Framework  : Python + Streamlit

Jalankan:  streamlit run app.py
"""

import io
import random

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# 1. BASIS PENGETAHUAN (RULE BASE)
# ---------------------------------------------------------------------------
SHIFTS = [
    {"id": 1, "nama": "Shift 1", "hari": "Senin", "jam": "13:00 - 15:00"},
    {"id": 2, "nama": "Shift 2", "hari": "Selasa", "jam": "10:00 - 12:00"},
    {"id": 3, "nama": "Shift 3", "hari": "Kamis", "jam": "13:00 - 15:00"},
    {"id": 4, "nama": "Shift 4", "hari": "Jumat", "jam": "08:00 - 10:00"},
]
SEMUA_HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu"]
DEFAULT_KAPASITAS = 15  # Rule 1

KOLOM_WAJIB = ["Nama", "NIM", "Kelompok", "Hari Bentrok"]


def label_shift(s):
    return f"{s['nama']} - {s['hari']} ({s['jam']})"


# ---------------------------------------------------------------------------
# 2. PRA-PEMROSESAN DATA
# ---------------------------------------------------------------------------
def parse_hari(teks):
    """Ubah 'Senin, rabu' / 'Senin;Rabu' / kosong menjadi himpunan nama hari baku."""
    if teks is None or (isinstance(teks, float) and pd.isna(teks)):
        return set()
    teks = str(teks).replace(";", ",").replace("/", ",")
    hasil = set()
    for bagian in teks.split(","):
        h = bagian.strip().capitalize()
        if h in ("Jum'at", "Jumat"):
            h = "Jumat"
        if h in SEMUA_HARI:
            hasil.add(h)
    return hasil


def normalisasi_kolom(df):
    """Samakan nama kolom (abaikan huruf besar/kecil dan variasi umum)."""
    peta = {}
    for c in df.columns:
        k = str(c).strip().lower()
        if k in ("nama", "nama mahasiswa"):
            peta[c] = "Nama"
        elif k == "nim":
            peta[c] = "NIM"
        elif k in ("kelompok", "id kelompok", "group"):
            peta[c] = "Kelompok"
        elif k in ("hari bentrok", "hari bentrok/kuliah lain", "hari_bentrok", "bentrok"):
            peta[c] = "Hari Bentrok"
    df = df.rename(columns=peta)
    kurang = [k for k in KOLOM_WAJIB if k not in df.columns]
    if kurang:
        raise ValueError(
            "Kolom wajib tidak ditemukan: " + ", ".join(kurang)
            + ". Kolom yang dibutuhkan: " + ", ".join(KOLOM_WAJIB)
        )
    df = df[KOLOM_WAJIB].copy()
    df["NIM"] = df["NIM"].astype(str).str.strip()
    df["Kelompok"] = df["Kelompok"].astype(str).str.strip()
    df["Nama"] = df["Nama"].astype(str).str.strip()
    df["Hari Bentrok"] = df["Hari Bentrok"].fillna("")
    return df


# ---------------------------------------------------------------------------
# 3. GENERATOR DATA SIMULASI
# ---------------------------------------------------------------------------
NAMA_DEPAN = ["Andi", "Budi", "Citra", "Dewi", "Eka", "Fajar", "Gita", "Hadi", "Intan",
              "Joko", "Kiki", "Lina", "Maya", "Nanda", "Oki", "Putri", "Rani", "Sari",
              "Tono", "Umar", "Vina", "Wawan", "Yuni", "Zaki", "Dian", "Rizky", "Agus", "Siti"]
NAMA_BELAKANG = ["Pratama", "Saputra", "Wijaya", "Lestari", "Kusuma", "Santoso", "Hidayat",
                 "Nugroho", "Rahmawati", "Setiawan", "Permata", "Ramadhan", "Utami", "Firmansyah"]


def buat_data_dummy(jumlah_kelompok=14, seed=42):
    rnd = random.Random(seed)
    baris, nim = [], 2024001001
    for k in range(1, jumlah_kelompok + 1):
        for _ in range(rnd.randint(2, 4)):  # 2-4 anggota per kelompok
            n_bentrok = rnd.choices([0, 1, 2], weights=[45, 40, 15])[0]
            hari = rnd.sample(["Senin", "Selasa", "Rabu", "Kamis", "Jumat"], n_bentrok)
            baris.append({
                "Nama": f"{rnd.choice(NAMA_DEPAN)} {rnd.choice(NAMA_BELAKANG)}",
                "NIM": str(nim),
                "Kelompok": f"K{k:02d}",
                "Hari Bentrok": ", ".join(hari),
            })
            nim += 1
    return pd.DataFrame(baris)


# ---------------------------------------------------------------------------
# 4. MESIN INFERENSI (RULE-MATCHING + BACKTRACKING CSP)
# ---------------------------------------------------------------------------
def bangun_kelompok(df):
    """Langkah 1 & 2: kelompokkan data, gabungkan hari bentrok seluruh anggota."""
    kelompok = {}
    for _, r in df.iterrows():
        g = kelompok.setdefault(r["Kelompok"], {"anggota": [], "hari_bentrok": set()})
        g["anggota"].append(r["NIM"])
        g["hari_bentrok"] |= parse_hari(r["Hari Bentrok"])  # Rule 3: batasan gabungan
    for gid, g in kelompok.items():
        g["ukuran"] = len(g["anggota"])
        # Rule 2 + Rule 4: shift yang harinya tidak bentrok dengan siapa pun di kelompok
        g["shift_valid"] = [s["id"] for s in SHIFTS if s["hari"] not in g["hari_bentrok"]]
    return kelompok


def diagnosa_gagal(gid, g, beban, kapasitas):
    """Alasan yang bisa dibaca admin mengapa sebuah kelompok gagal dijadwalkan."""
    if g["ukuran"] > kapasitas:
        return f"Jumlah anggota ({g['ukuran']}) melebihi kapasitas shift ({kapasitas})."
    if not g["shift_valid"]:
        return ("Semua shift bentrok dengan hari berikut: "
                + ", ".join(sorted(g["hari_bentrok"], key=SEMUA_HARI.index)) + ".")
    return "Shift yang tidak bentrok sudah penuh (kapasitas tidak mencukupi)."


def selesaikan(kelompok, kapasitas, batas_node=200_000):
    """
    Mengembalikan (penugasan, gagal, metode)
    - penugasan: {kelompok: shift_id}
    - gagal    : {kelompok: alasan}
    Strategi: backtracking dengan heuristik MRV (kelompok paling terbatas dulu),
    pemilihan shift paling longgar lebih dulu agar distribusi seimbang.
    Jika tidak ada solusi lengkap, jatuh ke mode greedy parsial dan melaporkan
    kelompok yang gagal (kondisi deadlock).
    """
    urutan = sorted(kelompok, key=lambda x: (len(kelompok[x]["shift_valid"]), -kelompok[x]["ukuran"]))
    beban = {s["id"]: 0 for s in SHIFTS}
    penugasan = {}
    node = [0]

    def bt(i):
        if i == len(urutan):
            return True
        node[0] += 1
        if node[0] > batas_node:
            return False
        gid = urutan[i]
        g = kelompok[gid]
        for sid in sorted(g["shift_valid"], key=lambda s: beban[s]):
            if beban[sid] + g["ukuran"] <= kapasitas:  # Rule 1
                beban[sid] += g["ukuran"]
                penugasan[gid] = sid
                if bt(i + 1):
                    return True
                beban[sid] -= g["ukuran"]
                del penugasan[gid]
        return False

    if bt(0):
        return penugasan, {}, "Backtracking CSP (solusi lengkap)"

    # Fallback: deadlock -> penjadwalan parsial + laporan kegagalan
    beban = {s["id"]: 0 for s in SHIFTS}
    penugasan, gagal = {}, {}
    for gid in urutan:
        g = kelompok[gid]
        kandidat = [s for s in g["shift_valid"] if beban[s] + g["ukuran"] <= kapasitas]
        if kandidat:
            sid = min(kandidat, key=lambda s: beban[s])
            beban[sid] += g["ukuran"]
            penugasan[gid] = sid
        else:
            gagal[gid] = diagnosa_gagal(gid, g, beban, kapasitas)
    return penugasan, gagal, "Greedy parsial (deadlock terdeteksi)"


def susun_hasil(df, penugasan):
    peta = {s["id"]: s for s in SHIFTS}
    baris = []
    for _, r in df.iterrows():
        sid = penugasan.get(r["Kelompok"])
        if sid is None:
            continue
        s = peta[sid]
        baris.append({
            "Nama": r["Nama"], "NIM": r["NIM"], "Kelompok": r["Kelompok"],
            "Hari Bentrok": r["Hari Bentrok"] if r["Hari Bentrok"] else "-",
            "Shift": s["nama"], "Hari": s["hari"], "Jam": s["jam"],
        })
    out = pd.DataFrame(baris)
    if not out.empty:
        out = out.sort_values(["Shift", "Kelompok", "Nama"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# 5. VERIFIKASI (UJI BEBAS BENTROK)
# ---------------------------------------------------------------------------
def verifikasi(hasil, kapasitas):
    """Periksa ulang seluruh rule secara independen dari solver."""
    cek = []
    if hasil.empty:
        return [("Rule 1 - Kapasitas", True, "Tidak ada data terjadwal"),
                ("Rule 2 - Bebas bentrok individu", True, "Tidak ada data terjadwal"),
                ("Rule 3 - Kesatuan kelompok", True, "Tidak ada data terjadwal")]
    per_shift = hasil.groupby("Shift").size()
    ok1 = bool((per_shift <= kapasitas).all())
    cek.append(("Rule 1 - Kapasitas maksimal", ok1,
                f"Isi terbesar {per_shift.max()} dari {kapasitas} mahasiswa"))
    pelanggar = [r["NIM"] for _, r in hasil.iterrows()
                 if r["Hari"] in parse_hari(r["Hari Bentrok"] if r["Hari Bentrok"] != "-" else "")]
    cek.append(("Rule 2 - Bebas bentrok individu", len(pelanggar) == 0,
                "0 pelanggaran" if not pelanggar else f"{len(pelanggar)} pelanggaran"))
    terpisah = hasil.groupby("Kelompok")["Shift"].nunique()
    terpisah = terpisah[terpisah > 1]
    cek.append(("Rule 3 - Kesatuan kelompok", terpisah.empty,
                "Semua kelompok utuh" if terpisah.empty else f"{len(terpisah)} kelompok terpisah"))
    return cek


def ke_excel(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Jadwal")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 6. ANTARMUKA STREAMLIT
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Penjadwalan Shift Praktikum AI", page_icon="🗓️", layout="wide")

    st.title("🗓️ Sistem Pakar Penjadwalan Shift Praktikum Otomatis")
    st.markdown(
        "Penjadwalan shift praktikum secara manual rawan bentrok dan memakan waktu. "
        "Aplikasi ini memakai **penalaran berbasis aturan (rule-based)** dan "
        "**Constraint Satisfaction Problem (CSP)** untuk membagi kelompok ke shift "
        "secara otomatis, adil, dan bebas bentrok."
    )

    # ---- Sidebar
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        kapasitas = st.number_input("Kapasitas maksimal per shift", min_value=1, max_value=100,
                                    value=DEFAULT_KAPASITAS, step=1)
        st.divider()
        st.subheader("Data simulasi")
        n_kel = st.slider("Jumlah kelompok (dummy)", 4, 30, 14)
        seed = st.number_input("Seed acak", min_value=0, value=42, step=1)
        st.divider()
        st.subheader("Shift tersedia")
        for s in SHIFTS:
            st.caption(f"• {label_shift(s)}")
        st.divider()
        st.subheader("Aturan sistem")
        st.caption(f"1. Maksimal {kapasitas} mahasiswa per shift")
        st.caption("2. Tidak boleh pada hari bentrok individu")
        st.caption("3. Satu kelompok wajib satu shift")
        st.caption("4. Hanya 4 shift yang tersedia")

    # ---- Input data
    st.header("1. Input Data")
    if "df" not in st.session_state:
        st.session_state.df = None

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Opsi A - Unggah file CSV / Excel**")
        berkas = st.file_uploader("Kolom: Nama, NIM, Kelompok, Hari Bentrok",
                                  type=["csv", "xlsx", "xls"], label_visibility="collapsed")
        if berkas is not None:
            try:
                mentah = pd.read_csv(berkas) if berkas.name.lower().endswith(".csv") else pd.read_excel(berkas)
                st.session_state.df = normalisasi_kolom(mentah)
                st.session_state.sumber = f"Berkas: {berkas.name}"
            except Exception as e:
                st.error(f"Gagal membaca berkas: {e}")
    with col_b:
        st.markdown("**Opsi B - Muat data simulasi**")
        if st.button("🎲 Load Dummy Data", use_container_width=True):
            st.session_state.df = buat_data_dummy(n_kel, int(seed))
            st.session_state.sumber = f"Data simulasi ({n_kel} kelompok, seed {int(seed)})"
            st.session_state.pop("hasil", None)

    df = st.session_state.df
    if df is None:
        st.info("Unggah berkas atau klik **Load Dummy Data** untuk memulai.")
        return

    st.caption(f"Sumber data: {st.session_state.get('sumber', '-')}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Jumlah mahasiswa", len(df))
    m2.metric("Jumlah kelompok", df["Kelompok"].nunique())
    m3.metric("Kapasitas total", f"{int(kapasitas) * len(SHIFTS)}")
    with st.expander("Lihat data input", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ---- Eksekusi
    st.header("2. Eksekusi")
    if st.button("🚀 Jalankan Optimasi Jadwal AI", type="primary", use_container_width=True):
        kelompok = bangun_kelompok(df)
        penugasan, gagal, metode = selesaikan(kelompok, int(kapasitas))
        st.session_state.hasil = {
            "kelompok": kelompok, "penugasan": penugasan, "gagal": gagal,
            "metode": metode, "kapasitas": int(kapasitas),
            "tabel": susun_hasil(df, penugasan),
        }

    h = st.session_state.get("hasil")
    if not h:
        return

    # ---- Hasil
    st.header("3. Hasil Penjadwalan")
    tabel, gagal = h["tabel"], h["gagal"]
    if gagal:
        st.error(f"⚠️ Deadlock: {len(gagal)} kelompok gagal dijadwalkan. "
                 "Sesuaikan kapasitas atau data bentrok, lalu jalankan ulang.")
        st.dataframe(
            pd.DataFrame([{"Kelompok": k, "Anggota": h["kelompok"][k]["ukuran"],
                           "Hari Bentrok Gabungan": ", ".join(sorted(h["kelompok"][k]["hari_bentrok"], key=SEMUA_HARI.index)) or "-",
                           "Alasan": v} for k, v in gagal.items()]),
            use_container_width=True, hide_index=True)
    else:
        st.success(f"✅ Seluruh {len(h['kelompok'])} kelompok berhasil dijadwalkan.")
    st.caption(f"Metode: {h['metode']}")

    st.subheader("Tabel jadwal akhir")
    st.dataframe(tabel, use_container_width=True, hide_index=True)
    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button("⬇️ Unduh CSV", tabel.to_csv(index=False).encode("utf-8"),
                       "jadwal_praktikum.csv", "text/csv", use_container_width=True)
    d2.download_button("⬇️ Unduh Excel", ke_excel(tabel), "jadwal_praktikum.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)

    st.subheader("Distribusi mahasiswa per shift")
    terisi = tabel.groupby("Shift").size() if not tabel.empty else pd.Series(dtype=int)
    dist = pd.DataFrame({
        "Shift": [f"{s['nama']} ({s['hari']})" for s in SHIFTS],
        "Terisi": [int(terisi.get(s["nama"], 0)) for s in SHIFTS],
    })
    dist["Kapasitas"] = h["kapasitas"]
    dist["Sisa"] = dist["Kapasitas"] - dist["Terisi"]
    dist["Utilisasi (%)"] = (dist["Terisi"] / dist["Kapasitas"] * 100).round(1)
    c1, c2 = st.columns([3, 2])
    c1.bar_chart(dist.set_index("Shift")[["Terisi", "Sisa"]])
    c2.dataframe(dist, use_container_width=True, hide_index=True)

    st.subheader("Verifikasi aturan (uji bebas bentrok)")
    for nama, ok, ket in verifikasi(tabel, h["kapasitas"]):
        (st.success if ok else st.error)(f"{'✅' if ok else '❌'} {nama}: {ket}")


if __name__ == "__main__":
    main()
