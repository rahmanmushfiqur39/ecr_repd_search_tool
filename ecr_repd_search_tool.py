import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import geopandas as gpd
import re


# ----------------------------
# Default Parameters
# ----------------------------
DEFAULT_BUFFER_KM = 2.0
DEFAULT_CAP_TOL = 0.10
DEFAULT_TEXT_THRESH = 70

# ----------------------------
# Helpers
# ----------------------------
def safe_to_numeric(df, cols):
    for col in cols:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def joined_text(row, cols):
    vals = [str(row[c]) for c in cols if c in row and pd.notna(row[c])]
    return " ".join(vals).strip()

def normalize_postcode(s):
    return str(s).replace(" ", "").lower() if pd.notna(s) else ""

def ecr_effective_capacity(row, status_col, already_col, accepted_col):
    status = str(row.get(status_col, "")).strip().lower()
    if status == "connected":
        return pd.to_numeric(row.get(already_col, np.nan), errors="coerce")
    if status == "accepted to connect":
        return pd.to_numeric(row.get(accepted_col, np.nan), errors="coerce")
    return np.nan

def ordered_reasons(flags):
    order = ["Spatial", "Text (GrpA)", "Text (GrpB)", "Capacity", "Postcode"]
    return ", ".join([r for r in order if r in flags])

def clean_text(s):
    """
    Cleans a text string for fuzzy matching.
    Removes punctuation, lowercases, and ignores placeholders like 'data not available'.
    """
    s = str(s).strip().lower()
    if not s or s in ["data not available", "n/a", "na", "none", "no data"]:
        return ""
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = " ".join(s.split())
    # Also remove internal occurrences like "solar farm data not available ltd"
    s = s.replace("data not available", "").strip()
    return s

def compute_match(base_row, search_row, text_thresh, base_cols, search_cols,
                  is_search_ecr, ecr_status_col, ecr_alr_col, ecr_acc_col, cap_tolerance):
    reasons = {"Spatial"}
    base_details, search_details = [], []

    # --- Search capacity ---
    if is_search_ecr:
        # search dataset is ECR
        search_cap = ecr_effective_capacity(search_row, ecr_status_col, ecr_alr_col, ecr_acc_col)
        # base dataset is REPD
        base_cap = pd.to_numeric(base_row.get(base_cols["capacity"], np.nan), errors="coerce")
    else:
        # search dataset is REPD
        search_cap = pd.to_numeric(search_row.get(search_cols["capacity"], np.nan), errors="coerce")
        # base dataset is ECR
        base_cap = ecr_effective_capacity(base_row, ecr_status_col, ecr_alr_col, ecr_acc_col)
      
    if pd.notna(base_cap) and pd.notna(search_cap):
        if abs(search_cap - base_cap) <= cap_tolerance * base_cap:
            reasons.add("Capacity")
            base_details.append(f"C: {base_cap}")
            search_details.append(f"C: {search_cap}")

    # --- Text Group A ---
    base_text_a = clean_text(joined_text(base_row, base_cols["text_a"]))
    search_text_a = clean_text(joined_text(search_row, search_cols["text_a"]))

    if base_text_a or search_text_a:
        token_ratio_a= fuzz.token_sort_ratio(base_text_a, search_text_a)
        partial_ratio_a= fuzz.partial_ratio(base_text_a, search_text_a)
        text_score_a = max(token_ratio_a, partial_ratio_a)
        if text_score_a >= text_thresh:
            reasons.add("Text (GrpA)")
            base_details.append(f"tA: {base_text_a}")
            search_details.append(f"tA: {search_text_a}")
    
    # --- Text Group B ---
    base_text_b = clean_text(joined_text(base_row, base_cols["text_b"]))
    search_text_b = clean_text(joined_text(search_row, search_cols["text_b"]))
    if base_text_b or search_text_b:
        token_ratio_b= fuzz.token_sort_ratio(base_text_b, search_text_b)
        partial_ratio_b= fuzz.partial_ratio(base_text_b, search_text_b)
        text_score_b = max(token_ratio_b, partial_ratio_b)
        if text_score_b >= text_thresh:
            reasons.add("Text (GrpB)")
            base_details.append(f"tB: {base_text_b}")
            search_details.append(f"tB: {search_text_b}")

    # Postcode
    base_pc = normalize_postcode(base_row.get(base_cols["postcode"], ""))
    search_pc = normalize_postcode(search_row.get(search_cols["postcode"], ""))
    if base_pc and search_pc and base_pc == search_pc:
        reasons.add("Postcode")
        base_details.append(f"PC: {base_pc}")
        search_details.append(f"PC: {search_pc}")

    return len(reasons), ordered_reasons(reasons), base_details, search_details


# ----------------------------
# Caching Functions
# ----------------------------
@st.cache_data
def load_excel(uploaded, sheet=None):
    return pd.read_excel(uploaded, sheet_name=sheet)

@st.cache_data
def make_geodf(df, x_col, y_col):
    return gpd.GeoDataFrame(df,
                            geometry=gpd.points_from_xy(df[x_col], df[y_col], crs="EPSG:27700"))


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="ECR–REPD Matching Tool", layout="wide")
st.title("⚡ ECR–REPD Matching Tool")

# --- Step 1: File Input
file_option = st.radio(
    "Do you have one file with two sheets or two separate files?",
    ("One file, two sheets", "Two separate files")
)

repd_df, ecr_df = None, None

if file_option == "One file, two sheets":
    uploaded = st.file_uploader("Upload Excel file", type=["xlsx"])
    if uploaded:
        if "repd_df" not in st.session_state:
            try:
                st.session_state.repd_df = load_excel(uploaded, sheet="REPD")
                st.session_state.ecr_df = load_excel(uploaded, sheet="ECR")
                st.success("✅ Loaded REPD and ECR sheets.")
            except Exception as e:
                st.error(f"Error reading file: {e}")
        repd_df = st.session_state.repd_df
        ecr_df = st.session_state.ecr_df
else:
    repd_file = st.file_uploader("Upload REPD Excel", type=["xlsx"], key="repd")
    ecr_file = st.file_uploader("Upload ECR Excel", type=["xlsx"], key="ecr")
    if repd_file and ecr_file:
        if "repd_df" not in st.session_state:
            try:
                st.session_state.repd_df = load_excel(repd_file)
                st.session_state.ecr_df = load_excel(ecr_file)
                st.success("✅ Loaded REPD and ECR files.")
            except Exception as e:
                st.error(f"Error reading files: {e}")
        repd_df = st.session_state.repd_df
        ecr_df = st.session_state.ecr_df


# ----------------------------
# Proceed only if both loaded
# ----------------------------
if repd_df is not None and ecr_df is not None:
    # --- Step 2: Direction
    st.subheader("Search Direction")
    direction = st.radio(
        "Choose what to search:",
        (
            "Search for REPD projects in ECR (Base = REPD, Search = ECR)",
            "Search for ECR projects in REPD (Base = ECR, Search = REPD)"
        )
    )
    base_is_repd = "Base = REPD" in direction

    repd_cols, ecr_cols = list(repd_df.columns), list(ecr_df.columns)

    # --- Step 3: Parameters
    st.subheader("Matching Parameters")
    cols_params = st.columns(3)
    with cols_params[0]:
        buffer_km = st.number_input(" Spatial search buffer distance (km)", 0.0, 100.0, DEFAULT_BUFFER_KM, 0.5)
    with cols_params[1]:
        cap_tol = st.number_input("Capacity tolerance (fraction)", 0.0, 1.0, DEFAULT_CAP_TOL, 0.01)
    with cols_params[2]:
        text_thresh = st.slider("Text matching factor (0–100)", 0, 100, DEFAULT_TEXT_THRESH)
    
    keep_spatial_only_m = st.number_input(
    "Keep entries within the buffer area failing other matching criteria (no text, capacity, or postcode) only if within (m)",
    0.0, 5000.0, 100.0, 100.0,
    help="Entries within the buffer with no matching (no text, capacity, or postcode) will be kept only if they are within this distance in metres."
    )


    # --- Step 4: Column mapping (same structure)
    st.subheader("Column Mapping")

   # --- REPD columns ---
    st.markdown("**REPD columns**")
    repd_id_col = st.selectbox("REPD ID", [""] + repd_cols, index=0, help="Unique project ID column in the REPD dataset (e.g. 'REPD_ID').")
    repd_cap_col = st.selectbox("REPD Capacity", [""] + repd_cols, index=0, help="Column containing the installed capacity of the project in MW (e.g. 'Installed Capacity (MWelec)').")
    repd_text_a_cols = st.multiselect("REPD Text Group A", repd_cols, help="Text fields used for Group A name matching (commonly 'Operator (or Applicant)' and 'Site Name').")
    repd_text_b_cols = st.multiselect("REPD Text Group B", repd_cols, help="Text fields used for Group B name matching (commonly 'Site Name' and 'Address').")
    repd_pc_col = st.selectbox("REPD Postcode", [""] + repd_cols, index=0, help="Column containing the site postcode in the REPD dataset (e.g. 'Post Code').")
    repd_x_col = st.selectbox("REPD X (Easting)", [""] + repd_cols, index=0, help="Easting (X-coordinate) column for REPD project locations.")
    repd_y_col = st.selectbox("REPD Y (Northing)", [""] + repd_cols, index=0, help="Northing (Y-coordinate) column for REPD project locations.")
    
    # --- ECR columns ---
    st.markdown("---")
    st.markdown("**ECR columns**")
    ecr_id_col = st.selectbox("ECR ID", [""] + ecr_cols, index=0, help="Unique project ID column in the ECR dataset (e.g. 'ECR_ID').")
    ecr_text_a_cols = st.multiselect("ECR Text Group A", ecr_cols, help="Text fields used for Group A name matching (commonly 'Customer Name' and 'Customer Site').")
    ecr_text_b_cols = st.multiselect("ECR Text Group B", ecr_cols, help="Text fields used for Group B name matching (commonly 'Customer Site' and 'Address Line 1').")
    ecr_status_col = st.selectbox("ECR Connection Status", [""] + ecr_cols, index=0, help="Column indicating connection status (e.g. 'Connected', 'Accepted To Connect'). Used to determine which capacity column to apply.")
    ecr_already_col = st.selectbox("ECR Already Connected Capacity", [""] + ecr_cols, index=0, help="Column containing the capacity (MW) for already connected projects.")
    ecr_accepted_col = st.selectbox("ECR Accepted to Connect Capacity", [""] + ecr_cols, index=0, help="Column containing the capacity (MW) for projects accepted to connect but not yet connected.")
    ecr_pc_col = st.selectbox("ECR Postcode", [""] + ecr_cols, index=0, help="Column containing the site postcode in the ECR dataset.")
    ecr_x_col = st.selectbox("ECR X (Easting)", [""] + ecr_cols, index=0, help="Easting (X-coordinate) column for ECR connection locations.")
    ecr_y_col = st.selectbox("ECR Y (Northing)", [""] + ecr_cols, index=0, help="Northing (Y-coordinate) column for ECR connection locations.")


    # --- Step 5: Column to pull
    st.subheader("Column to Pull")
    if base_is_repd:
        pull_source_name = "ECR"
        pull_col = st.selectbox("Select column from ECR to pull", [""] + ecr_cols, index=0, help="E.g. ECR_ID or Project name. This column will be added to the output table.")
    else:
        pull_source_name = "REPD"
        pull_col = st.selectbox("Select column from REPD to pull", [""] + repd_cols, index=0, help="E.g. Ref ID or Development Status. This column will be added to the output table.")
    dynamic_pull_col_name = f"Matched_{pull_source_name}_{pull_col}"

    # --- Step 9: Ignore matches
    st.subheader("Ignore These Matches")
    ignore_options = [
        "Spatial",
        "Spatial, Text (GrpA)",
        "Spatial, Text (GrpB)",
        "Spatial, Capacity",
    ]
    ignore_selected = st.multiselect("Choose combinations to ignore", ignore_options, [])

    # --- Step 7: Range input
    st.subheader("Base ID Range")
    base_id_col = repd_id_col if base_is_repd else ecr_id_col
    start_id = st.text_input(f"Start ID ({base_id_col})")
    end_id = st.text_input(f"End ID ({base_id_col})")

    # ----------------------------
    # Run button
    # ----------------------------
    if start_id and end_id and st.button("🔍 Run Matching"):
        try:
            start_id_num, end_id_num = int(start_id), int(end_id)
        except ValueError:
            st.error("IDs must be numeric.")
            st.stop()

        base_df = repd_df.copy() if base_is_repd else ecr_df.copy()
        search_df = ecr_df.copy() if base_is_repd else repd_df.copy()

        base_df[base_id_col] = pd.to_numeric(base_df[base_id_col], errors="coerce")
        base_df = base_df[(base_df[base_id_col] >= start_id_num) & (base_df[base_id_col] <= end_id_num)]

        # Clean numeric columns
        if base_is_repd:
            base_df = safe_to_numeric(base_df, [repd_x_col, repd_y_col, repd_cap_col])
            search_df = safe_to_numeric(search_df, [ecr_x_col, ecr_y_col, ecr_already_col, ecr_accepted_col])
        else:
            base_df = safe_to_numeric(base_df, [ecr_x_col, ecr_y_col, ecr_already_col, ecr_accepted_col])
            search_df = safe_to_numeric(search_df, [repd_x_col, repd_y_col, repd_cap_col])

        # Add output columns
        out_cols = [ dynamic_pull_col_name, "Matched Details REPD","Matched Details ECR", "Matching Reason"]
        for c in out_cols:
            base_df.insert(0, c, "NF")

        # Build geodataframes (cached)
        if base_is_repd:
            base_gdf = make_geodf(base_df, repd_x_col, repd_y_col)
            search_gdf = make_geodf(search_df, ecr_x_col, ecr_y_col)
        else:
            base_gdf = make_geodf(base_df, ecr_x_col, ecr_y_col)
            search_gdf = make_geodf(search_df, repd_x_col, repd_y_col)

        base_cols_map = {
            "capacity": repd_cap_col if base_is_repd else None,
            "text_a": repd_text_a_cols if base_is_repd else ecr_text_a_cols,
            "text_b": repd_text_b_cols if base_is_repd else ecr_text_b_cols,
            "postcode": repd_pc_col if base_is_repd else ecr_pc_col,
        }
        search_cols_map = {
            "capacity": repd_cap_col if not base_is_repd else None,
            "text_a": ecr_text_a_cols if base_is_repd else repd_text_a_cols,
            "text_b": ecr_text_b_cols if base_is_repd else repd_text_b_cols,
            "postcode": ecr_pc_col if base_is_repd else repd_pc_col,
        }

        # Matching
        results = base_df.copy()
        buffer_m = buffer_km * 1000
        for idx, b in base_gdf.iterrows():
            geom = b.geometry
            if geom is None or pd.isna(geom.x) or pd.isna(geom.y):
                continue
            buf = geom.buffer(buffer_m)
            candidates = search_gdf[search_gdf.intersects(buf)]
            if candidates.empty:
                continue

            best, best_score, best_reasons = None, -1, ""
            best_base_details, best_search_details = [], []
            best_distance = None
            for _, s in candidates.iterrows():
                distance_m = geom.distance(s.geometry)
                score, reasons, bd, sd = compute_match(
                    b, s, text_thresh, base_cols_map, search_cols_map, base_is_repd, ecr_status_col, ecr_already_col, ecr_accepted_col, cap_tol
                )

                # Add distance to Spatial label
                if "Spatial" in reasons:
                    reasons = reasons.replace("Spatial", f"Spatial ({distance_m:.1f} m)")
                if score > best_score:
                    best, best_score, best_reasons = s, score, reasons
                    best_base_details, best_search_details = bd, sd
                    best_distance = distance_m   # ✅ capture best match distance here
            
            # --- Handle spatial-only filtering here ---
            if best is not None:
                if best_score == 1 and best_distance > keep_spatial_only_m:
                    continue  # skip too-distant spatial-only matches
                    
                results.at[idx, dynamic_pull_col_name] = best.get(pull_col, "NF")
                results.at[idx, "Matching Reason"] = best_reasons
                if base_is_repd:
                    results.at[idx, "Matched Details REPD"] = "; ".join(best_base_details)
                    results.at[idx, "Matched Details ECR"] = "; ".join(best_search_details)
                else:
                    results.at[idx, "Matched Details ECR"] = "; ".join(best_base_details)
                    results.at[idx, "Matched Details REPD"] = "; ".join(best_search_details)

        # Ignore matches
        if ignore_selected:
            mask = results["Matching Reason"].isin(ignore_selected)
            results.loc[mask, "Matching Reason"] = "Ignored"
            results.loc[mask, dynamic_pull_col_name] = "Ignored"

        # Display + Save
        st.subheader("Results")
        st.dataframe(results, use_container_width=True)

        base_name = "REPD" if base_is_repd else "ECR"
        search_name = "ECR" if base_is_repd else "REPD"
        out_file = f"Search_{base_name}_projectID_{start_id_num}_{end_id_num}_in_{search_name}.xlsx"

        results.to_excel(out_file, index=False)
        with open(out_file, "rb") as f:
            st.download_button("📥 Download Results", f, file_name=out_file)

        if st.button("🔄 Clear and Start Again"):
            st.session_state.clear()
            st.experimental_rerun()
