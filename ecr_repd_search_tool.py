import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import geopandas as gpd

# ----------------------------------
# Constants (defaults; now user-editable in the UI)
# ----------------------------------
DEFAULT_BUFFER_KM = 2.0
DEFAULT_CAP_TOL = 0.10          # 10%
DEFAULT_TEXT_THRESH = 70        # 0-100

# ----------------------------------
# Helpers
# ----------------------------------
def safe_to_numeric(df, cols):
    for col in cols:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def joined_text(row, cols):
    vals = []
    for c in cols:
        if c and c in row and pd.notna(row[c]):
            vals.append(str(row[c]))
    return " ".join(vals).strip()

def normalize_postcode(s):
    if pd.isna(s):
        return ""
    return str(s).replace(" ", "").lower()

def ecr_effective_capacity(row, status_col, already_col, accepted_col):
    """
    Returns a numeric capacity based on ECR connection status rules:
      - if status == 'Connected' => use 'Already Connected Capacity'
      - if status == 'Accepted To Connect' => use 'Accepted to Connect Capacity'
    If fields are missing or unparsable, returns NaN.
    """
    status = str(row.get(status_col, "")).strip().lower()
    if status == "connected":
        return pd.to_numeric(row.get(already_col, np.nan), errors="coerce")
    if status == "accepted to connect":
        return pd.to_numeric(row.get(accepted_col, np.nan), errors="coerce")
    # For any other status, return NaN (won't pass capacity test)
    return np.nan

def build_geo(df, x_col, y_col, epsg="EPSG:27700"):
    try:
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[x_col], df[y_col], crs=epsg)
        )
        return gdf
    except Exception as e:
        st.error(f"Coordinate error for columns [{x_col}], [{y_col}]: {e}")
        st.stop()

def ordered_reasons(flags):
    """
    Make reason strings deterministic to support 'Ignore these matches' step.
    Order: Spatial, Text (GrpA), Text (GrpB), Capacity, Postcode
    """
    order = ["Spatial", "Text (GrpA)", "Text (GrpB)", "Capacity", "Postcode"]
    present = [r for r in order if r in flags]
    return ", ".join(present)

def compute_match(
    base_row,
    search_row,
    text_thresh,
    base_cols,
    search_cols,
    is_search_ecr,
    ecr_status_col,
    ecr_alr_col,
    ecr_acc_col,
    cap_tolerance
):
    """
    Returns:
      (score:int, reasons:list[str], base_details:list[str], search_details:list[str])
    """
    reasons = set(["Spatial"])  # spatial is pre-qualified by geometry filter
    base_details = []
    search_details = []

    # Capacity
    base_cap = pd.to_numeric(base_row.get(base_cols["capacity"], np.nan), errors="coerce")
    if is_search_ecr:
        search_cap = ecr_effective_capacity(search_row, ecr_status_col, ecr_alr_col, ecr_acc_col)
    else:
        search_cap = pd.to_numeric(search_row.get(search_cols["capacity"], np.nan), errors="coerce")

    if pd.notna(base_cap) and pd.notna(search_cap):
        if abs(search_cap - base_cap) <= cap_tolerance * base_cap:
            reasons.add("Capacity")
            base_details.append(f"capacity: {base_cap}")
            search_details.append(f"capacity: {search_cap}")

    # Text Group A
    base_text_a = joined_text(base_row, base_cols["text_a"])
    search_text_a = joined_text(search_row, search_cols["text_a"])
    if base_text_a or search_text_a:
        if fuzz.token_sort_ratio(base_text_a, search_text_a) >= text_thresh:
            reasons.add("Text (GrpA)")
            base_details.append(f"textA: {base_text_a}")
            search_details.append(f"textA: {search_text_a}")

    # Text Group B
    base_text_b = joined_text(base_row, base_cols["text_b"])
    search_text_b = joined_text(search_row, search_cols["text_b"])
    if base_text_b or search_text_b:
        if fuzz.token_sort_ratio(base_text_b, search_text_b) >= text_thresh:
            reasons.add("Text (GrpB)")
            base_details.append(f"textB: {base_text_b}")
            search_details.append(f"textB: {search_text_b}")

    # Postcode
    base_pc = normalize_postcode(base_row.get(base_cols["postcode"], ""))
    search_pc = normalize_postcode(search_row.get(search_cols["postcode"], ""))
    if base_pc and search_pc and base_pc == search_pc:
        reasons.add("Postcode")
        base_details.append(f"postcode: {base_pc}")
        search_details.append(f"postcode: {search_pc}")

    # Score = count of reasons
    score = len(reasons)
    return score, ordered_reasons(reasons), base_details, search_details

# ----------------------------------
# UI
# ----------------------------------
st.set_page_config(page_title="ECR–REPD Matching Tool", layout="wide")
st.title("⚡ ECR–REPD Matching Tool (Dynamic)")

# --- Step 1: Ask file format (kept the same) ---
file_option = st.radio(
    "Do you have one file with two sheets or two separate files?",
    ("One file, two sheets", "Two separate files")
)

repd_df = None
ecr_df = None

# --- Step 2: File Upload (kept the same) ---
if file_option == "One file, two sheets":
    uploaded = st.file_uploader("Upload Excel file", type=["xlsx"])
    if uploaded:
        try:
            repd_df = pd.read_excel(uploaded, sheet_name="REPD")
            ecr_df = pd.read_excel(uploaded, sheet_name="ECR")
            st.success("✅ Loaded REPD and ECR sheets.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
else:
    repd_file = st.file_uploader("Upload REPD Excel", type=["xlsx"], key="repd")
    ecr_file = st.file_uploader("Upload ECR Excel", type=["xlsx"], key="ecr")
    if repd_file and ecr_file:
        try:
            repd_df = pd.read_excel(repd_file)
            ecr_df = pd.read_excel(ecr_file)
            st.success("✅ Loaded REPD and ECR files.")
        except Exception as e:
            st.error(f"Error reading files: {e}")

# Proceed if both datasets are present
if repd_df is not None and ecr_df is not None:

    # --- Step 2 (new): Direction of search ---
    st.subheader("Search Direction")
    direction = st.radio(
        "Choose what to search:",
        (
            "Search for REPD projects in ECR (Base = REPD, Search = ECR)",
            "Search for ECR projects in REPD (Base = ECR, Search = REPD)"
        )
    )
    base_is_repd = "Base = REPD" in direction

    # Column lists for dropdowns
    repd_cols = list(repd_df.columns)
    ecr_cols  = list(ecr_df.columns)

    # --- Step 3: Parameter inputs ---
    st.subheader("Matching Parameters")
    cols_params = st.columns(3)
    with cols_params[0]:
        buffer_km = st.number_input("Buffer distance (km)", min_value=0.0, value=DEFAULT_BUFFER_KM, step=0.5)
    with cols_params[1]:
        cap_tol = st.number_input("Capacity tolerance (fraction)", min_value=0.0, value=DEFAULT_CAP_TOL, step=0.01,
                                  help="0.10 means ±10% of the base project's capacity")
    with cols_params[2]:
        text_thresh = st.slider("Text matching factor (0–100)", min_value=0, max_value=100, value=DEFAULT_TEXT_THRESH)

    # --- Step 4: Dynamic column selectors ---
    st.subheader("Column Mapping")

    st.markdown("**REPD columns**")
    repd_id_col = st.selectbox("REPD ID", repd_cols, index=repd_cols.index("REPD_ID") if "REPD_ID" in repd_cols else 0)
    repd_cap_col = st.selectbox("REPD Capacity", repd_cols, index=(repd_cols.index("Installed Capacity (MWelec)")
                                                                   if "Installed Capacity (MWelec)" in repd_cols else 0))
    repd_text_a_cols = st.multiselect("REPD Text Group A (e.g., Operator (or Applicant), Site Name)",
                                      repd_cols, default=[c for c in ["Operator (or Applicant)", "Site Name"] if c in repd_cols])
    repd_text_b_cols = st.multiselect("REPD Text Group B (e.g., Site Name, Address)",
                                      repd_cols, default=[c for c in ["Site Name", "Address"] if c in repd_cols])
    repd_pc_col = st.selectbox("REPD Postcode", repd_cols, index=(repd_cols.index("Post Code") if "Post Code" in repd_cols else 0))
    repd_x_col = st.selectbox("REPD X (Easting)", repd_cols, index=(repd_cols.index("X-coordinate") if "X-coordinate" in repd_cols else 0))
    repd_y_col = st.selectbox("REPD Y (Northing)", repd_cols, index=(repd_cols.index("Y-coordinate") if "Y-coordinate" in repd_cols else 0))

    st.markdown("---")
    st.markdown("**ECR columns**")
    ecr_id_col = st.selectbox("ECR ID", ecr_cols, index=(ecr_cols.index("ECR_ID") if "ECR_ID" in ecr_cols else 0))
    ecr_cap_col = st.selectbox("ECR Capacity (use only if NOT using Connection Status rule)", ecr_cols,
                               index=(ecr_cols.index("Accepted_to_Connect_Registered_")
                                      if "Accepted_to_Connect_Registered_" in ecr_cols else 0))
    ecr_text_a_cols = st.multiselect("ECR Text Group A (e.g., Customer Name, Customer Site)",
                                     ecr_cols, default=[c for c in ["Customer_Name", "Customer_Site"] if c in ecr_cols])
    ecr_text_b_cols = st.multiselect("ECR Text Group B (e.g., Customer Site, Address Line 1)",
                                     ecr_cols, default=[c for c in ["Customer_Site", "Address_Line_1"] if c in ecr_cols])
    ecr_status_col = st.selectbox("ECR Connection Status", ecr_cols,
                                  index=(ecr_cols.index("Connection_Status") if "Connection_Status" in ecr_cols else 0))
    ecr_already_col = st.selectbox("ECR Already Connected Capacity", ecr_cols,
                                   index=(ecr_cols.index("Already_Connected_Registered_")
                                          if "Already_Connected_Registered_" in ecr_cols else 0))
    ecr_accepted_col = st.selectbox("ECR Accepted to Connect Capacity", ecr_cols,
                                    index=(ecr_cols.index("Accepted_to_Connect_Registered_")
                                           if "Accepted_to_Connect_Registered_" in ecr_cols else 0))
    ecr_pc_col = st.selectbox("ECR Postcode", ecr_cols, index=(ecr_cols.index("Postcode") if "Postcode" in ecr_cols else 0))
    ecr_x_col = st.selectbox("ECR X (Easting)", ecr_cols,
                             index=(ecr_cols.index("Location__X_coordinate___Eastin") if "Location__X_coordinate___Eastin" in ecr_cols else 0))
    ecr_y_col = st.selectbox("ECR Y (Northing)", ecr_cols,
                             index=(ecr_cols.index("Location__y_coordinate___Northi") if "Location__y_coordinate___Northi" in ecr_cols else 0))

    # --- Step 5: "Column to pull" (from SEARCH df depending on direction) ---
    st.subheader("Column to Pull from Matched Results")
    if base_is_repd:
        pull_source_name = "ECR"
        pull_col = st.selectbox("Select a column from ECR to pull into results", ecr_cols,
                                index=(ecr_cols.index("ECR_ID") if "ECR_ID" in ecr_cols else 0))
    else:
        pull_source_name = "REPD"
        pull_col = st.selectbox("Select a column from REPD to pull into results", repd_cols,
                                index=(repd_cols.index("REPD_ID") if "REPD_ID" in repd_cols else 0))

    dynamic_pull_col_name = f"Matched_{pull_source_name}_{pull_col}"

    # --- Step 9: Ignore these matches (before range) ---
    st.subheader("Ignore These Matches (optional)")
    ignore_options = [
        "Spatial",
        "Spatial, Text (GrpA)",
        "Spatial, Text (GrpB)",
        "Spatial, Capacity",
    ]
    ignore_selected = st.multiselect("Pick exact combinations of 'Matching Reason' to mark as Ignored", ignore_options, default=[])

    # --- Step 7: Range input (based on BASE df) ---
    st.subheader("Base ID Range")
    base_id_label = f"{'REPD' if base_is_repd else 'ECR'} ID column"
    base_id_col = repd_id_col if base_is_repd else ecr_id_col
    start_id = st.text_input(f"Enter start {base_id_label}")
    end_id = st.text_input(f"Enter end {base_id_label}")

    # --- Run ---
    if start_id and end_id and st.button("🔍 Run Matching"):
        # Validate numeric
        try:
            start_id_num = int(start_id)
            end_id_num   = int(end_id)
        except ValueError:
            st.error("ID range must be numeric.")
            st.stop()

        # Build BASE and SEARCH frames
        base_df  = repd_df.copy() if base_is_repd else ecr_df.copy()
        search_df = ecr_df.copy() if base_is_repd else repd_df.copy()

        # Force base ID numeric & filter by range
        base_df[base_id_col] = pd.to_numeric(base_df[base_id_col], errors="coerce")
        base_df = base_df[(base_df[base_id_col] >= start_id_num) & (base_df[base_id_col] <= end_id_num)].copy()

        # Cast numerics for coords & capacities
        if base_is_repd:
            base_df  = safe_to_numeric(base_df,  [repd_x_col, repd_y_col, repd_cap_col])
            search_df = safe_to_numeric(search_df, [ecr_x_col, ecr_y_col, ecr_cap_col, ecr_already_col, ecr_accepted_col])
        else:
            base_df  = safe_to_numeric(base_df,  [ecr_x_col, ecr_y_col, ecr_cap_col, ecr_already_col, ecr_accepted_col])
            search_df = safe_to_numeric(search_df, [repd_x_col, repd_y_col, repd_cap_col])

        # Insert required columns at the start (point 5 + original three)
        out_cols = [dynamic_pull_col_name, "Matching Reason", "Matched Details REPD", "Matched Details ECR"]
        for col in reversed(out_cols):
            base_df.insert(0, col, "NF")

        # Build GeoDataFrames
        if base_is_repd:
            base_gdf = build_geo(base_df, repd_x_col, repd_y_col)
            search_gdf = build_geo(search_df, ecr_x_col, ecr_y_col)
        else:
            base_gdf = build_geo(base_df, ecr_x_col, ecr_y_col)
            search_gdf = build_geo(search_df, repd_x_col, repd_y_col)

        # Column maps for matching
        base_cols_map = {
            "capacity": repd_cap_col if base_is_repd else ecr_cap_col,
            "text_a":   repd_text_a_cols if base_is_repd else ecr_text_a_cols,
            "text_b":   repd_text_b_cols if base_is_repd else ecr_text_b_cols,
            "postcode": repd_pc_col if base_is_repd else ecr_pc_col,
        }
        search_cols_map = {
            "capacity": ecr_cap_col if base_is_repd else repd_cap_col,
            "text_a":   ecr_text_a_cols if base_is_repd else repd_text_a_cols,
            "text_b":   ecr_text_b_cols if base_is_repd else repd_text_b_cols,
            "postcode": ecr_pc_col if base_is_repd else repd_pc_col,
        }

        # Matching
        results = base_df.copy()
        buffer_m = float(buffer_km) * 1000.0

        for idx, base_row in base_gdf.iterrows():
            geom = base_row.geometry
            if geom is None or pd.isna(geom.x) or pd.isna(geom.y):
                continue

            # Spatial pre-filter
            buf = geom.buffer(buffer_m)
            candidates = search_gdf[search_gdf.intersects(buf)]
            if candidates.empty:
                continue

            best = None
            best_score = -1
            best_reasons = ""
            best_base_details = []
            best_search_details = []

            for _, srow in candidates.iterrows():
                score, reasons, base_details, search_details = compute_match(
                    base_row=base_row,
                    search_row=srow,
                    text_thresh=text_thresh,
                    base_cols=base_cols_map,
                    search_cols=search_cols_map,
                    is_search_ecr=(not base_is_repd),
                    ecr_status_col=ecr_status_col,
                    ecr_alr_col=ecr_already_col,
                    ecr_acc_col=ecr_accepted_col,
                    cap_tolerance=cap_tol
                )
                if score > best_score:
                    best_score = score
                    best = srow
                    best_reasons = reasons
                    best_base_details = base_details
                    best_search_details = search_details

            if best is not None:
                # Fill pull column
                if base_is_repd:
                    results.at[idx, dynamic_pull_col_name] = best.get(pull_col, "NF")
                else:
                    results.at[idx, dynamic_pull_col_name] = best.get(pull_col, "NF")

                results.at[idx, "Matching Reason"] = best_reasons

                # Keep detail column names as requested ("like before")
                # Populate REPD / ECR detail sides consistently by dataset
                if base_is_repd:
                    results.at[idx, "Matched Details REPD"] = "; ".join(best_base_details)
                    results.at[idx, "Matched Details ECR"] = "; ".join(best_search_details)
                else:
                    results.at[idx, "Matched Details ECR"] = "; ".join(best_base_details)
                    results.at[idx, "Matched Details REPD"] = "; ".join(best_search_details)

        # --- Step 9: Apply "Ignore these matches"
        if ignore_selected:
            mask_ignore = results["Matching Reason"].isin(ignore_selected)
            results.loc[mask_ignore, "Matching Reason"] = "Ignored"
            results.loc[mask_ignore, dynamic_pull_col_name] = "Ignored"

        # --- Show results
        st.subheader("Results")
        st.dataframe(results, use_container_width=True)

        # --- Step 10: Output filename
        base_name = "REPD" if base_is_repd else "ECR"
        search_name = "ECR" if base_is_repd else "REPD"
        out_file = f"Search_{base_name}_projectID_{start_id_num}_{end_id_num}_in_{search_name}.xlsx"

        results.to_excel(out_file, index=False)
        with open(out_file, "rb") as f:
            st.download_button("📥 Download Results as Excel", f, file_name=out_file)

        # Reset
        if st.button("🔄 Clear and Start Again"):
            st.experimental_rerun()
