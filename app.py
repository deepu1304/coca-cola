import streamlit as st
import pandas as pd
import plotly.express as px

from planner.production import allocate_production, optimize_allocation
from planner.shipment import plan_shipments
from planner.metrics import calculate_metrics, highlight_violations
from planner.forecasting import forecast_demand
from planner.anomaly import detect_anomalies
from planner.clustering import cluster_skus

st.set_page_config(page_title="Cola Planning Dashboard", layout="wide", page_icon="🥤")

with open('assets/style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #1769aa;'>Cola Company: Production & Deployment Planning</h1>", unsafe_allow_html=True)
st.subheader("📈 Upload Weekly Demand Data / Historical Demand Data")

with st.sidebar:
    st.header("Adjust Planning Parameters")
    max_capacity = st.slider("Max Plant Capacity (units per week)", 100000, 200000, 150000, 5000)
    truck_size = st.selectbox("Truck Size (units)", [5000, 10000, 20000], index=1)
    safety_stock = st.slider("Safety Stock (min units per SKU per DC)", 1000, 10000, 5000, 500)

    use_advanced = st.checkbox("Enable Advanced ML Features")

    if use_advanced:
        cost_production = st.number_input("Production Cost per unit", 0.0, 10.0, 1.0, 0.1)
        cost_transport = st.number_input("Transportation Cost per truck", 0.0, 1000.0, 200.0, 10.0)
        cost_inventory = st.number_input("Inventory Holding Cost per unit", 0.0, 5.0, 0.5, 0.1)
        lead_time_north = st.slider("Lead Time North DC (weeks)", 1, 4, 1)
        lead_time_south = st.slider("Lead Time South DC (weeks)", 1, 4, 2)
        lead_time_map = {'North': lead_time_north, 'South': lead_time_south}
    else:
        lead_time_map = {'North': 1, 'South': 2}

    with open("data/sample_demand.csv", "rb") as file:
        st.download_button("📥 Download Template CSV", data=file, file_name="sample_demand.csv", mime="text/csv")

    uploaded_file = st.file_uploader("Choose CSV", type="csv")
    st.markdown("After downloading the template, fill it with your demand data, then upload it here.", unsafe_allow_html=True)

if uploaded_file:
    demand_df = pd.read_csv(uploaded_file)
    st.success("Demand Data Loaded ✅")
    st.dataframe(demand_df, use_container_width=True)

    if not use_advanced:
        # ORIGINAL SIMPLE DASHBOARD CODE
        if st.sidebar.button("Simulate Scenario"):
            result_df = allocate_production(demand_df, max_capacity)
            result_df = plan_shipments(result_df, truck_size, safety_stock, lead_time_map)
        else:
            result_df = allocate_production(demand_df, max_capacity)
            result_df = plan_shipments(result_df, truck_size, safety_stock, lead_time_map)

        metrics = calculate_metrics(result_df)

        st.markdown("### 🔍 Executive Summary & Recommendations")
        if not metrics['all_safety_met']:
            st.warning("⚠️ Safety stock not met in some SKUs or DCs. Consider increasing production or safety stock.")
        if metrics['service_level'] < 90:
            st.info("ℹ️ Service level below 90%. Optimize allocation or capacity.")
        else:
            st.success("✅ All KPIs within acceptable thresholds.")

        st.markdown("### 🛠️ Production & Shipment Recommendations")
        c1, c2, c3 = st.columns(3)
        c1.metric("Service Level Achieved", f"{metrics['service_level']:.0f}%")
        c2.metric("Truck Utilization", f"{metrics['truck_utilization']:.0f}%")
        c3.metric("Safety Stock Met", "✔️" if metrics['all_safety_met'] else "❌")

        week_options = sorted(result_df['week'].unique())
        week_filter = st.multiselect("Select Week(s) to View", options=week_options, default=week_options)
        dc_filter = st.multiselect("Select Distribution Centers", options=result_df['dc'].unique(), default=result_df['dc'].unique())
        filtered_df = result_df[(result_df['week'].isin(week_filter)) & (result_df['dc'].isin(dc_filter))]

        st.dataframe(highlight_violations(filtered_df), use_container_width=True)

        st.download_button("Download Plan as CSV", result_df.to_csv(index=False).encode('utf-8'), "shipment_plan.csv", "text/csv")

        if st.button("View Dashboard"):
            fig_pie = px.pie(result_df, names='sku', values='demand', title='Demand Distribution by SKU')
            st.plotly_chart(fig_pie, use_container_width=True)

            weekly_alloc = result_df.groupby(['week', 'dc'])['allocated'].sum().reset_index()
            fig_bar = px.bar(weekly_alloc, x='week', y='allocated', color='dc', barmode='group', title='Allocated Production by Week and Distribution Center')
            st.plotly_chart(fig_bar, use_container_width=True)

            safety_rate = result_df.groupby('week')['safety_met'].mean().reset_index()
            fig_line = px.line(safety_rate, x='week', y='safety_met', labels={'safety_met': 'Safety Stock Compliance (%)'}, title='Weekly Safety Stock Compliance Rate')
            fig_line.update_yaxes(tickformat=".0%", range=[0, 1])
            st.plotly_chart(fig_line)
    else:
        # ADVANCED ENHANCED DASHBOARD CODE
        forecast_df = forecast_demand(demand_df)
        st.markdown("### 📊 Forecasted Demand (Next 8 Weeks)")
        st.dataframe(forecast_df)

        forecast_data = forecast_df.rename(columns={'demand_forecast': 'demand'})[['week', 'demand']]
        forecast_data['sku'] = 'Regular'
        forecast_data['dc'] = 'North'

        if st.sidebar.button("Simulate Advanced Scenario"):
            costs = {
                'production': cost_production,
                'transport': cost_transport,
                'inventory': cost_inventory
            }
            optimized_df = optimize_allocation(forecast_data, max_capacity, costs, truck_size, safety_stock)
            shipment_df = plan_shipments(optimized_df, truck_size, safety_stock, lead_time_map)
            anomaly_df = detect_anomalies(shipment_df)
            clusters_df = cluster_skus(shipment_df)
            metrics = calculate_metrics(shipment_df)

            st.markdown("### 🔍 Executive Summary & Recommendations")
            if not metrics['all_safety_met']:
                st.warning("⚠️ Safety stock not met in some SKUs or DCs. Consider increasing production or safety stock.")
            if metrics['service_level'] < 90:
                st.info("ℹ️ Service level below 90%. Optimize allocation or capacity.")
            else:
                st.success("✅ All KPIs within acceptable thresholds.")

            st.markdown("### 🛠️ Production & Shipment Recommendations")
            c1, c2, c3 = st.columns(3)
            c1.metric("Service Level Achieved", f"{metrics['service_level']:.0f}%")
            c2.metric("Truck Utilization", f"{metrics['truck_utilization']:.0f}%")
            c3.metric("Safety Stock Met", "✔️" if metrics['all_safety_met'] else "❌")

            st.markdown("### 🚨 Anomalies Detected")
            st.dataframe(anomaly_df[anomaly_df['anomaly']])

            st.markdown("### 🏷️ SKU Clusters")
            st.dataframe(clusters_df)

            week_options = sorted(shipment_df['week'].unique())
            week_filter = st.multiselect("Select Week(s) to View", options=week_options, default=week_options)
            dc_filter = st.multiselect("Select Distribution Centers", options=shipment_df['dc'].unique(), default=shipment_df['dc'].unique())
            filtered_df = shipment_df[(shipment_df['week'].isin(week_filter)) & (shipment_df['dc'].isin(dc_filter))]

            st.dataframe(highlight_violations(filtered_df), use_container_width=True)
            st.download_button("Download Final Plan as CSV", shipment_df.to_csv(index=False).encode('utf-8'), "shipment_plan.csv", "text/csv")

            fig_pie = px.pie(shipment_df, names='sku', values='demand', title='Demand Distribution by SKU')
            st.plotly_chart(fig_pie, use_container_width=True)

            weekly_alloc = shipment_df.groupby(['week', 'dc'])['allocated'].sum().reset_index()
            fig_bar = px.bar(weekly_alloc, x='week', y='allocated', color='dc', barmode='group', title='Allocated Production by Week and Distribution Center')
            st.plotly_chart(fig_bar, use_container_width=True)

            safety_rate = shipment_df.groupby('week')['safety_met'].mean().reset_index()
            fig_line = px.line(safety_rate, x='week', y='safety_met', labels={'safety_met': 'Safety Stock Compliance (%)'}, title='Weekly Safety Stock Compliance Rate')
            fig_line.update_yaxes(tickformat=".0%", range=[0, 1])
            st.plotly_chart(fig_line)

            if 'dc_latitude' in shipment_df.columns and 'dc_longitude' in shipment_df.columns:
                fig_map = px.scatter_mapbox(shipment_df, lat='dc_latitude', lon='dc_longitude', color='dc', size='allocated', hover_name='sku', zoom=4, mapbox_style="open-street-map", title="Shipment Routes and Allocations")
                st.plotly_chart(fig_map, use_container_width=True)

            heat_data = shipment_df.pivot_table(index='week', columns='dc', values='safety_met', aggfunc='mean')
            fig_heat = px.imshow(heat_data, color_continuous_scale='RdYlGn', labels=dict(x="Distribution Center", y="Week", color="Safety Stock Compliance"))
            st.plotly_chart(fig_heat)

        else:
            st.info("Click 'Simulate Advanced Scenario' to run forecasting, optimization, and analytics.")
else:
    st.info("Upload your demand CSV to display results.")

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Designed by Aniket • Powered by Streamlit")
