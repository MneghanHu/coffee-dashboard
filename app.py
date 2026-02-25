# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d

st.set_page_config(page_title="Paz Roast MVP", layout="wide")
st.title("☕️ Paz Coffee Energy Efficiency Monitoring System")

@st.cache_data
def load_data():
    ref_ts = pd.read_csv('processed_ref.csv')
    ref_sum = pd.read_csv('ref_summary.csv')
    master_ts = pd.read_csv('master_timeseries.csv')
    master_sum = pd.read_csv('master_summary.csv')
    return ref_ts, ref_sum, master_ts, master_sum

try:
    ref_ts, ref_sum, master_ts, master_sum = load_data()
except Exception as e:
    st.error(f"数据加载失败: {e}")
    st.stop()

ref_sum = ref_sum.iloc[0]

# 数据预处理
master_sum = master_sum.copy()
master_sum['start_gas_num'] = pd.to_numeric(master_sum['start_gas'], errors='coerce')
master_sum['start_gas_int'] = master_sum['start_gas_num'].round(0).astype('Int64')

# 在侧边栏显示调试信息
available_gas = master_sum['start_gas_int'].dropna().unique()
available_gas = sorted(available_gas)
st.sidebar.write("**Debug - Available start gas values:**", available_gas)

if available_gas:
    st.sidebar.write("**Counts per value:**")
    for g in available_gas:
        cnt = (master_sum['start_gas_int'] == g).sum()
        st.sidebar.write(f"  {g}%: {cnt} batches")

# 视图选择
view = st.sidebar.radio("Select View", ["Single Batch Comparison", "Statistical Analysis by Start Gas"])

if view == "Single Batch Comparison":
    batches = master_sum['batch_id'].tolist()
    selected = st.sidebar.selectbox("Select batch to compare:", batches)
    current_sum = master_sum[master_sum['batch_id'] == selected].iloc[0]
    current_ts = master_ts[master_ts['batch_id'] == selected].sort_values('time_sec')

    col_left, col_right = st.columns([2,1])
    with col_left:
        st.subheader("Temperature Curve Comparison")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ref_ts['time_sec'], y=ref_ts['beantemp'], name='Standard', line=dict(color='lightblue', dash='dot')))
        fig.add_trace(go.Scatter(x=current_ts['time_sec'], y=current_ts['beantemp'], name='Current', line=dict(color='darkblue')))
        fig.update_layout(xaxis_title="Time (seconds)", yaxis_title="Bean Temperature (°C)", hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("📊 Energy Efficiency Metrics")
        total_gas_ref = ref_sum['total_gas']
        total_gas_cur = current_sum['total_gas']
        energy_dev = ((total_gas_cur - total_gas_ref) / total_gas_ref) * 100 if total_gas_ref != 0 else 0
        st.metric("Total Energy Deviation", f"{energy_dev:+.1f}%", delta=f"{energy_dev:+.1f}%", delta_color="inverse")

        st.markdown("**Phase-wise Gas Consumption vs Standard**")
        phases = ['dry', 'mail', 'dev']
        phase_names = ['Drying', 'Maillard', 'Development']
        for phase, name in zip(phases, phase_names):
            ref_gas = ref_sum[f'{phase}_gas']
            cur_gas = current_sum[f'{phase}_gas']
            dev = ((cur_gas - ref_gas) / ref_gas) * 100 if ref_gas != 0 else 0
            st.write(f"{name}: {dev:+.1f}%")

        ref_eff = ref_sum['total_efficiency']
        cur_eff = current_sum['total_efficiency']
        eff_dev = ((cur_eff - ref_eff) / ref_eff) * 100 if ref_eff != 0 else 0
        st.metric("Total Thermal Efficiency (°C/100%·s)", f"{cur_eff:.2f}", delta=f"{eff_dev:+.1f}%")

        st.markdown("**Phase-wise Thermal Efficiency**")
        for phase, name in zip(phases, phase_names):
            ref_eff_phase = ref_sum[f'{phase}_efficiency']
            cur_eff_phase = current_sum[f'{phase}_efficiency']
            eff_dev_phase = ((cur_eff_phase - ref_eff_phase) / ref_eff_phase) * 100 if ref_eff_phase != 0 else 0
            st.write(f"{name}: {cur_eff_phase:.2f} ({eff_dev_phase:+.1f}%)")

        dev_val = current_sum['deviation']
        st.metric("Curve Deviation (℃·s)", f"{dev_val:.0f}")

        # Ranking
        master_sum['dev_rank'] = master_sum['deviation'].rank(ascending=True, method='min')
        master_sum['gas_rank'] = master_sum['total_gas'].rank(ascending=True, method='min')
        st.markdown("### Batch Performance Ranking")
        st.write(f"Curve Deviation Rank: **{int(master_sum[master_sum['batch_id']==selected]['dev_rank'].iloc[0])} / {len(master_sum)}** (1 = best)")
        st.write(f"Energy Consumption Rank: **{int(master_sum[master_sum['batch_id']==selected]['gas_rank'].iloc[0])} / {len(master_sum)}** (1 = best)")

        st.markdown("### Suggestions")
        advice = []
        if current_sum['dev_gas'] > master_sum['dev_gas'].mean():
            advice.append("• Development phase gas is above average. Consider reducing gas earlier.")
        if current_sum['dry_efficiency'] < master_sum['dry_efficiency'].mean():
            advice.append("• Drying phase efficiency is below average. Check charge temperature or airflow.")
        if current_sum['deviation'] > master_sum['deviation'].median():
            advice.append("• Curve deviation is larger than typical. Try to follow the standard profile more closely.")
        if not advice:
            advice.append("✅ This batch is performing well in all aspects.")
        for a in advice:
            st.write(a)

        st.caption("*Lower energy deviation and lower curve deviation indicate better performance.*")

else:
    st.subheader("📈 Statistical Analysis by Start Gas Value")
    if not available_gas:
        st.warning("No valid start gas values found.")
        st.stop()

    selected_gas = st.sidebar.selectbox("Select Start Gas (%) to analyze:", available_gas)
    group_df = master_sum[master_sum['start_gas_int'] == selected_gas].copy()
    st.write(f"**{len(group_df)} batches found with start gas = {selected_gas}%**")
    with st.expander("Show batch IDs in this group"):
        st.write(group_df['batch_id'].tolist())

    st.markdown("### Summary Statistics")
    metrics = ['total_gas', 'dry_gas', 'mail_gas', 'dev_gas',
               'total_efficiency', 'dry_efficiency', 'mail_efficiency', 'dev_efficiency', 'deviation']
    rows = []
    for m in metrics:
        vals = group_df[m].dropna()
        if vals.empty:
            continue
        baseline = ref_sum[m]
        pct = ((vals.mean() - baseline) / baseline * 100) if baseline != 0 else np.nan
        rows.append({
            'Metric': m,
            'Mean': vals.mean(),
            'Std': vals.std(),
            'Min': vals.min(),
            'Max': vals.max(),
            'Baseline (50%)': baseline,
            'Mean vs Baseline (%)': pct
        })
    st.dataframe(pd.DataFrame(rows).round(2), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Box(y=group_df['total_gas'], name=f'{selected_gas}% batches', boxmean='sd', marker_color='lightblue'))
        fig.add_hline(y=ref_sum['total_gas'], line_dash='dash', line_color='red', annotation_text='Baseline (50%)')
        fig.update_layout(title='Total Gas Consumption Distribution', yaxis_title='Gas Integral')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Box(y=group_df['deviation'], name=f'{selected_gas}% batches', boxmean='sd', marker_color='lightblue'))
        fig.add_hline(y=ref_sum['deviation'], line_dash='dash', line_color='red', annotation_text='Baseline (50%)')
        fig.update_layout(title='Curve Deviation Distribution', yaxis_title='Deviation (℃·s)')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Average Temperature Curve Comparison")
    time_uniform = np.arange(0, 801)
    curves = []
    for bid in group_df['batch_id']:
        ts = master_ts[master_ts['batch_id'] == bid].sort_values('time_sec')
        f = interp1d(ts['time_sec'], ts['beantemp'], kind='linear', fill_value='extrapolate')
        curves.append(f(time_uniform))
    if curves:
        mean_c = np.mean(curves, axis=0)
        std_c = np.std(curves, axis=0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_uniform, y=mean_c, name=f'{selected_gas}% Average', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=time_uniform, y=mean_c+std_c, mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=time_uniform, y=mean_c-std_c, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,0,255,0.2)', name='±1 Std Dev'))
        fig.add_trace(go.Scatter(x=ref_ts['time_sec'], y=ref_ts['beantemp'], name='Baseline (50%)', line=dict(color='red', dash='dash')))
        fig.update_layout(xaxis_title='Time (seconds)', yaxis_title='Bean Temperature (°C)')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Energy vs. Quality Scatter")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=group_df['total_gas'], y=group_df['deviation'], mode='markers', text=group_df['batch_id'], name=f'{selected_gas}% batches', marker=dict(color='blue', size=8)))
    fig.add_trace(go.Scatter(x=[ref_sum['total_gas']], y=[ref_sum['deviation']], mode='markers', name='Baseline (50%)', marker=dict(color='red', size=12, symbol='x')))
    fig.update_layout(xaxis_title='Total Gas Consumption', yaxis_title='Curve Deviation (℃·s)', hovermode='closest')
    st.plotly_chart(fig, use_container_width=True)

    st.caption("*Lower energy consumption and lower curve deviation indicate better performance.*")