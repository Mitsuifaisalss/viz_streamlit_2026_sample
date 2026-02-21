import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 画面全体を広く使う設定
st.set_page_config(page_title="日本の製造業ポテンシャルマップ", layout="wide")

st.title("🗾 日本の製造業ポテンシャルマップ")
st.write("e-Statのデータと最新の機械学習（クラスタリング・異常検知）を駆使し、日本の製造業の真の姿を浮き彫りにします。BY 三ツ井ファイサルシャフザード")

# --- 47都道府県の緯度経度データ ---
pref_coords = {
    "北海道": [43.0642, 141.3469], "青森県": [40.8244, 140.7400], "岩手県": [39.7036, 141.1525],
    "宮城県": [38.2682, 140.8694], "秋田県": [39.7186, 140.1025], "山形県": [38.2404, 140.3633],
    "福島県": [37.7503, 140.4675], "茨城県": [36.3414, 140.4468], "栃木県": [36.5658, 139.8836],
    "群馬県": [36.3911, 139.0608], "埼玉県": [35.8572, 139.6490], "千葉県": [35.6047, 140.1232],
    "東京都": [35.6895, 139.6917], "神奈川県": [35.4478, 139.6425], "新潟県": [37.9022, 139.0236],
    "富山県": [36.6953, 137.2113], "石川県": [36.5944, 136.6256], "福井県": [36.0641, 136.2219],
    "山梨県": [35.6639, 138.5683], "長野県": [36.6513, 138.1812], "岐阜県": [35.3912, 136.7223],
    "静岡県": [34.9756, 138.3828], "愛知県": [35.1802, 136.9066], "三重県": [34.7303, 136.5086],
    "滋賀県": [35.0045, 135.8686], "京都府": [35.0116, 135.7681], "大阪府": [34.6937, 135.5023],
    "兵庫県": [34.6913, 135.1830], "奈良県": [34.6851, 135.8048], "和歌山県": [34.2260, 135.1675],
    "鳥取県": [35.5011, 134.2351], "島根県": [35.4723, 133.0505], "岡山県": [34.6618, 133.9350],
    "広島県": [34.3963, 132.4594], "山口県": [34.1859, 131.4714], "徳島県": [34.0657, 134.5594],
    "香川県": [34.3401, 134.0433], "愛媛県": [33.8416, 132.7661], "高知県": [33.5597, 133.5311],
    "福岡県": [33.5902, 130.4017], "佐賀県": [33.2635, 130.2988], "長崎県": [32.7503, 129.8777],
    "熊本県": [32.7898, 130.7417], "大分県": [33.2382, 131.6126], "宮崎県": [31.9111, 131.4239],
    "鹿児島県": [31.5602, 130.5581], "沖縄県": [26.2124, 127.6809]
}

with st.sidebar:
    st.header("データ読み込み")
    uploaded_file = st.file_uploader("CSVをアップロード", type=["csv"])

if uploaded_file is not None:
    try:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='cp932')
            
        # e-Statデータの自動整形
        if "Unnamed: 5" in df.columns:
            df = df.iloc[8:].reset_index(drop=True)
            df = df.rename(columns={
                "Unnamed: 3": "産業名",
                "Unnamed: 5": "都道府県名",
                "Unnamed: 6": "事業所数",
                "Unnamed: 7": "従業者数",
                "Unnamed: 10": "製造品出荷額",
                "Unnamed: 11": "付加価値額"
            })
            df = df[df["都道府県名"] != "全国計"]
            df = df.dropna(subset=["都道府県名"])

        st.subheader("1. フィルター設定")
        
        if "産業名" in df.columns:
            industry_list = df["産業名"].dropna().unique()
            selected_industry = st.selectbox("🔍 分析したい産業ジャンルを選択してください:", industry_list)
            plot_df = df[df["産業名"] == selected_industry].copy()
        else:
            plot_df = df.copy()
            selected_industry = "全体"

        col1, col2 = st.columns(2)
        with col1:
            pref_col = st.selectbox("「都道府県名」の列:", plot_df.columns, index=list(plot_df.columns).index("都道府県名") if "都道府県名" in plot_df.columns else 0)
        with col2:
            val_col = st.selectbox("主役となる指標（出荷額など）:", plot_df.columns, index=list(plot_df.columns).index("製造品出荷額") if "製造品出荷額" in plot_df.columns else 0)

        if st.button("🚀 フルデータアナリティクスを実行"):
            # データ前処理
            plot_df['Clean_Pref'] = plot_df[pref_col].astype(str).str.replace(r'^[0-9]+[\s　]*', '', regex=True)
            coord_df = pd.DataFrame.from_dict(pref_coords, orient='index', columns=['lat', 'lon']).reset_index()
            coord_df.rename(columns={'index': 'Clean_Pref'}, inplace=True)
            merged_df = pd.merge(plot_df, coord_df, on='Clean_Pref', how='inner')
            
            for col in [val_col, "事業所数", "従業者数", "付加価値額"]:
                if col in merged_df.columns:
                    merged_df[col] = pd.to_numeric(merged_df[col].astype(str).str.replace(',', ''), errors='coerce')
            merged_df = merged_df.dropna(subset=[val_col])

            if not merged_df.empty:
                # --- KPIダッシュボード（超プロフェッショナル風） ---
                st.divider()
                st.subheader(f"📊 {selected_industry} のエグゼクティブ・サマリー")
                
                total_val = merged_df[val_col].sum()
                top_pref = merged_df.loc[merged_df[val_col].idxmax(), 'Clean_Pref']
                top_val = merged_df[val_col].max()
                
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("全国総計", f"{total_val:,.0f}")
                kpi2.metric("第1位の都道府県", top_pref)
                kpi3.metric("第1位のシェア", f"{(top_val / total_val)*100:.1f} %")
                
                # --- ここからタブで画面を切り替える ---
                tab1, tab2, tab3 = st.tabs(["🗺️ 空間＆基本分析", "🤖 AIクラスタリング", "🚨 ビジネス分析＆異常検知"])
                
                with tab1:
                    st.write("### ① 地理的分布とランキング")
                    fig_map = px.scatter_mapbox(
                        merged_df, lat="lat", lon="lon", hover_name="Clean_Pref",
                        hover_data={val_col: True, "lat": False, "lon": False},
                        size=val_col, color=val_col, color_continuous_scale=px.colors.sequential.Plasma,
                        size_max=50, zoom=4.5, mapbox_style="carto-positron",
                        title=f"都道府県別 {val_col} の分布"
                    )
                    fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
                    st.plotly_chart(fig_map, use_container_width=True)

                    ranked_df = merged_df.sort_values(by=val_col, ascending=False)
                    fig_bar = px.bar(ranked_df, x='Clean_Pref', y=val_col, text_auto='.2s', color=val_col, color_continuous_scale='Blues')
                    fig_bar.update_xaxes(tickangle=-45)
                    st.plotly_chart(fig_bar, use_container_width=True)

                    st.write("### ② 産業の「質」の分析")
                    scatter_df = merged_df.dropna(subset=["事業所数", "従業者数"])
                    if not scatter_df.empty:
                        fig_scatter = px.scatter(
                            scatter_df, x="事業所数", y=val_col, hover_name="Clean_Pref",
                            size="従業者数", color="Clean_Pref", log_x=True, log_y=True,
                            title="事業所数 vs 出荷額（円の大きさは従業者数）", template="plotly_white"
                        )
                        fig_scatter.update_layout(showlegend=False)
                        st.plotly_chart(fig_scatter, use_container_width=True)

                with tab2:
                    st.write("### 機械学習による都道府県クラスタリング (K-Means)")
                    st.write("事業所数・従業者数・出荷額のバランスから、AIが47都道府県を4つのタイプに自動分類しました。")
                    ml_features = ["事業所数", "従業者数", val_col]
                    ml_df = merged_df.dropna(subset=ml_features).copy()
                    
                    if len(ml_df) > 4:
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(ml_df[ml_features])
                        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                        ml_df['クラスター'] = kmeans.fit_predict(scaled_data)
                        ml_df['クラスター'] = 'Type ' + ml_df['クラスター'].astype(str)
                        
                        col_ml1, col_ml2 = st.columns([2, 1])
                        with col_ml1:
                            fig_3d = px.scatter_3d(
                                ml_df, x="事業所数", y="従業者数", z=val_col,
                                color="クラスター", hover_name="Clean_Pref",
                                color_discrete_sequence=px.colors.qualitative.Set1
                            )
                            fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0))
                            st.plotly_chart(fig_3d, use_container_width=True)
                        with col_ml2:
                            for cluster_name in sorted(ml_df['クラスター'].unique()):
                                prefs = ml_df[ml_df['クラスター'] == cluster_name]['Clean_Pref'].tolist()
                                st.info(f"**{cluster_name} ({len(prefs)}県):**\n {', '.join(prefs)}")

                with tab3:
                    st.write("### ① 寡占化の証明：パレート分析（ABC分析）")
                    st.write("「少数のトップ県だけで、全国の大部分を占めているのではないか？」を検証します。赤い線が累積のシェア（%）を表します。")
                    
                    # パレート図の計算と描画
                    pareto_df = ranked_df.copy()
                    pareto_df['累積割合(%)'] = pareto_df[val_col].cumsum() / pareto_df[val_col].sum() * 100
                    
                    fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_pareto.add_trace(go.Bar(x=pareto_df['Clean_Pref'], y=pareto_df[val_col], name=val_col, marker_color='royalblue'), secondary_y=False)
                    fig_pareto.add_trace(go.Scatter(x=pareto_df['Clean_Pref'], y=pareto_df['累積割合(%)'], name="累積割合(%)", marker_color='red', mode='lines+markers'), secondary_y=True)
                    fig_pareto.update_yaxes(title_text="累積割合(%)", range=[0, 105], secondary_y=True)
                    fig_pareto.update_layout(title_text="パレート図 (80:20の法則の検証)", template="plotly_white")
                    st.plotly_chart(fig_pareto, use_container_width=True)
                    
                    # 80%ラインに到達する県数を計算
                    top_80_count = len(pareto_df[pareto_df['累積割合(%)'] <= 80]) + 1
                    st.success(f"💡 分析結果: 上位 **{top_80_count}県** だけで、全国の出荷額の約80%を占めていることが証明されました。")

                    st.divider()
                    st.write("### ② AIによる異常検知（Isolation Forest）")
                    st.write("他の県と明らかに違う、特異なバランス（異常値）を持つ都道府県をAIが自動で検知します。")
                    
                    if len(ml_df) > 10:
                        # 異常検知アルゴリズムの実行
                        iso_forest = IsolationForest(contamination=0.05, random_state=42) # 全体の5%を異常と判定
                        ml_df['異常判定'] = iso_forest.fit_predict(scaled_data)
                        ml_df['状態'] = ml_df['異常判定'].map({1: '通常', -1: '⚠️ 特異（異常検知）'})
                        
                        anomalies = ml_df[ml_df['異常判定'] == -1]
                        
                        fig_iso = px.scatter(
                            ml_df, x="事業所数", y=val_col, hover_name="Clean_Pref",
                            color="状態", color_discrete_map={'通常': 'lightgray', '⚠️ 特異（異常検知）': 'red'},
                            log_x=True, log_y=True, title="異常検知プロット（赤い点が特異な県）"
                        )
                        st.plotly_chart(fig_iso, use_container_width=True)
                        
                        if not anomalies.empty:
                            st.warning(f"🚨 以下の県が「特異な動きをしている」とAIが判定しました:\n **{', '.join(anomalies['Clean_Pref'].tolist())}**\n\n (理由の仮説：工場が少ないのに出荷額が異常に高い、またはその逆など、他の県にはない特殊なビジネス構造を持っている可能性があります)")
                        else:
                            st.info("特異な動きをしている県は検知されませんでした。")
                    else:
                        st.warning("異常検知を行うためのデータが不足しています。")

            else:
                st.error("分析できるデータが見つかりませんでした。")
                
    except Exception as e:
         st.error(f"エラーが発生しました: {e}")
else:
    st.info("左側のサイドバーから、CSVファイルをアップロードしてください。")
