# 地理空間特徴量のテストスクリプト
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# 1. 橋梁座標確認
print("=" * 60)
print("橋梁座標範囲の確認")
print("=" * 60)
bridges = pd.read_excel('data/YamaguchiPrefBridgeListOpen251122_154891.xlsx', header=[0,1])
bridges.columns = [col[0] if 'Unnamed' in str(col[1]) else col[0] for col in bridges.columns]
bridges['緯度'] = pd.to_numeric(bridges['緯度'], errors='coerce')
bridges['経度'] = pd.to_numeric(bridges['経度'], errors='coerce')
valid = bridges[bridges['緯度'].notna() & bridges['経度'].notna()]
print(f"有効な座標: {len(valid)}件")
print(f"経度範囲: {valid['経度'].min():.5f} 〜 {valid['経度'].max():.5f}")
print(f"緯度範囲: {valid['緯度'].min():.5f} 〜 {valid['緯度'].max():.5f}")

# 2. 河川データ確認
print("\n" + "=" * 60)
print("河川データの確認")
print("=" * 60)
rivers = gpd.read_file('data/RiverDataKokudo/W05-08_35_GML/W05-08_35-g_Stream.shp')
print(f"CRS: {rivers.crs}")
print(f"河川数: {len(rivers)}件")
print(f"Bounds: {rivers.total_bounds}")

# CRS設定
if rivers.crs is None:
    rivers.set_crs("EPSG:4326", inplace=True)
    print("CRSをEPSG:4326に設定しました")

# 3. 海岸線データ確認
print("\n" + "=" * 60)
print("海岸線データの確認")
print("=" * 60)
coastline = gpd.read_file('data/KaigansenDataKokudo/C23-06_35_GML/C23-06_35-g_Coastline.shp')
print(f"CRS: {coastline.crs}")
print(f"海岸線数: {len(coastline)}件")
print(f"Bounds: {coastline.total_bounds}")

if coastline.crs is None:
    coastline.set_crs("EPSG:4326", inplace=True)
    print("CRSをEPSG:4326に設定しました")

# 4. サンプル橋梁でテスト
print("\n" + "=" * 60)
print("サンプル橋梁でのテスト（先頭10件）")
print("=" * 60)
sample = valid.head(10).copy()
gdf_sample = gpd.GeoDataFrame(
    sample,
    geometry=gpd.points_from_xy(sample['経度'], sample['緯度']),
    crs="EPSG:4326"
)

# 河川判定
rivers_buffer = rivers.buffer(0.0005).unary_union
gdf_sample['under_river'] = gdf_sample.geometry.apply(
    lambda point: 1 if rivers_buffer.contains(point) else 0
)

# 海岸線距離
coastline_union = coastline.unary_union
gdf_sample['distance_to_coast_km'] = gdf_sample.geometry.apply(
    lambda point: coastline_union.distance(point) * 111
)

print("\n結果サマリー:")
for idx, row in gdf_sample.iterrows():
    print(f"  {idx}: 経度={row['経度']:.5f}, 緯度={row['緯度']:.5f}, 河川={row['under_river']}, 海岸距離={row['distance_to_coast_km']:.2f}km")

print(f"\n河川上の橋梁: {gdf_sample['under_river'].sum()}件 / {len(gdf_sample)}件")
print(f"海岸線距離: 最小={gdf_sample['distance_to_coast_km'].min():.2f}km, 最大={gdf_sample['distance_to_coast_km'].max():.2f}km, 平均={gdf_sample['distance_to_coast_km'].mean():.2f}km")
