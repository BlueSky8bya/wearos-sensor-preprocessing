"""PPG 로딩 (전처리)
원본: ppg_data_visualization.ipynb
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

def load_ppg_data_from_folder(folder_path, sensor_prefix="PpgGreen_"):
    """
    폴더 안의 PpgGreen_*.csv 파일을 전부 읽어 하나의 DataFrame으로 합쳐 반환.
    (os.listdir + startswith 사용)
    """
    try:
        all_files = os.listdir(folder_path)
    except FileNotFoundError:
        print(f"❌ 폴더를 찾을 수 없습니다: {folder_path}")
        return None
    except Exception as e:
        print(f"❌ os.listdir 오류: {folder_path}, {e}")
        return None

    # PpgGreen_로 시작하는 파일만 골라서 읽기
    sensor_files = [f for f in all_files if f.startswith(sensor_prefix)]
    if not sensor_files:
        print(f"⚠️ {folder_path} 폴더에 {sensor_prefix} 관련 CSV 파일이 없습니다.")
        return None

    df_list = []
    for file_name in sensor_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            df = pd.read_csv(file_path, header=0, names=["time", "value"])
            df_list.append(df)
        except Exception as e:
            print(f"❌ 파일 읽기 오류: {file_name} - {e}")

    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        return combined_df
    else:
        return None
