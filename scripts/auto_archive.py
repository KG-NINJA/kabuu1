import json, shutil
from datetime import datetime
from pathlib import Path

def auto_archive_forecast():
    src = Path("data/forecast_combined.json")
    if not src.exists():
        print("❌ forecast_combined.json が存在しません。終了。")
        return
    dest_folder = Path("data/history")
    dest_folder.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    dest = dest_folder / f"forecast_{date_str}.json"
    shutil.copy2(src, dest)
    print(f"✅ 予測データを保存: {dest}")

if __name__ == "__main__":
    auto_archive_forecast()
