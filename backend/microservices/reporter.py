import io
import re
from collections import Counter
from typing import Any

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def get_filtered_accidents(
    accidents: list[dict[str, Any]],
    start_year: int | None = None,
    end_year: int | None = None,
    rain_only: bool | None = None,
    vehicle_type: str | None = None,
    city: str | None = None
) -> list[dict[str, Any]]:
    filtered = []
    for acc in accidents:
        # Date & Year filter
        year = None
        date_iso = acc.get("date_iso")
        if date_iso:
            m = re.search(r"(\d{4})", str(date_iso))
            if m:
                year = int(m.group(1))
        
        if year is not None:
            if start_year is not None and year < start_year:
                continue
            if end_year is not None and year > end_year:
                continue

        # Rain / Weather filter
        is_rainy = False
        data_orig = acc.get("data_original")
        if data_orig:
            orig_str = str(data_orig).upper()
            if "LLUVIA" in orig_str or "LLUVIOSO" in orig_str or "HUMEDO" in orig_str:
                is_rainy = True
        
        if rain_only is not None:
            if rain_only and not is_rainy:
                continue
            if not rain_only and is_rainy:
                continue

        # Vehicle filter
        vehicles_str = str(acc.get("vehicles", "")).upper()
        if vehicle_type:
            if vehicle_type.upper() not in vehicles_str:
                continue

        # City filter
        extr = acc.get("extraccion")
        muni_val = None
        if isinstance(extr, dict) and extr.get("BARRIO_O_MUNICIPIO"):
            muni_val = extr["BARRIO_O_MUNICIPIO"].get("value")
        
        if muni_val is None and data_orig:
            for k, v in data_orig.items():
                if "MUNICIPIO" in k.upper() or "CIUDAD" in k.upper():
                    muni_val = str(v)
                    break
        
        if city:
            if not muni_val or city.upper() not in muni_val.upper():
                continue

        filtered.append(acc)
    return filtered


def generate_report_chart(accidents: list[dict[str, Any]]) -> bytes:
    years = []
    cities = []
    vehicles_list = []
    rain_counts = {"Lluvia / Humedad": 0, "Despejado / Seco": 0}

    for acc in accidents:
        # Year
        date_iso = acc.get("date_iso")
        if date_iso:
            m = re.search(r"(\d{4})", str(date_iso))
            if m:
                years.append(m.group(1))
        
        # City
        extr = acc.get("extraccion")
        muni_val = None
        if isinstance(extr, dict) and extr.get("BARRIO_O_MUNICIPIO"):
            muni_val = extr["BARRIO_O_MUNICIPIO"].get("value")
        if not muni_val and acc.get("data_original"):
            for k, v in acc["data_original"].items():
                if "MUNICIPIO" in k.upper() or "CIUDAD" in k.upper():
                    muni_val = str(v)
                    break
        if muni_val:
            muni_upper = muni_val.upper().strip()
            if muni_upper not in ["DESCONOCIDO", "NO REGISTRA", "SIN DATOS", "PENDIENTE", "DESCONOCIDA", "UNKNOWN"]:
                cities.append(muni_upper)

        # Rain
        is_rainy = False
        if acc.get("data_original"):
            orig_str = str(acc["data_original"]).upper()
            if "LLUVIA" in orig_str or "LLUVIOSO" in orig_str or "HUMEDO" in orig_str:
                is_rainy = True
        
        if is_rainy:
            rain_counts["Lluvia / Humedad"] += 1
        else:
            rain_counts["Despejado / Seco"] += 1

        # Vehicles
        v_str = str(acc.get("vehicles", "NO REGISTRA")).upper()
        for part in v_str.split(","):
            part_cleaned = re.sub(r"\(\d+\)", "", part).strip()
            if part_cleaned and part_cleaned not in ["NO REGISTRA", "DESCONOCIDO", "SIN DATOS", "UNKNOWN"]:
                if ":" in part_cleaned:
                    part_cleaned = part_cleaned.split(":")[-1].strip()
                vehicles_list.append(part_cleaned)

    # Plot figure setup
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor('#1e1e2e')  # Sleek dark blue-grey theme background

    for ax in axs.flat:
        ax.set_facecolor('#252538')
        ax.tick_params(colors='#cdd6f4')
        ax.xaxis.label.set_color('#cdd6f4')
        ax.yaxis.label.set_color('#cdd6f4')
        ax.title.set_color('#cdd6f4')
        for spine in ax.spines.values():
            spine.set_color('#45475a')

    # Subplot 1: Accidents by Year
    year_counter = Counter(years)
    sorted_years = sorted(year_counter.keys())
    year_vals = [year_counter[y] for y in sorted_years]
    axs[0, 0].bar(sorted_years, year_vals, color='#89b4fa', edgecolor='#74c7ec')
    axs[0, 0].set_title("Accidentes por Año", fontsize=12, fontweight='bold')
    axs[0, 0].set_ylabel("Cantidad")
    axs[0, 0].grid(axis='y', linestyle='--', alpha=0.3)

    # Subplot 2: Accidents by City
    city_counter = Counter(cities)
    top_cities = city_counter.most_common(5)
    city_names = [c[0][:15] for c in top_cities]
    city_vals = [c[1] for c in top_cities]
    axs[0, 1].bar(city_names, city_vals, color='#f5c2e7', edgecolor='#f5bde6')
    axs[0, 1].set_title("Top 5 Ciudades", fontsize=12, fontweight='bold')
    axs[0, 1].set_ylabel("Cantidad")
    axs[0, 1].grid(axis='y', linestyle='--', alpha=0.3)

    # Subplot 3: Accidents by Vehicle Type
    vehicle_counter = Counter(vehicles_list)
    top_vehicles = vehicle_counter.most_common(5)
    veh_names = [v[0][:15] for v in top_vehicles]
    veh_vals = [v[1] for v in top_vehicles]
    axs[1, 0].barh(veh_names, veh_vals, color='#a6e3a1', edgecolor='#94e2d5')
    axs[1, 0].set_title("Top 5 Tipo de Vehículos", fontsize=12, fontweight='bold')
    axs[1, 0].set_xlabel("Cantidad")
    axs[1, 0].grid(axis='x', linestyle='--', alpha=0.3)

    # Subplot 4: Weather Conditions (Lluvia vs Dry)
    labels = list(rain_counts.keys())
    sizes = list(rain_counts.values())
    colors = ['#74c7ec', '#f9e2af']
    if sum(sizes) > 0:
        patches, texts, autotexts = axs[1, 1].pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=140,
            colors=colors
        )
        for t in texts:
            t.set_color('#cdd6f4')  # Outer label is white for dark bg contrast
        for at in autotexts:
            at.set_color('#11111b')  # Inner percentage is dark/black for contrast on light slice
            at.set_weight('bold')
    axs[1, 1].set_title("Condición de Lluvia", fontsize=12, fontweight='bold')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
