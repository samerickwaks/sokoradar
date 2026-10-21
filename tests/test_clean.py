import numpy as np
import pandas as pd
import pytest

from sokoradar.clean import clean_prices, commodity_map, market_base, unit_to_kg


@pytest.mark.parametrize("unit,kg", [("KG", 1.0), ("90 KG", 90.0), ("200 G", 0.2), ("500 ML", 0.5), ("L", 1.0)])
def test_unit_to_kg(unit, kg):
    assert unit_to_kg(unit) == pytest.approx(kg)


@pytest.mark.parametrize("unit", ["Unit", "Bunch", None, "Head"])
def test_unit_to_kg_rejects_counts(unit):
    assert np.isnan(unit_to_kg(unit))


def test_market_base_merges_place_hints_but_keeps_numbered_markets():
    assert market_base("Garissa town (Garissa)") == market_base("Garissa") == "garissa"
    assert market_base("Marigat (Baringo)") == market_base("Marigat town (Baringo)")
    assert market_base("Kalobeyei (Village 1)") != market_base("Kalobeyei (Village 2)")
    assert market_base("Kakuma 3") != market_base("Kakuma 4")


def test_commodity_map_covers_maize_variants():
    cmap = commodity_map()
    assert cmap["Maize (white)"] == cmap["Maize"] == "maize"
    assert cmap["Maize flour (white)"] == "maize_flour"
    assert "Sugar" not in cmap


def _raw():
    months = pd.date_range("2018-01-15", periods=40, freq="MS") + pd.Timedelta(days=14)
    rows = []
    for i, d in enumerate(months):
        # Same market under two spellings, one in 90 kg bags; one obvious unit slip.
        rows.append(dict(date=d, admin1="North Eastern", admin2="Mandera", market="Mandera", market_id="1",
                         latitude=3.94, longitude=41.86, category="cereals", commodity="Maize (white)",
                         commodity_id="67", unit="KG", priceflag="actual", pricetype="Retail", currency="KES",
                         price=50 + i % 5, usdprice=0.5))
        rows.append(dict(date=d, admin1="North Eastern", admin2="Mandera", market="Mandera town", market_id="2",
                         latitude=3.935, longitude=41.855, category="cereals", commodity="Maize",
                         commodity_id="51", unit="90 KG", priceflag="actual", pricetype="Retail", currency="KES",
                         price=90 * (50 + i % 5) * (40 if i == 20 else 1), usdprice=45.0))
    return pd.DataFrame(rows)


def test_clean_prices_merges_units_names_and_drops_outliers():
    res = clean_prices(_raw())
    p = res.prices
    assert p["series_id"].nunique() == 1
    assert p["series_id"].iloc[0] == "mandera|maize|retail"
    assert p["price"].between(45, 60).all()           # 90 kg bags converted to KES/kg; slip removed
    assert p["month"].is_monotonic_increasing
    assert res.markets.loc[0, "county"] == "Mandera"
    assert res.quality["markets_merged"] == 1


def test_clean_prices_enforces_minimum_history():
    raw = _raw()
    short = raw[raw["date"] < "2019-06-01"]
    assert clean_prices(short).prices.empty
