from siv_monitor import INCIDENT_TYPES, SENSORS, SimulationConfig, incidents, simulate


def test_simulation_is_reproducible():
    a = simulate(SimulationConfig(n_minutes=600, seed=3))
    b = simulate(SimulationConfig(n_minutes=600, seed=3))
    assert a.equals(b)


def test_simulation_shape_and_labels():
    df = simulate(SimulationConfig(n_minutes=2000, seed=1, incident_rate_per_day=20))
    assert len(df) == 2000
    assert set(SENSORS) <= set(df.columns)
    assert set(df["incident"]) <= set(INCIDENT_TYPES) | {"aucun"}
    assert df["label"].sum() > 0
    assert (df.loc[df["label"] == 0, "incident"] == "aucun").all()


def test_incidents_are_contiguous_segments():
    df = simulate(SimulationConfig(n_minutes=3000, seed=2, incident_rate_per_day=20))
    for start, end, kind in incidents(df):
        assert df["label"].iloc[start:end].all()
        assert (df["incident"].iloc[start:end] == kind).all()
