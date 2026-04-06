CREATE TABLE chemical_compounds (
    compound_key INTEGER PRIMARY KEY,
    registry_code TEXT NOT NULL UNIQUE,
    common_name TEXT NOT NULL,
    formula_notation TEXT,
    phase_state TEXT,
    discovery_year INTEGER,
    is_synthetic INTEGER NOT NULL DEFAULT 0 CHECK (is_synthetic IN (0,1))
);

CREATE TABLE element_nodes (
    element_key INTEGER PRIMARY KEY,
    symbol_code TEXT NOT NULL,
    element_name TEXT NOT NULL,
    atomic_index INTEGER NOT NULL,
    group_label TEXT,
    period_number INTEGER,
    electronegativity_score REAL,
    UNIQUE(symbol_code, atomic_index)
);

CREATE TABLE molecular_bonds (
    linkage_key INTEGER PRIMARY KEY,
    parent_compound_key INTEGER NOT NULL,
    source_element_key INTEGER NOT NULL,
    target_element_key INTEGER NOT NULL,
    bond_order TEXT NOT NULL,
    connection_strength REAL,
    is_aromatic INTEGER NOT NULL DEFAULT 0 CHECK (is_aromatic IN (0,1)),
    FOREIGN KEY (parent_compound_key) REFERENCES chemical_compounds(compound_key),
    FOREIGN KEY (source_element_key) REFERENCES element_nodes(element_key),
    FOREIGN KEY (target_element_key) REFERENCES element_nodes(element_key)
);

CREATE TABLE compound_properties (
    property_key INTEGER PRIMARY KEY,
    subject_compound_key INTEGER NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    metric_unit TEXT,
    measurement_method TEXT,
    observed_on DATE,
    FOREIGN KEY (subject_compound_key) REFERENCES chemical_compounds(compound_key)
);

CREATE TABLE safety_classifications (
    class_key INTEGER PRIMARY KEY,
    related_compound_key INTEGER NOT NULL,
    class_system TEXT NOT NULL,
    hazard_tier TEXT,
    risk_phrase TEXT,
    threshold_percent REAL CHECK (threshold_percent >= 0 AND threshold_percent <= 100),
    denominator_scope TEXT,
    FOREIGN KEY (related_compound_key) REFERENCES chemical_compounds(compound_key)
);

CREATE TABLE structural_features (
    feature_key INTEGER PRIMARY KEY,
    owner_compound_key INTEGER NOT NULL,
    feature_label TEXT NOT NULL,
    feature_count INTEGER,
    ring_system_type TEXT,
    topology_rank INTEGER,
    FOREIGN KEY (owner_compound_key) REFERENCES chemical_compounds(compound_key)
);
