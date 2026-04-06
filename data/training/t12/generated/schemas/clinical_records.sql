PRAGMA foreign_keys = ON;

CREATE TABLE patient_registry (
  patient_key INTEGER PRIMARY KEY,
  registry_code TEXT NOT NULL UNIQUE,
  birth_date TEXT,
  sex_at_birth TEXT,
  ethnicity_group TEXT,
  residence_region TEXT,
  enrollment_date TEXT,
  mortality_date TEXT
);

CREATE TABLE clinical_encounters (
  encounter_key INTEGER PRIMARY KEY,
  patient_key INTEGER NOT NULL,
  visit_start_ts TEXT NOT NULL,
  visit_end_ts TEXT,
  care_setting TEXT,
  admitting_reason TEXT,
  clinician_unit TEXT,
  discharge_disposition TEXT,
  FOREIGN KEY (patient_key) REFERENCES patient_registry(patient_key)
);

CREATE TABLE laboratory_tests (
  lab_result_key INTEGER PRIMARY KEY,
  patient_key INTEGER NOT NULL,
  encounter_key INTEGER,
  specimen_collected_ts TEXT NOT NULL,
  analyte_name TEXT NOT NULL,
  measured_value REAL,
  value_text TEXT,
  unit_label TEXT,
  reference_low REAL,
  reference_high REAL,
  abnormal_flag TEXT,
  FOREIGN KEY (patient_key) REFERENCES patient_registry(patient_key),
  FOREIGN KEY (encounter_key) REFERENCES clinical_encounters(encounter_key)
);

CREATE TABLE diagnosis_records (
  diagnosis_key INTEGER PRIMARY KEY,
  patient_key INTEGER NOT NULL,
  encounter_key INTEGER,
  diagnosis_date TEXT NOT NULL,
  diagnosis_code TEXT NOT NULL,
  diagnosis_term TEXT,
  diagnosis_context TEXT,
  chronicity_label TEXT,
  FOREIGN KEY (patient_key) REFERENCES patient_registry(patient_key),
  FOREIGN KEY (encounter_key) REFERENCES clinical_encounters(encounter_key)
);

CREATE TABLE treatment_plans (
  treatment_key INTEGER PRIMARY KEY,
  patient_key INTEGER NOT NULL,
  encounter_key INTEGER,
  treatment_start_date TEXT NOT NULL,
  treatment_end_date TEXT,
  intervention_name TEXT NOT NULL,
  dose_amount REAL,
  dose_unit TEXT,
  administration_route TEXT,
  active_flag INTEGER NOT NULL DEFAULT 1 CHECK (active_flag IN (0, 1)),
  FOREIGN KEY (patient_key) REFERENCES patient_registry(patient_key),
  FOREIGN KEY (encounter_key) REFERENCES clinical_encounters(encounter_key)
);

CREATE TABLE vital_measurements (
  vital_key INTEGER PRIMARY KEY,
  patient_key INTEGER NOT NULL,
  encounter_key INTEGER,
  observed_ts TEXT NOT NULL,
  measure_name TEXT NOT NULL,
  numeric_reading REAL,
  reading_unit TEXT,
  reading_category TEXT,
  FOREIGN KEY (patient_key) REFERENCES patient_registry(patient_key),
  FOREIGN KEY (encounter_key) REFERENCES clinical_encounters(encounter_key)
);

CREATE INDEX idx_encounters_patient_time ON clinical_encounters(patient_key, visit_start_ts);
CREATE INDEX idx_labs_patient_time ON laboratory_tests(patient_key, specimen_collected_ts);
CREATE INDEX idx_labs_encounter ON laboratory_tests(encounter_key);
CREATE INDEX idx_diagnosis_patient_date ON diagnosis_records(patient_key, diagnosis_date);
CREATE INDEX idx_treatment_patient_dates ON treatment_plans(patient_key, treatment_start_date, treatment_end_date);
CREATE INDEX idx_vitals_patient_time ON vital_measurements(patient_key, observed_ts);
