PRAGMA foreign_keys = ON;

CREATE TABLE racing_series (
  series_key INTEGER PRIMARY KEY,
  series_title TEXT NOT NULL UNIQUE,
  governing_region TEXT,
  first_season_year INTEGER,
  points_rule_summary TEXT
);

CREATE TABLE race_venues (
  venue_key INTEGER PRIMARY KEY,
  venue_title TEXT NOT NULL,
  municipality TEXT NOT NULL,
  nation TEXT NOT NULL,
  latitude_deg REAL,
  longitude_deg REAL,
  altitude_meters INTEGER,
  timezone_name TEXT,
  UNIQUE (venue_title, municipality, nation)
);

CREATE TABLE competition_events (
  event_key INTEGER PRIMARY KEY,
  series_key INTEGER NOT NULL,
  venue_key INTEGER NOT NULL,
  season_year INTEGER NOT NULL,
  round_number INTEGER NOT NULL,
  event_label TEXT NOT NULL,
  opening_date TEXT,
  closing_date TEXT,
  scheduled_laps INTEGER,
  scheduled_distance_km REAL,
  FOREIGN KEY (series_key) REFERENCES racing_series(series_key),
  FOREIGN KEY (venue_key) REFERENCES race_venues(venue_key),
  UNIQUE (series_key, season_year, round_number)
);

CREATE TABLE competition_phases (
  phase_key INTEGER PRIMARY KEY,
  event_key INTEGER NOT NULL,
  phase_type TEXT NOT NULL CHECK (phase_type IN ('practice', 'qualifying', 'race')),
  phase_sequence INTEGER NOT NULL,
  phase_name TEXT NOT NULL,
  phase_start_ts TEXT,
  phase_end_ts TEXT,
  FOREIGN KEY (event_key) REFERENCES competition_events(event_key),
  UNIQUE (event_key, phase_type, phase_sequence)
);

CREATE TABLE team_entries (
  entry_key INTEGER PRIMARY KEY,
  series_key INTEGER NOT NULL,
  entrant_name TEXT NOT NULL,
  chassis_model TEXT,
  power_unit_supplier TEXT,
  base_nation TEXT,
  FOREIGN KEY (series_key) REFERENCES racing_series(series_key),
  UNIQUE (series_key, entrant_name, chassis_model)
);

CREATE TABLE driver_profiles (
  driver_key INTEGER PRIMARY KEY,
  given_name TEXT NOT NULL,
  family_name TEXT NOT NULL,
  date_of_birth TEXT,
  home_nation TEXT,
  permanent_number INTEGER,
  profile_tag TEXT UNIQUE
);

CREATE TABLE race_results (
  result_key INTEGER PRIMARY KEY,
  event_key INTEGER NOT NULL,
  phase_key INTEGER NOT NULL,
  entry_key INTEGER NOT NULL,
  driver_key INTEGER NOT NULL,
  grid_slot INTEGER,
  finishing_rank INTEGER,
  result_status TEXT,
  elapsed_millis INTEGER,
  completed_laps INTEGER,
  points_awarded REAL,
  completion_fraction REAL CHECK (completion_fraction >= 0 AND completion_fraction <= 1),
  FOREIGN KEY (event_key) REFERENCES competition_events(event_key),
  FOREIGN KEY (phase_key) REFERENCES competition_phases(phase_key),
  FOREIGN KEY (entry_key) REFERENCES team_entries(entry_key),
  FOREIGN KEY (driver_key) REFERENCES driver_profiles(driver_key),
  UNIQUE (event_key, phase_key, driver_key)
);

CREATE TABLE lap_records (
  lap_record_key INTEGER PRIMARY KEY,
  event_key INTEGER NOT NULL,
  phase_key INTEGER NOT NULL,
  driver_key INTEGER NOT NULL,
  entry_key INTEGER NOT NULL,
  lap_index INTEGER NOT NULL,
  lap_millis INTEGER NOT NULL,
  sector1_millis INTEGER,
  sector2_millis INTEGER,
  sector3_millis INTEGER,
  pit_flag INTEGER NOT NULL DEFAULT 0 CHECK (pit_flag IN (0, 1)),
  FOREIGN KEY (event_key) REFERENCES competition_events(event_key),
  FOREIGN KEY (phase_key) REFERENCES competition_phases(phase_key),
  FOREIGN KEY (driver_key) REFERENCES driver_profiles(driver_key),
  FOREIGN KEY (entry_key) REFERENCES team_entries(entry_key),
  UNIQUE (event_key, phase_key, driver_key, lap_index)
);

CREATE TABLE qualifying_sessions (
  qualifying_key INTEGER PRIMARY KEY,
  event_key INTEGER NOT NULL,
  phase_key INTEGER NOT NULL,
  driver_key INTEGER NOT NULL,
  entry_key INTEGER NOT NULL,
  segment_one_millis INTEGER,
  segment_two_millis INTEGER,
  segment_three_millis INTEGER,
  best_segment_millis INTEGER,
  qualifying_rank INTEGER,
  FOREIGN KEY (event_key) REFERENCES competition_events(event_key),
  FOREIGN KEY (phase_key) REFERENCES competition_phases(phase_key),
  FOREIGN KEY (driver_key) REFERENCES driver_profiles(driver_key),
  FOREIGN KEY (entry_key) REFERENCES team_entries(entry_key),
  UNIQUE (event_key, phase_key, driver_key)
);

CREATE INDEX idx_events_series_season ON competition_events(series_key, season_year);
CREATE INDEX idx_events_venue ON competition_events(venue_key);
CREATE INDEX idx_phases_event_type ON competition_phases(event_key, phase_type);
CREATE INDEX idx_results_event_rank ON race_results(event_key, finishing_rank);
CREATE INDEX idx_results_driver ON race_results(driver_key);
CREATE INDEX idx_laps_phase_driver ON lap_records(phase_key, driver_key);
CREATE INDEX idx_qual_phase_rank ON qualifying_sessions(phase_key, qualifying_rank);
