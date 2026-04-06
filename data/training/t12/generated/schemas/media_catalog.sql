PRAGMA foreign_keys = ON;

CREATE TABLE catalog_items (
    item_key INTEGER PRIMARY KEY,
    item_code TEXT NOT NULL UNIQUE,
    canonical_title TEXT NOT NULL,
    franchise_code INTEGER,
    origin_language_code INTEGER,
    first_publication_year INTEGER,
    default_rating_key INTEGER,
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1))
);

CREATE TABLE content_ratings (
    rating_key INTEGER PRIMARY KEY,
    rating_code TEXT NOT NULL UNIQUE,
    authority_name TEXT NOT NULL,
    maturity_band INTEGER NOT NULL,
    descriptor_text TEXT
);

CREATE TABLE edition_releases (
    release_key INTEGER PRIMARY KEY,
    item_key INTEGER NOT NULL,
    release_code TEXT NOT NULL UNIQUE,
    release_title TEXT NOT NULL,
    publisher_code INTEGER NOT NULL,
    release_date DATE,
    print_wave INTEGER,
    language_code INTEGER,
    default_format_key INTEGER,
    FOREIGN KEY (item_key) REFERENCES catalog_items(item_key),
    FOREIGN KEY (default_format_key) REFERENCES format_variants(format_key)
);

CREATE TABLE format_variants (
    format_key INTEGER PRIMARY KEY,
    release_key INTEGER NOT NULL,
    format_code TEXT NOT NULL,
    medium_type TEXT NOT NULL,
    package_style_code INTEGER,
    list_price NUMERIC,
    barcode_text TEXT,
    FOREIGN KEY (release_key) REFERENCES edition_releases(release_key),
    UNIQUE (release_key, format_code)
);

CREATE TABLE regional_availability (
    availability_key INTEGER PRIMARY KEY,
    format_key INTEGER NOT NULL,
    territory_code INTEGER NOT NULL,
    channel_code INTEGER NOT NULL,
    availability_status_code INTEGER NOT NULL,
    available_from DATE,
    available_to DATE,
    stock_bucket INTEGER,
    FOREIGN KEY (format_key) REFERENCES format_variants(format_key)
);

CREATE TABLE translation_records (
    translation_key INTEGER PRIMARY KEY,
    item_key INTEGER NOT NULL,
    locale_code INTEGER NOT NULL,
    translated_title TEXT NOT NULL,
    translated_subtitle TEXT,
    script_code INTEGER,
    translator_credit TEXT,
    is_official INTEGER NOT NULL DEFAULT 1 CHECK (is_official IN (0, 1)),
    FOREIGN KEY (item_key) REFERENCES catalog_items(item_key),
    UNIQUE (item_key, locale_code, translated_title)
);

CREATE TABLE collector_notes (
    note_key INTEGER PRIMARY KEY,
    release_key INTEGER,
    format_key INTEGER,
    note_kind_code INTEGER NOT NULL,
    note_sequence INTEGER NOT NULL,
    note_text TEXT NOT NULL,
    source_region_code INTEGER,
    effective_on DATE,
    retired_on DATE,
    FOREIGN KEY (release_key) REFERENCES edition_releases(release_key),
    FOREIGN KEY (format_key) REFERENCES format_variants(format_key)
);

CREATE INDEX idx_catalog_items_rating
    ON catalog_items (default_rating_key);

CREATE INDEX idx_edition_releases_item
    ON edition_releases (item_key, release_date);

CREATE INDEX idx_format_variants_release
    ON format_variants (release_key, medium_type);

CREATE INDEX idx_regional_availability_format
    ON regional_availability (format_key, territory_code, channel_code);

CREATE INDEX idx_translation_records_item_locale
    ON translation_records (item_key, locale_code);

CREATE INDEX idx_collector_notes_targets
    ON collector_notes (release_key, format_key, note_kind_code);
