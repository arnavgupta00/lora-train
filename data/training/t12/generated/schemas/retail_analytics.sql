PRAGMA foreign_keys = ON;

CREATE TABLE shoppers (
    shopper_key INTEGER PRIMARY KEY,
    shopper_code TEXT NOT NULL UNIQUE,
    given_name TEXT NOT NULL,
    family_name TEXT NOT NULL,
    signup_date DATE NOT NULL,
    birth_year INTEGER,
    home_region_code INTEGER NOT NULL,
    segment_code INTEGER NOT NULL,
    contact_opt_in INTEGER NOT NULL DEFAULT 1 CHECK (contact_opt_in IN (0, 1))
);

CREATE TABLE store_locations (
    store_key INTEGER PRIMARY KEY,
    store_code TEXT NOT NULL UNIQUE,
    store_name TEXT NOT NULL,
    market_code INTEGER NOT NULL,
    region_code INTEGER NOT NULL,
    opened_on DATE,
    floor_area_sqft INTEGER
);

CREATE TABLE product_catalog (
    product_key INTEGER PRIMARY KEY,
    sku_code TEXT NOT NULL UNIQUE,
    product_title TEXT NOT NULL,
    department_code INTEGER NOT NULL,
    category_code INTEGER NOT NULL,
    subcategory_code INTEGER,
    standard_price NUMERIC NOT NULL,
    launch_date DATE,
    is_private_label INTEGER NOT NULL DEFAULT 0 CHECK (is_private_label IN (0, 1))
);

CREATE TABLE payment_methods (
    payment_key INTEGER PRIMARY KEY,
    payment_code TEXT NOT NULL UNIQUE,
    payment_channel TEXT NOT NULL,
    issuer_group_code INTEGER,
    is_card_present INTEGER NOT NULL DEFAULT 1 CHECK (is_card_present IN (0, 1))
);

CREATE TABLE loyalty_memberships (
    membership_key INTEGER PRIMARY KEY,
    shopper_key INTEGER NOT NULL,
    loyalty_program_code INTEGER NOT NULL,
    tier_code INTEGER NOT NULL,
    enrolled_on DATE NOT NULL,
    suspended_on DATE,
    status_code INTEGER NOT NULL,
    points_balance INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (shopper_key) REFERENCES shoppers(shopper_key)
);

CREATE TABLE purchase_records (
    purchase_key INTEGER PRIMARY KEY,
    receipt_number TEXT NOT NULL,
    line_sequence INTEGER NOT NULL,
    shopper_key INTEGER NOT NULL,
    store_key INTEGER NOT NULL,
    product_key INTEGER NOT NULL,
    payment_key INTEGER NOT NULL,
    purchased_at DATETIME NOT NULL,
    business_year INTEGER NOT NULL,
    business_month INTEGER NOT NULL CHECK (business_month BETWEEN 1 AND 12),
    units INTEGER NOT NULL,
    gross_amount NUMERIC NOT NULL,
    markdown_amount NUMERIC NOT NULL DEFAULT 0,
    net_amount NUMERIC NOT NULL,
    FOREIGN KEY (shopper_key) REFERENCES shoppers(shopper_key),
    FOREIGN KEY (store_key) REFERENCES store_locations(store_key),
    FOREIGN KEY (product_key) REFERENCES product_catalog(product_key),
    FOREIGN KEY (payment_key) REFERENCES payment_methods(payment_key),
    UNIQUE (receipt_number, line_sequence)
);

CREATE TABLE monthly_summaries (
    summary_key INTEGER PRIMARY KEY,
    shopper_key INTEGER,
    store_key INTEGER,
    category_code INTEGER,
    summary_year INTEGER NOT NULL,
    summary_month INTEGER NOT NULL CHECK (summary_month BETWEEN 1 AND 12),
    total_visits INTEGER NOT NULL,
    distinct_products INTEGER NOT NULL,
    total_units INTEGER NOT NULL,
    total_net_amount NUMERIC NOT NULL,
    average_ticket NUMERIC,
    segment_code INTEGER,
    generated_at DATETIME NOT NULL,
    FOREIGN KEY (shopper_key) REFERENCES shoppers(shopper_key),
    FOREIGN KEY (store_key) REFERENCES store_locations(store_key)
);

CREATE INDEX idx_purchase_records_shopper_period
    ON purchase_records (shopper_key, business_year, business_month, purchased_at);

CREATE INDEX idx_purchase_records_store_period
    ON purchase_records (store_key, business_year, business_month);

CREATE INDEX idx_purchase_records_product
    ON purchase_records (product_key);

CREATE INDEX idx_product_catalog_category
    ON product_catalog (department_code, category_code);

CREATE INDEX idx_monthly_summaries_period
    ON monthly_summaries (summary_year, summary_month, category_code);
