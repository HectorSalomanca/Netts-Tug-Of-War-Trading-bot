-- ============================================================
-- Tug-Of-War V3 Schema Additions
-- Run after schema.sql (V2) to add V3 columns and tables
-- ============================================================

-- ── New columns on existing trades table ────────────────────
ALTER TABLE trades ADD COLUMN IF NOT EXISTS tbl_label INTEGER;
-- tbl_label: 1 = win (upper barrier), -1 = loss (lower barrier), 0 = time stop (vertical)

ALTER TABLE trades ADD COLUMN IF NOT EXISTS stockformer_score REAL;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS ofi_regression_score REAL;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS hmm_posterior REAL;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS ensemble_score REAL;

-- ── New columns on existing signals table ───────────────────
ALTER TABLE signals ADD COLUMN IF NOT EXISTS ticker_personality TEXT;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS spread_percentile REAL;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS stacked_imbalance BOOLEAN DEFAULT FALSE;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS trapped_exhaustion BOOLEAN DEFAULT FALSE;

-- ── New columns on existing regime_states table ─────────────
ALTER TABLE regime_states ADD COLUMN IF NOT EXISTS state_v3 TEXT;
ALTER TABLE regime_states ADD COLUMN IF NOT EXISTS vix_proxy REAL;
ALTER TABLE regime_states ADD COLUMN IF NOT EXISTS momentum_accel REAL;

-- ── New table: institutional_flows ──────────────────────────
CREATE TABLE IF NOT EXISTS institutional_flows (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now(),
    ticker TEXT NOT NULL,
    fund TEXT NOT NULL,
    action TEXT NOT NULL,        -- 'accumulation', 'distribution', 'new_position', 'exit', 'hold'
    shares_current BIGINT,
    shares_previous BIGINT,
    pct_change REAL,
    quarter TEXT,                -- e.g. 'Q4 2025'
    user_id UUID REFERENCES auth.users(id)
);

CREATE INDEX IF NOT EXISTS idx_inst_flows_ticker ON institutional_flows(ticker);
CREATE INDEX IF NOT EXISTS idx_inst_flows_fund ON institutional_flows(fund);

-- ── New table: alt_data_signals ─────────────────────────────
CREATE TABLE IF NOT EXISTS alt_data_signals (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now(),
    ticker TEXT NOT NULL,
    source TEXT NOT NULL,        -- 'nvd_cve', 'fda_rss', 'tsmc_guidance', 'gov_contracts', etc.
    trigger_keyword TEXT,
    direction TEXT CHECK (direction IN ('buy', 'sell', 'neutral')),
    confidence REAL,
    text_snippet TEXT,
    user_id UUID REFERENCES auth.users(id)
);

CREATE INDEX IF NOT EXISTS idx_alt_signals_ticker ON alt_data_signals(ticker);
CREATE INDEX IF NOT EXISTS idx_alt_signals_source ON alt_data_signals(source);

-- ── New table: ensemble_scores (per-cycle meta-model output) ─
CREATE TABLE IF NOT EXISTS ensemble_scores (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now(),
    symbol TEXT NOT NULL,
    ensemble_score REAL,
    ensemble_direction TEXT,
    stockformer_component REAL,
    ofi_component REAL,
    hmm_component REAL,
    weights JSONB,
    spy_beta REAL,
    user_id UUID REFERENCES auth.users(id)
);

CREATE INDEX IF NOT EXISTS idx_ensemble_symbol ON ensemble_scores(symbol);
