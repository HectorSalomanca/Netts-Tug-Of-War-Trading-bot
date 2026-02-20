-- Tug-Of-War System: Supabase Schema
-- Apply via Supabase MCP or dashboard SQL editor

-- ============================================================
-- SIGNALS: Raw bot outputs
-- ============================================================
CREATE TABLE IF NOT EXISTS public.signals (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at timestamptz NOT NULL DEFAULT now(),
  symbol text NOT NULL,
  bot text NOT NULL CHECK (bot IN ('sovereign', 'madman')),
  direction text NOT NULL CHECK (direction IN ('buy', 'sell', 'neutral')),
  confidence numeric(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
  signal_type text NOT NULL,
  raw_data jsonb,
  source_url text,
  user_id uuid NOT NULL REFERENCES auth.users(id)
);

ALTER TABLE public.signals ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Owner full access on signals"
  ON public.signals FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE INDEX idx_signals_symbol ON public.signals(symbol);
CREATE INDEX idx_signals_bot ON public.signals(bot);
CREATE INDEX idx_signals_created_at ON public.signals(created_at DESC);

-- ============================================================
-- TUG_RESULTS: Referee verdicts
-- ============================================================
CREATE TABLE IF NOT EXISTS public.tug_results (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at timestamptz NOT NULL DEFAULT now(),
  symbol text NOT NULL,
  sovereign_direction text NOT NULL CHECK (sovereign_direction IN ('buy', 'sell', 'neutral')),
  madman_direction text NOT NULL CHECK (madman_direction IN ('buy', 'sell', 'neutral')),
  conflict boolean NOT NULL DEFAULT false,
  verdict text NOT NULL CHECK (verdict IN ('execute', 'crowded_skip', 'no_signal')),
  sovereign_confidence numeric(5,4) NOT NULL CHECK (sovereign_confidence >= 0 AND sovereign_confidence <= 1),
  madman_confidence numeric(5,4) NOT NULL CHECK (madman_confidence >= 0 AND madman_confidence <= 1),
  tug_score numeric(5,4) NOT NULL CHECK (tug_score >= 0 AND tug_score <= 1),
  user_id uuid NOT NULL REFERENCES auth.users(id)
);

ALTER TABLE public.tug_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Owner full access on tug_results"
  ON public.tug_results FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Public read on tug_results"
  ON public.tug_results FOR SELECT USING (true);

CREATE INDEX idx_tug_results_symbol ON public.tug_results(symbol);
CREATE INDEX idx_tug_results_created_at ON public.tug_results(created_at DESC);

-- ============================================================
-- TRADES: Alpaca executions
-- ============================================================
CREATE TABLE IF NOT EXISTS public.trades (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at timestamptz NOT NULL DEFAULT now(),
  tug_result_id uuid REFERENCES public.tug_results(id),
  symbol text NOT NULL,
  side text NOT NULL CHECK (side IN ('buy', 'sell')),
  qty numeric NOT NULL,
  order_type text NOT NULL CHECK (order_type IN ('market', 'limit')),
  limit_price numeric,
  alpaca_order_id text,
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'filled', 'cancelled', 'rejected')),
  fill_price numeric,
  fill_qty numeric,
  pnl numeric,
  user_id uuid NOT NULL REFERENCES auth.users(id)
);

ALTER TABLE public.trades ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Owner full access on trades"
  ON public.trades FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Public read on trades"
  ON public.trades FOR SELECT USING (true);

CREATE INDEX idx_trades_symbol ON public.trades(symbol);
CREATE INDEX idx_trades_created_at ON public.trades(created_at DESC);
