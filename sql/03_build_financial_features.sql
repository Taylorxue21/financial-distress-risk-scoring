-- Clean the tables before running, make sure the idempotency.
DROP TABLE IF EXISTS derived_financial_features CASCADE;

CREATE TABLE derived_financial_features AS
WITH base_metrics AS (
    SELECT
        r.ticker,
        r.as_of_date AS date, 
        c.sector,
        
        r.total_revenue,
        r.net_income,
        
        -- 1. Cash to Assets
        (r.cash_and_equivalents::NUMERIC / NULLIF(r.total_assets, 0)) AS cash_to_assets,
        
        -- 2. Debt to Assets
        (r.total_debt::NUMERIC / NULLIF(r.total_assets, 0)) AS debt_to_assets,
        
        -- 3. OCF to Debt
        (r.operating_cashflow::NUMERIC / NULLIF(r.total_debt, 0)) AS ocf_to_debt,

        -- 4. Extract realized historical volatility over the past 6 months up to the financial report disclosure date (as_of_date)
        (
            SELECT STDDEV(p."adjClose") FROM raw_stock_prices p 
            WHERE p.ticker = r.ticker 
              AND p.date >= (r.as_of_date - INTERVAL '6 months')::DATE 
              AND p.date < r.as_of_date
        ) AS volatility_6m

    FROM raw_financials r
    LEFT JOIN company_profiles c ON r.ticker = c.ticker
    WHERE r.as_of_date IS NOT NULL
),

interaction_layer AS (
    SELECT 
        *,
        -- Interaction 1: High leverage * High volatility (extreme tail-risk multiplier)
        (debt_to_assets * volatility_6m) AS leverage_vol_interaction,
        
        -- Interaction 2: Debt / Cash (Liquidity Crunch Index)
        (debt_to_assets / NULLIF(cash_to_assets, 0)) AS debt_to_cash_interaction
    FROM base_metrics
),

factor_scoring AS (
    SELECT
        *,
        -- Industry-neutralized Z-score calculation: Z = (X - μ) / σ
        (leverage_vol_interaction - AVG(leverage_vol_interaction) OVER (PARTITION BY date, sector)) 
        / NULLIF(STDDEV(leverage_vol_interaction) OVER (PARTITION BY date, sector), 0) AS lev_vol_z,

        -- Industry-neutralized Z-score calculation for interaction_2
        (debt_to_cash_interaction - AVG(debt_to_cash_interaction) OVER (PARTITION BY date, sector)) 
        / NULLIF(STDDEV(debt_to_cash_interaction) OVER (PARTITION BY date, sector), 0) AS debt_cash_z
        
    FROM interaction_layer
)

-- inal output: Apply hard clipping to prevent XGBoost from fitting extreme noise values
SELECT
    ticker,
    date,
    sector,
    total_revenue,
    net_income,
    cash_to_assets,
    debt_to_assets,
    ocf_to_debt,
    volatility_6m,
    
    -- Apply hard clipping to prevent XGBoost from fitting extreme noise values
    LEAST(GREATEST(lev_vol_z, -3), 3) AS robust_lev_vol_interaction,
    LEAST(GREATEST(debt_cash_z, -3), 3) AS robust_debt_cash_interaction

FROM factor_scoring;


CREATE INDEX idx_derived_fin_ticker ON derived_financial_features(ticker);
CREATE INDEX idx_derived_fin_date ON derived_financial_features(date);
CREATE INDEX idx_derived_fin_sector ON derived_financial_features(sector);
