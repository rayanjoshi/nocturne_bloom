        WITH stock_data AS (
        SELECT 
            date,
            permno,
            ABS(prc) / NULLIF(cfacpr, 0) as close,
            CASE WHEN openprc IS NOT NULL THEN ABS(openprc) / NULLIF(cfacpr, 0) ELSE ABS(prc) / NULLIF(cfacpr, 0) END as open,
            CASE WHEN askhi IS NOT NULL THEN askhi / NULLIF(cfacpr, 0) ELSE ABS(prc) / NULLIF(cfacpr, 0) END as high,
            CASE WHEN bidlo IS NOT NULL THEN bidlo / NULLIF(cfacpr, 0) ELSE ABS(prc) / NULLIF(cfacpr, 0) END as low,
            vol * cfacshr as volume
        FROM crsp.dsf
        WHERE permno = '{PERMNO}'
        AND date >= '{START_DATE}' AND date <= '{END_DATE}'
        AND prc IS NOT NULL
        AND cfacpr IS NOT NULL AND cfacpr != 0
    ),
    spy_data AS (
        SELECT 
            d.date,
            ABS(d.prc) / NULLIF(d.cfacpr, 0) as spy_close
        FROM crsp.dsf d
        JOIN crsp.dsenames n ON d.permno = n.permno
        WHERE n.ticker = 'SPY'
        AND d.date >= n.namedt AND d.date <= n.nameendt
        AND d.date >= '{START_DATE}' AND d.date <= '{END_DATE}'
        AND d.prc IS NOT NULL
        AND d.cfacpr IS NOT NULL AND d.cfacpr != 0
    ),
    soxx_data AS (
        SELECT 
            d.date,
            ABS(d.prc) / NULLIF(d.cfacpr, 0) as soxx_close
        FROM crsp.dsf d
        JOIN crsp.dsenames n ON d.permno = n.permno
        WHERE n.ticker = 'SOXX'
        AND d.date >= n.namedt AND d.date <= n.nameendt
        AND d.date >= '{START_DATE}' AND d.date <= '{END_DATE}'
        AND d.prc IS NOT NULL
        AND d.cfacpr IS NOT NULL AND d.cfacpr != 0
    ),
    qqq_data AS (
        SELECT 
            d.date,
            ABS(d.prc) / NULLIF(d.cfacpr, 0) as qqq_close,
            d.ret as qqq_return
        FROM crsp.dsf d
        JOIN crsp.dsenames n ON d.permno = n.permno
        WHERE n.ticker = 'QQQ'
        AND d.date >= n.namedt AND d.date <= n.nameendt
        AND d.date >= '{START_DATE}' AND d.date <= '{END_DATE}'
        AND d.prc IS NOT NULL
        AND d.cfacpr IS NOT NULL AND d.cfacpr != 0
    ),
    vix_proxy_data AS (
        SELECT 
            d.date,
            ABS(d.prc) / NULLIF(d.cfacpr, 0) as vix_proxy
        FROM crsp.dsf d
        JOIN crsp.dsenames n ON d.permno = n.permno
        WHERE n.ticker = 'VIXY'
        AND d.date >= n.namedt AND d.date <= n.nameendt
        AND d.date >= '{START_DATE}' AND d.date <= '{END_DATE}'
        AND d.prc IS NOT NULL
        AND d.cfacpr IS NOT NULL AND d.cfacpr != 0
    ),
    treasury_data AS (
        SELECT 
            date,
            dgs10 as treasury_10y
        FROM frb.rates_daily
        WHERE date >= '{START_DATE}' AND date <= '{END_DATE}'
        AND dgs10 IS NOT NULL
    ),
    fundamental_data AS (
        SELECT 
            l.lpermno as permno,
            f.datadate,
            CASE WHEN f.ceqq > 0 AND f.cshoq > 0 THEN f.ceqq / f.cshoq ELSE NULL END as book_value_per_share,
            CASE WHEN f.niq IS NOT NULL AND f.cshoq > 0 THEN (f.niq * 4) / f.cshoq ELSE NULL END as earnings_per_share
        FROM comp.fundq f
        JOIN crsp_q_ccm.ccm_lookup l ON f.gvkey = l.gvkey
        WHERE l.lpermno = '{PERMNO}'
        AND f.datadate >= '{START_DATE}' AND f.datadate <= '{END_DATE}'
        AND f.cshoq IS NOT NULL
    )
    SELECT 
        s.date,
        s.permno,
        s.high,
        s.low, 
        s.open,
        s.close,
        s.volume,
        spy.spy_close,
        qqq.qqq_close,
        qqq.qqq_return,
        soxx.soxx_close,
        vix_proxy.vix_proxy,
        t.treasury_10y,
        f.book_value_per_share,
        f.earnings_per_share
    FROM stock_data s
    LEFT JOIN spy_data spy ON s.date = spy.date
    LEFT JOIN qqq_data qqq ON s.date = qqq.date
    LEFT JOIN soxx_data soxx ON s.date = soxx.date
    LEFT JOIN vix_proxy_data vix_proxy ON s.date = vix_proxy.date
    LEFT JOIN treasury_data t ON s.date = t.date
    LEFT JOIN fundamental_data f ON s.permno = f.permno 
        AND f.datadate = (
            SELECT MAX(f2.datadate) 
            FROM fundamental_data f2 
            WHERE f2.permno = s.permno 
            AND f2.datadate <= s.date
        )
    ORDER BY s.date;