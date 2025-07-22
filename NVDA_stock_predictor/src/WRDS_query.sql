WITH stock_data AS (
        SELECT 
            date,
            permno,
            ABS(prc) / cfacpr as close,
            CASE WHEN openprc IS NOT NULL THEN ABS(openprc) / cfacpr ELSE ABS(prc) / cfacpr END as open,
            CASE WHEN askhi IS NOT NULL THEN askhi / cfacpr ELSE ABS(prc) / cfacpr END as high,
            CASE WHEN bidlo IS NOT NULL THEN bidlo / cfacpr ELSE ABS(prc) / cfacpr END as low,
            vol * cfacshr as volume
        FROM crsp.dsf
        WHERE permno = '{PERMNO}'
        AND date >= '{START_DATE}' AND date <= '{END_DATE}'
        AND prc IS NOT NULL
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
        f.book_value_per_share,
        f.earnings_per_share
    FROM stock_data s
    LEFT JOIN fundamental_data f ON s.permno = f.permno 
        AND f.datadate = (
            SELECT MAX(f2.datadate) 
            FROM fundamental_data f2 
            WHERE f2.permno = s.permno 
            AND f2.datadate <= s.date
        )
    ORDER BY s.date;