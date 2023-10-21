SELECT
    user_id,
    item_id,
    COUNT(units) AS qty,
    global_avg.price AS price
FROM default.karpov_express_orders o
JOIN (
    SELECT
        item_id,
        ROUND(AVG(price), 2) AS price
    FROM default.karpov_express_orders
    GROUP BY item_id
) AS global_avg ON o.item_id = global_avg.item_id
WHERE toDate({{start_date}}) <= toDate(o.timestamp) AND toDate(o.timestamp) <= toDate({{end_date}})
GROUP BY user_id, item_id, global_avg.price
ORDER BY user_id, item_id;
