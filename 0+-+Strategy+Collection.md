
# Hypothesis Summary

3 strategies exist:
1. Find and filter strategy to find stocks to buy today
2. Buy strategy to determine how many shares to buy of each stock returned by strategy 1.
3. Sell strategy to determine how many shares to sell of each stock in the portfolio

List of all my null and alternative hypothesis (and results):

Always goal: 25% annualised return over 4 years

Constrains:
- Max. leverage = 1. Thus, don't use negative funds

## Filter strategies to find stocks worth buying

Motivation: we want to find stocks that have the highest likelihood of stock price rise (and the lowest likelihood of stock price decline)

A filter consists of 3 components: a factor (e.g. 1 year revenue growth), a logical operator (e.g. greater than), and a threshold (e.g. 10%)




## All kinds of strategies (filter/find, buy, sell)
Strategies
1. Calculate sortino ratio; get top/bottom 30; buy/sell each week
2. Calculate sortino ratio; get top/bottom 30; buy/sell each month
3. Calculate sortino ratio; get top 30; buy each week; sell all positions every 3 months
4. Calc. sortino ratio; get top 20; buy every week; hold each position for 2 months & sell
5. Calc. EBIT/Revenue of each company; get top 25
- Get TipRanks highest upside potential; get top 25; buy, hold, and sell
- Get companies with Zacks rank number = 1; sort by EBIT/Revenue; get top 25; buy/hold/sell
- Sort companies by EPS; get top 25; buy/hold/sell
- Sort by positive earnings surprise (every day); get top 5; buy/hold/sell

Filter 1
- market cap < 300
- last earnings surpise positive
- earnings per share > 0
- sort by EBIT/Revenue (desc)

Filter 2
- Find most efficient frontier; use 20 companies each; run simulation with all companies

Filter 3
- Sort industries by tipranks highest upside potential (requires custom data upload / API integration)
- Get top 5 industries
- Get top 5 companies of each industry
- Buy/hold/sell

Filter 4
- Sort by Beta (desc)
- Get top 25 (highest beta) companies
- Buy/hold/sell

Filter 5
- Calc. 15-min return using pre-market price of all companies
- Sort by 15-min return (desc)
- Get top 5
- Buy/hold/sell

Filter 6
- Buy all companies where TipRanks sentiment score = 1; buy each day

Filter 7
- Every minute: get companies where current price > 52-week high
- Short-sell

Filter 8
- Monitor Zacks rank number every 15 min
- If company gets upgraded from 2 to 1, then buy

Filter 9
- Sort industries by highest average bullish sentiment score from tipranks
- Get top 5 industries
- Sort companies in top 5 industries by bullish sentiment score
- Get top 5 companies in each industry

Filter 10
- Get companies where 5 < beta < 10
- Sort by highest best
- Buy top 25

Filter 11
- Sort by iex.shortInterest (highest first)
- Shorten top 25

Filter 12
- Sort by annual revenue growth (highest first)
- Buy top 25

Filter 13
- Stocks annual return > 0
- Stocks 3-month revenue growth > industry revenue 
- Stock EPS growth > 0
- Zacks rank = 1
- Lowest short interest within industry group

Filter 14
- Exit 1 week before next earnings announcement

Filter 15
- Get all industries
- Calculate avg. EPS growth of each industry
- Select top 5 industries where EPS growth is highest
- For each industry, sort companies by EPS growth
- Select top 2 companies of each top industry

Filter 16
- Top 25 growth_grade in morningstar data

Filter 17
- If simpleMovingAverage(SP500, 200days) > simpleMovingAverage(SP500, 60days), then invest in companies having negative Beta (market is in down trend)
- Else, buy companies having positive beta

Filter 18
- company.PriceToBook ratio < company.industry.avgPriceToBook

Filter 19
- Buy/Sell companies in top 5 industries with highest avg. earnings surprise 

Filter 20
- 1r Revenue Growth > 0

Filter 21
- 1y EBITDA growth > 0

Filter 22
- 1y R&D expense growth > 0

Filter 23
- Buy/Sell companies in top 5 industries with highest avg. bullish sentiment score

Filter 24
- No spearman rank correlation between stocks

Filter 25
- Quick ratio > 0
- Current ratio > 0

Filter 26
- 3m earnings growth > 0%

Filter 27
- Total debt/EBITDA < 3

Filter 28
- Price-Earnings-Growth ratio

Filter 29
- Only apply industry-relevant filters to companies in a particular industry, e.g. don't apply price-to-book ratio to Internet companies

Filter 30
- Company's bullish sentiment score > industry avg. bullish sentiment score

# Rankings
Ranking 1
- Industries with highest avg. bullish sentiment score

Ranking 2
- Sort by short interest (highest first), group by industry 

Ranking 3
- Sory by profit margin (highest first), group by industry

Ranking 4
- Sort by 1y Revenue / Latest Reported Total Assets (highest first)

Ranking 5
- Sort by Dividends paid per share (highest first)

Ranking 6
- % of net sales spent on R&D (4-11% per year is good)

# Buy/Sell Triggers/Conditions

If condition A is met, then execute action B. Conditions can be linked with logical expressions, for example AND, and OR.

Measurement Frequency: every minute or day

## Buy (all)
How many shares should be bought of a filtered universe of stocks? The filtered universe is generated by a find/filter/rank strategy.

## Sell (all)
- Particular sell conditions/actions might only apply to specific industries (e.g. only biotechnology)


| Condition | Action |
|---|---|
| Holding period = 20 days | Sell all |

Time-based: 
- Condition: holding period of stock = 20 days
- Action: sell all

Return-based: 
- gain: if cumulative return of stock = 10%, then sell 50% of stock; if cumulative return = 15%, then sell another 50%; 
- gain: if cumulative return of stock = 10%, then sell 25% of stock; if cumulative return = 12.5%, then sell 25% of remaining amount; if cumulative return = 15%, then sell 25%; if return = 20%, then sell all positions
- avoid losing: cumulative return of stock = -10%

External: examples
- stock gets downgraded from Zacks rank 1 to 2
- current stock price > 52-week high price
- stock price drops by more than 2 standard deviations of daily return
- company's sentiment score drops below industry avg. (before it was above industry avg.)

# Rules/Conditions

If condition A is met, then execute aciton B. This applies to filter/buy and selling stocks.
