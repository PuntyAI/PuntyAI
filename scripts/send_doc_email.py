"""Send the scheduler/analysis framework document email to punty@punty.ai."""
import asyncio
from punty.delivery.email import send_email

HTML = """
<div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 800px; margin: 0 auto; color: #e0e0e0; background: #0a0a0f; padding: 32px;">

<h1 style="color: #e91e8c; border-bottom: 2px solid #e91e8c; padding-bottom: 12px;">PuntyAI &mdash; Scheduler Process &amp; Analysis Framework</h1>

<h2 style="color: #00d4ff;">1. Daily Scheduler Flow</h2>

<table style="width: 100%; border-collapse: collapse; margin: 16px 0;">
<tr style="background: rgba(233,30,140,0.1);"><th style="padding: 8px; text-align: left; color: #e91e8c; border-bottom: 1px solid #333;">Time (AEDT)</th><th style="padding: 8px; text-align: left; color: #e91e8c; border-bottom: 1px solid #333;">Job</th><th style="padding: 8px; text-align: left; color: #e91e8c; border-bottom: 1px solid #333;">What It Does</th></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #222;">00:05</td><td style="padding: 8px; border-bottom: 1px solid #222;">Daily Calendar Scrape</td><td style="padding: 8px; border-bottom: 1px solid #222;">Scrapes racing.com for meetings, auto-selects, full data scrape, schedules per-meeting jobs. Retries hourly until noon if empty.</td></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #222;">2h before R1</td><td style="padding: 8px; border-bottom: 1px solid #222;">Pre-Race Job</td><td style="padding: 8px; border-bottom: 1px solid #222;">Re-scrape scratchings/gear/odds/track, scrape speed maps, generate Early Mail, auto-approve if valid, post to Twitter.</td></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #222;">Every 15min</td><td style="padding: 8px; border-bottom: 1px solid #222;">Odds Updates</td><td style="padding: 8px; border-bottom: 1px solid #222;">Live odds from TAB, track scratchings.</td></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #222;">Every 30min</td><td style="padding: 8px; border-bottom: 1px solid #222;">Speed Maps</td><td style="padding: 8px; border-bottom: 1px solid #222;">Predicted running positions from Punters.com.</td></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #222;">30min after last</td><td style="padding: 8px; border-bottom: 1px solid #222;">Post-Race Job</td><td style="padding: 8px; border-bottom: 1px solid #222;">Wait for settlement (3 retries, 2h max), generate wrap-up, auto-approve, post Twitter.</td></tr>
</table>

<h2 style="color: #00d4ff;">2. Analysis Weights (25 Factors)</h2>
<p style="color: #999;">Each weight is set from Low &rarr; Low-Med &rarr; Med &rarr; Med-High &rarr; High. Controls how much emphasis the AI gives each factor.</p>

<h3 style="color: #ff6b35;">Core Form &amp; Fitness (HIGH)</h3>
<table style="width: 100%; border-collapse: collapse; margin: 16px 0;">
<tr style="background: rgba(255,107,53,0.1);"><th style="padding: 6px 8px; text-align: left; color: #ff6b35; border-bottom: 1px solid #333;">#</th><th style="padding: 6px 8px; text-align: left; color: #ff6b35; border-bottom: 1px solid #333;">Weight</th><th style="padding: 6px 8px; text-align: left; color: #ff6b35; border-bottom: 1px solid #333;">Description</th></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">1</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">recent_form</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Recent race results and form line</td></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">2</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">class</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Race classification and quality level</td></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">3</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">track_conditions</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Surface (good/soft/heavy), rail position effects</td></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">4</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">distance_fit</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Suitability to race distance</td></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">5</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">barrier_draw</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Starting barrier effectiveness</td></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">6</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">jockey</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Jockey form and track effectiveness</td></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">7</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">trainer</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Trainer form and track expertise</td></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">8</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">weight</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Weight carried, handicap burden</td></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">9</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">sectionals</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Sectional times from past races (VIC only)</td></tr>
</table>

<h3 style="color: #ff6b35;">Race Dynamics (HIGH)</h3>
<table style="width: 100%; border-collapse: collapse; margin: 16px 0;">
<tr style="background: rgba(255,107,53,0.1);"><th style="padding: 6px 8px; text-align: left; color: #ff6b35; border-bottom: 1px solid #333;">#</th><th style="padding: 6px 8px; text-align: left; color: #ff6b35; border-bottom: 1px solid #333;">Weight</th><th style="padding: 6px 8px; text-align: left; color: #ff6b35; border-bottom: 1px solid #333;">Description</th></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">10</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">speed_map_race_shape</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Predicted positions, pace setup, race flow advantage</td></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">11</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">barrier_track_bias</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Barrier effectiveness at specific tracks</td></tr>
</table>

<h3 style="color: #ff6b35;">Form Patterns (Med to Med-High)</h3>
<table style="width: 100%; border-collapse: collapse; margin: 16px 0;">
<tr style="background: rgba(255,107,53,0.1);"><th style="padding: 6px 8px; text-align: left; color: #ff6b35; border-bottom: 1px solid #333;">#</th><th style="padding: 6px 8px; text-align: left; color: #ff6b35; border-bottom: 1px solid #333;">Weight</th><th style="padding: 6px 8px; text-align: left; color: #ff6b35; border-bottom: 1px solid #333;">Description</th></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">12</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">head_to_head</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Direct comparisons to competitors</td></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">13</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">first_second_up</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Fresh vs second-up performance</td></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">14</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">course</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Track/venue form history</td></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">15</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">market</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Market movements and odds changes</td></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">16</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">trials</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Trial form and preparation</td></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">17</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">gear</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Equipment changes (blinkers, etc.)</td></tr>
</table>

<h3 style="color: #ff6b35;">Pedigree &amp; Background (Low-Med)</h3>
<table style="width: 100%; border-collapse: collapse; margin: 16px 0;">
<tr style="background: rgba(255,107,53,0.1);"><th style="padding: 6px 8px; text-align: left; color: #ff6b35; border-bottom: 1px solid #333;">#</th><th style="padding: 6px 8px; text-align: left; color: #ff6b35; border-bottom: 1px solid #333;">Weight</th><th style="padding: 6px 8px; text-align: left; color: #ff6b35; border-bottom: 1px solid #333;">Description</th></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">18</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">pedigree</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Breeding suitability, sire/dam factors</td></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">19</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">money</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Stable support, prize money indicators</td></tr>
</table>

<h3 style="color: #ff6b35;">Expert &amp; Speed Factors (Med-High)</h3>
<table style="width: 100%; border-collapse: collapse; margin: 16px 0;">
<tr style="background: rgba(255,107,53,0.1);"><th style="padding: 6px 8px; text-align: left; color: #ff6b35; border-bottom: 1px solid #333;">#</th><th style="padding: 6px 8px; text-align: left; color: #ff6b35; border-bottom: 1px solid #333;">Weight</th><th style="padding: 6px 8px; text-align: left; color: #ff6b35; border-bottom: 1px solid #333;">Description</th></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">20</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">tipsters_analysis</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Professional analysis consensus</td></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">21</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">pf_map_factor</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Punting Form pace advantage (&gt;1.0 = advantage)</td></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">22</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">pf_speed_rank</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Early speed rating (1=fastest, 25=slowest)</td></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">23</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">pf_settle_position</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Historical settling position</td></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">24</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">pf_jockey_factor</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Jockey effectiveness metric</td></tr>
</table>

<h3 style="color: #ff6b35;">Profile &amp; Signals (Med-High to Low)</h3>
<table style="width: 100%; border-collapse: collapse; margin: 16px 0;">
<tr style="background: rgba(255,107,53,0.1);"><th style="padding: 6px 8px; text-align: left; color: #ff6b35; border-bottom: 1px solid #333;">#</th><th style="padding: 6px 8px; text-align: left; color: #ff6b35; border-bottom: 1px solid #333;">Weight</th><th style="padding: 6px 8px; text-align: left; color: #ff6b35; border-bottom: 1px solid #333;">Description</th></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">25</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">horse_profile</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Age, sex, peak age indicators (4-5yo peak)</td></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">26</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">odds_fluctuations</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Historical odds movement patterns</td></tr>
<tr><td style="padding: 6px 8px; border-bottom: 1px solid #222;">27</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">stewards_comments</td><td style="padding: 6px 8px; border-bottom: 1px solid #222;">Stewards notes on race incidents</td></tr>
</table>

<h2 style="color: #00d4ff;">3. Data Points Per Runner (85+)</h2>

<h3 style="color: #ff6b35;">Identity &amp; Position</h3>
<p>saddlecloth, barrier, horse_name, jockey, trainer, trainer_location, weight, horse_age, horse_sex, horse_colour, sire, dam, dam_sire</p>

<h3 style="color: #ff6b35;">Performance</h3>
<p>handicap_rating, days_since_last_run, career_prize_money, career_record, form (last starts), last_five, form_history (detailed)</p>

<h3 style="color: #ff6b35;">Track &amp; Distance Stats</h3>
<p>track_dist_stats, track_stats, distance_stats, first_up_stats, second_up_stats, good_track_stats, soft_track_stats, heavy_track_stats, jockey_stats, trainer_stats, class_stats</p>

<h3 style="color: #ff6b35;">Odds &amp; Market (7 bookmakers)</h3>
<p>current_odds, place_odds, opening_odds, odds_tab, odds_sportsbet, odds_bet365, odds_ladbrokes, odds_betfair, odds_flucs (history), market_movement (heavy_support &rarr; big_drift)</p>

<h3 style="color: #ff6b35;">Speed Map &amp; Pace</h3>
<p>speed_map_position (leader/on_pace/midfield/backmarker), speed_value, pf_speed_rank (1-25), pf_settle, pf_map_factor (&gt;1.0 = pace advantage), pf_jockey_factor</p>

<h3 style="color: #ff6b35;">Gear &amp; Comments</h3>
<p>gear, gear_changes, comments, comment_long, stewards_comment</p>

<h2 style="color: #00d4ff;">4. Data Sources</h2>
<table style="width: 100%; border-collapse: collapse; margin: 16px 0;">
<tr style="background: rgba(0,212,255,0.1);"><th style="padding: 8px; text-align: left; color: #00d4ff; border-bottom: 1px solid #333;">Source</th><th style="padding: 8px; text-align: left; color: #00d4ff; border-bottom: 1px solid #333;">Data</th></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #222;">Racing.com GraphQL</td><td style="padding: 8px; border-bottom: 1px solid #222;">Runners, form, odds (5 bookies), jockeys, trainers, gear, stewards, career stats</td></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #222;">TAB API</td><td style="padding: 8px; border-bottom: 1px solid #222;">Live odds, place odds, scratchings, results, dividends</td></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #222;">Punters.com (Playwright)</td><td style="padding: 8px; border-bottom: 1px solid #222;">Speed maps, predicted positions, tips consensus</td></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #222;">CloudFront CSV</td><td style="padding: 8px; border-bottom: 1px solid #222;">Sectional times (VIC tracks only)</td></tr>
<tr><td style="padding: 8px; border-bottom: 1px solid #222;">TRC Rankings</td><td style="padding: 8px; border-bottom: 1px solid #222;">Trainer global rankings, group win counts, strike rates</td></tr>
</table>

<h2 style="color: #00d4ff;">5. Learning Loop (RAG)</h2>
<p>After settlement:</p>
<ol>
<li><strong>RaceMemory</strong> &mdash; Stores each pick with context (track, distance, class, odds, barrier, form, speed map) + actual result + P&amp;L</li>
<li><strong>RaceAssessment</strong> &mdash; Full LLM assessment with 12 filterable attributes (track, distance, class, going, age/sex restriction, weight type, field size, prize money, penetrometer, state, weather, temperature)</li>
<li><strong>Embedding</strong> &mdash; Key learnings vectorized for similarity search</li>
<li><strong>Retrieval</strong> &mdash; Next time a similar race appears, retrieves matching past assessments (same track &plusmn;200m &rarr; same state &rarr; any track)</li>
</ol>

<h2 style="color: #00d4ff;">6. Auto-Approval Validation</h2>
<h3 style="color: #ff6b35;">Early Mail</h3>
<ul>
<li>Min 2000 characters, no placeholder text</li>
<li>Contains Big 3 section &amp; Sequence Lanes</li>
<li>All races covered (Race N patterns)</li>
<li>Min 10 odds references ($X.XX), min 5 saddlecloth numbers (No.X)</li>
</ul>
<h3 style="color: #ff6b35;">Wrap-up</h3>
<ul>
<li>Min 1000 characters, no placeholder text</li>
<li>Venue name mentioned, Punty Ledger section</li>
<li>Quick Hits or Race-by-Race section, min 3 P&amp;L figures</li>
</ul>

<hr style="border: 1px solid #333; margin: 24px 0;">
<p style="color: #666; font-size: 12px;">Generated by PuntyAI &mdash; 10 Feb 2026</p>
</div>
"""

async def main():
    result = await send_email(
        to_email="punty@punty.ai",
        subject="PuntyAI - Scheduler Process & Analysis Framework Document",
        body_html=HTML,
    )
    print(result)

asyncio.run(main())
