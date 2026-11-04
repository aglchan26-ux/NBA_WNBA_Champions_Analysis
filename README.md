# NBA & WNBA Champions Analysis

**Which player archetypes drive winning in the NBA and WNBA, how has the championship formula drifted, and does pay track value?**

Site: https://aglchan26-ux.github.io/NBA_WNBA_Champions_Analysis/

NBA 2003-04 to 2025-26 and WNBA 2009 to 2026, built from play-by-play, box scores and salary sheets. Each league
is measured against itself and compared side by side. Everything reported is association, not causation.

## What's here

| Folder | Contents |
|---|---|
| `scripts/fetch_data.py` | downloads the public ESPN game data (sportsdataverse releases) and external files |
| `src/` | the pipeline: player-season table, small-sample shrinkage, archetypes, team and title models, league eras |
| `results/` | model outputs and summary tables (CSV) |
| `site/` | the Quarto website (pre-rendered results in `site/_freeze`) |
| `tests/` | data checks |
| `config/` | reviewed archetype sub-type reference profiles |

Raw and processed data are **not** in this repository. Salary sheets (Her Hoop Stats, Spotrac,
Basketball-Reference, ESPN) and NBA.com mirrors are third-party data and are not redistributed; see the
site's Data page for sources.

## Run order

```bash
python scripts/fetch_data.py
cd src
python build_player_season.py && python build_shots.py && python build_pbp_extras.py
python step2_fit.py && python step2_validate.py
python participation.py
python -c "import step3_archetypes as s; s.fit_final(); s.fit_wnba_model()"
python step3_subtypes.py && python step3_layers.py && python step3_segments.py && python step3_validation.py
python step5_team_profiles.py && python step5_checklist.py && python step5_models.py
python step5_league_eras.py && python step5_title_no_net.py
python export_site_data.py
cd ../site && quarto render
```

Salary steps (`parse_wnba_salaries.py`, `build_salaries.py`) need the manually supplied salary files in
`data/raw/manual/` and `data/raw/uploads/`.

## Publishing

Pushing to `main` runs `.github/workflows/publish.yml`, which assembles the pre-rendered site and publishes it to
the `gh-pages` branch. Re-render locally (`quarto render` in `site/`) after the results change, then commit
`site/_freeze`.
