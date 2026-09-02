# Methodology — Toronto Transit Equity Map

This document explains how the map's three scores are calculated, every place the
method deliberately departs from a simpler approach and why, and what the results
can and cannot tell you. It is written for a general reader — a journalist, a
planner, a resident — not just for someone reading the code.

The map covers all **158 City of Toronto neighbourhoods** (the current 2021
scheme, not the older 140-neighbourhood model).

---

## 1. The three scores

All three scores are computed across the 158 neighbourhoods together. Every raw
input is first **min–max normalized** to a 0–1 range —

```
normalize(x) = (x − minimum across the 158) / (maximum across the 158 − minimum across the 158)
```

— so a neighbourhood's normalized value is simply where it sits between the
citywide lowest (0) and highest (1). This makes quantities that are otherwise not
comparable (dollars, percentages, people per km²) safe to average together.

### 1a. Access score — how well transit serves the neighbourhood

**In plain language.** Two ingredients: how many transit stops the neighbourhood
has relative to its size, and how much service those stops get in the weekday
morning rush — with bigger, higher-capacity vehicles counting for more, and with
partial credit for major stations just outside the neighbourhood's edge.

**The math.**

```
# Per stop, over the weekday 07:00–09:00 peak (GTFS service_id = 1):

capacity_multiplier(mode) = { bus: 1.0, streetcar: 2.5, lrt: 6.0, subway: 15.0 }

capacity_weighted_trips_per_hour(stop)
    = ( Σ over the stop's qualifying scheduled trips of capacity_multiplier(that trip's mode) ) / 2

# Per neighbourhood:

own_frequency              = Σ capacity_weighted_trips_per_hour over stops located inside the neighbourhood
walkshed_credit            = Σ distance-decayed credit from nearby rapid-transit stations   (see §2b)
total_effective_frequency  = own_frequency + walkshed_credit

stop_density               = (number of stops inside the neighbourhood) / (neighbourhood area in km²)

freq_norm     = normalize(total_effective_frequency)
density_norm  = normalize(stop_density)

access_score  = 0.4 × freq_norm + 0.6 × density_norm      # already in 0–1
```

**Direction.** More frequent service and more stops per km² → higher access.

**Observed range in this snapshot:** about **0.02** (Guildwood) to **0.91**
(Yonge-Bay Corridor). The ends aren't 0 and 1 because no single neighbourhood is
top-ranked on frequency *and* density at once.

**How `mode` is decided.** Per route, from the GTFS `route_type` and route name:
`route_type` 1 → subway; `route_type` 3 → bus; `route_type` 0 → **lrt** if the
route name begins "Line \<number\>" (in this feed, *Line 5 Eglinton* and *Line 6
Finch West*), otherwise **streetcar**. An unrecognized mode falls back to the bus
multiplier and prints a warning.

### 1b. Need score — how dependent the neighbourhood is on transit

**In plain language.** The average of three signals, each ranked across the city:
how many residents have low incomes, how many people commute without a car, and
how densely people live.

**The math.**

```
# Per neighbourhood, from the 2021 Census neighbourhood profile:

low_income_pct       = LIM-AT low-income prevalence (%)                                   # share of people below the Low-income Measure, after tax
non_car_commute_pct  = (all commuters − car/truck/van commuters) / all commuters × 100
population_density    = neighbourhood population / neighbourhood area in km²

need_score = mean( normalize(low_income_pct),
                   normalize(non_car_commute_pct),
                   normalize(population_density) )        # equal weight, result in 0–1
```

**Direction.** Higher low-income share, higher non-car commute share, higher
density → higher need.

**Observed range:** about **0.03** to **0.87**. Raw input ranges across the 158
neighbourhoods: `low_income_pct` 4.3–29.0%, `non_car_commute_pct` 17.2–80.3%,
`population_density` roughly 860–43,700 people/km².

### 1c. Equity gap — the headline layer

**In plain language.** Need minus access. A positive gap means the neighbourhood
leans on transit more than transit currently delivers. A negative gap means
service runs ahead of need.

**The math.**

```
equity_gap = need_score − access_score        # range roughly −1 … +1
```

**Observed range:** about **−0.29 to +0.46**.

- **Largest positive gaps (most underserved):** North St. James Town (+0.46),
  North Toronto (+0.43), Harbourfront–CityPlace (+0.42), Yonge–Doris (+0.38),
  South Parkdale (+0.35).
- **Largest negative gaps (service ahead of need):** Annex (−0.29), Yonge-Bay
  Corridor (−0.23), Runnymede–Bloor West Village (−0.16), Danforth (−0.15),
  Rosedale–Moore Park (−0.14).

The equity gap is a **screening signal** — "where is the mismatch largest" — not
a service-planning verdict on any individual route or neighbourhood.

---

## 2. Where this deviates from a naive approach, and why

A naive version of this map would count raw scheduled trips per neighbourhood,
take Census variables at face value, and combine everything at raw magnitude.
Each change below is a deliberate departure from that. **None of them is the
single objectively correct choice** — they are defensible calls, written down so
you can disagree with them specifically.

### 2a. Transit service is weighted by vehicle capacity

**Naive:** every scheduled trip counts equally.

**Why that fails:** a six-car subway train and a single bus are not equivalent
service, and a neighbourhood built around one busy subway station has far fewer
physical stops than one laced with a bus grid. Counting trips — or stops — alone
made subway- and LRT-served neighbourhoods look *underserved* next to dense bus
networks.

**The fix:** each scheduled trip is scaled by the approximate in-service
passenger capacity of its vehicle, normalized to a standard bus = 1.0:

| Mode | Multiplier | Basis (approximate nominal in-service capacity) |
|---|---|---|
| Bus | 1.0 | ~50-passenger standard TTC bus — the baseline |
| Streetcar | 2.5 | ~130-passenger low-floor Flexity Outlook streetcar |
| LRT | 6.0 | Line 5 / Line 6 Flexity Freedom vehicles run coupled, ~295–300 passengers per train |
| Subway | 15.0 | ~800–900-passenger six-car T1 / Toronto Rocket trainset |

These are **order-of-magnitude service-capacity ratios**, not crush loads or
measured ridership. They were set by hand from published vehicle specifications.
Every trip is weighted by its own vehicle before the sum, so a stop served by
several modes is not averaged down to the smallest.

**A note on the LRT / streetcar split.** This GTFS feed codes both the legacy
streetcar network and the modern Flexity Freedom light-rail lines under the same
`route_type` (0). They get different multipliers because the vehicles and service
model genuinely differ. The split is made by route name ("Line 5", "Line 6"), and
the feed's `route_type` 0 routes are printed during the run so the classification
can be checked.

### 2b. Rapid-transit access gets a 500 m walkshed credit, deduplicated to stations

**Naive:** a stop counts only for the neighbourhood whose polygon contains it.

**Why that fails:** neighbourhood boundaries are administrative lines, not
travel-behaviour lines. Eglinton Station sits roughly 20 metres *outside* the
North Toronto boundary — strict point-in-polygon gave North Toronto no credit for
a subway station its residents obviously use. The reverse also happens: a
boundary can scoop in a station that mostly serves the neighbourhood next door.

**The fix:** point-in-polygon assignment stays primary — it still decides which
neighbourhood "owns" each stop for the stop-count and own-frequency terms — but
every neighbourhood *additionally* receives a distance-decayed credit for nearby
rapid-transit stations:

```
for each rapid-transit station S, and each neighbourhood polygon P
    whose interior does NOT contain S:

        d = distance from S to the nearest edge of P, in metres
        if d ≤ 500:
            walkshed_credit(P) += total_station_frequency(S) × (1 − d / 500)
```

- **500 m** ≈ a six-minute walk, the standard transit catchment distance.
- Decay is **linear** — full credit at the boundary line, zero at 500 m. Linear
  (rather than exponential or Gaussian) is the simplest defensible shape and
  avoids adding a second tuning parameter.
- One station can credit several neighbourhoods. That is intended — it mirrors
  walkable access — and it never changes a stop's primary assignment.

**Station de-duplication.** GTFS lists every platform as its own "stop", so a
four-platform interchange would otherwise contribute four separate
distance-decayed credits for what is physically one walk to one place. Platforms
are first grouped into physical stations (by GTFS `parent_station` where present
— unpopulated in this feed — otherwise by normalized station name), each
station's frequency is summed, and its location is taken as the centroid of its
platforms. In this feed, **234 rapid-transit platforms collapse to 110
stations**, and about **86 of the 158 neighbourhoods** receive some walkshed
credit.

### 2c. Low-income and no-vehicle inputs are proxies — the exact variables don't exist

The original design named "% low-income households" and "% households with no
vehicle available". **Neither exists in the 2021 Census neighbourhood profile.**

- **Low income.** The 2021 profile has no household-level low-income figure at
  all — every low-income measure in it is person-based. The map uses **LIM-AT
  prevalence** (the share of *people* below the Low-income Measure, after tax),
  which is the standard low-income indicator for Toronto neighbourhoods. The
  deviation is one of unit (people, not households); the intent — a low-income
  *share*, not a median dollar figure — is unchanged.
- **No vehicle.** The 2021 Census **dropped the household vehicle-ownership
  question entirely**. There is no direct or near substitute. The only
  transportation variable in the profile is *main mode of commuting*, for the
  employed labour force with a usual workplace. The map uses the **non-car
  commute share** — everyone who commutes by walking, cycling, transit, or other
  means — as a stand-in for not having a car. It is measured on commuters, not
  households, and it cannot separate "can't drive" from "chooses not to".
- **Density.** The profile publishes no land area or density figure, so
  population (from the profile) is divided by the *same* polygon areas the access
  score uses, keeping the two scores geometrically consistent.

In this data, none of the three source variables had any suppressed values, so
the fallback path (fill a suppressed cell with the citywide median and flag the
neighbourhood) never ran.

### 2d. Inputs are normalized before being weighted, not after

**Naive:** `access = 0.6 × frequency + 0.4 × stop_density`, then rescale the
result to 0–1.

**Why that fails:** the raw frequency and stop-density numbers differ by roughly
a factor of 37 in spread. Combining them at raw magnitude and normalizing once
let frequency completely dominate — stop density barely moved the final score,
regardless of its nominal weight.

**The fix:** each term is min–max normalized to 0–1 across the 158 neighbourhoods
*first*, and only then combined. The weights were also deliberately flipped to
**0.4 frequency / 0.6 density**, giving stop density the larger share — a
neighbourhood blanketed with stops is well served even if each route is only
moderately frequent. The same normalize-then-combine order is used for the need
score's three equally weighted terms.

---

## 3. Known limitations

Stated plainly, because the map is only useful if its blind spots are visible.

- **Bloor-Yonge counts as two stations.** The GTFS feed names the two halves of
  the Bloor-Yonge interchange separately ("Bloor Station" for Line 2, "Yonge
  Station" for Line 1), about 100 m apart, with no `parent_station` link to join
  them. The de-duplication step therefore leaves them as two stations.
  Neighbourhoods near that interchange collect walkshed credit from both and may
  look slightly better served than they should.

- **The density input can't tell two very different kinds of "dense" apart.**
  Population density is one of the three equally weighted need inputs, and it
  rewards *any* densely populated neighbourhood. A low-income high-rise community
  and an affluent downtown condo district score identically on that term. An
  affluent, dense, heavily car-free neighbourhood — where many residents commute
  by transit or on foot **by choice, not necessity** — can post a high need score
  on the density and non-car-commute terms without its residents being
  transit-dependent in the way the map is trying to measure. Only the LIM-AT term
  speaks to economic vulnerability, and it is outvoted two to one. Read a high
  need score alongside its low-income component, not on its own.

- **West Humber-Clairville is a genuine outlier that stretches the access
  scale.** It is by far the largest neighbourhood by area — mostly Pearson
  airport lands and industrial parks, with relatively few residents — so its
  area-derived figures sit far from every other neighbourhood's. Because min–max
  normalization is anchored on the extreme values, this one neighbourhood
  compresses the normalized range that the other 157 are scored within. It is
  left in as-is: excluding it, or clipping its values, would be its own arbitrary
  intervention, and it is a real Toronto neighbourhood.

- **It is a single point-in-time snapshot.** The demographics are from the **2021
  Census**. The transit service is from one TTC GTFS feed **valid 6 September –
  31 October 2026**, retrieved **1 September 2026**, evaluated on a representative
  weekday morning peak. Routes, schedules, demographics and neighbourhood
  boundaries all change; this map does not update. Re-running the pipeline
  against newer data produces a new snapshot — the code is what's maintained, not
  the published map.

- **Every boundary, proxy and weight is a judgment call.** The 500 m radius, the
  linear decay, the four capacity multipliers, the 0.4 / 0.6 access split, the
  equal weighting of the three need inputs, the 7–9 am window, the choice of
  LIM-AT and non-car-commute as proxies — all defensible, all deliberately
  simple, **none derived from a single objectively correct method**. Reasonable
  analysts would choose differently and get a somewhat different map. The value
  here is that every choice is written down and the pipeline is reproducible, not
  that the numbers are the last word.

- **Minor mechanical notes.** Stops are joined to neighbourhoods by geometric
  intersection with de-duplication, so a stop sitting exactly on a shared
  boundary is never double-counted. Stops up to 25 m outside every polygon are
  snapped to the nearest one (GPS and boundary-generalization slack); stops
  further out — TTC service into Mississauga and York Region — are left
  unassigned and credited to no neighbourhood. A neighbourhood with zero stops
  would keep an access score of 0 rather than being dropped, but in this data
  every neighbourhood has transit stops of its own, so that path never runs.

---

## 4. Data sources and licence

All three datasets are published by the City of Toronto on its Open Data portal
([open.toronto.ca](https://open.toronto.ca)), served through the CKAN API, and
licensed under the
**[Open Government Licence – Toronto](https://open.toronto.ca/open-data-license/)**.
The published map credits the licence in its footer.

| Dataset | CKAN package ID | Used for |
|---|---|---|
| TTC Routes and Schedules (GTFS) | `ttc-routes-and-schedules` | stops, trips, schedules → access score |
| Neighbourhoods | `neighbourhoods` | the 158 neighbourhood boundary polygons → areas, spatial joins, map geometry |
| Neighbourhood Profiles | `neighbourhood-profiles` | 2021 Census income, commuting, and population → need score |

Each package is queried at:

```
https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/package_show?id=<package-id>
```

The pipeline discovers the right resource in each package at download time rather
than hard-coding filenames. The specific files used in this snapshot are the GTFS
zip (`opendata_ttc_schedules.zip`), the 158-neighbourhood boundary file in
EPSG:4326 (`neighbourhoods-4326.geojson`), and the 2021 / 158-neighbourhood
profile workbook (`nbhd_2021_census_profile_full_158model.xlsx` — the only format
in which that profile is published).

**Boundary note.** Toronto has two neighbourhood schemes — an older
140-neighbourhood model and the current 158. This project uses the 2021 / 158
model throughout, and joins the boundary file to the profile workbook on the
neighbourhood *number*, not the name (seven names differ only by spacing or
punctuation between the two files).
