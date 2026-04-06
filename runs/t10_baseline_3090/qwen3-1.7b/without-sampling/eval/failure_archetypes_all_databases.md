# Failure Archetypes Review: All Databases

Model/run:

- `qwen3-1.7b`
- `runs/t10_baseline_3090/qwen3-1.7b/without-sampling/eval`

Scope:

- This review covers all `11` BIRD dev databases in the run.
- It focuses on failed examples only.
- The emphasis is not just on error labels, but on the actual question families the model is failing to solve.

Primary source files:

- `per_example_results.jsonl`
- `bird_dev_t10.jsonl`

## Global Reading

Across the full benchmark, the main pattern is not syntax collapse. It is semantic drift.

- Most failures are `wrong_result`, meaning the SQL runs but answers the wrong question.
- The model usually preserves the rough query skeleton.
- The failure is typically one of:
  - wrong return column
  - wrong join path
  - wrong entity counted
  - wrong denominator or cohort
  - wrong table ownership of a field
  - dropped filter in a multi-condition question

That distinction matters. If the model were mostly failing by syntax, the fix would be post-processing or SQL grammar repair. Here the fix needs stronger schema grounding, better decomposition, and more supervision on result shape and granularity.

## Cross-Database Failure Families

The recurring failure families across the benchmark are:

1. Rank by `X`, return `Y`
   - The model identifies the right extreme item but returns the wrong field.
   - Strong in `california_schools`, `formula_1`, `codebase_community`, `superhero`.

2. Join path confusion
   - The model knows the relevant nouns, but chooses the wrong bridge table or wrong API/id key.
   - Strong in `financial`, `formula_1`, `student_club`, `superhero`, `toxicology`, `european_football_2`.

3. Row count vs entity count
   - The model counts joined rows instead of distinct logical entities.
   - Strong in `card_games`, `financial`, `toxicology`, `thrombosis_prediction`.

4. Cohort and denominator drift
   - The model answers a nearby percentage question, but over the wrong population.
   - Strong in `financial`, `thrombosis_prediction`, `toxicology`, `debit_card_specializing`, `superhero`.

5. Output contract violation
   - The model returns `name` instead of `id`, or diagnosis instead of symptoms, or race URL instead of season URL.
   - Strong in `card_games`, `formula_1`, `thrombosis_prediction`, `codebase_community`.

6. Hidden two-stage retrieval
   - The question first identifies an entity, then asks for some other property of that entity.
   - Strong in `california_schools`, `financial`, `debit_card_specializing`, `codebase_community`.

## Database Reviews

## california_schools

Overall profile:

- Accuracy: `12.36%` (`11/89`)
- Dominant failure labels: `wrong_result 56`, `column_error 21`
- Core weakness: education queries that combine ranking, derived rates, and district/school metadata are frequently answered with the right shape but wrong field or wrong formula.

Classic failures:

- "What is the unabbreviated mailing street address of the school with the highest FRPM count for K-12 students?"
  - Gold intent: rank schools by `FRPM Count (K-12)`, then return `MailStreet`.
  - Model failure: switches to `Free Meal Count (K-12)` and a wrong mailing column.

- "What is the highest eligible free rate for K-12 students in the schools in Alameda County?"
  - Gold intent: compute `Free Meal Count / Enrollment`, then sort by that ratio.
  - Model failure: returns the maximum meal count instead of the maximum rate.

- "Please list the zip code of all the charter schools in Fresno County Office of Education."
  - Gold intent: district-level filter plus charter flag, then join to school zip.
  - Model failure: replaces district scope with county scope.

Failure classes:

### Ranking and top-k school retrieval

- Typical question:
  - "Find the school with the extreme value of a metric, then return another field from that school."
- What correct SQL must do:
  - rank by the requested metric
  - keep the metric name exact
  - project a different answer column from the identified top row
- Typical failure:
  - returns the max metric itself
  - swaps `FRPM Count` and `Free Meal Count`
  - selects the wrong address/test-taker field
- Why this class breaks:
  - the model understands extreme-value retrieval but not the separation between ranking field and answer field

### Derived education rates and formula-grounded metrics

- Typical question:
  - "Compute a rate such as free-meal-count divided by enrollment, then rank or filter on it."
- What correct SQL must do:
  - construct the numerator/denominator pair explicitly
  - preserve null checks
  - keep the school-type or county restriction
- Typical failure:
  - keeps the sort but drops the denominator
  - returns numerator instead of ratio
  - drops one of the constraining predicates
- Why this class breaks:
  - the model grounds to the right table region but not to the explicit formula definition

### Cross-table school metadata lookup

- Typical question:
  - "Join meal-program records with school metadata under district, charter, funding, or date constraints."
- What correct SQL must do:
  - join on `CDSCode`
  - distinguish county, district, and school attributes
  - preserve all compound filters
- Typical failure:
  - county substituted for district
  - funding-type constraint removed
  - answer field taken from the wrong table
- Why this class breaks:
  - the schema has several overlapping institutional and geographic descriptors

Most actionable fixes:

- train more "rank by X, return Y" examples
- add explicit formula decomposition for rate questions
- add schema notes for county vs district vs school-level fields

## financial

Overall profile:

- Accuracy: `17.92%` (`19/106`)
- Dominant failure labels: `wrong_result 69`, `column_error 16`
- Core weakness: banking questions with multi-hop joins and opaque domain-code columns are often answered with the right business nouns but the wrong table ownership.

Classic failures:

- "How many accounts who choose issuance after transaction are staying in East Bohemia region?"
  - Gold intent: region from `district`, issuance behavior from `account`.
  - Model failure: moves both signals into the wrong table mix.

- "List out the id number of client who choose statement of issuance after transaction are Disponent?"
  - Gold intent: `account -> disp -> client`.
  - Model failure: anchors on `trans` and loses the proper role chain.

- "For the client whose loan was approved first in 1993/7/5, what is the increase rate of his/her account balance from 1993/3/22 to 1998/12/27?"
  - Gold intent: identify cohort by loan date, then aggregate balance change across two dates.
  - Model failure: rewrites into direct row subtraction over the wrong tables.

Failure classes:

### Aggregation with region, account, and client-role filters

- Typical question:
  - "Count accounts, clients, or districts after combining region, salary, gender, and account-behavior constraints."
- What correct SQL must do:
  - know which table owns regional codes
  - know which table owns account frequency
  - preserve distinctness when counting districts or accounts
- Typical failure:
  - attaches `A3`, `A11`, or `frequency` to the wrong relation
  - counts joined rows instead of the target entity
- Why this class breaks:
  - the schema uses opaque code columns whose meanings are not obvious from names alone

### Multi-hop account/client/disposition lookup

- Typical question:
  - "Find the account or client through an indirect relationship, then return a property of the resolved entity."
- What correct SQL must do:
  - choose the right bridge path
  - keep track of whether the answer is an account-level or client-level field
- Typical failure:
  - substitutes `trans` for `disp`
  - returns transaction facts instead of account facts
- Why this class breaks:
  - there are several valid-looking join paths and the model chooses the most locally salient one

### Derived percentages and increase-rate computations

- Typical question:
  - "Compute a percentage or growth rate over a carefully selected cohort."
- What correct SQL must do:
  - define numerator and denominator over the same business population
  - preserve the target branch/client/date conditions
- Typical failure:
  - computes over the wrong population
  - turns multi-date aggregation into direct subtraction
- Why this class breaks:
  - the model sees "percentage" but does not preserve the hidden cohort logic

### Youngest/oldest plus salary composites

- Typical question:
  - "Pick the youngest or oldest client, connect them to district salary context, then project an account."
- What correct SQL must do:
  - resolve age extremum first
  - carry district salary context through the join
  - return the correct entity type
- Typical failure:
  - flattens the logic into average salary ranking
  - returns client ID instead of account ID
- Why this class breaks:
  - several reasoning steps are composed, and the model compresses them too early

Most actionable fixes:

- add table-role documentation for `district`, `account`, `disp`, `client`, `trans`
- force numerator/denominator explanation before SQL on percentage questions
- add more nested-selection exemplars

## formula_1

Overall profile:

- Accuracy: `20.69%` (`36/174`)
- Dominant failure labels: `wrong_result 92`, `column_error 40`
- Core weakness: motorsport semantics are broadly understood, but the model repeatedly chooses the wrong F1 table family or the wrong output field.

Classic failures:

- "For the driver who set the fastest lap speed in race No.933, where does he come from?"
  - Gold intent: order `results` by fastest lap speed, then return driver nationality.
  - Model failure: returns driver name instead of nationality.

- "What is his number of the driver who finished 0:01:54 in the Q3 of qualifying race No.903?"
  - Gold intent: `qualifying -> drivers`.
  - Model failure: drifts into `results` and invents extra result conditions.

- "How many races in the year 2010 are held on grand prixs outside Asia and Europe?"
  - Gold intent: use circuit country through `circuits`, then filter races.
  - Model failure: answers from `races` as if geography lived there.

Failure classes:

### Fastest/top driver ranking

- Typical question:
  - "Rank by race-performance metric, then return a driver metadata field."
- What correct SQL must do:
  - sort on lap/result metric
  - join to drivers
  - return the requested metadata, not just any driver descriptor
- Typical failure:
  - correct ranking, wrong answer column
  - near-miss column hallucinations
- Why this class breaks:
  - the model sees the driver retrieval pattern but does not maintain answer-field discipline

### Event-phase table selection

- Typical question:
  - "Choose the correct F1 table among `qualifying`, `results`, `races`, `seasons`, `drivers`."
- What correct SQL must do:
  - keep the phase-specific fact table
  - join to metadata only when necessary
- Typical failure:
  - `results` used where `qualifying` is needed
  - race metadata used where season metadata is needed
- Why this class breaks:
  - multiple F1 tables share overlapping IDs and event language

### Aggregation with race, geography, and finish-state filters

- Typical question:
  - "Count races or drivers after combining event identity, geography, and non-finish logic."
- What correct SQL must do:
  - use circuits for geography
  - use results for finish status
  - use races for event identity and year
- Typical failure:
  - removes a required join
  - keeps only the most obvious condition and drops the rest
- Why this class breaks:
  - the model over-compresses the question into a single-table answer

### Percentage and completion-rate computations

- Typical question:
  - "Compare events or compute completion rates over correctly defined race slices."
- What correct SQL must do:
  - define the participant set
  - define the numerator set
  - compute the denominator at the right granularity
- Typical failure:
  - row arithmetic instead of aggregate logic
  - date/race fields attached to the wrong table
- Why this class breaks:
  - denominator construction is unstable when race identity and result rows are both in play

Most actionable fixes:

- add explicit table-role summaries for F1 event-phase tables
- add "sort by metric, return different metadata" examples
- add repair checks for whether geography/finish-state live in the selected tables

## thrombosis_prediction

Overall profile:

- Accuracy: `26.99%` (`44/163`)
- Dominant failure labels: `wrong_result 77`, `column_error 37`
- Core weakness: clinical queries combine temporal reasoning, patient cohorts, and measurement tables; the model often keeps the right medical concepts but computes against the wrong anchor or wrong cohort.

Classic failures:

- "What was the age of the youngest patient when they initially arrived at the hospital?"
  - Gold intent: age at first date using SQLite-compatible year extraction.
  - Model failure: uses non-SQLite `YEAR()` function.

- "What is the percentage of female patient had total protein not within the normal range?"
  - Gold intent: female cohort from `Patient`, lab value from `Laboratory`, percentage over that female cohort.
  - Model failure: tries to filter `sex` from `Laboratory` and distorts denominator logic.

- "State the sex and birthday of patient ID '163109'. When was the examination taken and what symptom does the patient had."
  - Gold intent: retrieve symptom.
  - Model failure: returns diagnosis instead.

Failure classes:

### Date-aware aggregation over patient history

- Typical question:
  - "Compute age or exam lag relative to admission or first visit."
- What correct SQL must do:
  - use SQLite date extraction
  - anchor the time difference to the right event
  - preserve admission-state constraints when present
- Typical failure:
  - wrong date function
  - wrong time anchor
  - dropped admission filter
- Why this class breaks:
  - temporal semantics and SQL dialect constraints interact here

### Patient/examination detail retrieval

- Typical question:
  - "Join patient and examination, then return the exact requested clinical fields."
- What correct SQL must do:
  - preserve field-level answer contract
  - keep ordering if age extremum is involved
- Typical failure:
  - diagnosis substituted for symptoms
  - one requested field omitted
- Why this class breaks:
  - there are several clinically plausible text fields, and the model picks a nearby one

### Percentage and cohort calculations

- Typical question:
  - "Define a demographic or admission cohort, then aggregate a measurement over it."
- What correct SQL must do:
  - isolate the patient cohort first
  - then apply lab/exam measurement conditions
  - then compute over the right denominator
- Typical failure:
  - demographic field moved to lab table
  - denominator population altered
  - age computed from wrong date
- Why this class breaks:
  - cohort logic is hidden in the wording rather than explicit in schema names

### Distinct patient set extraction

- Typical question:
  - "List or count unique patients satisfying lab/exam conditions."
- What correct SQL must do:
  - decide whether the answer is an ID list or a count
  - use `DISTINCT` when repeated exams/labs can duplicate the patient
- Typical failure:
  - count returned when list requested
  - duplicate-prone rows counted as unique patients
- Why this class breaks:
  - result shape tracking is weak under join multiplicity

Most actionable fixes:

- add SQLite-specific temporal training examples
- add explicit cohort formulation before SQL
- add post-generation check for question nouns vs selected columns

## debit_card_specializing

Overall profile:

- Accuracy: `31.25%` (`20/64`)
- Dominant failure labels: `wrong_result 34`, `column_error 9`
- Core weakness: business analytics over year-month consumption gets simplified into loose filters, with grouping and comparison logic dropped.

Classic failures:

- "In 2012, who had the least consumption in LAM?"
  - Gold intent: filter to LAM, group by customer for 2012, rank by summed consumption.
  - Model failure: drops segment condition and returns ranked customer from yearmonth only.

- "How much did customer 6 consume in total between August and November 2013?"
  - Gold intent: bounded year-month range.
  - Model failure: rewrites to a chain of `LIKE` clauses with broken precedence.

- "Which of the three segments—SME, LAM and KAM—has the biggest and lowest percentage increases in consumption paid in EUR between 2012 and 2013?"
  - Gold intent: compute per-segment percentage change.
  - Model failure: collapses into unrelated gas-station counts.

Failure classes:

### Segment-aware yearly ranking

- Typical question:
  - "Aggregate annual consumption by customer inside a segment or currency slice, then choose the top or bottom customer."
- What correct SQL must do:
  - filter year
  - filter segment/currency
  - group by customer
  - rank by aggregate consumption
- Typical failure:
  - omits group-by or omits segment/currency
- Why this class breaks:
  - the model treats the task as row retrieval instead of grouped analytics

### Windowed consumption aggregation

- Typical question:
  - "Sum over a precise year-month interval or compute a country difference inside a business category."
- What correct SQL must do:
  - use exact range boundaries
  - preserve arithmetic structure
  - keep category conditions
- Typical failure:
  - loose `LIKE`-based date logic
  - broken `AND`/`OR` precedence
  - dropped business filter
- Why this class breaks:
  - year-month strings invite brittle lexical rewrites

### Cross-segment comparative metrics

- Typical question:
  - "Compare annual averages or growth percentages across SME, LAM, and KAM, sometimes under an additional subset constraint."
- What correct SQL must do:
  - define the subset
  - compute segment-specific metrics
  - compare them in the requested output format
- Typical failure:
  - replaces multi-output analytics with simple sums or unrelated comparisons
- Why this class breaks:
  - this class composes cohorting, time slicing, and comparative metrics

### Transaction-history linkage

- Typical question:
  - "Use a transaction to identify a customer, then look up some other historical business fact."
- What correct SQL must do:
  - anchor on `transactions_1k`
  - resolve customer ID
  - fetch the target from `yearmonth` or `customers`
- Typical failure:
  - wrong anchor table
  - transaction fields and monthly-consumption fields mixed
- Why this class breaks:
  - the query is implicitly two-stage, but the model tries to answer in one table family

Most actionable fixes:

- add "group then rank" examples over yearmonth facts
- add canonical year-month range templates
- add explicit two-stage retrieval exemplars

## card_games

Overall profile:

- Accuracy: `35.60%` (`68/191`)
- Dominant failure labels: `wrong_result 87`, `column_error 33`
- Core weakness: metadata-rich card schemas trigger answer-column drift and table-role confusion, especially around `cards`, `legalities`, `foreign_data`, and `rulings`.

Classic failures:

- "Which are the cards that have incredibly powerful foils."
  - Gold intent: return `id`.
  - Model failure: returns `name`.

- "State the alternative languages available for card named Annul numbered 29."
  - Gold intent: return `foreign_data.language`.
  - Model failure: returns `foreign_data.name`.

- "How many cards of legalities whose status is restricted have text boxes?"
  - Gold intent: `COUNT(DISTINCT cards.id)`.
  - Model failure: counts raw joined rows.

Failure classes:

### Identifier-vs-name retrieval

- Typical question:
  - "Filter the right cards, but return database IDs rather than natural-language names."
- What correct SQL must do:
  - keep the result field exact
- Typical failure:
  - `name` returned instead of `id`
- Why this class breaks:
  - natural language makes `name` feel more salient than the requested identifier field

### Distinct legalities and format retrieval

- Typical question:
  - "Join `cards` and `legalities`, apply format/status conditions, and often deduplicate."
- What correct SQL must do:
  - keep card attributes on `cards`
  - keep format/status on `legalities`
  - use `DISTINCT` when multiple legalities rows duplicate the card
- Typical failure:
  - format moved to the wrong table
  - output field changed
  - `DISTINCT` omitted
- Why this class breaks:
  - metadata schemas with many side tables encourage table-role confusion

### Side-table metadata retrieval

- Typical question:
  - "Resolve the card instance, then pull language, ruling text, or other metadata from a linked table."
- What correct SQL must do:
  - pick the correct side table
  - return the requested side-table field
- Typical failure:
  - wrong side-table field returned
  - drift to another card instance
- Why this class breaks:
  - many answer columns are all text-like and semantically nearby

### Aggregation over legal subsets

- Typical question:
  - "Count unique cards satisfying legality and another card property such as textlessness or starter-deck membership."
- What correct SQL must do:
  - count cards, not joined rows
- Typical failure:
  - raw row count instead of distinct card count
- Why this class breaks:
  - the model often tracks filter logic better than counting granularity

Most actionable fixes:

- add answer-column supervision for `id` vs `name`
- add table-role hints for `cards` vs `legalities` vs `rulings` vs `foreign_data`
- add row-count vs entity-count contrastive training pairs

## toxicology

Overall profile:

- Accuracy: `37.93%` (`55/145`)
- Dominant failure labels: `wrong_result 62`, `column_error 28`
- Core weakness: chemistry graph questions are broadly understood conceptually, but the model often counts at the wrong level or traverses the wrong graph path.

Classic failures:

- "In the non-carcinogenic molecules, how many contain chlorine atoms?"
  - Gold intent: count distinct molecules.
  - Model failure: counts joined rows.

- "What atoms are connected in single type bonds?"
  - Gold intent: use `bond` + `connected` and return atom pairs.
  - Model failure: returns one atom identifier.

- "What is the percentage of carbon in double-bond molecules?"
  - Gold intent: atom-level numerator over atom-level denominator within qualifying molecules.
  - Model failure: counts bonds and rewrites bond type symbol.

Failure classes:

### Molecule counting with atom/bond membership filters

- Typical question:
  - "Count how many molecules contain some atom or structural motif."
- What correct SQL must do:
  - identify qualifying molecules via atom/bond tables
  - count distinct molecules
- Typical failure:
  - counts atom rows rather than molecule IDs
- Why this class breaks:
  - the model confuses structural evidence rows with the entity being counted

### Bond and connection graph lookup

- Typical question:
  - "Recover which atoms are connected or what bond type links an atom pair."
- What correct SQL must do:
  - use `bond` for edge label
  - use `connected` for edge endpoints
  - sometimes preserve bidirectionality
- Typical failure:
  - wrong structural table selected
  - wrong answer type returned
- Why this class breaks:
  - graph semantics are spread over multiple relations with different responsibilities

### Distinct structural entity retrieval

- Typical question:
  - "Return unique molecules or elements satisfying a bond-type or carcinogenicity condition."
- What correct SQL must do:
  - choose minimal necessary join path
  - return the requested entity type
- Typical failure:
  - over-joins the graph and changes the answer set
- Why this class breaks:
  - the model over-traverses because every graph table looks relevant

### Percentage composition calculations

- Typical question:
  - "Compute percentage of atoms or molecules satisfying some structural property."
- What correct SQL must do:
  - identify numerator unit
  - identify denominator unit
  - keep symbolic bond encoding exact
- Typical failure:
  - atom-vs-molecule denominator confusion
  - lexicalizing symbolic bond types (`=` -> `double`) in a way the schema does not support
- Why this class breaks:
  - the model does not reliably hold the level of measurement

Most actionable fixes:

- add denominator annotation in training data
- add minimal-path graph traversal examples
- add count-at-atom vs count-at-molecule contrastive examples

## codebase_community

Overall profile:

- Accuracy: `59.68%` (`111/186`)
- Dominant failure labels: `wrong_result 50`, `column_error 24`
- Core weakness: community Q&A queries often fail because the model uses denormalized-looking fields from `posts` instead of joining `users`, or mishandles ratios over badges, posts, views, and votes.

Classic failures:

- "Who is the owner of the post 'Eliciting priors from experts'?"
  - Gold intent: join `posts.OwnerUserId` to `users.Id`, return `users.DisplayName`.
  - Model failure: returns `OwnerDisplayName` directly from `posts`.

- "How many posts does the user csgillespie own?"
  - Gold intent: join `users` to `posts`.
  - Model failure: counts a non-existent owner column on `users`.

- "What is the average number of badges obtained by a user with over 200 views?"
  - Gold intent: count badges divided by distinct users in the filtered cohort.
  - Model failure: wrong denominator field and wrong distinctness.

Failure classes:

### Post/user ownership and editor lookup

- Typical question:
  - "Resolve the user associated with a post through owner or last-editor IDs."
- What correct SQL must do:
  - join posts to users
  - return user metadata from `users`
- Typical failure:
  - uses denormalized-looking display fields from `posts`
- Why this class breaks:
  - the model prefers directly named columns over relationally correct joins

### Aggregation with author/date filters

- Typical question:
  - "Count posts or users after joining across posts, users, comments, or badges with date or ownership constraints."
- What correct SQL must do:
  - count the correct entity
  - preserve date casting when needed
  - respect ownership through the correct join
- Typical failure:
  - date comparison without normalization
  - counting from wrong table
- Why this class breaks:
  - community schemas have many denormalized-looking columns that tempt shortcut queries

### Popularity/comment ranking with metadata projection

- Typical question:
  - "Find the most viewed/favorited/commented item, then return user or badge metadata."
- What correct SQL must do:
  - rank on one table
  - project another table’s field
- Typical failure:
  - ranking is mostly right, but answer field is pulled from the wrong table or wrong relation
- Why this class breaks:
  - another instance of "rank by X, return Y"

### Ratio and average questions over badges, views, votes, and posts

- Typical question:
  - "Compute badges-per-user or posts-per-vote style ratios over the correct cohort."
- What correct SQL must do:
  - identify the correct numerator entity
  - identify the distinct denominator entity
- Typical failure:
  - denominator chosen from the wrong table or without `DISTINCT`
- Why this class breaks:
  - the model handles the join skeleton but not the unit accounting

Most actionable fixes:

- add explicit supervision for `users` joins instead of denormalized-looking post columns
- add ratio questions with clear numerator/denominator annotation
- add date-normalization examples on timestamp fields

## european_football_2

Overall profile:

- Accuracy: `51.16%` (`66/129`)
- Dominant failure labels: `wrong_result 37`, `column_error 24`
- Core weakness: football queries often require switching correctly between `Player`, `Player_Attributes`, `Team`, `Team_Attributes`, `Match`, and `League`; the model repeatedly confuses API IDs, row IDs, and attribute table ownership.

Classic failures:

- "Which home team had lost the fewest matches in the 2016 season?"
  - Gold intent: identify losing home teams in the 2015/2016 season, group by `home_team_api_id`, return team name.
  - Model failure: joins on team row `id` rather than `team_api_id`.

- "Among all the players whose weight is under 130, how many of them preferred foot in attacking is left?"
  - Gold intent: weight from `Player`, preferred foot from `Player_Attributes`.
  - Model failure: assumes both live in `Player_Attributes`.

- "What was the build up play speed class for 'Willem II' on 2011/2/22?"
  - Gold intent: join `Team.team_api_id` to `Team_Attributes.team_api_id` and match the correct date.
  - Model failure: switches to FIFA API key and even changes the date.

Failure classes:

### Match/team/league ranking and win/loss retrieval

- Typical question:
  - "Identify which team won/lost most/fewest under league, season, and home/away conditions."
- What correct SQL must do:
  - preserve home vs away semantics
  - join through league when league context is in the question
  - join teams by `team_api_id`
- Typical failure:
  - wrong team key
  - home and away conditions drift
- Why this class breaks:
  - the schema uses several team identifiers and role-specific match columns

### Player and player-attributes disambiguation

- Typical question:
  - "Combine demographic fields from `Player` with skill attributes from `Player_Attributes`."
- What correct SQL must do:
  - keep weight/name/birthday on `Player`
  - keep preferred foot, crossing, overall rating on `Player_Attributes`
  - join by `player_api_id`
- Typical failure:
  - player attributes queried from `Player`
  - wrong ID key used between player tables
- Why this class breaks:
  - the schema contains `id`, `player_api_id`, and `player_fifa_api_id`, which are easy to misuse

### Team and team-attributes temporal lookup

- Typical question:
  - "Retrieve tactical/team style attributes for a named team on a specific date."
- What correct SQL must do:
  - use `team_api_id`
  - filter the exact date
- Typical failure:
  - uses FIFA key instead of team API key
  - date changes or is normalized incorrectly
- Why this class breaks:
  - several team identifiers coexist and date semantics live only on the attribute table

### Percentages and distinct player retrieval over time windows

- Typical question:
  - "Compute percentages or list distinct players under attribute constraints and year windows."
- What correct SQL must do:
  - preserve date window on the attribute table
  - deduplicate players
  - keep numerator/denominator at player level
- Typical failure:
  - answer attempted from the wrong table
  - wrong distinctness
  - incorrect key join
- Why this class breaks:
  - cohort definition spans both static player metadata and time-varying attributes

Most actionable fixes:

- add schema-role notes for `id` vs `player_api_id` vs `player_fifa_api_id`
- add training examples that contrast `Player` and `Player_Attributes`
- add team/team-attributes temporal lookup exemplars

## student_club

Overall profile:

- Accuracy: `53.16%` (`84/158`)
- Dominant failure labels: `wrong_result 48`, `column_error 25`
- Core weakness: event, attendance, member, budget, income, and expense questions are often answered with the right nouns but the wrong business table or an unnecessary `position = 'Student_Club'` filter.

Classic failures:

- "How many students in the Student_Club are from the College of Engineering?"
  - Gold intent: count members joined to majors with engineering college.
  - Model failure: injects an unnecessary `position = 'Student_Club'` condition.

- "How much did the Student_Club members spend on food in September Meeting?"
  - Gold intent: event + budget.
  - Model failure: drifts into `income`.

- "What percentage was the budget for Parking to the total budget for the 'November Speaker'?"
  - Gold intent: budget allocation within one event.
  - Model failure: substitutes `income` table and loses event-budget structure.

Failure classes:

### Event/attendance/member aggregation

- Typical question:
  - "Count attendance or participation for a named event or member under date conditions."
- What correct SQL must do:
  - use `attendance` as the bridge
  - use `event` for event identity/date
  - use `member` only when person identity is required
- Typical failure:
  - adds unnecessary membership-role constraints
  - counts the wrong entity
- Why this class breaks:
  - the phrase "Student_Club" tempts the model to use `position` even when the schema question is broader

### Budget/income/expense table confusion

- Typical question:
  - "Retrieve or compare club spending on events, categories, and costs."
- What correct SQL must do:
  - distinguish `budget`, `income`, and `expense`
  - keep event linkage on the right relation
- Typical failure:
  - uses `income` when the task is about budget
  - uses `expense` directly without the budget bridge
- Why this class breaks:
  - several finance-related tables are semantically close but represent different business processes

### Ratio and budget-share calculations

- Typical question:
  - "Compare category budgets between events or compute category share within an event."
- What correct SQL must do:
  - preserve event identity
  - preserve budget category
  - build the right numerator and denominator
- Typical failure:
  - event context drifts
  - wrong table enters denominator
- Why this class breaks:
  - the model sees "budget" and "amount" but not the event-scoped denominator

### Result-shape confusion in member/event retrieval

- Typical question:
  - "List members, categories, or countries under attendance/location conditions."
- What correct SQL must do:
  - decide whether the answer is a list of IDs, names, or categories
  - use `DISTINCT` when one event/member can appear many times
- Typical failure:
  - returns names when IDs requested
  - returns counts when lists requested
- Why this class breaks:
  - result-shape discipline weakens when event and member joins multiply rows

Most actionable fixes:

- add table-role summaries for `attendance`, `budget`, `income`, `expense`
- discourage reflexive use of `position='Student_Club'` unless the question explicitly asks for role
- add result-shape supervision for event/member list vs count questions

## superhero

Overall profile:

- Accuracy: `54.26%` (`70/129`)
- Dominant failure labels: `column_error 30`, `wrong_result 29`
- Core weakness: the model repeatedly confuses `hero_power` and `hero_attribute`, and more broadly mixes structural hero metadata with publisher/colour/gender side tables.

Classic failures:

- "Please list all the superpowers of 3-D Man."
  - Gold intent: `superhero -> hero_power -> superpower`.
  - Model failure: swaps in `hero_attribute`.

- "Among the superheroes with the super power of 'Super Strength', how many of them have a height of over 200cm?"
  - Gold intent: height from superhero, power through `hero_power`.
  - Model failure: power relation again moved to `hero_attribute`.

- "What is the percentage of superheroes who act in their own self-interest or make decisions based on their own moral code? Indicate how many of the said superheroes were published by Marvel Comics."
  - Gold intent: alignment-defined cohort, then overall percentage and Marvel count.
  - Model failure: misplaces alignment and publisher fields, breaking both numerator and count.

Failure classes:

### Power lookup vs attribute lookup

- Typical question:
  - "Retrieve or count heroes by superpower."
- What correct SQL must do:
  - use `hero_power` for hero-to-power membership
  - use `hero_attribute` only for numerical attributes like speed or strength-style metrics
- Typical failure:
  - `hero_attribute` used as if it were the power bridge table
- Why this class breaks:
  - the names `hero_power` and `hero_attribute` are semantically close and both connect to hero-related properties

### Publisher/colour/gender side-table grounding

- Typical question:
  - "Filter heroes by eye colour, hair colour, publisher, gender, race, or alignment."
- What correct SQL must do:
  - keep each property on its proper side table
  - preserve multi-side-table joins
- Typical failure:
  - uses publisher as if it were a power table
  - mixes colour/race/gender ownership
- Why this class breaks:
  - many short side tables with parallel `id` columns encourage accidental cross-wiring

### Derived percentages over hero cohorts

- Typical question:
  - "Compute the percentage of heroes in some alignment/publisher/height cohort."
- What correct SQL must do:
  - define cohort from joined metadata tables
  - compute numerator over the right condition
  - compute denominator over the intended base set
- Typical failure:
  - numerator/denominator use wrong tables
  - fields such as alignment or publisher attached to `superhero` directly when they require joins
- Why this class breaks:
  - the model sees the right concepts but collapses schema indirection

### Ranking heroes by attribute or powers

- Typical question:
  - "Find the slowest hero, tallest Marvel hero, or hero with the most powers."
- What correct SQL must do:
  - use the right join path to the relevant attribute/power table
  - rank by the correct measure
  - return the requested metadata
- Typical failure:
  - wrong join path
  - group by wrong entity
  - wrong return field
- Why this class breaks:
  - these questions combine ranking, join-path choice, and answer-type precision

Most actionable fixes:

- add explicit contrastive training: `hero_power` vs `hero_attribute`
- add side-table ownership checks for publisher/colour/gender/race/alignment
- add repair checks for whether powers are being sourced through the right relation

## Summary Of Most Important Database-Specific Weaknesses

- `california_schools`: formulas and district-vs-county grounding
- `financial`: multi-hop role resolution and denominator drift
- `formula_1`: table-family confusion across F1 event phases
- `thrombosis_prediction`: temporal anchors and patient-cohort definition
- `debit_card_specializing`: grouped analytics over year-month windows
- `card_games`: answer-column discipline and distinct card counting
- `toxicology`: graph traversal and molecule-vs-atom counting
- `codebase_community`: denormalized-looking post fields vs proper user joins
- `european_football_2`: player/team API-key disambiguation
- `student_club`: budget vs income vs expense confusion
- `superhero`: `hero_power` vs `hero_attribute` confusion

## Highest-Yield Fix Strategies Across The Whole Benchmark

1. Add a pre-SQL decomposition step in prompts or training targets:
   - target entity
   - answer column
   - source tables
   - join path
   - distinctness requirement
   - numerator
   - denominator

2. Add contrastive examples for:
   - `return id` vs `return name`
   - `COUNT(*)` vs `COUNT(DISTINCT entity_id)`
   - rank by one field, return another
   - two-stage retrieval
   - percentages over explicitly defined cohorts

3. Add schema-specific notes for the hardest databases:
   - `financial`
   - `formula_1`
   - `card_games`
   - `toxicology`
   - `superhero`
   - `european_football_2`

4. Add a semantic repair pass after SQL generation:
   - verify the selected answer column matches the question noun
   - verify the join path reaches the actual owner of each filtered field
   - verify whether deduplication is needed
   - verify denominator scope on percentage/rate questions
