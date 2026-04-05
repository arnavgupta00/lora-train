# Failure Archetypes Review

Databases reviewed:

- `california_schools`
- `financial`
- `formula_1`
- `thrombosis_prediction`
- `debit_card_specializing`
- `card_games`
- `toxicology`

Source files:

- Eval results: `per_example_results.jsonl`
- Dataset metadata: `bird_dev_t10.jsonl`

## How To Read This

Each failure class is written to be reproducible and operational:

- `Archetype pattern`: what the question is really asking the model to do
- `What the correct SQL must do`: the minimal reasoning steps needed to answer correctly
- `Typical model failure`: the recurring bad move seen in the failed predictions
- `Why it fails`: the root cause
- `Likely fix strategy`: what to target in prompting, training, or repair

The counts below are approximate class counts over failed examples. They are used to identify the dominant failure families, not to define exact taxonomy boundaries.

## Compact Matrix

| Database | Failure class | What these questions ask for | Root cause | Likely fix strategy |
|---|---|---|---|---|
| `california_schools` | Ranking and top-k school retrieval | Find the single school with the extreme value of a metric, then return a different attribute from that school | Metric and target field get swapped | Train on "rank by one field, return another field" patterns |
| `california_schools` | Derived rates and calculated education metrics | Compute a rate such as meal-count divided by enrollment, then rank/filter on that rate | Numerator returned instead of ratio; school-type filter dropped | Add formula-grounding examples with explicit numerator/denominator supervision |
| `california_schools` | Cross-table institutional lookup | Join `frpm` with `schools`, then apply district, charter, funding, and date filters exactly | County and district filters get conflated | Add schema notes that distinguish district vs county vs school attributes |
| `california_schools` | Aggregation over school subsets | Count schools satisfying score + school-property conditions | Wrong entity counted; irrelevant SAT filters injected | Add count-granularity checks in repair step |
| `financial` | Aggregation with region/account/client roles | Count accounts, clients, or districts after combining region, gender, salary, and account behavior | Filters attached to wrong table; dedup lost | Add examples emphasizing table ownership of domain codes like `A3`, `A11`, `frequency` |
| `financial` | Multi-hop account/client/disp lookup | Traverse account -> disposition -> client or account -> transaction correctly | Wrong bridge table chosen | Add chain-of-join demonstrations with role labels |
| `financial` | Derived percentages and increase rates | Compute ratios over the right population and the right date slices | Wrong denominator or wrong time anchor | Add templates for "percentage of subgroup within filtered cohort" |
| `financial` | Youngest/oldest plus salary composites | Combine age extremum, district salary, and account retrieval | Nested logic flattened into simple ranking | Train on nested-selection + return-field compositions |
| `formula_1` | Fastest/top driver ranking | Rank drivers/results by lap metric, then return nationality, number, or other metadata | Right ranking, wrong output column | Add examples where the ordered field and returned field differ |
| `formula_1` | Join-based metadata lookup | Use the correct F1 fact table among `qualifying`, `results`, `races`, `seasons`, `drivers` | Wrong table family selected | Add schema disambiguation for event phase tables |
| `formula_1` | Aggregation with geographic/finish filters | Count race or driver rows after combining race, circuit, and finish-state conditions | Joins removed; event filters simplified | Add decomposition prompts: identify tables before writing SQL |
| `formula_1` | Percentage and completion calculations | Compare events or compute completion rates over result sets | Direct row arithmetic replaces aggregate logic | Add repair heuristics for denominator construction |
| `thrombosis_prediction` | Date-aware patient aggregation | Compute age gaps, first visit, exam lag, and admission-conditioned counts | Wrong SQLite date logic; wrong time anchor | Add SQLite-specific temporal function exemplars |
| `thrombosis_prediction` | Patient/examination detail retrieval | Return the exact requested clinical field after patient-exam join | Diagnosis/symptom/date fields swapped | Add output-field verification against question nouns |
| `thrombosis_prediction` | Percentage and cohort calculations | Define a precise medical cohort, then aggregate lab/exam values over it | Cohort boundaries drift across tables | Add cohort extraction intermediate step in prompts |
| `thrombosis_prediction` | Distinct patient set extraction | Return unique patients satisfying multi-table medical constraints | Count vs list confusion; `DISTINCT` omitted | Add result-shape supervision: list IDs vs count IDs |
| `debit_card_specializing` | Segment-aware yearly ranking | Aggregate annual consumption by customer inside a segment/currency and choose top/bottom | Group-by dropped; segment filter dropped | Add "group then rank" examples over monthly fact tables |
| `debit_card_specializing` | Windowed consumption aggregation | Sum over bounded year-month windows or compare exact business counts | Date window rewritten loosely with `LIKE`; precedence errors | Add canonical year-month range templates |
| `debit_card_specializing` | Cross-segment comparative metrics | Compare SME/LAM/KAM averages or growth rates within a conditioned subset | Multi-output analytic question collapsed | Add structured reasoning over subgroup comparisons |
| `debit_card_specializing` | Transaction-history linkage | Use a transaction fact to identify customer, then retrieve history from another table | Wrong anchor table and wrong second-stage lookup | Add two-stage retrieval exemplars |
| `card_games` | Identifier-vs-name retrieval | Return `id`, not `name`, after card-level filtering | Output contract violated | Add explicit supervision for return column discipline |
| `card_games` | Distinct legalities retrieval | Join `cards` and `legalities`, keep format/status semantics straight, often with `DISTINCT` | Table ownership and dedup errors | Add table-role summaries for metadata schemas |
| `card_games` | Card metadata joins | Retrieve language, ruling text, or foreign text from the correct side table | Wrong text field selected | Add output-field verification for side-table retrieval |
| `card_games` | Aggregation over legal subsets | Count unique cards under legality + deck/text constraints | Distinctness and table-role errors | Add count-of-entity vs count-of-row training pairs |
| `toxicology` | Molecule counting with atom/bond filters | Count distinct molecules that contain atoms or bond types under carcinogenicity labels | Wrong counting unit | Add training that contrasts atom count vs molecule count |
| `toxicology` | Bond/connection graph lookup | Traverse `bond` and `connected` correctly to recover atom pairs or bond types | Wrong structural table chosen | Add graph-schema traversal exemplars |
| `toxicology` | Distinct molecule retrieval by structure | Return unique molecules or elements satisfying bond-structure constraints | Over-joined path returns wrong entity set | Add path-selection examples with minimal join sets |
| `toxicology` | Percentage composition calculations | Compute shares over atoms or molecules with the right denominator | Denominator and symbolic bond-type mistakes | Add percentage supervision with explicit denominator annotation |

## california_schools

Overall failure shape:

- Dominant errors: `wrong_result` >> `column_error`
- Main pattern: the model usually builds executable SQL, but it binds the wrong school metric, the wrong geographic field, or the wrong formula.

### Class 1: Ranking and top-k school retrieval

- Approx failed count: `25`
- Archetype pattern:
  - "Find the school with the highest or lowest value of a school-meal or SAT-related metric, then return another attribute of that exact school."
  - This is not "return the max metric"; it is "identify the row by ranking, then project a different field."
- Representative prompts:
  - "What is the unabbreviated mailing street address of the school with the highest FRPM count for K-12 students?"
  - "What is the number of SAT test takers of the schools with the highest FRPM count for K-12 students?"
- What the correct SQL must do:
  - Rank schools by the requested metric.
  - Keep the correct metric name, such as `FRPM Count (K-12)` vs `Free Meal Count (K-12)`.
  - Return a non-ranking field from the top-ranked row.
- Typical model failure:
  - Returns the maximum metric itself instead of the requested attribute.
  - Swaps `FRPM Count` and `Free Meal Count`.
  - Chooses the wrong address/test-taker column.
- Why it fails:
  - The model recognizes the extreme-value intent, but not the separation between ranking column and answer column.
- Likely fix strategy:
  - Add more supervision on "order by X, select Y" examples where `X` and `Y` are semantically related but different.

### Class 2: Derived rates and calculated education metrics

- Approx failed count: `9`
- Archetype pattern:
  - "Compute a school-level rate or ratio, often meal-count divided by enrollment, then rank or filter on that computed value."
- Representative prompts:
  - "What is the highest eligible free rate for K-12 students in the schools in Alameda County?"
  - "Please list the lowest three eligible free rates for students aged 5-17 in continuation schools."
- What the correct SQL must do:
  - Construct the correct numerator and denominator.
  - Preserve null checks when the denominator-derived metric can be null.
  - Keep school-type restrictions such as continuation school.
- Typical model failure:
  - Returns the numerator alone.
  - Keeps the sort order but drops the denominator or filtering clause.
  - Uses the wrong county string or school type filter.
- Why it fails:
  - The model maps "rate" to the relevant table region, but not to the explicit formula.
- Likely fix strategy:
  - Add prompt scaffolding that explicitly asks for "formula before SQL" on ratio questions.

### Class 3: Cross-table institutional lookup with precise filters

- Approx failed count: `13`
- Archetype pattern:
  - "Join school metadata with meal-program records and apply several precise institutional filters such as district, charter status, funding type, and open date."
- Representative prompts:
  - "Please list the zip code of all the charter schools in Fresno County Office of Education."
  - "Please list the phone numbers of the direct charter-funded schools that are opened after 2000/1/1."
- What the correct SQL must do:
  - Join the correct tables on `CDSCode`.
  - Respect district vs county vs school fields.
  - Keep all compound conditions, not just the most salient one.
- Typical model failure:
  - Replaces district-level scope with county-level scope.
  - Drops funding-type constraints.
  - Returns a semantically nearby field from the wrong table.
- Why it fails:
  - The schema has several overlapping geography and institution descriptors that invite near-miss grounding.
- Likely fix strategy:
  - Add schema-grounding notes that explicitly contrast county, district, and school-level columns.

### Class 4: Aggregation over school performance subsets

- Approx failed count: `13`
- Archetype pattern:
  - "Count schools after combining SAT-score thresholds with school properties such as virtual status or charter funding type."
- Representative prompts:
  - "How many schools with an average score in Math greater than 400 in the SAT test are exclusively virtual?"
  - "Among the schools with the average score in Math over 560 in the SAT test, how many schools are directly charter-funded?"
- What the correct SQL must do:
  - Join the SAT table to the right school/program metadata table.
  - Count the right entity at the right granularity.
  - Apply score filters and school-property filters together.
- Typical model failure:
  - Counts the wrong table key.
  - Injects irrelevant SAT-type filters.
  - Switches the requested property to a nearby but wrong one.
- Why it fails:
  - The model gets the broad join pattern right but loses exact schema ownership inside the count.
- Likely fix strategy:
  - Add repair rules that verify count target and filter ownership before execution.

## financial

Overall failure shape:

- Dominant errors: `wrong_result` with a clear secondary band of `column_error`
- Main pattern: the model knows the broad banking graph, but it repeatedly attaches domain codes and filters to the wrong table.

### Class 1: Aggregation with region, account, and client-role filters

- Approx failed count: `27`
- Archetype pattern:
  - "Count accounts, clients, or districts after combining a regional attribute, an account-behavior code, and sometimes demographic constraints."
- Representative prompts:
  - "How many accounts who choose issuance after transaction are staying in East Bohemia region?"
  - "List out the no. of districts that have female average salary is more than 6000 but less than 10000?"
- What the correct SQL must do:
  - Know that district-region fields live on `district`.
  - Know that issuance behavior like `frequency` lives on `account`.
  - Preserve distinctness when the count target is district or account rather than joined rows.
- Typical model failure:
  - Places `frequency` or region filters on the wrong table.
  - Counts raw joined rows instead of distinct business entities.
- Why it fails:
  - Banking-domain codes like `A3`, `A11`, and `frequency` are opaque and easy to misattach.
- Likely fix strategy:
  - Add schema annotations for domain-code columns and more count-vs-distinct training examples.

### Class 2: Multi-hop account, disposition, client, and transaction lookup

- Approx failed count: `25`
- Archetype pattern:
  - "Recover a client or account attribute through a multi-hop relationship, where several valid-looking paths exist."
- Representative prompts:
  - "List out the id number of client who choose statement of issuance after transaction are Disponent?"
  - "The transaction of 840 USD happened in 1998/10/14, when was this account opened?"
- What the correct SQL must do:
  - Choose the right bridge path, such as `account -> disp -> client`.
  - Use the transaction to identify the account, then project the account attribute.
- Typical model failure:
  - Anchors on `trans` too long and forgets to return to the account table.
  - Returns the transaction date rather than the account opening date.
- Why it fails:
  - The schema contains multiple fact tables with shared keys and role ambiguity.
- Likely fix strategy:
  - Encourage an intermediate reasoning step: "identify the anchor entity first, then project the answer."

### Class 3: Derived percentages and increase-rate computations

- Approx failed count: `13`
- Archetype pattern:
  - "Compute a percentage or growth rate over the correct cohort, often after selecting a specific branch, region, or approval date."
- Representative prompts:
  - "For the branch which located in the south Bohemia with biggest number of inhabitants, what is the percentage of the male clients?"
  - "For the client whose loan was approved first in 1993/7/5, what is the increase rate of his/her account balance from 1993/3/22 to 1998/12/27?"
- What the correct SQL must do:
  - Build the right numerator and denominator over the same cohort.
  - Slice the right dates and preserve the branch/client selection condition.
- Typical model failure:
  - Computes over the wrong population.
  - Replaces multi-date aggregation with direct row subtraction.
  - Loses the condition that defines the target client or branch.
- Why it fails:
  - The model recognizes "percentage" but not the full cohort-and-time definition.
- Likely fix strategy:
  - Add prompts that force the model to state numerator, denominator, and cohort before SQL.

### Class 4: Youngest/oldest plus salary/ranking composites

- Approx failed count: `11`
- Archetype pattern:
  - "Select the youngest or oldest client, connect them to district salary context, and then return an account-level answer."
- Representative prompts:
  - "List out the account numbers of female clients who are oldest and has lowest average salary, calculate the gap between this lowest average salary with the highest average salary?"
  - "List out the account numbers of clients who are youngest and have highest average salary?"
- What the correct SQL must do:
  - Resolve the age extremum first.
  - Preserve district salary context.
  - Return account IDs, not client IDs.
- Typical model failure:
  - Flattens the nested selection into a generic average-salary ranking.
  - Returns the wrong entity ID type.
- Why it fails:
  - This class mixes demographic extremum, business context, and answer-type discipline.
- Likely fix strategy:
  - Train more examples with nested entity selection followed by projection of a different linked entity.

## formula_1

Overall failure shape:

- Dominant errors: `wrong_result`, plus a large `column_error` block
- Main pattern: the model understands the racing semantics, but it repeatedly chooses the wrong F1 table or the wrong output field.

### Class 1: Fastest, earliest, or top driver ranking

- Approx failed count: `28`
- Archetype pattern:
  - "Order results by a lap or race-performance metric, then return a driver metadata field such as nationality or number."
- Representative prompts:
  - "For the driver who set the fastest lap speed in race No.933, where does he come from?"
  - "For the driver who set the fastest lap speed, what is his nationality?"
- What the correct SQL must do:
  - Rank results on the right metric.
  - Join to the drivers table.
  - Return the requested metadata field, not just any driver descriptor.
- Typical model failure:
  - Returns driver name instead of nationality.
  - Hallucinates a near-match column like `fasterLapSpeed`.
- Why it fails:
  - The model gets the ordering intent but not the exact target field.
- Likely fix strategy:
  - Add examples where ranking column and answer column differ but are both salient.

### Class 2: Join-based metadata lookup across race tables

- Approx failed count: `26`
- Archetype pattern:
  - "Use the correct event-phase table (`qualifying`, `results`, `races`, `seasons`) to recover a metadata field."
- Representative prompts:
  - "What is his number of the driver who finished 0:01:54 in the Q3 of qualifying race No.903?"
  - "Show me the season page of year when the race No. 901 took place."
- What the correct SQL must do:
  - Choose the correct F1 event table.
  - Preserve format-specific conditions like `q3` or season-year linkage.
  - Return the field from the right table.
- Typical model failure:
  - Uses `results` instead of `qualifying`.
  - Returns the race URL instead of the season URL.
- Why it fails:
  - Several F1 tables have overlapping identifiers but very different semantics.
- Likely fix strategy:
  - Add a schema cheat-sheet that distinguishes qualifying-phase, race-result, race, and season metadata.

### Class 3: Aggregation with geographic or finish-state filters

- Approx failed count: `25`
- Archetype pattern:
  - "Count races or drivers after combining race metadata, circuit geography, and completion-state conditions."
- Representative prompts:
  - "How many races in the year 2010 are held on grand prixs outside Asia and Europe?"
  - "For the Bahrain Grand Prix in 2007, how many drivers not finished the game?"
- What the correct SQL must do:
  - Join races to circuits when geography is involved.
  - Keep event-year and event-name constraints.
  - Respect finish-state logic such as `time IS NULL`.
- Typical model failure:
  - Removes the circuit join and invents geographic fields in `races`.
  - Drops one event constraint or finish-state condition.
- Why it fails:
  - The model simplifies to a single-table answer even when the schema requires multiple table families.
- Likely fix strategy:
  - Use prompting that explicitly asks: "Which table contains geography? Which table contains finish state?"

### Class 4: Percentage and completion-rate calculations

- Approx failed count: `16`
- Archetype pattern:
  - "Compute a cross-event percentage or completion rate using the right race slice and the right denominator."
- Representative prompts:
  - "Paul di Resta was in the No. 853 race, what percent faster did he finish in the 853rd race than the next race for the fastest lap speed?"
  - "For the drivers who took part in the race in 1983/7/16, what's their race completion rate?"
- What the correct SQL must do:
  - Select the right races or dates.
  - Aggregate the right result rows.
  - Compute numerator and denominator over the same participant set.
- Typical model failure:
  - Performs row-level subtraction instead of aggregate comparison.
  - Attaches race-date fields to the wrong table.
- Why it fails:
  - Temporal comparison and denominator construction are both unstable for the model here.
- Likely fix strategy:
  - Add SQL repairs that validate denominator population and race/event anchor before execution.

## thrombosis_prediction

Overall failure shape:

- Dominant errors: `wrong_result`, with several `column_error` and date-function issues
- Main pattern: the model keeps the medical nouns, but time logic, cohort logic, and clinical field selection drift.

### Class 1: Date-aware aggregation over patient history

- Approx failed count: `50`
- Archetype pattern:
  - "Compute age or elapsed time relative to first admission, first examination, or another clinically meaningful event."
- Representative prompts:
  - "What was the age of the youngest patient when they initially arrived at the hospital?"
  - "How many patients hadn't undergone a medical examination until at least a year following their initial hospital visit?"
- What the correct SQL must do:
  - Use SQLite-compatible date extraction.
  - Compute time difference from the right anchor date.
  - Keep admission-related filters when required.
- Typical model failure:
  - Uses non-SQLite date functions.
  - Computes age relative to the wrong event.
  - Drops the admission-status condition.
- Why it fails:
  - Medical questions rely heavily on temporal semantics that are easy to approximate incorrectly.
- Likely fix strategy:
  - Add SQLite temporal exemplars and explicit "anchor date" reasoning in prompts.

### Class 2: Patient and examination detail retrieval

- Approx failed count: `15`
- Archetype pattern:
  - "Join patient and examination data, then return the exact requested clinical details."
- Representative prompts:
  - "State the sex and birthday of patient ID '163109'. When was the examination taken and what symptom does the patient had."
  - "What are the symptoms observed by the youngest patient to ever did a medical examination? Identify their diagnosis."
- What the correct SQL must do:
  - Join the patient and exam tables on patient ID.
  - Return the exact requested fields.
  - Preserve youngest/oldest ordering where present.
- Typical model failure:
  - Returns diagnosis when the question asks for symptoms.
  - Drops one requested field or flips the age ordering.
- Why it fails:
  - The model recognizes the right patient row but not the exact output contract.
- Likely fix strategy:
  - Add a post-generation check that aligns requested nouns in the question with selected columns.

### Class 3: Percentage and cohort calculations across patient plus lab/exam tables

- Approx failed count: `22`
- Archetype pattern:
  - "Define a medical cohort first, then aggregate a lab or exam statistic over exactly that cohort."
- Representative prompts:
  - "What is the percentage of female patient had total protein not within the normal range?"
  - "For in-patient age 50 and above, what is their average anti-cardiolipin antibody (IgG) concentration?"
- What the correct SQL must do:
  - Keep patient attributes on the patient table.
  - Keep lab/exam values on the right fact table.
  - Apply age and admission filters before aggregation.
- Typical model failure:
  - Pushes demographic fields into the lab table.
  - Computes age from examination date rather than birthday.
  - Uses the wrong denominator cohort.
- Why it fails:
  - Cohort definition and measurement source are both multi-table, making "who is in the denominator" unstable.
- Likely fix strategy:
  - Train with intermediate cohort-formulation text before SQL decoding.

### Class 4: Distinct patient set extraction with medical filters

- Approx failed count: `14`
- Archetype pattern:
  - "Return the unique patient set satisfying one or more lab/exam conditions, sometimes along with a derived attribute like age."
- Representative prompts:
  - "State the ID and age of patient with positive degree of coagulation."
  - "How many patients who were examined between 1987/7/6 and 1996/1/31 had a GPT level greater than 30 and an ALB level less than 4? List them by their ID."
- What the correct SQL must do:
  - Distinguish between listing IDs and counting IDs.
  - Use `DISTINCT` when multiple exams or lab rows can duplicate the patient.
  - Compute age correctly.
- Typical model failure:
  - Counts where it should list.
  - Lists duplicated patient rows.
  - Miscomputes age.
- Why it fails:
  - The model does not reliably track requested result shape when duplicates are possible.
- Likely fix strategy:
  - Add explicit supervision on output modality: scalar count vs unique ID list vs ID-plus-attribute list.

## debit_card_specializing

Overall failure shape:

- Dominant errors: `wrong_result`
- Main pattern: business-analytics questions are simplified too aggressively; grouping, date windows, and segment/currency constraints get dropped.

### Class 1: Segment-aware yearly ranking

- Approx failed count: `12`
- Archetype pattern:
  - "Within a year and segment/currency slice, aggregate consumption by customer, then choose the highest or lowest customer."
- Representative prompts:
  - "In 2012, who had the least consumption in LAM?"
  - "Which customers, paying in CZK, consumed the most gas in 2011?"
- What the correct SQL must do:
  - Filter by year.
  - Filter by segment or currency.
  - Group by customer.
  - Rank by summed consumption.
- Typical model failure:
  - Drops segment or currency constraints.
  - Produces a filtered customer list without the aggregation-and-ranking step.
- Why it fails:
  - The model treats these as record retrieval instead of analytical grouping queries.
- Likely fix strategy:
  - Add exemplars of "group by customer over yearmonth, then order by sum."

### Class 2: Windowed consumption aggregation

- Approx failed count: `9`
- Archetype pattern:
  - "Sum or compare quantities over a bounded year-month range or exact business category window."
- Representative prompts:
  - "How much did customer 6 consume in total between August and November 2013?"
  - "How many more 'discount' gas stations does the Czech Republic have compared to Slovakia?"
- What the correct SQL must do:
  - Use exact year-month bounds.
  - Keep arithmetic comparison structure intact.
  - Preserve category filters such as `Segment = 'Discount'`.
- Typical model failure:
  - Rewrites the bounded range as loose `LIKE` clauses.
  - Gets `AND`/`OR` precedence wrong.
  - Drops the business category predicate.
- Why it fails:
  - The model chooses lexical date matching over robust range logic.
- Likely fix strategy:
  - Add year-month range templates and repair checks for operator precedence.

### Class 3: Cross-segment comparative metrics and percentage deltas

- Approx failed count: `8`
- Archetype pattern:
  - "Compare SME, LAM, and KAM on annual average consumption or year-over-year percentage change, often for a special subset."
- Representative prompts:
  - "What is the difference in the annual average consumption of the customers with the least amount of consumption paid in CZK for 2013 between SME and LAM, LAM and KAM, and KAM and SME?"
  - "Which of the three segments—SME, LAM and KAM—has the biggest and lowest percentage increases in consumption paid in EUR between 2012 and 2013?"
- What the correct SQL must do:
  - Define the subset first.
  - Compute per-segment metrics.
  - Return multiple comparative outputs.
- Typical model failure:
  - Collapses the query into plain sums.
  - Ignores the conditioning subset.
  - Replaces multi-output comparison with a binary or unrelated answer.
- Why it fails:
  - This is the most compositional class here: subgrouping, date windows, conditioning, and comparison all interact.
- Likely fix strategy:
  - Train more multi-output analytical SQL with explicit subgroup metric decomposition.

### Class 4: Transaction-history linkage

- Approx failed count: `4`
- Archetype pattern:
  - "Use a transaction event to identify a customer, then retrieve some historical consumption or segment information from another table."
- Representative prompts:
  - "What segment did the customer have at 2012/8/23 21:20:00?"
  - "For the customer who paid 124.05 in 2012/8/24, how much did he/she spend during the January of 2012?"
- What the correct SQL must do:
  - Anchor on the transaction fact table.
  - Resolve the customer.
  - Use that customer to retrieve the second-stage answer from the right history table.
- Typical model failure:
  - Starts from the wrong fact table.
  - Mixes transaction date/amount with monthly-consumption date/value.
- Why it fails:
  - The query is two-stage, but the model tries to answer it in one pass from one table family.
- Likely fix strategy:
  - Add retrieval chains in training data that explicitly separate "identify entity" from "lookup target information."

## card_games

Overall failure shape:

- Dominant errors: `wrong_result`, then `column_error`
- Main pattern: the model knows the right card universe, but it often returns the wrong identifier or the wrong text field from the wrong side table.

### Class 1: Identifier-vs-name retrieval

- Approx failed count: `22`
- Archetype pattern:
  - "Filter a subset of cards correctly, but return `id` rather than `name`."
- Representative prompts:
  - "Which are the cards that have incredibly powerful foils."
  - "What are the borderless cards available without powerful foils?"
- What the correct SQL must do:
  - Stay on the `cards` table if no other metadata is needed.
  - Preserve the exact return field requested by the question.
- Typical model failure:
  - Returns `name` because it is more natural in language than `id`.
  - Adds unnecessary joins that change the semantics.
- Why it fails:
  - Output-field discipline is weak when the question names an entity but wants a database identifier.
- Likely fix strategy:
  - Add more "natural-language entity, database-ID answer" supervision.

### Class 2: Distinct legalities and format-based retrieval

- Approx failed count: `21`
- Archetype pattern:
  - "Join `cards` with `legalities`, apply format/status restrictions, and often deduplicate results."
- Representative prompts:
  - "List all the mythic rarity print cards banned in gladiator format."
  - "For artifact type of cards that do not have multiple faces on the same card, state its legalities status for vintage play format."
- What the correct SQL must do:
  - Keep format and legal status on the legalities table.
  - Keep card attributes such as rarity/type/side on the cards table.
  - Use `DISTINCT` where multiple legalities rows can duplicate results.
- Typical model failure:
  - Moves format or type to the wrong table.
  - Returns the wrong output field, such as `name` instead of `status`.
- Why it fails:
  - This schema has many side tables with overlapping card semantics.
- Likely fix strategy:
  - Add table-role descriptions and explicit dedup examples for metadata joins.

### Class 3: Card metadata joins to foreign data or rulings

- Approx failed count: `21`
- Archetype pattern:
  - "Find the card row, then retrieve a specific text or metadata attribute from a linked side table."
- Representative prompts:
  - "State the alternative languages available for card named Annul numbered 29."
  - "What is the description about the ruling of card 'Condemn'?"
- What the correct SQL must do:
  - Resolve the exact card instance.
  - Join to the correct side table.
  - Return the right side-table field such as language or ruling text.
- Typical model failure:
  - Returns foreign card name instead of language.
  - Returns card text instead of ruling text.
  - Drifts to a different card instance.
- Why it fails:
  - Several text-like columns exist, and the model overgeneralizes "description" to the wrong one.
- Likely fix strategy:
  - Add examples that contrast `cards.text`, `rulings.text`, and foreign-data fields explicitly.

### Class 4: Aggregation over legal subsets

- Approx failed count: `21`
- Archetype pattern:
  - "Count unique cards satisfying legality status plus another card property like textlessness or starter-deck membership."
- Representative prompts:
  - "How many cards of legalities whose status is restricted have text boxes?"
  - "How many cards of legalities whose status is restricted are found in a starter deck?"
- What the correct SQL must do:
  - Join cards and legalities.
  - Filter status correctly.
  - Count unique cards, not joined rows.
- Typical model failure:
  - Uses plain `COUNT` instead of `COUNT(DISTINCT ...)`.
  - Swaps table ownership of deck/text properties and status.
- Why it fails:
  - The model gets the subset mostly right but not the unit of counting.
- Likely fix strategy:
  - Add contrastive examples that show why joined-row counts overcount cards.

## toxicology

Overall failure shape:

- Dominant errors: `wrong_result`, with a notable `column_error` band
- Main pattern: the model understands the chemistry concept, but it maps to the wrong graph path or the wrong counting granularity.

### Class 1: Molecule counting with atom or bond membership filters

- Approx failed count: `19`
- Archetype pattern:
  - "Count how many molecules satisfy a structural condition like containing chlorine or sodium, often under a carcinogenicity label."
- Representative prompts:
  - "In the non-carcinogenic molecules, how many contain chlorine atoms?"
  - "In the molecule containing sodium atoms, how many are non-carcinogenic?"
- What the correct SQL must do:
  - Use the atom table to identify candidate molecules.
  - Join to the molecule table for the label.
  - Count distinct molecules, not atom rows.
- Typical model failure:
  - Counts joined rows rather than distinct molecules.
- Why it fails:
  - The model confuses "molecules that contain X" with "rows about X."
- Likely fix strategy:
  - Add training pairs that contrast atom-level counts and molecule-level counts on the same schema.

### Class 2: Bond and connection graph lookup

- Approx failed count: `17`
- Archetype pattern:
  - "Recover which atoms are connected by a bond or what bond type links a given atom pair."
- Representative prompts:
  - "What atoms are connected in single type bonds?"
  - "What type of bond is there between the atoms TR004_8 and TR004_20?"
- What the correct SQL must do:
  - Use `bond` for bond type.
  - Use `connected` for the atom pair relation.
  - Handle both pair directions when needed.
- Typical model failure:
  - Queries the wrong structural table.
  - Returns one atom ID instead of an atom pair.
  - Misses the bidirectional condition.
- Why it fails:
  - The graph schema requires a specific traversal order that the model often overcomplicates.
- Likely fix strategy:
  - Add graph-traversal exemplars that explicitly name which table represents edges vs edge labels.

### Class 3: Distinct molecule retrieval by structure

- Approx failed count: `17`
- Archetype pattern:
  - "Return unique molecules or elements satisfying a bond-type or carcinogenicity structure condition."
- Representative prompts:
  - "Find the triple-bonded molecules which are carcinogenic."
  - "What elements are in a double type bond?"
- What the correct SQL must do:
  - Choose the minimal join path needed.
  - Return the requested entity type, either molecule IDs or elements.
  - Deduplicate when many graph edges point to the same entity.
- Typical model failure:
  - Adds unnecessary joins via `connected`.
  - Returns the wrong entity type.
- Why it fails:
  - The model over-traverses the graph and changes the answer space.
- Likely fix strategy:
  - Add path-selection examples that compare minimal and over-joined SQL on the same question.

### Class 4: Percentage composition and share calculations

- Approx failed count: `12`
- Archetype pattern:
  - "Compute the percentage of atoms or molecules meeting a structural property, with careful denominator choice."
- Representative prompts:
  - "What is the percentage of carbon in double-bond molecules?"
  - "What percentage of carcinogenic-type molecules does not contain fluorine?"
- What the correct SQL must do:
  - Choose the denominator at the right level: atoms or molecules.
  - Keep symbolic bond types exact, such as `=` and `-`.
  - Use distinct counts when required.
- Typical model failure:
  - Counts bonds when it should count atoms.
  - Counts atoms when it should count molecules.
  - Rewrites symbolic bond types into lexical descriptions that do not match the schema.
- Why it fails:
  - The model does not reliably track the level of measurement.
- Likely fix strategy:
  - Add denominator annotation during training, such as "count distinct molecules" vs "count atom rows."

## Cross-Database Takeaways

The seven databases fail in different surface forms, but the recurring root causes are consistent:

1. Wrong-but-executable SQL
   - Most failures are `wrong_result`, not syntax failure.
   - The model usually keeps the rough query skeleton and misses one critical semantic choice.

2. Schema ownership confusion
   - The most common failure across `california_schools`, `financial`, `formula_1`, and `card_games`.
   - The model knows the relevant concepts, but not which table owns them.

3. Count granularity and deduplication mistakes
   - Especially strong in `financial`, `card_games`, and `toxicology`.
   - The model often counts joined rows instead of the intended entity.

4. Denominator and cohort-definition instability
   - Strong in `financial`, `thrombosis_prediction`, `debit_card_specializing`, and `toxicology`.
   - Percentage questions break when the model does not first define the exact cohort.

5. Output-field mismatch
   - Strong in `formula_1`, `card_games`, and `california_schools`.
   - The model often returns a semantically nearby field rather than the requested one.

## Most Actionable Fixes

If you want the highest-yield improvements for these seven databases, the best targets are:

1. Add a pre-SQL reasoning scaffold:
   - target entity
   - source tables
   - return column
   - numerator
   - denominator
   - distinctness requirement

2. Add contrastive training examples where:
   - ranking column differs from returned column
   - row count differs from distinct entity count
   - patient/account/customer lookup is two-stage
   - percentage questions require explicit cohort selection

3. Add schema-grounding notes for the most failure-prone databases:
   - `district vs account vs client` in `financial`
   - `races vs results vs qualifying vs seasons` in `formula_1`
   - `cards vs legalities vs foreign_data vs rulings` in `card_games`
   - `bond vs connected vs atom vs molecule` in `toxicology`

4. Add a repair pass that checks:
   - whether the selected column matches the asked-for noun
   - whether `DISTINCT` is needed
   - whether the denominator matches the intended population
   - whether the query uses the table that actually owns the referenced field
